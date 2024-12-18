import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_max, scatter_sum
import ase

from simgen.calculators import MaceSimilarityCalculator

from .model import MolDiff
from .diffusion import log_sample_categorical


def to_ase_single(node_types, pos_prev, featurizer):
    numbers = [featurizer.nodetype_to_ele[x.item()] for x in node_types]
    pos = pos_prev.detach().cpu().numpy()
    return ase.Atoms(numbers=numbers, positions=pos, cell=None)


def to_ase(node_types, pos_prev, batch_idx, featurizer):
    atoms = []
    mol_indices = torch.unique(batch_idx)
    for mol_idx in mol_indices:
        mask = batch_idx == mol_idx
        node_type = node_types[mask]
        pos = pos_prev[mask]
        atoms.append(to_ase_single(node_type, pos, featurizer))
    return atoms


def prepare_atoms(
    logits: torch.Tensor,
    pos: torch.Tensor,
    batch_idx: torch.Tensor,
    featurizer,
    num_replicas: int = 1,
):
    if num_replicas > 1:
        num_graphs = batch_idx.max().item() + 1
        logits_rep = logits.repeat(num_replicas, 1)
        pos_rep = pos.repeat(num_replicas, 1)
        batch_idx_rep = torch.cat(
            [batch_idx + i * num_graphs for i in range(num_replicas)]
        )
        node_types = log_sample_categorical(logits_rep)
        atoms = to_ase(node_types, pos_rep, batch_idx_rep, featurizer)
        graph_indices = torch.arange(num_graphs, device=batch_idx.device).repeat(
            num_replicas
        )
        return atoms, node_types, graph_indices, batch_idx_rep
    else:
        node_types = log_sample_categorical(logits)
        atoms = to_ase(node_types, pos, batch_idx, featurizer)
        return atoms, None, None, batch_idx


def _simgen_guidance(
    simgen_calc,
    featurizer,
    logits,
    pos_prev,
    batch_idx,
    gui_scale,
    noise_level,
    use_importance_sampling=False,
    num_replicas=1,
):
    mean_moldiff_step_size = torch.norm((pos_prev - pos_prev).detach(), dim=-1).mean()
    with torch.enable_grad():
        assert featurizer is not None, "featurizer required"
        assert simgen_calc is not None, "simgen_calc required"
        num_replicas = num_replicas if use_importance_sampling else 1
        atoms, node_types_repeated, graph_indices, batch_idx_sum = prepare_atoms(
            logits, pos_prev, batch_idx, featurizer, num_replicas
        )
        simgen_batch = simgen_calc.batch_atoms(atoms)
        embeddings = simgen_calc._get_node_embeddings(simgen_batch)
        log_dens = simgen_calc._calculate_log_k(
            embeddings, simgen_batch.node_attrs, noise_level
        )
        log_dens = scatter_sum(log_dens, batch_idx_sum, dim=0)
        if use_importance_sampling:
            assert graph_indices is not None
            assert isinstance(node_types_repeated, torch.Tensor)
            log_dens, argmax = scatter_max(log_dens, graph_indices, dim=0)
            force_mask = torch.cat([torch.where(batch_idx_sum==x)[0] for x in argmax])
        else:
            force_mask = torch.arange(len(batch_idx_sum), device=batch_idx.device)
        simgen_force = simgen_calc._get_gradient(simgen_batch.positions, log_dens)
        simgen_force = simgen_force[force_mask]
        force_norms = (simgen_force**2).sum(dim=-1).sqrt()
        force_norms = scatter_max(force_norms, batch_idx, dim=0)[0][batch_idx]
        simgen_scale = mean_moldiff_step_size * gui_scale
        mult = torch.where(
            force_norms > simgen_scale,
            simgen_scale / force_norms,
            torch.ones_like(force_norms),
        )
        delta = simgen_force * mult[:, None]
    return delta


class GuidedMolDiff(MolDiff):
    @torch.no_grad()
    def sample(
        self,
        n_graphs: int,
        batch_node: torch.Tensor,
        halfedge_index: torch.Tensor,
        batch_halfedge: torch.Tensor,
        bond_predictor: None | nn.Module = None,
        featurizer=None,
        simgen_calc: MaceSimilarityCalculator | None = None,
        simgen_gui_scale=0.0,  # additional param for simgen guidance scale
        use_simgen_importance_sampling=False,  # remove this and just check num replicas
        num_replicates=1,  # additional param for simgen importance sampling
        bond_gui_scale=0.0,  # additional param for bond guidance scale
    ):
        device = batch_node.device
        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)

        node_init = self.node_transition.sample_init(n_nodes_all)
        pos_init = self.pos_transition.sample_init([n_nodes_all, 3])
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
        if self.categorical_space == "discrete":
            _, h_node_init, log_node_type = node_init
            _, h_halfedge_init, log_halfedge_type = halfedge_init
        else:
            h_node_init = node_init
            h_halfedge_init = halfedge_init

        node_traj = torch.zeros(
            [self.num_timesteps + 1, n_nodes_all, h_node_init.shape[-1]],
            dtype=h_node_init.dtype,
            device=device,
        )
        pos_traj = torch.zeros(
            [self.num_timesteps + 1, n_nodes_all, 3],
            dtype=pos_init.dtype,
            device=device,
        )
        halfedge_traj = torch.zeros(
            [self.num_timesteps + 1, n_halfedges_all, h_halfedge_init.shape[-1]],
            dtype=h_halfedge_init.dtype,
            device=device,
        )
        node_traj[0] = h_node_init
        pos_traj[0] = pos_init
        halfedge_traj[0] = h_halfedge_init

        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        for i, step in tqdm(
            enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps
        ):
            time_step = torch.full((n_graphs,), step, dtype=torch.long, device=device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            preds = self(
                h_node_pert,
                pos_pert,
                batch_node,
                h_edge_pert,
                edge_index,
                batch_edge,
                time_step,
            )
            pred_node = preds["pred_node"]
            pred_pos = preds["pred_pos"]
            pred_halfedge = preds["pred_halfedge"]

            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=batch_node
            )

            if self.categorical_space == "discrete":
                log_node_recon = F.log_softmax(pred_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(
                    log_node_recon, log_node_type, time_step, batch_node, v0_prob=True
                )
                node_type_prev = log_sample_categorical(log_node_type)
                h_node_prev = self.node_transition.onehot_encode(node_type_prev)

                log_edge_recon = F.log_softmax(pred_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(
                    log_edge_recon,
                    log_halfedge_type,
                    time_step,
                    batch_halfedge,
                    v0_prob=True,
                )
                halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                h_halfedge_prev = self.edge_transition.onehot_encode(halfedge_type_prev)
            else:
                h_node_prev = self.node_transition.get_prev_from_recon(
                    x_t=h_node_pert, x_recon=pred_node, t=time_step, batch=batch_node
                )
                h_halfedge_prev = self.edge_transition.get_prev_from_recon(
                    x_t=h_halfedge_pert,
                    x_recon=pred_halfedge,
                    t=time_step,
                    batch=batch_halfedge,
                )

            # SiMGen guidance
            if simgen_calc is not None and simgen_gui_scale > 0.0:
                delta = _simgen_guidance(
                    simgen_calc,
                    featurizer,
                    pred_node,
                    pos_prev,
                    batch_node,
                    simgen_gui_scale,
                    i/self.num_timesteps+1e-3,
                    use_simgen_importance_sampling,
                    num_replicates,
                )
                pos_prev = pos_prev + delta

            # Bond guidance
            if bond_predictor is not None and bond_gui_scale > 0.0:
                with torch.enable_grad():
                    h_node_in = h_node_pert.detach()
                    pos_in = pos_pert.detach().requires_grad_(True)
                    pred_bond = bond_predictor(
                        h_node_in,
                        pos_in,
                        batch_node,
                        edge_index,
                        batch_edge,
                        time_step,
                    )
                    # Use a simple metric: increase probability of valid bonds
                    uncertainty = torch.sigmoid(-torch.logsumexp(pred_bond, dim=-1))
                    uncertainty = uncertainty.log().sum()
                    delta_bond = (
                        -torch.autograd.grad(uncertainty, pos_in)[0] * bond_gui_scale
                    )
                pos_prev = pos_prev + delta_bond

            node_traj[i + 1] = h_node_prev
            pos_traj[i + 1] = pos_prev
            halfedge_traj[i + 1] = h_halfedge_prev

            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev

        return {
            "pred": [pred_node, pred_pos, pred_halfedge],
            "traj": [node_traj, pos_traj, halfedge_traj],
        }
