import enum
import logging
from dataclasses import dataclass
from typing import Mapping

import ase
import torch
import torch.nn.functional as F
from simgen.calculators import MaceSimilarityCalculator
from torch import nn
from torch_scatter import scatter_logsumexp, scatter_max, scatter_sum
from tqdm import tqdm

from .diffusion import log_sample_categorical
from .model import MolDiff


class ScaleMode(enum.Enum):
    FRACTIONAL = "fractional"
    ABSOLUTE = "absolute"


class NoiseSchedule(enum.Enum):
    CONSTANT = "constant"
    MATCHING = "matching"


@dataclass
class SiMGenGuidanceParams:
    sim_calc: MaceSimilarityCalculator
    node_to_element_map: Mapping[float | int, int]
    simgen_scale_mode: ScaleMode = ScaleMode.FRACTIONAL
    simgen_gui_scale: float = 0.2
    min_gui_scale: float = 0.05
    sigma_schedule_type: NoiseSchedule = NoiseSchedule.CONSTANT
    constant_sigma_value: float = 1.0


@dataclass
class ImportanceSamplingConfig:
    frequency: int = 1
    inverse_temp: float = 1.0
    mini_batch: int = -1


@dataclass
class GenerationState:
    pos: torch.Tensor
    h_node: torch.Tensor
    h_halfedge: torch.Tensor
    batch_node: torch.Tensor
    batch_halfedge: torch.Tensor
    batch_edge: torch.Tensor
    halfedge_index: torch.Tensor
    edge_index: torch.Tensor
    n_graphs: int


def to_ase_single(
    node_types: torch.Tensor,
    pos_prev: torch.Tensor,
    node_to_element_map: Mapping[int | float, int],
) -> ase.Atoms:
    numbers = [node_to_element_map[x.item()] for x in node_types]
    pos = pos_prev.detach().cpu().numpy()
    return ase.Atoms(numbers=numbers, positions=pos, cell=None)


def to_ase(
    node_types: torch.Tensor,
    pos_prev: torch.Tensor,
    batch_idx: torch.Tensor,
    node_to_element_map: Mapping[int | float, int],
) -> list[ase.Atoms]:
    atoms = []
    mol_indices = torch.unique(batch_idx)
    for mol_idx in mol_indices:
        mask = batch_idx == mol_idx
        node_type = node_types[mask]
        pos = pos_prev[mask]
        atoms.append(to_ase_single(node_type, pos, node_to_element_map))
    return atoms


def prepare_atoms(
    logits: torch.Tensor,
    pos: torch.Tensor,
    batch_idx: torch.Tensor,
    node_to_element_map: Mapping[int | float, int],
) -> list[ase.Atoms]:
    node_types = log_sample_categorical(logits)
    atoms = to_ase(node_types, pos, batch_idx, node_to_element_map)
    return atoms


def _simgen_guidance(
    simgen_calc: MaceSimilarityCalculator,
    node_to_element_map: Mapping[int | float, int],
    logits: torch.Tensor,
    pos_prev: torch.Tensor,
    batch_idx: torch.Tensor,
    gui_scale: float | torch.Tensor,
    noise_level: float,
) -> torch.Tensor:
    with torch.enable_grad():
        assert simgen_calc is not None, "simgen_calc required"
        atoms = prepare_atoms(logits, pos_prev, batch_idx, node_to_element_map)
        simgen_batch = simgen_calc.batch_atoms(atoms)
        embeddings = simgen_calc._get_node_embeddings(simgen_batch)
        log_dens = simgen_calc._calculate_log_k(
            embeddings, simgen_batch.node_attrs, noise_level
        )
        log_dens = scatter_sum(log_dens, batch_idx, dim=0)
        simgen_force = simgen_calc._get_gradient(simgen_batch.positions, log_dens)

        force_norms = (simgen_force**2).sum(dim=-1).sqrt()
        force_norms = scatter_max(force_norms, batch_idx, dim=0)[0][batch_idx]

        mult = torch.where(
            force_norms > gui_scale,
            gui_scale / force_norms,
            torch.ones_like(force_norms),
        )
        delta = simgen_force * mult[:, None]
    return delta


def _simgen_guidance_inverse_sum_order(
    simgen_calc: MaceSimilarityCalculator,
    node_to_element_map: Mapping[int | float, int],
    logits: torch.Tensor,
    pos_prev: torch.Tensor,
    batch_idx: torch.Tensor,
    gui_scale: float | torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        assert simgen_calc is not None, "simgen_calc required"
        atoms = prepare_atoms(logits, pos_prev, batch_idx, node_to_element_map)
        simgen_batch = simgen_calc.batch_atoms(atoms)
        embeddings = simgen_calc._get_node_embeddings(simgen_batch)

        squared_distance_matrix = simgen_calc._calculate_distance_matrix(
            embeddings, simgen_batch.node_attrs
        )
        additional_multiplier = (
            119 * (1 - (noise_level / 10) ** 0.25) + 1 if noise_level <= 10 else 1
        )
        squared_distance_matrix = (
            squared_distance_matrix * additional_multiplier
        )  # (N_config_atoms, N_ref_atoms)
        log_dens = scatter_logsumexp(
            -squared_distance_matrix / 2, batch_idx, dim=0
        )  # (N_graphs, N_ref_atoms)
        log_dens = log_dens.sum(dim=-1)  # (N_graphs,)
        simgen_force = simgen_calc._get_gradient(simgen_batch.positions, log_dens)

        force_norms = (simgen_force**2).sum(dim=-1).sqrt()
        force_norms = scatter_max(force_norms, batch_idx, dim=0)[0][batch_idx]

        mult = torch.where(
            force_norms > gui_scale,
            gui_scale / force_norms,
            torch.ones_like(force_norms),
        )
        delta = simgen_force * mult[:, None]
    return delta, log_dens


class GuidedMolDiff(MolDiff):
    @torch.no_grad()
    def sample(
        self,
        n_graphs: int,
        batch_node: torch.Tensor,
        halfedge_index: torch.Tensor,
        batch_halfedge: torch.Tensor,
        bond_predictor: None | nn.Module = None,
        bond_gui_scale: float = 0.0,
        simgen_guidance: SiMGenGuidanceParams | None = None,
        importance_sampling_params: ImportanceSamplingConfig | None = None,
    ) -> dict[str, list[torch.Tensor]]:
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

        state = GenerationState(
            pos=pos_pert,
            h_node=h_node_pert,
            h_halfedge=h_halfedge_pert,
            batch_node=batch_node,
            batch_halfedge=batch_halfedge,
            batch_edge=batch_edge,
            halfedge_index=halfedge_index,
            edge_index=edge_index,
            n_graphs=n_graphs,
        )

        for i, step in tqdm(
            enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps
        ):
            time_step = torch.full((n_graphs,), step, dtype=torch.long, device=device)
            h_edge_pert = torch.cat([state.h_halfedge, state.h_halfedge], dim=0)
            preds = self(
                state.h_node,
                state.pos,
                state.batch_node,
                h_edge_pert,
                state.edge_index,
                state.batch_edge,
                time_step,
            )
            pred_node = preds["pred_node"]
            pred_pos = preds["pred_pos"]
            pred_halfedge = preds["pred_halfedge"]

            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=state.pos, x_recon=pred_pos, t=time_step, batch=state.batch_node
            )

            if self.categorical_space == "discrete":
                log_node_recon = F.log_softmax(pred_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(
                    log_node_recon,
                    log_node_type,
                    time_step,
                    state.batch_node,
                    v0_prob=True,
                )
                node_type_prev = log_sample_categorical(log_node_type)
                h_node_prev = self.node_transition.onehot_encode(node_type_prev)

                log_edge_recon = F.log_softmax(pred_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(
                    log_edge_recon,
                    log_halfedge_type,
                    time_step,
                    state.batch_halfedge,
                    v0_prob=True,
                )
                halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                h_halfedge_prev = self.edge_transition.onehot_encode(halfedge_type_prev)
            else:
                h_node_prev = self.node_transition.get_prev_from_recon(
                    x_t=state.h_node,
                    x_recon=pred_node,
                    t=time_step,
                    batch=state.batch_node,
                )
                h_halfedge_prev = self.edge_transition.get_prev_from_recon(
                    x_t=state.h_halfedge,
                    x_recon=pred_halfedge,
                    t=time_step,
                    batch=state.batch_halfedge,
                )

            pos_delta = 0.0
            if simgen_guidance is not None and simgen_guidance.simgen_gui_scale > 0.0:
                scale_mode = simgen_guidance.simgen_scale_mode
                sim_gui_scale = simgen_guidance.simgen_gui_scale
                if scale_mode == ScaleMode.FRACTIONAL:
                    avg_moldiff_force = (
                        torch.norm(pos_prev - state.pos, dim=-1).mean().item()
                    )
                    gui_scale = max(
                        avg_moldiff_force * sim_gui_scale, simgen_guidance.min_gui_scale
                    )
                elif scale_mode == ScaleMode.ABSOLUTE:
                    gui_scale = sim_gui_scale
                else:
                    raise ValueError(f"Invalid scale mode: {scale_mode}")

                if simgen_guidance.sigma_schedule_type == NoiseSchedule.CONSTANT:
                    noise_level = simgen_guidance.constant_sigma_value
                elif simgen_guidance.sigma_schedule_type == NoiseSchedule.MATCHING:
                    noise_level = step / self.num_timesteps + 1e-3
                else:
                    raise ValueError(
                        f"Invalid noise schedule: {simgen_guidance.sigma_schedule_type}"
                    )

                simgen_delta, log_dens = _simgen_guidance_inverse_sum_order(
                    simgen_guidance.sim_calc,
                    simgen_guidance.node_to_element_map,
                    pred_node[:, :-1],  # ignore the absorbing node type
                    state.pos,  # CHECK: pos_prev or pred_pos. Pred_pos is look-ahead guidance
                    state.batch_node,
                    gui_scale,
                    noise_level=noise_level,
                )
                if importance_sampling_params is not None:
                    state, simgen_delta = self._importance_sampling(
                        state=state,
                        log_density=log_dens,
                        simgen_forces=simgen_delta,
                        step=i,
                        cfg=importance_sampling_params,
                    )
                pos_delta += simgen_delta

            # Bond guidance
            if bond_predictor is not None and bond_gui_scale > 0.0:
                with torch.enable_grad():
                    h_node_in = state.h_node.detach()
                    pos_in = state.pos.detach().requires_grad_(True)
                    pred_bond = bond_predictor(
                        h_node_in,
                        pos_in,
                        state.batch_node,
                        state.edge_index,
                        state.batch_edge,
                        time_step,
                    )
                    uncertainty = torch.sigmoid(-torch.logsumexp(pred_bond, dim=-1))
                    uncertainty = uncertainty.log().sum()
                    delta_bond = (
                        -torch.autograd.grad(uncertainty, pos_in)[0] * bond_gui_scale
                    )
                pos_delta += delta_bond

            state.pos = pos_prev + pos_delta
            state.h_node = h_node_prev
            state.h_halfedge = h_halfedge_prev

            node_traj[i + 1] = state.h_node
            pos_traj[i + 1] = state.pos
            halfedge_traj[i + 1] = state.h_halfedge

        return {
            "pred": [pred_node, pred_pos, pred_halfedge],
            "traj": [node_traj, pos_traj, halfedge_traj],
        }

    @staticmethod
    @torch.no_grad()
    def _importance_sampling(
        state: GenerationState,
        log_density: torch.Tensor,
        simgen_forces: torch.Tensor,
        step: int,
        cfg: ImportanceSamplingConfig,
    ) -> tuple[GenerationState, torch.Tensor]:
        """
        Performs importance sampling on the generation state if the step matches the
        specified sampling frequency, otherwise returns the original state unmodified.
        """
        # Only do importance sampling if step > 0 and we hit the specified frequency
        if step <= 0 or (step % cfg.frequency != 0):
            return state, simgen_forces

        logging.info(f"Importance sampling at step {step}")
        logging.info(f"Log densities: {log_density}")

        # If mini-batch > 1, treat log densities in grouped fashion
        if cfg.mini_batch > 1:
            assert state.n_graphs % cfg.mini_batch == 0
            log_dens_mini = log_density.view(-1, cfg.mini_batch)
            weights = F.softmax(
                log_dens_mini * cfg.inverse_temp, dim=1
            )
            selected_batches = torch.cat(
                [
                    torch.multinomial(
                        w, cfg.mini_batch, replacement=True
                    )
                    + grp_i * cfg.mini_batch
                    for grp_i, w in enumerate(weights)
                ]
            )
        else:
            weights = F.softmax(
                log_density * cfg.inverse_temp, dim=0
            )
            selected_batches = torch.multinomial(
                weights, state.n_graphs, replacement=True
            )

        logging.info(f"Weights: {weights}")
        logging.info(f"Selected batches: {selected_batches}")

        selected_nodes = torch.cat(
            [torch.where(state.batch_node == x)[0] for x in selected_batches]
        )
        selected_halfedges = torch.cat(
            [torch.where(state.batch_halfedge == x)[0] for x in selected_batches]
        )

        pos = state.pos[selected_nodes]
        simgen_forces = simgen_forces[selected_nodes]
        h_node = state.h_node[selected_nodes]
        h_halfedge = state.h_halfedge[selected_halfedges]

        _, nodes_per_batch = torch.unique(state.batch_node, return_counts=True)
        batch_node = torch.repeat_interleave(
            torch.arange(state.n_graphs, device=pos.device),
            nodes_per_batch[selected_batches],
        )

        _, edge_counts = torch.unique(state.batch_halfedge, return_counts=True)
        batch_halfedge = torch.repeat_interleave(
            torch.arange(state.n_graphs, device=pos.device),
            edge_counts[selected_batches],
        )

        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        halfedge_index_list = []
        idx_start = 0
        for _, n_nodes in enumerate(nodes_per_batch[selected_batches]):
            halfedge_index_this_mol = torch.triu_indices(
                n_nodes, n_nodes, offset=1, device=pos.device
            )
            halfedge_index_list.append(halfedge_index_this_mol + idx_start)
            idx_start += n_nodes

        halfedge_index = torch.cat(halfedge_index_list, dim=1).to(pos.device)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1).to(
            pos.device
        )

        return (
            GenerationState(
                pos=pos,
                h_node=h_node,
                h_halfedge=h_halfedge,
                batch_node=batch_node,
                batch_halfedge=batch_halfedge,
                batch_edge=batch_edge,
                halfedge_index=halfedge_index,
                edge_index=edge_index,
                n_graphs=state.n_graphs,
            ),
            simgen_forces,
        )
