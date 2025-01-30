from typing import Callable
import torch
import torch.nn.functional as F
from torch_scatter import scatter_max
from tqdm import tqdm

from .diffusion import log_sample_categorical
from .model import MolDiff


class ScaffoldedMolDiff(MolDiff):
    @torch.no_grad()
    def sample(
        self,
        n_graphs,
        batch_node,
        halfedge_index,
        batch_halfedge,
        bond_predictor: None | Callable = None,
        guidance=None,
        scaffold_positions: torch.Tensor | None = None,
        scaffold_node_types: torch.Tensor | None = None,
        readd_noise: bool = False,
    ):
        device = batch_node.device
        # # 1. get the init values (position, node types)
        # n_graphs = len(n_nodes_list)
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

        # # 1.5 log init
        node_traj = torch.zeros(
            [self.num_timesteps + 1, n_nodes_all, h_node_init.shape[-1]],
            dtype=h_node_init.dtype,
        ).to(device)
        pos_traj = torch.zeros(
            [self.num_timesteps + 1, n_nodes_all, 3], dtype=pos_init.dtype
        ).to(device)
        halfedge_traj = torch.zeros(
            [self.num_timesteps + 1, n_halfedges_all, h_halfedge_init.shape[-1]],
            dtype=h_halfedge_init.dtype,
        ).to(device)
        node_traj[0] = h_node_init
        pos_traj[0] = pos_init
        halfedge_traj[0] = h_halfedge_init

        # # 2. sample loop
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        for i, step in tqdm(
            enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps
        ):
            time_step = torch.full(
                size=(n_graphs,), fill_value=step, dtype=torch.long
            ).to(device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

            # # 1 inference
            preds = self(
                h_node_pert,
                pos_pert,
                batch_node,
                h_edge_pert,
                edge_index,
                batch_edge,
                time_step,
            )
            pred_node = preds["pred_node"]  # (N, num_node_types)
            pred_pos = preds["pred_pos"]  # (N, 3)
            pred_halfedge = preds["pred_halfedge"]  # (E//2, num_bond_types)

            # # 2 get the t - 1 state
            # pos
            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=batch_node
            )
            if scaffold_positions is not None:
                scaffold_batch_node = torch.zeros(
                    len(scaffold_positions), dtype=torch.long
                ).to(device)
                if readd_noise:
                    scaffold_pert: torch.Tensor = self.pos_transition.add_noise(  # type: ignore
                        scaffold_positions, time_step, scaffold_batch_node
                    )
                    scaffold_repeated = scaffold_pert.repeat(n_graphs, 1)
                else:
                    scaffold_repeated = scaffold_positions.repeat(n_graphs, 1)
                offsets = torch.cumsum(torch.bincount(batch_node), -1)
                offsets = torch.cat([torch.zeros(1, device=device), offsets])[
                    :-1
                ].long()
                indices = torch.cat(
                    [
                        torch.arange(len(scaffold_positions), device=device) + offset
                        for offset in offsets
                    ]
                )
                pos_prev[indices] = scaffold_repeated
            if self.categorical_space == "discrete":
                # node types
                log_node_recon = F.log_softmax(pred_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(
                    log_node_recon, log_node_type, time_step, batch_node, v0_prob=True
                )
                node_type_prev = log_sample_categorical(log_node_type)
                h_node_prev = self.node_transition.onehot_encode(node_type_prev)

                # halfedge types
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

            if scaffold_node_types is not None:
                assert (
                    self.categorical_space == "discrete"
                )  # only discrete space is supported
                scaffold_batch_node = torch.zeros(
                    len(scaffold_node_types), dtype=torch.long
                ).to(device)
                if readd_noise:
                    scaffold_pert = self.node_transition.add_noise(  # type: ignore
                        scaffold_node_types, time_step, scaffold_batch_node
                    )[0].argmax(-1)
                    scaffold_repeated = scaffold_pert.repeat(n_graphs)
                else:
                    scaffold_repeated = scaffold_node_types.repeat(n_graphs)
                scaffold_repeated = self.node_transition.onehot_encode(
                    scaffold_repeated
                )
                offsets = torch.cumsum(torch.bincount(batch_node), -1)
                offsets = torch.cat([torch.zeros(1, device=device), offsets])[
                    :-1
                ].long()
                indices = torch.cat(
                    [
                        torch.arange(len(scaffold_node_types), device=device) + offset
                        for offset in offsets
                    ]
                )
                h_node_prev[indices] = scaffold_repeated

            # # use guidance to modify pos
            if guidance is not None:
                gui_type, gui_scale = guidance
                if gui_scale > 0:
                    assert bond_predictor is not None, (
                        "bond_predictor is required for guidance"
                    )
                    with torch.enable_grad():
                        h_node_in = h_node_pert.detach()
                        pos_in = pos_pert.detach().requires_grad_(True)
                        pred_bondpredictor = bond_predictor(
                            h_node_in,
                            pos_in,
                            batch_node,
                            edge_index,
                            batch_edge,
                            time_step,
                        )
                        if gui_type == "entropy":
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = -torch.sum(
                                prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1
                            )
                            entropy = entropy.log().sum()
                            delta = -torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == "uncertainty":
                            uncertainty = torch.sigmoid(
                                -torch.logsumexp(pred_bondpredictor, dim=-1)
                            )
                            uncertainty = uncertainty.log().sum()
                            delta = (
                                -torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                            )
                        elif (
                            gui_type == "uncertainty_bond"
                        ):  # only for the predicted real bond (not no bond)
                            prob = torch.softmax(pred_bondpredictor, dim=-1)
                            uncertainty = torch.sigmoid(
                                -torch.logsumexp(pred_bondpredictor, dim=-1)
                            )
                            uncertainty = uncertainty.log()
                            uncertainty = (
                                uncertainty * prob[:, 1:].detach().sum(dim=-1)
                            ).sum()
                            delta = (
                                -torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale
                            )
                        elif gui_type == "entropy_bond":
                            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
                            entropy = -torch.sum(
                                prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1
                            )
                            entropy = entropy.log()
                            entropy = (
                                entropy * prob_halfedge[:, 1:].detach().sum(dim=-1)
                            ).sum()
                            delta = -torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == "logit_bond":
                            ind_real_bond = (halfedge_type_prev >= 1) & (
                                halfedge_type_prev <= 4
                            )
                            idx_real_bond = ind_real_bond.nonzero().squeeze(-1)
                            pred_real_bond = pred_bondpredictor[
                                idx_real_bond, halfedge_type_prev[idx_real_bond]
                            ]
                            pred = pred_real_bond.sum()
                            delta = +torch.autograd.grad(pred, pos_in)[0] * gui_scale
                        elif gui_type == "logit":
                            ind_bond_notmask = halfedge_type_prev <= 4
                            idx_real_bond = ind_bond_notmask.nonzero().squeeze(-1)
                            pred_real_bond = pred_bondpredictor[
                                idx_real_bond, halfedge_type_prev[idx_real_bond]
                            ]
                            pred = pred_real_bond.sum()
                            delta = +torch.autograd.grad(pred, pos_in)[0] * gui_scale
                        elif gui_type == "crossent":
                            prob_halfedge_type = log_halfedge_type.exp()[
                                :, :-1
                            ]  # the last one is masked bond (not used in predictor)
                            entropy = F.cross_entropy(
                                pred_bondpredictor, prob_halfedge_type, reduction="none"
                            )
                            entropy = entropy.log().sum()
                            delta = -torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        elif gui_type == "crossent_bond":
                            prob_halfedge_type = log_halfedge_type.exp()[
                                :, 1:-1
                            ]  # the last one is masked bond. first one is no bond
                            entropy = F.cross_entropy(
                                pred_bondpredictor[:, 1:],
                                prob_halfedge_type,
                                reduction="none",
                            )
                            entropy = entropy.log().sum()
                            delta = -torch.autograd.grad(entropy, pos_in)[0] * gui_scale
                        else:
                            raise NotImplementedError(
                                f"Guidance type {gui_type} is not implemented"
                            )
                    pos_prev = pos_prev + delta

            # log update
            node_traj[i + 1] = h_node_prev
            pos_traj[i + 1] = pos_prev
            halfedge_traj[i + 1] = h_halfedge_prev

            # # 3 update t-1
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev

        # # 3. get the final positions
        return {
            "pred": [pred_node, pred_pos, pred_halfedge],
            "traj": [node_traj, pos_traj, halfedge_traj],
        }
