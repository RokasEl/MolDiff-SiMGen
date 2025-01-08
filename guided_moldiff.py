import logging
import pathlib
from dataclasses import dataclass
from typing import Mapping

import ase
import ase.io as aio
import numpy as np
import torch
import typer
import yaml
from mace.calculators import mace_off
from rdkit import Chem
from simgen.calculators import MaceSimilarityCalculator
from simgen.utils import setup_logger

from models.bond_predictor import BondPredictor
from models.guided_model import (
    GuidedMolDiff,
    ImportanceSamplingConfig,
    NoiseSchedule,
    ScaleMode,
    SiMGenGuidanceMode,
    SiMGenGuidanceParams,
)
from utils.reconstruct import MolReconsError, reconstruct_from_generated_with_edges
from utils.sample import seperate_outputs
from utils.transforms import FeaturizeMol, make_data_placeholder

app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass
class Config:
    experiment_name: str
    batch_size: int
    num_mols: int
    max_size: int = 20

    guidance_mode: SiMGenGuidanceMode = SiMGenGuidanceMode.INVERSE_SUM
    guidance_strength: float = 0.2
    min_gui_scale: float = 0.0  # 0 = no minimum, only used in FRACTIONAL mode
    scale_mode: ScaleMode = ScaleMode.FRACTIONAL
    bond_guidance_strength: float = 0

    default_sigma: float = 1.0
    element_sigmas: dict[int, float] | None = None
    # matching essentially reduces temperature as generation progresses for importance sampling
    sigma_schedule: NoiseSchedule = NoiseSchedule.MATCHING
    constant_sigma_value: float = 1.0  # only used in CONSTANT mode

    importance_sampling_freq: int = 50
    inverse_temperature: float = 1e-3
    mini_batch: int = 8

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["guidance_mode"] = SiMGenGuidanceMode(cfg["guidance_mode"])
        cfg["scale_mode"] = ScaleMode(cfg["scale_mode"])
        cfg["sigma_schedule"] = NoiseSchedule(cfg["sigma_schedule"])
        return cls(**cfg)

    def get_importance_sampling_config(self) -> ImportanceSamplingConfig | None:
        if self.importance_sampling_freq > 0:
            return ImportanceSamplingConfig(
                frequency=self.importance_sampling_freq,
                inverse_temp=self.inverse_temperature,
                mini_batch=self.mini_batch,
            )
        else:
            return None

    def get_simgen_guidance_params(
        self, simgen_calc: MaceSimilarityCalculator, element_mapping: Mapping
    ) -> SiMGenGuidanceParams:
        return SiMGenGuidanceParams(
            sim_calc=simgen_calc,
            node_to_element_map=element_mapping,
            guidance_mode=self.guidance_mode,
            simgen_scale_mode=self.scale_mode,
            simgen_gui_scale=self.guidance_strength,
            min_gui_scale=self.min_gui_scale,
            sigma_schedule_type=self.sigma_schedule,
            constant_sigma_value=self.constant_sigma_value,
        )


def traj_to_ase(out, featurizer, idx: int | None = None):
    """Convert trajectory to ASE Atoms list."""
    traj = []
    nodes_to_elements = featurizer.nodetype_to_ele.copy()
    nodes_to_elements[7] = 6
    if idx is not None:
        nodes, positions, _ = out
        nodes = nodes[idx]
        positions = positions[idx]
        numbers = [nodes_to_elements[n] for n in np.argmax(nodes, axis=1)]
        atoms = ase.Atoms(numbers, positions=positions)
        return atoms
    for nodes, positions, _ in zip(*out, strict=True):
        numbers = [nodes_to_elements[n] for n in np.argmax(nodes, axis=1)]
        atoms = ase.Atoms(numbers, positions=positions)
        traj.append(atoms)
    return traj


def load_mace_and_simgen_if_needed(config: Config, featurizer: FeaturizeMol):
    """Load MACE models and ASE data only if guidance_strength>0 or importance_sampling_freq>0."""
    if config.guidance_strength <= 0 and config.importance_sampling_freq <= 0:
        return None, None, None

    ref_atoms = aio.read("./penicillin_analogues.xyz", index=":")
    core_atoms = np.load("./penicillin_core_ids.npy")
    core_masks = []
    for atoms, mask in zip(ref_atoms, core_atoms, strict=True):
        core_mask = np.zeros(len(atoms), dtype=bool)
        core_mask[mask] = True
        core_masks.append(core_mask)

    calc = mace_off("medium", device="cuda", default_dtype="float32")
    z_table = calc.z_table

    element_sigma_array = (
        np.ones_like(z_table.zs, dtype=np.float32) * config.default_sigma
    )
    if config.element_sigmas is not None:
        for element, sigma in config.element_sigmas.items():
            element_sigma_array[z_table.z_to_index(element)] = sigma

    sim_calc = MaceSimilarityCalculator(
        calc.models[0],
        reference_data=ref_atoms,
        ref_data_mask=core_masks,
        device="cuda",
        alpha=0,
        max_norm=None,
        element_sigma_array=element_sigma_array,
    )

    simgen_guidance_params = config.get_simgen_guidance_params(
        sim_calc, featurizer.nodetype_to_ele
    )
    importance_sampling_config = config.get_importance_sampling_config()
    return sim_calc, simgen_guidance_params, importance_sampling_config


@app.command()
def main(config_path: str):
    config = Config.from_yaml(config_path)
    setup_logger(
        tag=f"guided_moldiff_{config.experiment_name}",
        level=logging.INFO,
        directory="./logs",
    )
    results_path = pathlib.Path(f"./results_production/{config.experiment_name}/")
    results_path.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load("./ckpt/MolDiff.pt", map_location="cuda")
    train_config = ckpt["config"]
    featurizer = FeaturizeMol(
        train_config.chem.atomic_numbers,
        train_config.chem.mol_bond_types,
        use_mask_node=train_config.transform.use_mask_node,
        use_mask_edge=train_config.transform.use_mask_edge,
    )
    model = GuidedMolDiff(
        config=train_config.model,
        num_node_types=featurizer.num_node_types,
        num_edge_types=featurizer.num_edge_types,
    ).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()

    if config.bond_guidance_strength > 0:
        logging.info("Loading bond predictor.")
        ckpt_bond = torch.load("./ckpt/bondpred.pt", map_location="cuda")
        bond_predictor = BondPredictor(
            ckpt_bond["config"]["model"],
            featurizer.num_node_types,
            featurizer.num_edge_types - 1,
        ).to("cuda")
        bond_predictor.load_state_dict(ckpt_bond["model"])
        bond_predictor.eval()
    else:
        bond_predictor = None

    sim_calc, simgen_guidance_params, importance_sampling_config = (
        load_mace_and_simgen_if_needed(config, featurizer)
    )

    logging.info(
        f"Experiment name: {config.experiment_name}. Initialized with configs:"
    )
    if sim_calc is not None:
        logging.info(f"SiMGen guidance params: {simgen_guidance_params}")
        logging.info(f"Importance sampling config: {importance_sampling_config}")
    else:
        logging.info(
            "Skipping MACE/ASE loading (guidance_strength <= 0 and importance_sampling_freq <= 0)."
        )

    pool = {"failed": [], "finished": [], "last_frames": []}
    while len(pool["finished"]) < config.num_mols:
        if len(pool["failed"]) > 256:
            logging.info(
                f"Experiment NO. {config.experiment_name}: Too many failed molecules. Stop sampling."
            )
            break
        batch_holder = make_data_placeholder(
            n_graphs=config.batch_size, device="cuda", max_size=config.max_size
        )
        outputs = model.sample(
            n_graphs=config.batch_size,
            batch_node=batch_holder["batch_node"],
            halfedge_index=batch_holder["halfedge_index"],
            batch_halfedge=batch_holder["batch_halfedge"],
            bond_predictor=bond_predictor,
            bond_gui_scale=config.bond_guidance_strength,
            simgen_guidance_params=simgen_guidance_params,
            importance_sampling_params=importance_sampling_config,
        )
        outputs = {
            key: [v.cpu().numpy() for v in value] for key, value in outputs.items()
        }
        batch_node = batch_holder["batch_node"].cpu().numpy()
        halfedge_index = batch_holder["halfedge_index"].cpu().numpy()
        batch_halfedge = batch_holder["batch_halfedge"].cpu().numpy()
        output_list = seperate_outputs(
            outputs, config.batch_size, batch_node, halfedge_index, batch_halfedge
        )

        for output_mol in output_list:
            mol_info = featurizer.decode_output(
                pred_node=output_mol["pred"][0],
                pred_pos=output_mol["pred"][1],
                pred_halfedge=output_mol["pred"][2],
                halfedge_index=output_mol["halfedge_index"],
            )
            try:
                rdmol = reconstruct_from_generated_with_edges(mol_info)
            except MolReconsError:
                pool["failed"].append(mol_info)
                logging.warning("Reconstruction error encountered.")
                continue
            mol_info["rdmol"] = rdmol
            smiles = Chem.MolToSmiles(rdmol)
            mol_info["smiles"] = smiles
            pool["finished"].append(mol_info)
            last_frame = traj_to_ase(output_mol["traj"], featurizer, -1)
            pool["last_frames"].append(last_frame)
            aio.write(results_path / "last_frames.xyz", last_frame, append=True)

            sdf_dir = results_path / "SDF"
            sdf_dir.mkdir(exist_ok=True)
            with open(results_path / "SMILES.txt", "a") as smiles_f:
                current_index = len(pool["finished"]) - 1
                smiles_f.write(mol_info["smiles"] + "\n")
                rdmol = mol_info["rdmol"]
                Chem.MolToMolFile(rdmol, str(sdf_dir / f"{current_index}.sdf"))
    torch.save(pool, results_path / "samples_all.pt")


if __name__ == "__main__":
    app()
