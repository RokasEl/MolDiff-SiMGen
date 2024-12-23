import torch
import typer
import numpy as np
import ase
import pandas as pd
import logging
import ase.io as aio
import pathlib
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from dataclasses import dataclass
import yaml

from simgen.calculators import MaceSimilarityCalculator
from simgen.utils import setup_logger
from mace.calculators import mace_off

from models.guided_model import GuidedMolDiff, ScaleMode
from models.bond_predictor import BondPredictor
from ase.optimize import LBFGS
from utils.reconstruct import reconstruct_from_generated_with_edges, MolReconsError
from utils.sample import seperate_outputs
from utils.transforms import FeaturizeMol, make_data_placeholder

app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass
class Config:
    experiment_name: str
    guidance_strength: float
    default_sigma: float
    num_mols: int
    batch_size: int
    element_sigmas: dict[int, float] | None = None
    bond_guidance_strength: float = 0
    num_replicas: int = 1
    max_size: int = 20
    scale_mode: ScaleMode = ScaleMode.FRACTIONAL


def load_molecule_from_smiles(smiles, removeHs=True):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    embedding_params = rdDistGeom.ETKDGv3()
    rdDistGeom.EmbedMolecule(mol, embedding_params)
    if removeHs:
        mol = Chem.RemoveHs(mol)
    return mol


def mol_to_ase_atoms(mol):
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = mol.GetConformer().GetPositions()
    return ase.Atoms(symbols, positions)


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


@app.command()
def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    config = Config(**cfg)
    config.scale_mode = ScaleMode(config.scale_mode)
    setup_logger(tag="guided_moldiff", level=logging.INFO, directory="./logs")
    results_path = pathlib.Path(f"./results_inverse_summation/{config.experiment_name}/")
    results_path.mkdir(parents=True, exist_ok=True)

    ref_atoms = aio.read("./penicillin_analogues.xyz", index=":")
    core_atoms = np.load("./penicillin_core_ids.npy")
    core_masks = []
    for atoms, mask in zip(ref_atoms, core_atoms, strict=True):
        core_mask = np.zeros(len(atoms), dtype=bool)
        core_mask[mask] = True
        core_masks.append(core_mask)

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
            featurizer=featurizer,
            simgen_calc=sim_calc,
            simgen_gui_scale=config.guidance_strength,
            simgen_scale_mode=config.scale_mode,
            num_replicas=config.num_replicas,
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
