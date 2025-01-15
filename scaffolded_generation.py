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
from rdkit import Chem
from simgen.utils import setup_logger

from models.bond_predictor import BondPredictor
from models.scaffolded_model import ScaffoldedMolDiff
from utils.data import traj_to_ase
from utils.reconstruct import MolReconsError, reconstruct_from_generated_with_edges
from utils.sample import seperate_outputs
from utils.transforms import FeaturizeMol, make_data_placeholder

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    experiment_name: str,
    batch_size: int = 16,
    mol_size: int = 20,
    num_mols: int = 100,
    use_bond_guidance: bool = False,
    readd_noise: bool = False,
    results_dir: str = "./results",
):
    penicillin_analogues = aio.read("./penicillin_analogues.xyz", index=":")
    core_ids = np.load("./penicillin_core_ids.npy")
    penicillin_core:ase.Atoms  = penicillin_analogues[0][core_ids[0]] # type: ignore
    setup_logger(
        tag=f"scaffolded_moldiff_{experiment_name}",
        level=logging.INFO,
        directory="./logs",
    )
    results_path = pathlib.Path(f"{results_dir}/{experiment_name}/")
    results_path.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load("./ckpt/MolDiff.pt", map_location="cuda")
    train_config = ckpt["config"]
    featurizer = FeaturizeMol(
        train_config.chem.atomic_numbers,
        train_config.chem.mol_bond_types,
        use_mask_node=train_config.transform.use_mask_node,
        use_mask_edge=train_config.transform.use_mask_edge,
    )
    model = ScaffoldedMolDiff(
        config=train_config.model,
        num_node_types=featurizer.num_node_types,
        num_edge_types=featurizer.num_edge_types,
    ).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()

    if use_bond_guidance:
        logging.info("Loading bond predictor.")
        ckpt_bond = torch.load("./ckpt/bondpred.pt", map_location="cuda")
        bond_predictor = BondPredictor(
            ckpt_bond["config"]["model"],
            featurizer.num_node_types,
            featurizer.num_edge_types - 1,
        ).to("cuda")
        bond_predictor.load_state_dict(ckpt_bond["model"])
        bond_predictor.eval()
        guidance = ("uncertainty", 1e-4)
    else:
        bond_predictor = None
        guidance = None

    core_positions = penicillin_core.get_positions()-penicillin_core.get_center_of_mass()
    core_positions = torch.tensor(core_positions, dtype=torch.float32).to("cuda")
    core_numbers = penicillin_core.get_atomic_numbers()
    core_node_types = [featurizer.ele_to_nodetype[x] for x in core_numbers]
    core_node_types = torch.tensor(core_node_types, dtype=torch.long).to("cuda")

    pool = {"failed": [], "finished": [], "last_frames": []}
    while len(pool["finished"]) < num_mols:
        if len(pool["failed"]) > 256:
            logging.info(
                f"Experiment NO. {experiment_name}: Too many failed molecules. Stop sampling."
            )
            break
        batch_holder = make_data_placeholder(
            n_graphs=batch_size, device="cuda", max_size=mol_size
        )
        outputs = model.sample(
            n_graphs=batch_size,
            batch_node=batch_holder["batch_node"],
            halfedge_index=batch_holder["halfedge_index"],
            batch_halfedge=batch_holder["batch_halfedge"],
            bond_predictor=bond_predictor,
            guidance=guidance,
            scaffold_positions = core_positions,
            scaffold_node_types = core_node_types,
            readd_noise=readd_noise,
        )
        outputs = {
            key: [v.cpu().numpy() for v in value] for key, value in outputs.items()
        }
        batch_node = batch_holder["batch_node"].cpu().numpy()
        halfedge_index = batch_holder["halfedge_index"].cpu().numpy()
        batch_halfedge = batch_holder["batch_halfedge"].cpu().numpy()
        output_list = seperate_outputs(
            outputs, batch_size, batch_node, halfedge_index, batch_halfedge
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
