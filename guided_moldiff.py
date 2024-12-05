import torch
import numpy as np
import matplotlib.pyplot as plt
import ase
import pandas as pd
import ase.io as aio
from pathlib import Path
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from simgen.calculators import MaceSimilarityCalculator
from mace.calculators import mace_off

from models.model import MolDiff
from models.bond_predictor import BondPredictor
from ase.optimize import LBFGS
from utils.sample import seperate_outputs
from utils.transforms import FeaturizeMol, make_data_placeholder


def load_molecule_from_smiles(smiles):
    """Load a molecule from SMILES and generate a 3D conformation."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    embedding_params = rdDistGeom.ETKDGv3()
    rdDistGeom.EmbedMolecule(mol, embedding_params)
    mol = Chem.RemoveHs(mol)
    return mol


def mol_to_ase_atoms(mol):
    """Convert RDKit molecule to ASE Atoms."""
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = mol.GetConformer().GetPositions()
    return ase.Atoms(symbols, positions)


def traj_to_ase(out, featurizer):
    """Convert trajectory to ASE Atoms list."""
    traj = []
    nodes_to_elements = featurizer.nodetype_to_ele.copy()
    nodes_to_elements[7] = 6
    for nodes, positions, _ in zip(*out):
        numbers = [nodes_to_elements[n] for n in np.argmax(nodes, axis=1)]
        atoms = ase.Atoms(numbers, positions=positions)
        traj.append(atoms)
    return traj


def main():
    # Load data and select molecule
    df = pd.read_csv("./fda_approved_drugs.txt", sep="\t")
    df = df.query("~smiles.isna()")
    penicillin_smiles = df.query("generic_name == 'Penicillin G'")["smiles"].values[0]
    penicillin_mol = load_molecule_from_smiles(penicillin_smiles)
    ase_mol = mol_to_ase_atoms(penicillin_mol)

    # Load models
    ckpt = torch.load("./ckpt/MolDiff.pt", map_location="cuda")
    train_config = ckpt["config"]

    featurizer = FeaturizeMol(
        train_config.chem.atomic_numbers,
        train_config.chem.mol_bond_types,
        use_mask_node=train_config.transform.use_mask_node,
        use_mask_edge=train_config.transform.use_mask_edge,
    )

    model = MolDiff(
        config=train_config.model,
        num_node_types=featurizer.num_node_types,
        num_edge_types=featurizer.num_edge_types,
    ).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ckpt_bond = torch.load("./ckpt/bondpred.pt", map_location="cuda")
    # bond_predictor = BondPredictor(
    #     ckpt_bond['config']['model'],
    #     featurizer.num_node_types,
    #     featurizer.num_edge_types-1
    # ).to("cuda")
    # bond_predictor.load_state_dict(ckpt_bond['model'])
    # bond_predictor.eval()

    # Setup similarity calculator
    calc = mace_off("medium", device="cuda", default_dtype="float32")
    z_table = calc.z_table
    ase_mol.calc = calc
    dyn = LBFGS(ase_mol)
    dyn.run(fmax=1e-3)
    ase_mol.calc = None

    element_sigma_array = np.ones_like(z_table.zs, dtype=np.float32) * 0.25
    sim_calc = MaceSimilarityCalculator(
        calc.models[0],
        reference_data=[ase_mol],
        device="cuda",
        alpha=0,
        max_norm=None,
        element_sigma_array=element_sigma_array,
    )

    # Molecule generation
    n_graphs = 12
    max_size = 12

    batch_holder = make_data_placeholder(
        n_graphs=n_graphs, device="cuda", max_size=max_size
    )
    outputs = model.sample(
        n_graphs=n_graphs,
        batch_node=batch_holder["batch_node"],
        halfedge_index=batch_holder["halfedge_index"],
        batch_halfedge=batch_holder["batch_halfedge"],
        bond_predictor=None,
        guidance=("simgen", 0.10),
        featurizer=featurizer,
        simgen_calc=sim_calc,
    )

    outputs = {key: [v.cpu().numpy() for v in value] for key, value in outputs.items()}
    batch_node = batch_holder["batch_node"].cpu().numpy()
    halfedge_index = batch_holder["halfedge_index"].cpu().numpy()
    batch_halfedge = batch_holder["batch_halfedge"].cpu().numpy()

    output_list = seperate_outputs(
        outputs, n_graphs, batch_node, halfedge_index, batch_halfedge
    )

    # Select and process a trajectory
    trajectories = [traj_to_ase(out["traj"], featurizer) for out in output_list]
    last_frames = [traj[-1] for traj in trajectories]
    aio.write("guided_moldiff_examples.xyz", last_frames)


if __name__ == "__main__":
    main()
