import pathlib
import argparse

from rdkit import Chem
from mace.calculators import mace_off
import numpy as np

import ase
import ase.io as aio
from ase.optimize import FIRE
import pandas as pd
from simgen.calculators import MaceSimilarityCalculator
from torch_scatter import scatter_logsumexp


def rdkit_mol_2_ase(mol):
    if mol is None:
        return None
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = mol.GetConformer().GetPositions()
    return ase.Atoms(symbols, positions)


def main(results_root:str, exp_name:str|None=None, ):
    # Load MACE model
    calc = mace_off("medium", device="cuda", default_dtype="float32")

    # Load penicillin data for ref set
    df = pd.read_csv("./fda_approved_drugs.txt", sep="\t")
    df = df.query("~smiles.isna()")
    penicillin_smiles = df.query("generic_name == 'Penicillin G'")["smiles"].values[0]
    penicillin_mol = Chem.MolFromSmiles(penicillin_smiles)
    penicillin_mol = Chem.AddHs(penicillin_mol)
    pdb_traj = aio.read("penicillin_trajectory.pdb", index=":")
    conformers = pdb_traj[::100]

    # Collect short MD trajectory to get multiple conformations
    for atoms in conformers:
        atoms.calc = calc
        dyn = FIRE(atoms)
        dyn.run(fmax=0.1)
        atoms.calc = None

    # define core ids
    patt = Chem.MolFromSmarts("O=C1CC2N1CCS2")
    core_ids = np.asarray(penicillin_mol.GetSubstructMatch(patt)).flatten()
    z_table = calc.z_table
    atoms = conformers[0]
    core_atom_mask = np.zeros(len(atoms), dtype=bool)
    core_atom_mask[core_ids] = True

    # Create similarity calculator
    element_sigma_array = np.ones_like(z_table.zs) * 1
    sim_calc = MaceSimilarityCalculator(
        model=calc.models[0],
        reference_data=conformers,
        ref_data_mask=[core_atom_mask] * len(conformers),
        element_sigma_array=element_sigma_array,
        max_norm=None,
        device="cuda",
    )

    # Process all experiments in the results folder
    results_path = pathlib.Path(results_root)
    if exp_name is not None:
        experiment_folders = [results_path / exp_name]
    else:
        experiment_folders = [
            folder
            for folder in results_path.iterdir()
            if folder.is_dir() and folder.name.startswith("exp_")
        ]

    for exp_folder in experiment_folders:
        sdf_path = exp_folder / "SDF"
        sdf_files = list(sdf_path.glob("*.sdf"))
        mols = []
        for sdf_file in sdf_files:
            suppl = Chem.SDMolSupplier(str(sdf_file))
            for mol in suppl:
                mols.append(mol)

        generated_densities = np.full(len(mols), np.nan)
        batch_size = 128
        noise_level = 1.0

        # Convert to ASE atoms, keeping None for invalid molecules
        atoms_list = [rdkit_mol_2_ase(mol) for mol in mols]

        for start_idx in range(0, len(atoms_list), batch_size):
            end_idx = min(start_idx + batch_size, len(atoms_list))
            atoms_batch = atoms_list[start_idx:end_idx]

            # Filter out None values for batching
            valid_atoms_batch = [atoms for atoms in atoms_batch if atoms is not None]
            valid_indices = [
                i for i, atoms in enumerate(atoms_batch) if atoms is not None
            ]

            if valid_atoms_batch:
                batch = sim_calc.batch_atoms(valid_atoms_batch)
                embeddings = sim_calc._get_node_embeddings(batch)
                squared_distance_matrix = sim_calc._calculate_distance_matrix(
                    embeddings, batch.node_attrs
                )
                additional_multiplier = (
                    119 * (1 - (noise_level / 10) ** 0.25) + 1
                    if noise_level <= 10
                    else 1
                )
                squared_distance_matrix = (
                    squared_distance_matrix * additional_multiplier
                )
                log_dens = scatter_logsumexp(
                    -squared_distance_matrix / 2, batch.batch, dim=0
                )
                log_dens = log_dens.sum(dim=-1)

                # Assign densities back to the correct indices
                generated_densities[np.array(valid_indices) + start_idx] = (
                    log_dens.detach().cpu().numpy()
                )

        # Print or save the densities for the current experiment
        print(f"Experiment: {exp_folder.name}")
        for i, density in enumerate(generated_densities):
            print(f"  Molecule {i+1}: Log Density = {density}")
        # Optionally save the generated densities to a file for each experiment
        np.save(exp_folder / "log_densities.npy", generated_densities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate log densities for molecules in a results folder."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        help="Path to the results folder containing experiment subfolders.",
    )
    parser.add_argument(
        "--exp_name", type=str, help="Name of the experiment.", default=None
    )
    args = parser.parse_args()
    main(args.results_root,args.exp_name)
