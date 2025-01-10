import pathlib
import argparse
from rdkit import Chem
from mace.calculators import mace_off
import numpy as np
import ase
import ase.io as aio
from simgen.calculators import MaceSimilarityCalculator
import torch
from torch_scatter import scatter_logsumexp

def rdkit_mol_2_ase(mol):
    if mol is None:
        return None
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = mol.GetConformer().GetPositions()
    return ase.Atoms(symbols, positions)

@torch.no_grad()
def calculate_log_dens(sim_calc, atoms_list, noise_level=1.0):
    batch = sim_calc.batch_atoms(atoms_list)
    embeddings = sim_calc._get_node_embeddings(batch)
    squared_distance_matrix = sim_calc._calculate_distance_matrix(embeddings, batch.node_attrs)
    additional_multiplier = 119 * (1 - (noise_level / 10) ** 0.25) + 1 if noise_level <= 10 else 1
    squared_distance_matrix = squared_distance_matrix * additional_multiplier
    log_dens = scatter_logsumexp(-squared_distance_matrix / 2, batch.batch, dim=0)
    return log_dens.sum(dim=-1)

def evaluate_mols(sim_calc, mols, batch_size=128, noise_level=1.0):
    generated_densities = np.full(len(mols), np.nan)
    atoms_list = [rdkit_mol_2_ase(mol) for mol in mols]
    for start_idx in range(0, len(atoms_list), batch_size):
        batch_slice = atoms_list[start_idx : start_idx + batch_size]
        valid_indices = [i for i, a in enumerate(batch_slice) if a is not None]
        if valid_indices:
            batch_atoms = [batch_slice[i] for i in valid_indices]
            log_dens = calculate_log_dens(sim_calc, batch_atoms, noise_level)
            generated_densities[start_idx + np.array(valid_indices)] = log_dens.cpu().numpy()
    return generated_densities

def evaluate_experiments(experiment_folders, sim_calc, save_name="log_densities.npy"):
    for exp_folder in experiment_folders:
        sdf_files = sorted(
            exp_folder.joinpath("SDF").glob("*.sdf"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        mols = []
        for sdf_file in sdf_files:
            suppl = Chem.SDMolSupplier(str(sdf_file))
            for mol in suppl:
                mols.append(mol)
        densities = evaluate_mols(sim_calc, mols, batch_size=128, noise_level=1.0)
        print(f"Experiment: {exp_folder.name}")
        for i, d in enumerate(densities):
            print(f"  Molecule {i+1}: Log Density = {d}")
        np.save(exp_folder / save_name, densities)
        
def get_ref_atoms_and_core_masks(keep_hs:bool=True):
    ref_atoms = aio.read("./penicillin_analogues.xyz", index=":")
    core_atoms = np.load("./penicillin_core_ids.npy")
    if not keep_hs:
        ref_atoms = [a[a.numbers != 1] for a in ref_atoms]
    core_masks = []
    for atoms, mask in zip(ref_atoms, core_atoms, strict=True):
        cm = np.zeros(len(atoms), dtype=bool)
        cm[mask] = True
        core_masks.append(cm)
    return ref_atoms, core_masks
    

def main(results_root: str, exp_name: str | None = None):
    calc = mace_off("medium", device="cuda", default_dtype="float32")
    z_table = calc.z_table

    ref_atoms, core_masks = get_ref_atoms_and_core_masks(keep_hs=True)

    element_sigma_array = np.ones_like(z_table.zs, dtype=np.float32) * 1.0
    sim_calc = MaceSimilarityCalculator(
        calc.models[0],
        reference_data=ref_atoms,
        ref_data_mask=core_masks,
        device="cuda",
        alpha=0,
        max_norm=None,
        element_sigma_array=element_sigma_array,
    )

    results_path = pathlib.Path(results_root)
    if exp_name is not None:
        experiment_folders = [results_path / exp_name]
    else:
        experiment_folders = [
            f for f in results_path.iterdir() if f.is_dir() and f.name.startswith("exp_")
        ]

    evaluate_experiments(experiment_folders, sim_calc)
    
    ref_atoms_no_h, core_masks_no_h = get_ref_atoms_and_core_masks(keep_hs=False)
    sim_calc_no_h = MaceSimilarityCalculator(
        calc.models[0],
        reference_data=ref_atoms_no_h,
        ref_data_mask=core_masks_no_h,
        device="cuda",
        alpha=0,
        max_norm=None,
        element_sigma_array=element_sigma_array,
    )
    evaluate_experiments(experiment_folders, sim_calc_no_h, save_name="log_densities_no_h.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, help="Path to results folder.")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()
    main(args.results_root, args.exp_name)
