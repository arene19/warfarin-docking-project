# ligand_library.py
"""
Warfarin and derivative SMILES library for docking studies.
Warfarin acts as vitamin K antagonist targeting VKORC1.
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
import numpy as np
import os
# -------------------------------------------------------
# Warfarin Derivative Library
# -------------------------------------------------------
target_ligands = [
    "/home/amar/warfarin_project/pdbqt_ligands/acenocoumarol6.pdbqt",
    "/home/amar/warfarin_project/pdbqt_ligands/Warfarin1.pdbqt",
]
def prepare_ligand_library(smiles_dict: dict) -> pd.DataFrame:
    """Processes a dictionary of SMILES strings into a DataFrame."""
    records = []
    
    for name, smiles in smiles_dict.items():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[ERROR] Could not parse SMILES for {name}")
                continue
                
            mol = Chem.AddHs(mol)
            Chem.SanitizeMol(mol)
            
            # Calculate descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            records.append({
                "Name": name,
                "SMILES": smiles,
                "MW": round(mw, 2),
                "LogP": round(logp, 2),
                "HBD": hbd,
                "HBA": hba,
                "Lipinski_Pass": (mw <= 500 and logp <= 5),
                "Mol": mol
            })
            
        except Exception as e:
            print(f"[ERROR] Processing failed for {name}: {e}")

    return pd.DataFrame(records)

# ligand_library.py

def generate_3d_conformers(mol):
    mol = Chem.AddHs(mol)
    # Use ETKDGv3 for more accurate bioactive conformers
    params = AllChem.ETKDGv3()
    AllChem.EmbedMolecule(mol, params)
    
    # CRITICAL: Force Geometry Optimization
    # This relaxes the molecule into its lowest energy state before docking
    AllChem.UFFOptimizeMolecule(mol) 
    
    return mol
# Add this snippet to your ligand_library.py function
def minimize_ligand(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol) # This relaxes the bonds
    return mol
    """
    Generate 3D conformers using ETKDG and energy minimize 
    using MMFF94 force field. Returns path to SDF file.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Add hydrogens
    mol_h = Chem.AddHs(mol)
    
    # Generate conformers using ETKDGv3
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0  # use all available cores
    
    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params)
    
    if len(conf_ids) == 0:
        raise ValueError(f"Could not generate conformers for {mol_name}")
    
    # Minimize with MMFF94 force field
    results = AllChem.MMFFOptimizeMoleculeConfs(
        mol_h, 
        mmffVariant="MMFF94s",
        numThreads=0
    )
# 1. Correctly filter and identify the best conformer
    # res[0] == 0 means optimization converged
    success_energies = [(i, res[1]) for i, res in enumerate(results) if res[0] == 0]

    if not success_energies:
        # Fallback: if optimization failed for all, use the first embedded conformer
        best_conf_id = 0
        best_energy = None
    else:
        # Use min() to find the tuple (ID, Energy) with the lowest energy
        best_tuple = min(success_energies, key=lambda x: x[1])
        best_conf_id = best_tuple[0]  # The ID for writing
        best_energy  = best_tuple[1]  # The Energy for printing

    # 2. Fix the print statement to be safe
    energy_msg = f" | Energy: {best_energy:.4f} kcal/mol" if best_energy is not None else " (Unoptimized)"
    print(f"  [{mol_name}] Best conformer: {best_conf_id}{energy_msg}")    
    # Write best conformer to SDF
    output_path = os.path.join(output_dir, f"{mol_name}.sdf")
    writer = Chem.SDWriter(output_path)
    writer.write(mol_h, confId=best_conf_id)
    writer.close()
    
    print(f"  Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("  Warfarin Derivative Library Preparation")
    print("=" * 60)
    
    df = prepare_ligand_library(WARFARIN_DERIVATIVES)
    
    print("\n--- Molecular Descriptors ---")
    display_cols = ["Name", "MW", "LogP", "HBD", "HBA", 
                    "TPSA", "RotBonds", "Lipinski_Pass"]
    print(df[display_cols].to_string(index=False))
    
    print("\n--- Generating 3D Conformers ---")
    sdf_paths = {}
    for _, row in df.iterrows():
        try:
            path = generate_3d_conformers(row["Mol"], row["Name"])
            sdf_paths[row["Name"]] = path
        except Exception as e:
            print(f"  [SKIP] {row['Name']}: {e}")
    
    print(f"\nGenerated {len(sdf_paths)} SDF files.")
if __name__ == "__main__":
    for ligand_path in target_ligands:
        if os.path.exists(ligand_path):
            print(f"Docking initiated for: {ligand_path}")
