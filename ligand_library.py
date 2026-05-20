# ligand_library.py
"""
Warfarin and derivative SMILES library for docking studies.
Warfarin acts as vitamin K antagonist targeting VKORC1.
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# -------------------------------------------------------
# Warfarin Derivative Library
# -------------------------------------------------------
WARFARIN_DERIVATIVES = {
    "S_Warfarin_ref": "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O",
    "R_Warfarin_ref": "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O"
}

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

            # Calculate all required descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotbonds = Descriptors.NumRotatableBonds(mol)

            records.append({
                "Name": name,
                "SMILES": smiles,
                "MW": round(mw, 2),
                "LogP": round(logp, 2),
                "HBD": hbd,
                "HBA": hba,
                "TPSA": round(tpsa, 2),
                "RotBonds": rotbonds,
                "Lipinski_Pass": (mw <= 500 and logp <= 5),
                "Mol": mol
            })

        except Exception as e:
            print(f"[ERROR] Processing failed for {name}: {e}")

    return pd.DataFrame(records)


def generate_3d_conformers(mol, mol_name, output_dir="ligands/sdf", num_confs=50):
    """
    Generate 3D conformers using ETKDG and energy minimize
    using MMFF94 force field. Returns path to SDF file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Add hydrogens (Crucial for 3D structure and docking)
    mol_h = Chem.AddHs(mol)

    # 2. Generate conformers using ETKDGv3
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0  # use all available cores

    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params)

    if len(conf_ids) == 0:
        raise ValueError(f"Could not generate conformers for {mol_name}")

    # 3. Minimize with MMFF94 force field to relax bonds
    results = AllChem.MMFFOptimizeMoleculeConfs(
        mol_h,
        mmffVariant="MMFF94s",
        numThreads=0
    )

    # 4. Correctly filter and identify the lowest energy conformer
    # res[0] == 0 means optimization converged successfully
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

    # 5. Print success and save to SDF
    energy_msg = f" | Energy: {best_energy:.4f} kcal/mol" if best_energy is not None else " (Unoptimized)"
    print(f"  [{mol_name}] Best conformer: {best_conf_id}{energy_msg}")
    
    output_path = os.path.join(output_dir, f"{mol_name}.sdf")
    writer = Chem.SDWriter(output_path)
    writer.write(mol_h, confId=best_conf_id)
    writer.close()

    return output_path


# -------------------------------------------------------
# Main Execution Block
# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Warfarin Derivative Library Preparation")
    print("=" * 60)

    df = prepare_ligand_library(WARFARIN_DERIVATIVES)

    if not df.empty:
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
    else:
        print("\n[WARNING] Library is empty. Check your SMILES dictionary.")

    print("\n--- Checking Target Ligands ---")
    for ligand_path in target_ligands:
        if os.path.exists(ligand_path):
            print(f"Docking initiated for: {ligand_path}")
        else:
            print(f"[MISSING] Cannot find: {ligand_path}")
