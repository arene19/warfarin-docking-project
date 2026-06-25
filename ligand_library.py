"""
Warfarin and derivative SMILES library parsing tool.
Handles molecular sanitization, calculations, and 3D conformer optimization.
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from config_utils import get_smiles

def prepare_ligand_library(smiles_dict: dict) -> pd.DataFrame:
    """Processes a dictionary of SMILES strings into a structured DataFrame."""
    records = []

    for name, entry in smiles_dict.items():
        smiles = get_smiles(entry)
        if not smiles:
            print(f"[ERROR] Could not resolve SMILES for {name}")
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[ERROR] Could not parse SMILES representation for {name}")
                continue

            # Descriptors are computed on the implicit-H molecule (RDKit convention).
            # Explicit hydrogens are added exactly once later, in generate_3d_conformers().
            Chem.SanitizeMol(mol)

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
                "Lipinski_Pass": bool(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
                "Mol": mol
            })

        except Exception as e:
            print(f"[ERROR] Processing failed for compound {name}: {e}")

    return pd.DataFrame(records)

def generate_3d_conformers(mol, mol_name: str, output_dir: str = "ligands", num_confs: int = 50) -> str:
    """
    Generates 3D conformers using ETKDGv3 and MMFF94s minimization.
    Writes output file directly to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    mol_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0 

    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params)

    if len(conf_ids) == 0:
        raise ValueError(f"No 3D conformer coordinates embedded for {mol_name}")

    results = AllChem.MMFFOptimizeMoleculeConfs(
        mol_h,
        mmffVariant="MMFF94s",
        numThreads=0
    )

    success_energies = [(i, res[1]) for i, res in enumerate(results) if res[0] == 0]

    if not success_energies:
        best_conf_id = 0
        best_energy = None
    else:
        best_tuple = min(success_energies, key=lambda x: x[1])
        best_conf_id = best_tuple[0]
        best_energy = best_tuple[1]

    energy_msg = f" | Energy: {best_energy:.4f} kcal/mol" if best_energy is not None else " (Unoptimized)"
    print(f"  [{mol_name}] Selected Conformer ID: {best_conf_id}{energy_msg}")
    
    output_path = os.path.join(output_dir, f"{mol_name}.sdf")
    with Chem.SDWriter(output_path) as writer:
        writer.write(mol_h, confId=best_conf_id)

    return output_path