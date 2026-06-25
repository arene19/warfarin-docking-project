#active_learning_merger.py
"""Merge RL_Gen flexible-docking affinities into the master multi-task CSV."""
from __future__ import annotations

import os

import pandas as pd
import yaml

from config_utils import ACTIVE_CONFIG, MASTER_CONFIG, canonicalize_smiles, get_smiles, load_config

print("=" * 50)
print("   ACTIVE LEARNING: MERGING VINA DATA INTO GNN")
print("=" * 50)

vina_path = "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv"
master_path = "data/coagulation_admet_multi_task.csv"
ligands_dir = "results/ligands"

if not os.path.exists(vina_path):
    raise FileNotFoundError(f"Could not find Vina results at {vina_path}")
if not os.path.exists(master_path):
    raise FileNotFoundError(f"Could not find master dataset at {master_path}")

config_path = MASTER_CONFIG if os.path.exists(MASTER_CONFIG) else ACTIVE_CONFIG
print(f"Reading SMILES mapping from {config_path}...")
config = load_config(config_path)
ligands = config.get("ligands", {}) or {}
ligand_dict = {name: get_smiles(entry) for name, entry in ligands.items()}

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False
    print("  Note: RDKit unavailable; SDF SMILES fallback disabled.")

_sdf_cache: dict[str, str | None] = {}


def smiles_from_sdf(name: str) -> str | None:
    if not _HAVE_RDKIT:
        return None
    if name in _sdf_cache:
        return _sdf_cache[name]
    path = os.path.join(ligands_dir, f"{name}.sdf")
    smi = None
    if os.path.exists(path):
        mol = Chem.MolFromMolFile(path, sanitize=True)
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
    _sdf_cache[name] = smi
    return smi


def resolve_smiles(name: str) -> str | None:
    return ligand_dict.get(name) or smiles_from_sdf(name)


df_vina = pd.read_csv(vina_path)
new_records = []
for _, row in df_vina.iterrows():
    ligand_name = row.get("ligand_name")
    dg_score = row.get("best_affinity")
    if not isinstance(ligand_name, str) or not ligand_name.startswith("RL_Gen_"):
        continue
    if dg_score is None or pd.isna(dg_score):
        continue
    smiles = resolve_smiles(ligand_name)
    if not smiles:
        print(
            f"  Warning: no SMILES for {ligand_name} "
            f"(not in {config_path} or {ligands_dir}/{ligand_name}.sdf), skipping."
        )
        continue
    canon = canonicalize_smiles(smiles) or smiles
    pxc50 = -(float(dg_score) / 1.36)
    new_records.append({"canonical_smiles": canon, "VKORC1_pXC50": pxc50})

print(f"Found {len(new_records)} AI-generated (RL_Gen_) leads with valid affinities.")

df_master = pd.read_csv(master_path)
smiles_col = "canonical_smiles" if "canonical_smiles" in df_master.columns else "SMILES"

# Index existing rows by canonical SMILES
canon_to_idx: dict[str, int] = {}
for idx, row in df_master.iterrows():
    smi = row.get(smiles_col)
    if pd.isna(smi):
        continue
    canon = canonicalize_smiles(str(smi)) or str(smi)
    canon_to_idx[canon] = idx

injected = 0
updated = 0
for rec in new_records:
    canon = rec["canonical_smiles"]
    new_val = rec["VKORC1_pXC50"]
    if canon in canon_to_idx:
        idx = canon_to_idx[canon]
        old = df_master.at[idx, "VKORC1_pXC50"]
        if pd.isna(old) or new_val > float(old):
            df_master.at[idx, "VKORC1_pXC50"] = new_val
            updated += 1
    else:
        new_row = {col: float("nan") for col in df_master.columns}
        new_row[smiles_col] = canon
        new_row["VKORC1_pXC50"] = new_val
        if "is_coumarin" in df_master.columns:
            new_row["is_coumarin"] = float("nan")
        df_master = pd.concat([df_master, pd.DataFrame([new_row])], ignore_index=True)
        canon_to_idx[canon] = len(df_master) - 1
        injected += 1

df_master.to_csv(master_path, index=False)
print(f"Injected {injected} new structures; updated VKORC1 on {updated} existing rows.")
print(f"Master dataset now contains {len(df_master)} unique molecules.")
print("Next step: python dynamic_gnn_pipeline.py")
