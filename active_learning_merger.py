#active_learning_merger.py
"""Merge RL_Gen flexible-docking affinities into the master multi-task CSV (train only)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from config_utils import ACTIVE_CONFIG, MASTER_CONFIG, canonicalize_smiles, get_smiles, load_config
from gnn_model import load_split_metadata

VINA_PATH = "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv"
MASTER_PATH = "data/coagulation_admet_multi_task.csv"
LIGANDS_DIR = "results/ligands"
SPLIT_PATH = Path("publication/data/gnn_scaffold_split.json")

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False

_sdf_cache: dict[str, str | None] = {}


def load_holdout_smiles() -> tuple[set[str], set[str]]:
    """Return (val_smiles, test_smiles) from frozen split; empty sets if missing."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Frozen scaffold split required for hold-out guard: {SPLIT_PATH}"
        )
    meta = load_split_metadata(SPLIT_PATH)
    val = {canonicalize_smiles(s) or s for s in meta.get("val_smiles", [])}
    test = {canonicalize_smiles(s) or s for s in meta.get("test_smiles", [])}
    return val, test


def smiles_from_sdf(name: str) -> str | None:
    if not _HAVE_RDKIT:
        return None
    if name in _sdf_cache:
        return _sdf_cache[name]
    path = os.path.join(LIGANDS_DIR, f"{name}.sdf")
    smi = None
    if os.path.exists(path):
        mol = Chem.MolFromMolFile(path, sanitize=True)
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
    _sdf_cache[name] = smi
    return smi


def resolve_smiles(name: str, ligand_dict: dict[str, str]) -> str | None:
    return ligand_dict.get(name) or smiles_from_sdf(name)


def main() -> None:
    print("=" * 50)
    print("   ACTIVE LEARNING: MERGING VINA DATA INTO GNN")
    print("=" * 50)

    if not os.path.exists(VINA_PATH):
        raise FileNotFoundError(f"Could not find Vina results at {VINA_PATH}")
    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(f"Could not find master dataset at {MASTER_PATH}")

    val_smiles, test_smiles = load_holdout_smiles()
    holdout = val_smiles | test_smiles
    if holdout:
        print(f"  Hold-out guard active: {len(val_smiles)} val + {len(test_smiles)} test SMILES protected.")

    config_path = MASTER_CONFIG if os.path.exists(MASTER_CONFIG) else ACTIVE_CONFIG
    print(f"Reading SMILES mapping from {config_path}...")
    config = load_config(config_path)
    ligands = config.get("ligands", {}) or {}
    ligand_dict = {name: get_smiles(entry) for name, entry in ligands.items()}

    df_vina = pd.read_csv(VINA_PATH)
    new_records = []
    for _, row in df_vina.iterrows():
        ligand_name = row.get("ligand_name")
        dg_score = row.get("best_affinity")
        if not isinstance(ligand_name, str) or not ligand_name.startswith("RL_Gen_"):
            continue
        if dg_score is None or pd.isna(dg_score):
            continue
        smiles = resolve_smiles(ligand_name, ligand_dict)
        if not smiles:
            print(
                f"  Warning: no SMILES for {ligand_name} "
                f"(not in {config_path} or {LIGANDS_DIR}/{ligand_name}.sdf), skipping."
            )
            continue
        canon = canonicalize_smiles(smiles) or smiles
        pxc50 = -(float(dg_score) / 1.36)
        new_records.append({"canonical_smiles": canon, "VKORC1_pXC50": pxc50})

    print(f"Found {len(new_records)} AI-generated (RL_Gen_) leads with valid affinities.")

    df_master = pd.read_csv(MASTER_PATH)
    smiles_col = "canonical_smiles" if "canonical_smiles" in df_master.columns else "SMILES"

    canon_to_idx: dict[str, int] = {}
    for idx, row in df_master.iterrows():
        smi = row.get(smiles_col)
        if pd.isna(smi):
            continue
        canon = canonicalize_smiles(str(smi)) or str(smi)
        canon_to_idx[canon] = idx

    injected = 0
    updated = 0
    skipped_holdout = 0
    for rec in new_records:
        canon = rec["canonical_smiles"]
        if canon in holdout:
            skipped_holdout += 1
            continue
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

    df_master.to_csv(MASTER_PATH, index=False)
    print(f"Injected {injected} new structures; updated VKORC1 on {updated} existing rows.")
    if skipped_holdout:
        print(f"Skipped {skipped_holdout} RL_Gen records targeting val/test hold-out SMILES.")
    print(f"Master dataset now contains {len(df_master)} unique molecules.")
    print("Next step: python dynamic_gnn_pipeline.py --split-from publication/data/gnn_scaffold_split.json")


if __name__ == "__main__":
    main()
