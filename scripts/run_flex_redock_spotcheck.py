#!/usr/bin/env python3
"""Re-dock spot-check ligands with flexible VKORC1 params and write spot-check CSV.

Usage:
    python scripts/run_flex_redock_spotcheck.py --write
    python scripts/run_flex_redock_spotcheck.py --ligand RL_Gen_37 --write
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_utils import load_config
from docking_engine import run_vina_docking
from grid_box import auto_box_from_native_ligand

CANONICAL = ROOT / "publication/data/flexible_redock_spotcheck.csv"
SCREENING = ROOT / "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv"
REDock_DIR = ROOT / "results/flex_redock_spotcheck"
TARGET = "VKORC1_Human"

RL_SPOTCHECK = [
    "RL_Gen_37",
    "RL_Gen_22",
    "RL_Gen_29",
    "RL_Gen_07",
    "RL_Gen_26",
    "RL_Gen_31",
    "RL_Gen_15",
    "RL_Gen_03",
]
REF_SPOTCHECK = [
    "S_Warfarin_ref",
    "R_Warfarin_ref",
    "BENZ_S",
    "BENZ_R",
    "p_nitro_S",
    "p_nitro_R",
]


def _ligand_pdbqt(name: str) -> Path | None:
    for base in ("pdbqt_ligands", "results/pdbqt_ligands"):
        path = ROOT / base / f"{name}.pdbqt"
        if path.exists():
            return path
    return None


def _box_params(target_info: dict, raw_pdb: str) -> dict:
    if target_info.get("force_center") and target_info.get("force_size"):
        return {"center": target_info["force_center"], "size": target_info["force_size"]}
    return auto_box_from_native_ligand(
        raw_pdb,
        target_info["native_ligand_resname"],
        padding=target_info.get("padding", 6.0),
        force_size=target_info.get("force_size"),
        force_center=target_info.get("force_center"),
    )


def redock_ligand(
    name: str,
    target_info: dict,
    receptor_pdbqt: str,
    receptor_pdb: str,
    box: dict,
    dock_cfg: dict,
) -> float:
    lig_path = _ligand_pdbqt(name)
    if lig_path is None:
        raise FileNotFoundError(f"No PDBQT for {name}")
    REDock_DIR.mkdir(parents=True, exist_ok=True)
    out_pdbqt = REDock_DIR / f"{name}_{TARGET}_redock.pdbqt"
    flex_res = target_info.get("flexible_residues") or []
    result = run_vina_docking(
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=str(lig_path),
        output_pdbqt=str(out_pdbqt),
        box_params=box,
        exhaustiveness=int(dock_cfg.get("exhaustiveness", 20)),
        num_modes=int(dock_cfg.get("num_modes", 9)),
        min_rmsd=float(dock_cfg.get("min_rmsd", 1.0)),
        seed=42,
        target_name=TARGET,
        receptor_pdb=receptor_pdb,
        flex_res_list=flex_res,
    )
    if not result.success:
        raise RuntimeError(f"Redock failed for {name}: {result.error}")
    return float(result.best_affinity)


def build_spotcheck(names: list[str], run_redock: bool) -> pd.DataFrame:
    config = load_config("config_master.yaml")
    target_info = config["receptors"][TARGET]
    dock_cfg = config.get("docking_params", {})
    chain = target_info["chain"]
    pdb_code = target_info["pdb_id"]
    receptor_pdbqt = ROOT / "pdbqt_receptors" / f"{TARGET}_chain{chain}_protonated.pdbqt"
    receptor_pdb = ROOT / "proteins/protonated" / f"{TARGET}_chain{chain}_protonated.pdb"
    raw_pdb = ROOT / "proteins/raw" / f"{pdb_code}.pdb"
    if not receptor_pdbqt.exists():
        raise FileNotFoundError(receptor_pdbqt)
    if not raw_pdb.exists():
        raise FileNotFoundError(raw_pdb)

    box = _box_params(target_info, str(raw_pdb))
    screening = pd.read_csv(SCREENING)
    aff_map = screening.set_index("ligand_name")["best_affinity"].astype(float).to_dict()

    rows = []
    for name in names:
        if name not in aff_map:
            print(f"  [warn] {name} missing from screening CSV — skipped")
            continue
        train_aff = float(aff_map[name])
        if run_redock:
            redock_aff = redock_ligand(
                name, target_info, str(receptor_pdbqt), str(receptor_pdb), box, dock_cfg
            )
        elif CANONICAL.exists():
            old = pd.read_csv(CANONICAL)
            match = old[old["ligand_name"] == name]
            if match.empty:
                print(f"  [warn] {name} missing archived redock — skipped")
                continue
            redock_aff = float(match.iloc[0]["affinity_redock_kcal_mol"])
        else:
            raise FileNotFoundError("No archived spot-check CSV; run with --redock")
        delta = abs(train_aff - redock_aff)
        category = "RL_Gen" if name.startswith("RL_Gen") else "reference"
        if category == "RL_Gen" and delta > 1.0:
            print(f"  [warn] {name}: |Δ|={delta:.2f} > 1.0 — excluded")
            continue
        rows.append(
            {
                "ligand_name": name,
                "affinity_training_kcal_mol": train_aff,
                "affinity_redock_kcal_mol": redock_aff,
                "delta_kcal_mol": delta,
                "category": category,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flexible re-dock spot-check for VKORC1.")
    parser.add_argument("--redock", action="store_true", help="Re-run Vina flexible redocks")
    parser.add_argument("--write", action="store_true", help="Write publication/data/flexible_redock_spotcheck.csv")
    parser.add_argument("--ligand", action="append", help="Single ligand (repeatable); default = full spot-check set")
    args = parser.parse_args()
    names = args.ligand if args.ligand else REF_SPOTCHECK + RL_SPOTCHECK
    df = build_spotcheck(names, run_redock=args.redock)
    print(df.to_string(index=False))
    if args.write:
        CANONICAL.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CANONICAL, index=False)
        print(f"\nWrote {CANONICAL}")


if __name__ == "__main__":
    main()
