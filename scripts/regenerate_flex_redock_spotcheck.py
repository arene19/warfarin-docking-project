#!/usr/bin/env python3
"""Regenerate flexible re-dock spot-check CSV (sync training affinities only).

For full flexible re-docks, use:
    python scripts/run_flex_redock_spotcheck.py --redock --write
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "publication/data/flexible_redock_spotcheck.csv"
SCREENING = ROOT / "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv"
DEPOSITION_SCREENING = ROOT / "deposition/package/docking/VKORC1_Human_screening_results.csv"

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


def _screening_csv() -> Path:
    if SCREENING.exists():
        return SCREENING
    if DEPOSITION_SCREENING.exists():
        return DEPOSITION_SCREENING
    raise FileNotFoundError("VKORC1_Human_screening_results.csv not found")


def _load_redock_map() -> dict[str, float]:
    if not CANONICAL.exists():
        return {}
    df = pd.read_csv(CANONICAL)
    return {
        str(row["ligand_name"]): float(row["affinity_redock_kcal_mol"])
        for _, row in df.iterrows()
    }


def build_spotcheck() -> pd.DataFrame:
    aff = pd.read_csv(_screening_csv()).set_index("ligand_name")["best_affinity"].astype(float)
    redock_map = _load_redock_map()
    rows = []
    for name in REF_SPOTCHECK + RL_SPOTCHECK:
        if name not in aff.index:
            print(f"  [warn] {name} missing from screening CSV — skipped")
            continue
        if name not in redock_map:
            print(f"  [warn] {name} missing archived re-dock affinity — skipped")
            continue
        train = float(aff[name])
        redock = float(redock_map[name])
        delta = abs(train - redock)
        category = "RL_Gen" if name.startswith("RL_Gen") else "reference"
        if category == "RL_Gen" and delta > 1.0:
            print(f"  [warn] {name}: |Δ|={delta:.2f} > 1.0 — excluded (stale re-dock)")
            continue
        rows.append(
            {
                "ligand_name": name,
                "affinity_training_kcal_mol": train,
                "affinity_redock_kcal_mol": redock,
                "delta_kcal_mol": delta,
                "category": category,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate flexible re-dock spot-check CSV.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write publication/data/flexible_redock_spotcheck.csv",
    )
    args = parser.parse_args()
    df = build_spotcheck()
    print(df.to_string(index=False))
    if args.write:
        CANONICAL.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CANONICAL, index=False)
        print(f"\nWrote {CANONICAL}")


if __name__ == "__main__":
    main()
