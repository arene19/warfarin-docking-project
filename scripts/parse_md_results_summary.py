#!/usr/bin/env python3
"""Merge MD package summary, per-run .xvg, and hbond JSON into publication/data/md_results_summary.json."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "publication/data/md_results_summary.json"
LEGACY_RL37 = ROOT / "publication/data/md_rl_gen_37_summary.json"
PACKAGE_SUMMARY = ROOT / "publication/data/md/MD_RESULTS_SUMMARY.json"
RUNS = ROOT / "md_gromacs/runs"

SYSTEMS = [
    {
        "run_id": "RL_Gen_37",
        "manuscript_label": "RL_Gen_37_isoA",
        "docking_flat": "RL_Gen_37",
        "docking_isoA": "RL_Gen_37_isoA",
    },
    {
        "run_id": "RL_Gen_29_isoA",
        "manuscript_label": "RL_Gen_29_isoA",
        "docking_flat": None,
        "docking_isoA": "RL_Gen_29_isoA",
    },
    {
        "run_id": "S_Warfarin_ref",
        "manuscript_label": "S-warfarin (ref)",
        "docking_flat": None,
        "docking_isoA": None,
    },
]


def read_xvg(path: Path) -> np.ndarray:
    ys: list[float] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                ys.append(float(parts[1]))
    return np.array(ys, dtype=float)


def summarize_rmsd(y: np.ndarray) -> dict[str, float]:
    if y.size == 0:
        return {
            "mean": float("nan"),
            "last25pct_mean": float("nan"),
            "second_half_mean": float("nan"),
            "max": float("nan"),
        }
    n = y.size
    half = n // 2
    last25 = max(1, n // 4)
    return {
        "mean": float(np.mean(y)),
        "last25pct_mean": float(np.mean(y[-last25:])),
        "second_half_mean": float(np.mean(y[half:])),
        "max": float(np.max(y)),
    }


def load_package_summary() -> dict[str, dict]:
    if not PACKAGE_SUMMARY.exists():
        return {}
    rows = json.loads(PACKAGE_SUMMARY.read_text(encoding="utf-8"))
    return {row["system_id"]: row for row in rows}


def load_hbond(run_dir: Path) -> list[dict]:
    path = run_dir / "analysis/hbond_summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("residues", [])


def build_system_entry(meta: dict, pkg: dict | None) -> dict:
    run_id = meta["run_id"]
    adir = RUNS / run_id / "analysis"
    entry: dict = {
        "run_id": run_id,
        "manuscript_label": meta["manuscript_label"],
        "docking_flat_ligand": meta["docking_flat"],
        "docking_isoA_ligand": meta["docking_isoA"],
        "status": "Complete",
    }

    if pkg:
        entry.update(
            {
                "production_ns": pkg["length_ns"],
                "temperature_K": round(pkg["temperature_K"], 2),
                "pressure_bar": round(pkg["pressure_bar"], 2),
                "protein_rmsd_mean_A": pkg["protein_rmsd"]["full_mean_A"],
                "protein_rmsd_second_half_mean_A": pkg["protein_rmsd"]["second_half_mean_A"],
                "protein_rmsd_last25pct_A": pkg["protein_rmsd"]["last25_mean_A"],
                "protein_rmsd_max_A": pkg["protein_rmsd"]["full_max_A"],
                "ligand_rmsd_mean_A": pkg["ligand_rmsd"]["full_mean_A"],
                "ligand_rmsd_2nd_half_mean_A": pkg["ligand_rmsd"]["second_half_mean_A"],
                "ligand_rmsd_last25pct_A": pkg["ligand_rmsd"]["last25_mean_A"],
                "ligand_rmsd_max_A": pkg["ligand_rmsd"]["full_max_A"],
                "trajectory": pkg.get("trajectory"),
            }
        )
    else:
        prot_path = next(
            (p for p in (adir / "rmsd_protein_ca.xvg", adir / "rmsd_protein.xvg") if p.exists()),
            None,
        )
        lig_path = adir / "rmsd_ligand.xvg" if (adir / "rmsd_ligand.xvg").exists() else None
        if prot_path and lig_path:
            prot = summarize_rmsd(read_xvg(prot_path))
            lig = summarize_rmsd(read_xvg(lig_path))
            entry.update(
                {
                    "protein_rmsd_mean_A": round(prot["mean"], 3),
                    "protein_rmsd_last25pct_A": round(prot["last25pct_mean"], 3),
                    "ligand_rmsd_2nd_half_mean_A": round(lig["second_half_mean"], 3),
                    "ligand_rmsd_max_A": round(lig["max"], 3),
                    "provenance_xvg": True,
                }
            )

    hbonds = load_hbond(RUNS / run_id)
    entry["hbond_residues"] = [
        {
            "plip_label": r["plip_label"],
            "gmx_resid": r["gmx_resid"],
            "occupancy_pct": r["occupancy_pct"],
        }
        for r in hbonds
    ]
    asn = next((r for r in hbonds if r["plip_label"] == "ASN80"), None)
    ser = next((r for r in hbonds if r["plip_label"] == "SER81"), None)
    if asn:
        entry["hbond_ASN80_occupancy_pct"] = asn["occupancy_pct"]
    if ser:
        entry["hbond_SER81_occupancy_pct"] = ser["occupancy_pct"]

    return entry


def legacy_rl37_payload(system: dict) -> dict:
    return {
        "system_id": "RL_Gen_37_isoA",
        "production_ns": system.get("production_ns", 100),
        "temperature_K": system.get("temperature_K"),
        "pressure_bar": system.get("pressure_bar"),
        "protein_rmsd_mean_A": system.get("protein_rmsd_mean_A"),
        "protein_rmsd_last25pct_A": system.get("protein_rmsd_last25pct_A"),
        "ligand_rmsd_2nd_half_mean_A": system.get("ligand_rmsd_2nd_half_mean_A"),
        "ligand_rmsd_max_A": system.get("ligand_rmsd_max_A"),
        "status": "Complete",
        "provenance": "parsed_from_md_results_summary",
        "source_notes": "Derived from publication/data/md_results_summary.json (RL_Gen_37 run folder).",
        "parsed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical MD results summary JSON.")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    pkg_by_id = load_package_summary()
    systems = [build_system_entry(meta, pkg_by_id.get(meta["run_id"])) for meta in SYSTEMS]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_package_summary": str(PACKAGE_SUMMARY.relative_to(ROOT)),
        "systems": systems,
        "naming_notes": "Run folder RL_Gen_37 corresponds to manuscript label RL_Gen_37_isoA.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

    rl37 = next(s for s in systems if s["run_id"] == "RL_Gen_37")
    LEGACY_RL37.write_text(json.dumps(legacy_rl37_payload(rl37), indent=2), encoding="utf-8")
    print(f"Wrote {LEGACY_RL37}")


if __name__ == "__main__":
    main()
