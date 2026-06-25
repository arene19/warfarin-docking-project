#!/usr/bin/env python3
"""Analyze GROMACS MD: protein Cα RMSD, ligand RMSD, H-bonds to VKORC1 key residues."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# VKORC1_Human chain A (6WV3) — PLIP interaction hotspots
HBOND_RESIDUES = {
    80: "ASN",
    81: "SER",
    139: "TYR",
    55: "PHE",
    59: "TRP",
    138: "THR",
    134: "VAL",
}

MDG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = MDG_ROOT / "runs"


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def find_trajectory(run_dir: Path) -> tuple[Path, Path]:
    for name in ("md_100ns", "md_20ns", "npt"):
        tpr = run_dir / f"{name}.tpr"
        xtc = run_dir / f"{name}.xtc"
        if tpr.exists() and xtc.exists():
            return tpr, xtc
    raise FileNotFoundError(f"No .tpr/.xtc pair found in {run_dir}")


def make_index(run_dir: Path, gmx: str) -> Path:
    """Build index with Protein_CA and Ligand groups."""
    ndx = run_dir / "analysis.ndx"
    if ndx.exists():
        return ndx
    gro = run_dir / "npt.gro"
    if not gro.exists():
        gro = run_dir / "step5_input.gro"
    script = "r CA & Protein\nname 13 Protein_CA\nr LIG\nname 14 Ligand\nq\n"
    subprocess.run(
        [gmx, "make_ndx", "-f", str(gro), "-o", str(ndx)],
        cwd=run_dir,
        input=script.encode(),
        check=True,
    )
    return ndx


def rmsd_analysis(run_dir: Path, gmx: str, tpr: Path, xtc: Path, ndx: Path, out: Path) -> None:
    xvg = out / "rmsd_protein_ca.xvg"
    run(
        [
            gmx,
            "rms",
            "-s",
            str(tpr),
            "-f",
            str(xtc),
            "-o",
            str(xvg),
            "-n",
            str(ndx),
        ],
        run_dir,
    )
    # Ligand RMSD (fit on protein, rms ligand)
    xvg_lig = out / "rmsd_ligand.xvg"
    run(
        [
            gmx,
            "rms",
            "-s",
            str(tpr),
            "-f",
            str(xtc),
            "-o",
            str(xvg_lig),
            "-n",
            str(ndx),
        ],
        run_dir,
    )


def hbond_analysis(run_dir: Path, gmx: str, tpr: Path, xtc: Path, out: Path) -> None:
    """H-bonds between ligand (LIG) and each key receptor residue."""
    summary_lines = ["# H-bond occupancy (ligand ↔ key VKORC1 residues)", ""]
    for resnum, resname in HBOND_RESIDUES.items():
        xvg = out / f"hbond_LIG_{resname}{resnum}.xvg"
        sel = f"resname LIG and (resid {resnum} or name *{resname}*)"
        try:
            run(
                [
                    gmx,
                    "hbond",
                    "-s",
                    str(tpr),
                    "-f",
                    str(xtc),
                    "-num",
                    str(xvg),
                    "-don",
                    "resname LIG",
                    "-acc",
                    f"resid {resnum}",
                ],
                run_dir,
            )
            summary_lines.append(f"{resname}{resnum}: {xvg.name}")
        except subprocess.CalledProcessError:
            summary_lines.append(f"{resname}{resnum}: FAILED (check resid numbering in topol)")
    (out / "hbond_summary.txt").write_text("\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, help="System ID, e.g. S_Warfarin_ref")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory (default: md_gromacs/runs/SYSTEM)")
    parser.add_argument("--gmx", default="gmx", help="GROMACS command")
    args = parser.parse_args()

    run_dir = args.run_dir or (DEFAULT_RUNS / args.system)
    if not run_dir.is_dir():
        sys.exit(f"Run directory not found: {run_dir}")

    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    tpr, xtc = find_trajectory(run_dir)
    print(f"Using trajectory: {xtc.name}")

    ndx = make_index(run_dir, args.gmx)
    rmsd_analysis(run_dir, args.gmx, tpr, xtc, ndx, out)
    hbond_analysis(run_dir, args.gmx, tpr, xtc, out)

    print(f"\nAnalysis outputs in: {out}")
    print("Plot with: xmgrace or python matplotlib on *.xvg files")


if __name__ == "__main__":
    main()
