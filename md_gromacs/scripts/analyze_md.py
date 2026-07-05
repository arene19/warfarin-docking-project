#!/usr/bin/env python3
"""Analyze GROMACS MD: protein Cα RMSD, ligand RMSD, H-bonds to VKORC1 key residues."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hbond_mapping import load_mapping

MDG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = MDG_ROOT / "runs"


def run(cmd: list[str], cwd: Path, input_text: str | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text.encode() if input_text else None,
        check=True,
    )


def find_trajectory(run_dir: Path) -> tuple[Path, Path]:
    for name in ("md_100ns", "md_20ns", "npt"):
        tpr = run_dir / "production" / f"{name}.tpr"
        xtc = run_dir / "production" / f"{name}.xtc"
        if tpr.exists() and xtc.exists():
            return tpr, xtc
        tpr = run_dir / f"{name}.tpr"
        xtc = run_dir / f"{name}.xtc"
        if tpr.exists() and xtc.exists():
            return tpr, xtc
    raise FileNotFoundError(f"No .tpr/.xtc pair found under {run_dir}")


def make_index(run_dir: Path, gmx: str) -> Path:
    """Build index with Protein_CA and Ligand groups."""
    ndx = run_dir / "analysis" / "analysis.ndx"
    if ndx.exists():
        return ndx
    gro = run_dir / "production" / "md_100ns.gro"
    if not gro.exists():
        gro = run_dir / "production" / "md_20ns.gro"
    if not gro.exists():
        gro = run_dir / "npt.gro"
    if not gro.exists():
        gro = run_dir / "charmm_gui_export" / "step5_input.gro"
    script = "r CA & Protein\nname 13 Protein_CA\nr LIG\nname 14 Ligand\nq\n"
    ndx.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [gmx, "make_ndx", "-f", str(gro), "-o", str(ndx)],
        cwd=run_dir,
        input=script.encode(),
        check=True,
    )
    return ndx


def parse_xvg_counts(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            values.append(float(parts[1]))
    return values


def summarize_hbond_xvg(path: Path) -> dict[str, float]:
    counts = parse_xvg_counts(path)
    if not counts:
        return {"frames": 0, "mean_count": 0.0, "occupancy_pct": 0.0, "max_count": 0.0}
    occupied = sum(1 for c in counts if c > 0)
    return {
        "frames": len(counts),
        "mean_count": round(sum(counts) / len(counts), 4),
        "occupancy_pct": round(100.0 * occupied / len(counts), 2),
        "max_count": max(counts),
    }


def rmsd_analysis(run_dir: Path, gmx: str, tpr: Path, xtc: Path, ndx: Path, out: Path) -> None:
    xvg = out / "rmsd_protein.xvg"
    run(
        [gmx, "rms", "-s", str(tpr), "-f", str(xtc), "-o", str(xvg), "-n", str(ndx)],
        run_dir,
        input_text="13\n13\n",
    )
    xvg_lig = out / "rmsd_ligand.xvg"
    run(
        [gmx, "rms", "-s", str(tpr), "-f", str(xtc), "-o", str(xvg_lig), "-n", str(ndx)],
        run_dir,
        input_text="13\n14\n",
    )


def hbond_analysis(run_dir: Path, gmx: str, tpr: Path, xtc: Path, out: Path) -> dict:
    """H-bonds between ligand (LIG) and each mapped VKORC1 binding-site residue."""
    residues = load_mapping(run_dir)
    summary_lines = [
        "# H-bond occupancy (ligand ↔ key VKORC1 residues)",
        "# plip_label: manuscript/PLIP name | gmx_resid: resid in production .tpr",
        "",
    ]
    results: list[dict] = []

    for entry in residues:
        plip = entry["plip_label"]
        resname = entry["resname"]
        gmx_resid = entry["gmx_resid"]
        xvg = out / f"hbond_LIG_{plip}_gmx{gmx_resid}.xvg"
        record = dict(entry)
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
                    "-r",
                    "resname LIG",
                    "-t",
                    f"resid {gmx_resid}",
                ],
                run_dir,
            )
            stats = summarize_hbond_xvg(xvg)
            record.update(stats)
            record["status"] = "ok"
            record["xvg"] = xvg.name
            summary_lines.append(
                f"{plip} (gmx resid {gmx_resid}): occupancy={stats['occupancy_pct']:.1f}% "
                f"mean={stats['mean_count']:.3f} max={stats['max_count']:.0f} -> {xvg.name}"
            )
        except subprocess.CalledProcessError:
            record["status"] = "failed"
            record["xvg"] = None
            summary_lines.append(f"{plip} (gmx resid {gmx_resid}): FAILED")
        results.append(record)

    summary_path = out / "hbond_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    payload = {"residues": results}
    json_path = out / "hbond_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, help="System ID, e.g. RL_Gen_37")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory")
    parser.add_argument("--gmx", default="gmx", help="GROMACS command")
    parser.add_argument("--hbond-only", action="store_true", help="Skip RMSD; run H-bond analysis only")
    parser.add_argument("--skip-rmsd", action="store_true", help="Alias for --hbond-only")
    args = parser.parse_args()

    hbond_only = args.hbond_only or args.skip_rmsd
    run_dir = args.run_dir or (DEFAULT_RUNS / args.system)
    if not run_dir.is_dir():
        sys.exit(f"Run directory not found: {run_dir}")

    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    tpr, xtc = find_trajectory(run_dir)
    print(f"Using trajectory: {tpr.name} + {xtc.name}")

    if not hbond_only:
        ndx = make_index(run_dir, args.gmx)
        rmsd_analysis(run_dir, args.gmx, tpr, xtc, ndx, out)

    payload = hbond_analysis(run_dir, args.gmx, tpr, xtc, out)
    print(f"\nAnalysis outputs in: {out}")
    for row in payload["residues"]:
        if row.get("status") == "ok":
            print(
                f"  {row['plip_label']:8s} gmx{row['gmx_resid']:3d}  "
                f"occupancy={row['occupancy_pct']:5.1f}%  mean={row['mean_count']:.3f}"
            )


if __name__ == "__main__":
    main()
