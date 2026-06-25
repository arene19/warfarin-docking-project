#!/usr/bin/env python3
"""Dock MD candidate ligands against HSA (corrected enantiomers)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from protein_preparation import clean_protein, add_hydrogens_pdbfixer, protein_to_pdbqt
from docking_engine import virtual_screening

MD_CANDIDATES = [
    "S_Warfarin_ref",
    "p_nitro_R",
    "dimethoxy_23_S",
    "m_bromo_R",
    "RL_Gen_37",
    "RL_Gen_29_isoA",
]

ARCHIVED_HSA = {
    "S_Warfarin_ref": -9.416,
    "p_nitro_R": -8.856,
    "dimethoxy_23_S": -9.065,
    "m_bromo_R": -8.916,
    "p_nitro_S": -9.139,
    "BENZ_S": -10.013,
}

VKOR_HUMAN = {
    "S_Warfarin_ref": -10.924,
    "p_nitro_R": -12.123,
    "dimethoxy_23_S": -11.490,
    "m_bromo_R": -11.373,
    "RL_Gen_37": -12.244,
    "RL_Gen_29_isoA": -11.419,
}


def ensure_hsa_receptor() -> tuple[str, str]:
    master = yaml.safe_load(open(ROOT / "config_master.yaml"))
    hsa = master["receptors"]["HSA"]
    chain = hsa["chain"]
    protonated_pdb = ROOT / "proteins" / "protonated" / f"HSA_chain{chain}_protonated.pdb"
    receptor_pdbqt = ROOT / "pdbqt_receptors" / f"HSA_chain{chain}_protonated.pdbqt"
    raw_pdb = ROOT / "proteins" / "raw" / f"{hsa['pdb_id']}.pdb"

    if not protonated_pdb.exists():
        print(f"[PREP] Building {protonated_pdb.name} ...")
        protonated_pdb.parent.mkdir(parents=True, exist_ok=True)
        clean_pdb = clean_protein(str(raw_pdb), chain)
        add_hydrogens_pdbfixer(clean_pdb, str(protonated_pdb))

    if not receptor_pdbqt.exists():
        print(f"[PREP] Building {receptor_pdbqt.name} ...")
        receptor_pdbqt.parent.mkdir(parents=True, exist_ok=True)
        protein_to_pdbqt(str(protonated_pdb), str(receptor_pdbqt))

    return str(protonated_pdb), str(receptor_pdbqt)


def main() -> None:
    os.chdir(ROOT)
    protonated_pdb, receptor_pdbqt = ensure_hsa_receptor()

    master = yaml.safe_load(open(ROOT / "config_master.yaml"))
    hsa = master["receptors"]["HSA"]
    box_params = {
        "center": hsa["force_center"],
        "size": hsa["force_size"],
        "exhaustiveness": master["docking_params"]["exhaustiveness"],
        "num_modes": master["docking_params"]["num_modes"],
    }

    output_dir = ROOT / "results" / "docked_poses" / "HSA"
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [n for n in MD_CANDIDATES if not (ROOT / "pdbqt_ligands" / f"{n}.pdbqt").exists()]
    if missing:
        raise SystemExit(f"Missing PDBQT ligands: {missing}")

    print(f"\n>>> Docking {len(MD_CANDIDATES)} MD candidates -> HSA")
    df = virtual_screening(
        ligand_pdbqt_dir="pdbqt_ligands",
        receptor_pdbqt=receptor_pdbqt,
        target_name="HSA",
        box_params=box_params,
        output_dir=str(output_dir),
        exhaustiveness=box_params["exhaustiveness"],
        receptor_pdb=protonated_pdb,
        flex_res_list=hsa.get("flexible_residues", []),
        needed_ligands=MD_CANDIDATES,
        num_modes=box_params["num_modes"],
        min_rmsd=master["docking_params"].get("min_rmsd", 1.0),
        n_cpu=master["docking_params"].get("n_cpu"),
        dock_timeout=master["docking_params"].get("dock_timeout_s", 900),
    )

    csv_path = output_dir / "HSA_screening_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Wrote {csv_path}")

    print("\n=== HSA DOCKING RESULTS (corrected enantiomers) ===")
    print(f"{'Ligand':<18} {'HSA_new':>8} {'HSA_old':>8} {'VKOR':>8} {'Sel_new':>8} {'Sel_old':>8}")
    print("-" * 70)
    rows = []
    for lig in MD_CANDIDATES:
        row = df[df["ligand_name"] == lig].iloc[0]
        hsa_new = float(row["best_affinity"])
        hsa_old = ARCHIVED_HSA.get(lig)
        vkor = VKOR_HUMAN[lig]
        sel_new = hsa_new - vkor
        sel_old = (hsa_old - vkor) if hsa_old is not None else None
        rows.append((lig, hsa_new, hsa_old, vkor, sel_new, sel_old))
        old_s = f"{hsa_old:8.3f}" if hsa_old is not None else "     n/a"
        old_sel = f"{sel_old:8.2f}" if sel_old is not None else "     n/a"
        print(f"{lig:<18} {hsa_new:8.3f} {old_s} {vkor:8.3f} {sel_new:8.2f} {old_sel}")

    out = ROOT / "results" / "md_hsa_redock_summary.txt"
    with open(out, "w") as f:
        f.write("MD candidate HSA re-dock (corrected enantiomers)\n")
        f.write("=" * 70 + "\n\n")
        for lig, hsa_new, hsa_old, vkor, sel_new, sel_old in rows:
            f.write(f"{lig}\n")
            f.write(f"  HSA (new): {hsa_new:.3f}\n")
            if hsa_old is not None:
                f.write(f"  HSA (archived): {hsa_old:.3f}  (delta {hsa_new - hsa_old:+.3f})\n")
            f.write(f"  VKOR Human: {vkor:.3f}\n")
            f.write(f"  Selectivity HSA-VKOR (new): {sel_new:.3f}\n")
            if sel_old is not None:
                f.write(f"  Selectivity HSA-VKOR (old): {sel_old:.3f}\n")
            f.write("\n")
    print(f"\n[OK] Summary -> {out}")


if __name__ == "__main__":
    main()
