#!/usr/bin/env python3
"""Build MD-ready VKORC1_Human + ligand complex PDBs (MODEL 1 poses)."""
from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LIGANDS = [
    "S_Warfarin_ref",
    "p_nitro_R",
    "p_nitro_S",
    "dimethoxy_23_S",
    "RL_Gen_37",
    "RL_Gen_29_isoA",
    "RL_Gen_22",
    "RL_Gen_45",
]

TARGET = "VKORC1_Human"
RECEPTOR_SRC = ROOT / "proteins/protonated/VKORC1_Human_chainA_protonated.pdb"
POSE_DIR = ROOT / "results/docked_poses/VKORC1_Human"
OUT = ROOT / "md_poses"
COMPLEX_DIR = OUT / "complexes"
RECEPTOR_DIR = OUT / "receptor"
LIGAND_DIR = OUT / "ligands"

LIGAND_RESNAME = "LIG"
LIGAND_CHAIN = "B"


def pdbqt_model1_to_ligand_lines(pdbqt_path: Path) -> list[str]:
    """Extract MODEL 1 from PDBQT as HETATM records with element columns."""
    lines: list[str] = []
    in_model = False
    for raw in pdbqt_path.read_text().splitlines():
        if raw.startswith("MODEL"):
            in_model = True
            continue
        if raw.startswith("ENDMDL"):
            break
        if not in_model:
            continue
        if not (raw.startswith("ATOM") or raw.startswith("HETATM")):
            continue
        element = raw[76:78].strip() if len(raw) >= 78 else ""
        if element:
            el = element[0].upper()
            if el not in {"C", "O", "N", "S", "H", "P", "F", "B", "I", "CL", "BR"}:
                el = "C"
        else:
            el = "C"
        coord = raw[:66]
        lines.append((coord, el))
    if not lines:
        raise ValueError(f"No MODEL 1 atoms in {pdbqt_path}")
    return lines


def read_protein_lines(receptor_pdb: Path) -> tuple[list[str], int]:
    protein: list[str] = []
    max_serial = 0
    for line in receptor_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                max_serial = max(max_serial, int(line[6:11]))
            except ValueError:
                pass
            protein.append(line)
        elif line.startswith("TER"):
            protein.append(line)
    return protein, max_serial


def format_ligand_atom(
    serial: int,
    coord_line: str,
    element: str,
    resname: str = LIGAND_RESNAME,
    chain: str = LIGAND_CHAIN,
    resseq: int = 1,
) -> str:
    x = float(coord_line[30:38])
    y = float(coord_line[38:46])
    z = float(coord_line[46:54])
    name = coord_line[12:16].strip()
    if len(name) > 4:
        name = name[:4]
    atom_field = f"{name:>4s}"[:4]
    return (
        f"HETATM{serial:5d} {atom_field} {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def write_complex(
    receptor_pdb: Path,
    pdbqt_path: Path,
    out_complex: Path,
    out_ligand: Path,
) -> int:
    protein_lines, max_serial = read_protein_lines(receptor_pdb)
    lig_atoms = pdbqt_model1_to_ligand_lines(pdbqt_path)

    lig_lines: list[str] = []
    serial = max_serial
    for coord, element in lig_atoms:
        serial += 1
        lig_lines.append(format_ligand_atom(serial, coord, element))

    out_ligand.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ligand, "w", encoding="utf-8") as f:
        f.writelines(lig_lines)
        f.write("END\n")

    out_complex.parent.mkdir(parents=True, exist_ok=True)
    with open(out_complex, "w", encoding="utf-8") as f:
        for line in protein_lines:
            f.write(line + "\n")
        f.write("TER\n")
        f.writelines(lig_lines)
        f.write("END\n")
    return len(lig_lines)


def parse_vina_affinity(pdbqt_path: Path) -> float | None:
    text = pdbqt_path.read_text()
    m = re.search(r"REMARK VINA RESULT:\s+([-\d.]+)", text)
    return float(m.group(1)) if m else None


def load_hsa_affinities() -> dict[str, float]:
    csv_path = ROOT / "results/docked_poses/HSA/HSA_screening_results.csv"
    out: dict[str, float] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[row["ligand_name"]] = float(row["best_affinity"])
            except (KeyError, ValueError):
                pass
    return out


def main() -> None:
    if not RECEPTOR_SRC.exists():
        raise SystemExit(f"Missing receptor: {RECEPTOR_SRC}")

    RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)
    receptor_dst = RECEPTOR_DIR / RECEPTOR_SRC.name
    shutil.copy2(RECEPTOR_SRC, receptor_dst)

    hsa = load_hsa_affinities()
    manifest_rows = []

    for lig in LIGANDS:
        pdbqt = POSE_DIR / f"{lig}_{TARGET}_docked.pdbqt"
        if not pdbqt.exists():
            raise SystemExit(f"Missing pose: {pdbqt}")

        complex_path = COMPLEX_DIR / f"{lig}_{TARGET}_model1_complex.pdb"
        ligand_path = LIGAND_DIR / f"{lig}_model1_ligand.pdb"
        n_atoms = write_complex(RECEPTOR_SRC, pdbqt, complex_path, ligand_path)
        vkor = parse_vina_affinity(pdbqt)
        hsa_aff = hsa.get(lig)
        sel = (hsa_aff - vkor) if hsa_aff is not None and vkor is not None else None

        manifest_rows.append(
            {
                "ligand_name": lig,
                "receptor": RECEPTOR_SRC.name,
                "pose_source": str(pdbqt.relative_to(ROOT)),
                "model": 1,
                "VKOR_dG_kcal_mol": vkor,
                "HSA_dG_kcal_mol": hsa_aff,
                "selectivity_HSA_minus_VKOR": sel,
                "complex_pdb": str(complex_path.relative_to(ROOT)),
                "ligand_pdb": str(ligand_path.relative_to(ROOT)),
                "ligand_atoms": n_atoms,
                "ligand_resname": LIGAND_RESNAME,
                "ligand_chain": LIGAND_CHAIN,
                "enantiomer_note": {
                    "RL_Gen_37": "RL_Gen_37: flat dock MODEL 1; 3D embed = CIP R (isoA)",
                    "RL_Gen_22": "RL_Gen_22: flat dock MODEL 1; CGenFF uses isoA SMILES",
                    "RL_Gen_45": "RL_Gen_45: flat dock MODEL 1; spirocyclic scaffold",
                }.get(lig, "corrected R/S per config.yaml"),
            }
        )
        print(f"[OK] {lig} -> {complex_path.name} ({n_atoms} ligand atoms, VKOR={vkor:.3f})")

    manifest_path = OUT / "md_pose_manifest.csv"
    fieldnames = list(manifest_rows[0].keys())
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    readme = OUT / "README_MD_POSES.md"
    readme.write_text(
        """# MD pose package — VKORC1_Human (corrected enantiomers)

Prepared for molecular dynamics starting structures.

## Contents

- `receptor/VKORC1_Human_chainA_protonated.pdb` — protonated receptor (6WV3, chain A)
- `complexes/*_VKORC1_Human_model1_complex.pdb` — receptor + ligand (MODEL 1 dock pose)
- `ligands/*_model1_ligand.pdb` — ligand only (chain B, resname `LIG`)
- `md_pose_manifest.csv` — affinities and file paths

## Ligands (8)

| Ligand | Role |
|--------|------|
| S_Warfarin_ref | Clinical S-warfarin reference |
| p_nitro_R | Top para-nitro (R) reference |
| p_nitro_S | Para-nitro (S) stereochemistry pair |
| dimethoxy_23_S | Dimethoxy reference (S) |
| RL_Gen_37 | Top GNN/REINVENT hit (manuscript MD-1) |
| RL_Gen_29_isoA | GNN hit; rich H-bond network (manuscript MD-2) |
| RL_Gen_22 | Third-ranked RL hit (manuscript MD-3) |
| RL_Gen_45 | Fourth-ranked RL hit; spirocyclic (manuscript MD-4) |

## Docking provenance

- **Target:** VKORC1_Human, flexible side-chains (A:217, A:276, A:269, A:272)
- **Pose:** AutoDock Vina MODEL 1 (best score) from corrected-enantiomer screening
- **HSA values:** from `results/docked_poses/HSA/` (corrected enantiomers, Jun 2026)

## Suggested MD workflow

1. Load complex PDB in GROMACS/AMBER/OpenMM
2. Assign ligand charges (AM1-BCC / RESP) — ligand PDB has no charges beyond 0.00 occupancy
3. Solvate, ionize, minimize, equilibrate, production
4. Optional: MM-PBSA on production frames

## RL_Gen_37 stereochemistry

Use the flat-dock MODEL 1 pose with **R** configuration (CIP R from 3D embed matches isoA).
""",
        encoding="utf-8",
    )
    print(f"\n[OK] Manifest -> {manifest_path}")
    print(f"[OK] README -> {readme}")


if __name__ == "__main__":
    main()
