"""Map PLIP-style VKORC1 hotspot labels to GROMACS resid numbers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MDG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = MDG_ROOT / "config" / "vkorc1_hbond_residues.json"

# PLIP manuscript labels -> expected amino acid at binding site
PLIP_HOTSPOTS: list[tuple[int, str]] = [
    (80, "ASN"),
    (81, "SER"),
    (139, "TYR"),
    (55, "PHE"),
    (59, "TRP"),
    (138, "THR"),
    (134, "VAL"),
]


def _parse_ca_rows(path: Path) -> list[tuple[int, str, str, tuple[float, float, float]]]:
    """Return (resid, resname, segment, xyz) for each protein CA."""
    rows: list[tuple[int, str, str, tuple[float, float, float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM") or " CA " not in line:
            continue
        resname = line[17:20].strip()
        if resname in {"LIG", "TIP3", "SOD", "CLA", "POT", "CAL", "CHL1"}:
            continue
        resnum = int(line[22:26])
        seg = line[72:76].strip() if len(line) >= 76 else ""
        xyz = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
        rows.append((resnum, resname, seg, xyz))
    return rows


def _chain_a_ca(path: Path) -> list[tuple[int, str, tuple[float, float, float]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM") or " CA " not in line:
            continue
        chain = line[21].strip() or "A"
        if chain != "A":
            continue
        resname = line[17:20].strip()
        resnum = int(line[22:26])
        xyz = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
        rows.append((resnum, resname, xyz))
    return rows


def _ligand_xyz(path: Path) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip()
        if resname != "LIG":
            continue
        name = line[12:16].strip()
        if name.startswith("H"):
            continue
        coords.append(
            (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        )
    return coords


def _min_dist(xyz: tuple[float, float, float], lig: list[tuple[float, float, float]]) -> float:
    return min(math.dist(xyz, l) for l in lig)


def resolve_from_complex(complex_pdb: Path, step5_pdb: Path) -> list[dict[str, Any]]:
    """Resolve PLIP labels to GMX resids using binding-site proximity + sequence index."""
    lig = _ligand_xyz(complex_pdb)
    if not lig:
        raise ValueError(f"No LIG heavy atoms found in {complex_pdb}")

    ref = _chain_a_ca(complex_pdb)
    gmx = _parse_ca_rows(step5_pdb)
    if len(ref) != len(gmx):
        raise ValueError(f"Protein length mismatch: ref={len(ref)} gmx={len(gmx)}")

    ref_index = {resnum: i for i, (resnum, _, _) in enumerate(ref)}
    mapping: list[dict[str, Any]] = []

    for plip_num, resname in PLIP_HOTSPOTS:
        cands = [
            (resnum, _min_dist(xyz, lig))
            for resnum, rn, xyz in [(r[0], r[1], r[2]) for r in ref]
            if rn == resname
        ]
        if not cands:
            raise ValueError(f"No {resname} found near ligand in {complex_pdb}")
        cands.sort(key=lambda x: x[1])
        ref_resid = cands[0][0]
        idx = ref_index[ref_resid]
        gmx_resid, gmx_rn, gmx_seg, _ = (
            gmx[idx][0],
            gmx[idx][1],
            gmx[idx][2],
            gmx[idx][3],
        )
        if gmx_rn != resname:
            raise ValueError(
                f"Sequence mismatch at index {idx}: expected {resname}, got {gmx_rn}"
            )
        mapping.append(
            {
                "plip_label": f"{resname}{plip_num}",
                "resname": resname,
                "ref_resid": ref_resid,
                "gmx_resid": gmx_resid,
                "gmx_segment": gmx_seg,
                "ligand_min_dist_A": round(cands[0][1], 3),
            }
        )
    return mapping


def load_mapping(run_dir: Path) -> list[dict[str, Any]]:
    cached = run_dir / "analysis" / "hbond_residue_map.json"
    complex_pdb = run_dir / "input" / "complex_clean_charmm.pdb"
    step5 = run_dir / "charmm_gui_export" / "step5_input.pdb"

    if complex_pdb.is_file() and step5.is_file():
        residues = resolve_from_complex(complex_pdb, step5)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"residues": residues}, indent=2) + "\n", encoding="utf-8")
        return residues

    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))["residues"]

    if DEFAULT_MAP.is_file():
        data = json.loads(DEFAULT_MAP.read_text(encoding="utf-8"))
        return data["residues"]

    raise FileNotFoundError(
        f"Cannot resolve H-bond residues for {run_dir.name}: "
        "need input/complex_clean_charmm.pdb and charmm_gui_export/step5_input.pdb"
    )
