#!/usr/bin/env python3
"""
Receptor setup validation: PDB availability, native ligand, grid box, flex residues,
reference RMSD, and distance-based pocket contacts (works without PLIP).

Used by app.py and runnable standalone:
  python receptor_validation.py --target VKORC1_Human
  python receptor_validation.py --target VKORC1_Human --check-pose S_Warfarin_ref
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from Bio.PDB import PDBParser

from analysis import calculate_docking_rmsd, run_plip_analysis
from grid_box import auto_box_from_native_ligand
from protein_preparation import download_pdb

# UniProt Q9BQB6 Asn80 -> PDB 6WV3 chain A Asn222 (warfarin pocket literature residue)
VKORC1_KEY_CONTACTS = [{"chain": "A", "resnum": 222, "label": "Asn222 (UniProt Asn80)"}]


@dataclass
class CheckResult:
    check_id: str
    label: str
    passed: bool
    detail: str = ""
    value: Any = None


@dataclass
class ValidationReport:
    target_name: str
    checks: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def overall_pass(self) -> bool:
        required = [c for c in self.checks if not c.check_id.startswith("optional_")]
        return bool(required) and all(c.passed for c in required)

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "overall_pass": self.overall_pass,
            "checks": [asdict(c) for c in self.checks],
            "warnings": self.warnings,
        }


def _parse_flex_residue(token: str) -> tuple[str, int] | None:
    token = token.strip()
    m = re.match(r"^([A-Za-z0-9]):(\d+)$", token)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _receptor_paths(target_name: str, chain: str, pdb_id: str) -> dict[str, str]:
    return {
        "raw_pdb": os.path.join("proteins/raw", f"{pdb_id}.pdb"),
        "protonated_pdb": os.path.join("proteins/protonated", f"{target_name}_chain{chain}_protonated.pdb"),
        "receptor_pdbqt": os.path.join("pdbqt_receptors", f"{target_name}_chain{chain}_protonated.pdbqt"),
        "flex_rigid": os.path.join("results/pdbqt_receptors", f"{target_name}_chain{chain}_protonated_rigid.pdbqt"),
        "flex_side": os.path.join("results/pdbqt_receptors", f"{target_name}_chain{chain}_protonated_flex.pdbqt"),
    }


def _native_ligand_info(pdb_path: str, resname: str, chain_id: str | None = None) -> dict:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    hits = []
    for model in structure:
        for chain in model:
            if chain_id and chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_resname().strip() == resname.strip():
                    atoms = [a for a in residue if a.element != "H"]
                    hits.append({
                        "chain": chain.id,
                        "resnum": residue.id[1],
                        "resname": residue.get_resname(),
                        "n_atoms": len(list(residue.get_atoms())),
                        "n_heavy": len(atoms),
                        "centroid": np.mean([a.coord for a in atoms], axis=0).tolist() if atoms else None,
                    })
    return {"found": bool(hits), "instances": hits}


def _residue_exists(pdb_path: str, chain_id: str, resnum: int) -> bool:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    for model in structure:
        if chain_id not in model:
            return False
        for residue in model[chain_id]:
            if residue.id[0] == " " and residue.id[1] == resnum:
                return True
    return False


def _residues_near_ligand(pdb_path: str, resname: str, chain_id: str, cutoff: float = 5.0) -> list[dict]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    lig_coords = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_resname().strip() == resname.strip():
                    lig_coords.extend(a.coord for a in residue if a.element != "H")
    if not lig_coords:
        return []
    lig_c = np.mean(lig_coords, axis=0)
    nearby = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                heavy = [a for a in residue if a.element != "H"]
                if not heavy:
                    continue
                d = min(float(np.linalg.norm(a.coord - lig_c)) for a in heavy)
                if d <= cutoff:
                    nearby.append({
                        "chain": chain.id,
                        "resnum": residue.id[1],
                        "resname": residue.get_resname(),
                        "distance_A": round(d, 2),
                    })
    nearby.sort(key=lambda x: x["distance_A"])
    return nearby


def _pdbqt_to_temp_pdb(pdbqt_path: str, out_pdb: str) -> bool:
    try:
        with open(pdbqt_path, "r") as f_in, open(out_pdb, "w") as f_out:
            for line in f_in:
                if line.startswith("ENDMDL"):
                    break
                if line.startswith(("ATOM", "HETATM")):
                    element = line[76:78].strip() or "C"
                    element = element[0].upper()
                    f_out.write(f"{line[:66]:<76}{element:>2}\n")
        return os.path.getsize(out_pdb) > 0
    except OSError:
        return False


def _min_ligand_residue_distance(ligand_pdb: str, receptor_pdb: str, chain_id: str, resnum: int) -> float | None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", receptor_pdb)
    target = None
    for model in structure:
        if chain_id not in model:
            return None
        for residue in model[chain_id]:
            if residue.id[0] == " " and residue.id[1] == resnum:
                target = residue
                break
    if target is None:
        return None
    rec_coords = [a.coord for a in target if a.element != "H"]
    if not rec_coords:
        return None

    lig = parser.get_structure("lig", ligand_pdb)
    lig_coords = [a.coord for a in lig.get_atoms() if a.element != "H"]
    if not lig_coords:
        return None

    return float(min(np.linalg.norm(rc - lc) for rc in rec_coords for lc in lig_coords))


def find_docked_pose(target_name: str, ligand_name: str) -> str | None:
    candidates = [
        os.path.join("results/docked_poses", target_name, f"{ligand_name}_{target_name}_docked.pdbqt"),
        os.path.join("results/docked_poses", target_name, f"{ligand_name}_docked.pdbqt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def validate_receptor(
    target_name: str,
    receptor_info: dict,
    ref_ligand_name: str | None = None,
    ref_ligand_smiles: str | None = None,
    docked_pose_path: str | None = None,
    key_contacts: list[dict] | None = None,
    download_if_missing: bool = True,
    rmsd_threshold: float = 2.0,
    contact_cutoff: float = 4.0,
) -> ValidationReport:
    """Run a structured validation checklist for one receptor target."""
    report = ValidationReport(target_name=target_name)
    pdb_id = str(receptor_info.get("pdb_id", "")).strip().upper()
    chain = str(receptor_info.get("chain", "A")).strip()
    native_res = str(receptor_info.get("native_ligand_resname", "")).strip()
    padding = float(receptor_info.get("padding", 6.0))
    flex_list = receptor_info.get("flexible_residues") or []
    paths = _receptor_paths(target_name, chain, pdb_id)

    # 1. PDB file
    raw_pdb = paths["raw_pdb"]
    if not os.path.isfile(raw_pdb) and download_if_missing and pdb_id:
        try:
            download_pdb(pdb_id)
        except Exception as exc:
            report.checks.append(CheckResult("pdb_file", "PDB structure available", False,
                                             f"Download failed for {pdb_id}: {exc}"))
        else:
            report.warnings.append(f"Downloaded {pdb_id} from RCSB.")

    pdb_ok = os.path.isfile(raw_pdb)
    report.checks.append(CheckResult(
        "pdb_file", "PDB structure available", pdb_ok,
        raw_pdb if pdb_ok else f"Missing: {raw_pdb}",
    ))
    if not pdb_ok:
        return report

    # 2. Native co-crystal ligand
    native = _native_ligand_info(raw_pdb, native_res, chain_id=chain)
    if native["found"]:
        inst = native["instances"][0]
        detail = (f"{native_res} on chain {inst['chain']} res {inst['resnum']} "
                  f"({inst['n_heavy']} heavy atoms)")
        report.checks.append(CheckResult("native_ligand", "Native ligand found in PDB", True, detail, inst))
    else:
        report.checks.append(CheckResult(
            "native_ligand", "Native ligand found in PDB", False,
            f"Resname '{native_res}' not found on chain {chain} in {raw_pdb}",
        ))
        return report

    # 3. Grid box
    try:
        if receptor_info.get("force_center") and receptor_info.get("force_size"):
            box = auto_box_from_native_ligand(
                raw_pdb, native_res, padding=padding,
                force_center=receptor_info["force_center"],
                force_size=receptor_info["force_size"],
            )
            box_note = "manual force_center/force_size"
        else:
            box = auto_box_from_native_ligand(raw_pdb, native_res, padding=padding)
            box_note = f"auto from {native_res} + {padding} Å padding"
        c, s = box["center"], box["size"]
        report.checks.append(CheckResult(
            "grid_box", "Grid box coordinates", True,
            f"{box_note} | center=[{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}] "
            f"size=[{s[0]:.1f}, {s[1]:.1f}, {s[2]:.1f}]",
            box,
        ))
    except Exception as exc:
        report.checks.append(CheckResult("grid_box", "Grid box coordinates", False, str(exc)))

    # 4. Pocket residues (informational)
    pocket = _residues_near_ligand(raw_pdb, native_res, chain, cutoff=5.0)
    pocket_str = ", ".join(f"{r['resname']}{r['resnum']}({r['distance_A']}Å)" for r in pocket[:8])
    report.checks.append(CheckResult(
        "optional_pocket", "Pocket residues near native ligand (≤5 Å)", True,
        pocket_str or "none detected",
        pocket,
    ))

    # 5. Flexible residues
    if flex_list:
        missing = []
        found = []
        for token in flex_list:
            parsed = _parse_flex_residue(str(token))
            if not parsed:
                missing.append(f"{token} (bad format — use A:217)")
                continue
            ch, rn = parsed
            if _residue_exists(raw_pdb, ch, rn):
                found.append(f"{ch}:{rn}")
            else:
                missing.append(f"{ch}:{rn}")
        ok = len(missing) == 0
        detail = f"{len(found)}/{len(flex_list)} valid"
        if missing:
            detail += f" | missing: {', '.join(missing)}"
        report.checks.append(CheckResult("flex_residues", "Flexible residues exist in structure", ok, detail))
    else:
        report.checks.append(CheckResult(
            "optional_flex", "Flexible residues", True, "None configured (rigid docking)",
        ))

    # 6. Cached receptor files
    for key, label in [
        ("protonated_pdb", "Protonated receptor PDB"),
        ("receptor_pdbqt", "Receptor PDBQT"),
    ]:
        p = paths[key]
        report.checks.append(CheckResult(
            f"cache_{key}", label, os.path.isfile(p),
            p if os.path.isfile(p) else f"Not yet built — will be created on first docking run",
        ))

    if flex_list:
        flex_ok = os.path.isfile(paths["flex_side"]) and os.path.isfile(paths["flex_rigid"])
        report.checks.append(CheckResult(
            "cache_flex", "Flexible receptor cache (rigid + flex PDBQT)", flex_ok,
            "Present" if flex_ok else "Not yet built — requires prody + Meeko on first flex run",
        ))

    # 7. Reference pose RMSD + contacts
    pose = docked_pose_path or (find_docked_pose(target_name, ref_ligand_name) if ref_ligand_name else None)
    if pose and os.path.isfile(pose):
        rmsd = calculate_docking_rmsd(pose, raw_pdb, native_res, ref_smiles=ref_ligand_smiles)
        if rmsd is not None:
            ok = rmsd < rmsd_threshold
            report.checks.append(CheckResult(
                "rmsd", f"Re-dock RMSD vs crystal {native_res}", ok,
                f"{rmsd:.2f} Å (threshold < {rmsd_threshold:.1f} Å) using {ref_ligand_name or Path(pose).stem}",
                round(rmsd, 3),
            ))
        else:
            report.checks.append(CheckResult(
                "rmsd", f"Re-dock RMSD vs crystal {native_res}", False,
                f"Could not compute RMSD for {pose}",
            ))

        # Distance-based contacts (PLIP-independent)
        temp_lig = f"validation/_tmp_{target_name}_{ref_ligand_name or 'pose'}.pdb"
        os.makedirs("validation", exist_ok=True)
        receptor_for_contacts = paths["protonated_pdb"] if os.path.isfile(paths["protonated_pdb"]) else raw_pdb

        if _pdbqt_to_temp_pdb(pose, temp_lig):
            # Pocket contact recovery: which crystal-pocket residues does the docked pose reach?
            pocket_res = _residues_near_ligand(raw_pdb, native_res, chain, cutoff=5.0)
            recovered = []
            for r in pocket_res[:12]:
                dist = _min_ligand_residue_distance(temp_lig, receptor_for_contacts, chain, r["resnum"])
                if dist is not None and dist <= contact_cutoff:
                    recovered.append(f"{r['resname']}{r['resnum']}({dist:.2f}Å)")

            report.checks.append(CheckResult(
                "pocket_contacts",
                f"Docked pose contacts pocket (≤{contact_cutoff:.0f} Å to native-neighbor residues)",
                bool(recovered),
                "Contacts: " + ", ".join(recovered) if recovered else
                "No pocket residue within cutoff — pose may be misaligned or pocket definition too strict",
                recovered,
            ))

            contacts_to_check = key_contacts or []
            if not contacts_to_check and "VKORC1" in target_name.upper():
                contacts_to_check = VKORC1_KEY_CONTACTS + [
                    {"chain": "A", "resnum": 277, "label": "Cys277 (disulfide pocket wall)"},
                    {"chain": "A", "resnum": 262, "label": "Leu262"},
                ]

            for contact in contacts_to_check:
                ch = contact.get("chain", chain)
                rn = int(contact["resnum"])
                label = contact.get("label", f"{contact.get('resname', '')}{rn}")
                dist = _min_ligand_residue_distance(temp_lig, receptor_for_contacts, ch, rn)
                if dist is None:
                    report.checks.append(CheckResult(
                        f"optional_contact_{ch}_{rn}", f"Key residue: {label}", False,
                        f"Residue {ch}:{rn} not found",
                    ))
                else:
                    ok = dist <= contact_cutoff
                    report.checks.append(CheckResult(
                        f"optional_contact_{ch}_{rn}", f"Key residue: {label}", ok,
                        f"min heavy-atom distance = {dist:.2f} Å (≤ {contact_cutoff:.1f} Å = contact)",
                        round(dist, 2),
                    ))
            try:
                os.remove(temp_lig)
            except OSError:
                pass

        # Optional PLIP if installed
        if os.path.isfile(receptor_for_contacts):
            plip_out = os.path.join("validation", f"{target_name}_{ref_ligand_name or 'pose'}_plip.csv")
            plip = run_plip_analysis(receptor_for_contacts, pose, plip_out)
            if plip:
                hb = plip.get("H_Bond_Residues", "")
                hp = plip.get("Hydrophobic_Residues", "")
                report.checks.append(CheckResult(
                    "optional_plip", "PLIP interaction profile", True,
                    f"H-bonds: {hb or 'none'} | Hydrophobic: {hp or 'none'}",
                    plip,
                ))
    elif ref_ligand_name:
        report.checks.append(CheckResult(
            "optional_rmsd", "Reference pose validation", False,
            f"No docked pose found for '{ref_ligand_name}' against {target_name}. "
            f"Dock the reference ligand first, then re-validate.",
        ))

    return report


def load_target_from_config(target_name: str, config_path: str = "config_master.yaml") -> dict | None:
    if not os.path.isfile(config_path):
        config_path = "config.yaml"
    if not os.path.isfile(config_path):
        return None
    cfg = yaml.safe_load(open(config_path))
    rec = (cfg.get("receptors") or {}).get(target_name)
    if not rec:
        return None
    out = dict(rec)
    out.pop("active", None)
    return out


def suggest_ref_ligand(target_name: str, ligands: dict | None = None) -> tuple[str | None, str | None]:
    """Pick the best reference ligand for RMSD validation."""
    if ligands is None:
        for path in ("config_master.yaml", "config.yaml"):
            if os.path.isfile(path):
                cfg = yaml.safe_load(open(path))
                ligands = cfg.get("ligands") or {}
                break
    if not ligands:
        return None, None

    # Prefer _ref ligands with existing docked poses
    refs = [n for n in ligands if str(n).endswith("_ref")]
    for name in refs:
        if find_docked_pose(target_name, name):
            smi = ligands[name]
            if isinstance(smi, dict):
                smi = smi.get("smiles", "")
            return name, smi
    return None, None


def format_report_text(report: ValidationReport) -> str:
    lines = [f"VALIDATION REPORT — {report.target_name}", ""]
    for c in report.checks:
        icon = "PASS" if c.passed else "FAIL"
        lines.append(f"[{icon}] {c.label}")
        if c.detail:
            lines.append(f"       {c.detail}")
    lines.append("")
    lines.append(f"Overall: {'PASS' if report.overall_pass else 'NEEDS ATTENTION'}")
    for w in report.warnings:
        lines.append(f"Note: {w}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate receptor setup for docking")
    parser.add_argument("--target", required=True, help="Target name from config")
    parser.add_argument("--config", default="config_master.yaml")
    parser.add_argument("--check-pose", default=None, help="Reference ligand name for RMSD/contacts")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    info = load_target_from_config(args.target, args.config)
    if not info:
        raise SystemExit(f"Target '{args.target}' not found in {args.config}")

    ref_name, ref_smi = args.check_pose, None
    if ref_name:
        cfg = yaml.safe_load(open(args.config if os.path.isfile(args.config) else "config.yaml"))
        raw = (cfg.get("ligands") or {}).get(ref_name, "")
        ref_smi = raw.get("smiles") if isinstance(raw, dict) else raw
    else:
        ref_name, ref_smi = suggest_ref_ligand(args.target)

    report = validate_receptor(args.target, info, ref_ligand_name=ref_name, ref_ligand_smiles=ref_smi)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report_text(report))
    raise SystemExit(0 if report.overall_pass else 1)


if __name__ == "__main__":
    main()
