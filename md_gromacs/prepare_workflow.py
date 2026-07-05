#!/usr/bin/env python3
"""Generate md_gromacs/ CHARMM-GUI inputs and ligand parameter files from md_poses/."""
from __future__ import annotations

import csv
import shutil
import textwrap
from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
MD_POSES = ROOT / "md_poses"
OUT = Path(__file__).resolve().parent

SYSTEMS = [
    "S_Warfarin_ref",
    "p_nitro_R",
    "p_nitro_S",
    "dimethoxy_23_S",
    "RL_Gen_37",
    "RL_Gen_29_isoA",
    "RL_Gen_22",
    "RL_Gen_45",
]

# ligand_id -> (yaml file relative to ROOT, smiles key)
SMILES_SOURCE = {
    "S_Warfarin_ref": ("config.yaml", "S_Warfarin_ref"),
    "p_nitro_R": ("config.yaml", "p_nitro_R"),
    "p_nitro_S": ("config.yaml", "p_nitro_S"),
    "dimethoxy_23_S": ("config.yaml", "dimethoxy_23_S"),
    "RL_Gen_37": ("config_master.yaml", "RL_Gen_37_isoA"),
    "RL_Gen_29_isoA": ("config_master.yaml", "RL_Gen_29_isoA"),
    "RL_Gen_22": ("config_master.yaml", "RL_Gen_22_isoA"),
    "RL_Gen_45": ("config_master.yaml", "RL_Gen_45"),
}

CGENFF_RISK = {
    "S_Warfarin_ref": "low",
    "p_nitro_R": "medium",
    "p_nitro_S": "medium",
    "dimethoxy_23_S": "medium",
    "RL_Gen_37": "high",
    "RL_Gen_29_isoA": "high",
    "RL_Gen_22": "high",
    "RL_Gen_45": "high",
}

CGENFF_NOTES = {
    "S_Warfarin_ref": textwrap.dedent(
        """\
        - Coumarin enolate ([O-] on lactone) at pH 7.4 — verify net charge in CGenFF output.
        - Compare atom names in ParamChem mol2 to ligand.pdb HETATM names before merging topologies.
        """
    ),
    "p_nitro_R": textwrap.dedent(
        """\
        - Enolate + para-nitro ([N+](=O)[O-]); check total charge after CGenFF.
        - Stereocenter: R (@ configuration in config.yaml SMILES).
        """
    ),
    "p_nitro_S": textwrap.dedent(
        """\
        - Enolate + para-nitro; stereocenter S (@@ in SMILES).
        - Same CGenFF atom typing as p_nitro_R except stereochemistry.
        """
    ),
    "dimethoxy_23_S": textwrap.dedent(
        """\
        - Enolate + dimethoxy aniline; stereocenter S.
        - Watch ether and amide torsions in ParamChem output.
        """
    ),
    "RL_Gen_37": textwrap.dedent(
        """\
        - **HIGH RISK** for automated CGenFF: spiro/piperidinone, cyclopropyl, benzhydryl-like center.
        - Use RL_Gen_37_isoA SMILES (CIP R) — matches md_poses flat-dock MODEL 1 embed.
        - **ParamChem fallback likely required**: upload ligand.sdf, download mol2+str, merge manually.
        - If CGenFF penalizes many torsions, consider shortening production until topology validated.
        """
    ),
    "RL_Gen_29_isoA": textwrap.dedent(
        """\
        - **HIGH RISK**: triazole, gem-difluoro, extended aromatic scaffold.
        - Use isoA SMILES only (explicit @ stereochemistry).
        - ParamChem manual submission recommended if CHARMM-GUI ligand reader fails.
        """
    ),
    "RL_Gen_22": textwrap.dedent(
        """\
        - **HIGH RISK**: fluorinated benzamide, carbamate, multi-ring scaffold.
        - Use RL_Gen_22_isoA SMILES (explicit @) — matches flat-dock MODEL 1 embed.
        - Lipinski MW violation (487 Da) in ADMET profile; verify topology before long production.
        - ParamChem fallback likely if CHARMM-GUI auto CGenFF penalizes torsions.
        """
    ),
    "RL_Gen_45": textwrap.dedent(
        """\
        - **HIGH RISK**: spirocyclic / bridged peroxide-like scaffold; unusual for CGenFF.
        - Flat-dock MODEL 1 pose; ParamChem manual submission strongly recommended.
        - Inspect penalty scores carefully; consider 20 ns pilot only until ligand RMSD stable.
        """
    ),
}


def load_smiles(yaml_name: str, key: str) -> str:
    path = ROOT / yaml_name
    data = yaml.safe_load(path.read_text())
    ligands = data.get("ligands", data)
    if key not in ligands:
        raise KeyError(f"{key} not in {path}")
    entry = ligands[key]
    if isinstance(entry, dict):
        return entry["smiles"]
    return entry


def split_pdb_lines(pdb_text: str) -> tuple[list[str], list[str]]:
    protein, ligand = [], []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            protein.append(line)
        elif line.startswith("HETATM"):
            ligand.append(line)
    return protein, ligand


def write_pdb(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines) + "\nEND\n"
    path.write_text(body)


def smiles_to_sdf_mol2(smiles: str, sdf_path: Path, mol2_path: Path) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(sdf_path))
    w.write(mol)
    w.close()
    mol2_block = Chem.MolToMolBlock(mol)
    mol2_path.write_text(f"# SMILES: {smiles}\n{mol2_block}\n")


def write_cgenff_notes(system: str, smiles: str, yaml_ref: str, key: str) -> None:
    notes_dir = OUT / "systems" / system / "ligand_params"
    risk = CGENFF_RISK[system]
    text = textwrap.dedent(
        f"""\
        # CGenFF / ParamChem notes — {system}

        | Field | Value |
        |-------|-------|
        | SMILES source | `{yaml_ref}` → `{key}` |
        | CGenFF auto risk | **{risk}** |
        | Resname in PDB | LIG |
        | Chain | B |

        ## SMILES
        ```
        {smiles}
        ```

        ## Submission checklist
        1. Upload `ligand.sdf` (or `ligand.mol2`) to [ParamChem](https://cgenff.umaryland.edu/) or CHARMM-GUI Ligand Reader.
        2. Set charge method consistent with pH 7.4 (coumarin refs: enolate).
        3. Download `.str` / `.rtf` + `.prm` (or GROMACS `.itp` from CHARMM-GUI).
        4. Rename ligand residue to **LIG** to match protein–ligand complex PDB.
        5. Merge `#include "ligand.itp"` into system topology after `#include "toppar_water_ions.str"` equivalent.

        ## Ligand-specific notes
        {CGENFF_NOTES[system]}

        ## ParamChem fallback (if auto CGenFF fails)
        1. Draw or paste SMILES in ParamChem; run CGenFF.
        2. Inspect penalty scores — accept < 50 for production; reparameterize if higher.
        3. Export GROMACS format via CHARMM-GUI Ligand Reader & Modeler using the ParamChem mol2.
        """
    )
    (notes_dir / "CGenFF_NOTES.md").write_text(text)


def copy_charmm_gui_pdbs(system: str) -> None:
    complex_src = MD_POSES / "complexes" / f"{system}_VKORC1_Human_model1_complex.pdb"
    lig_src = MD_POSES / "ligands" / f"{system}_model1_ligand.pdb"
    prot_src = MD_POSES / "receptor" / "VKORC1_Human_chainA_protonated.pdb"
    dest = OUT / "systems" / system / "charmm_gui"
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(complex_src, dest / "complex.pdb")
    shutil.copy2(lig_src, dest / "ligand.pdb")
    shutil.copy2(prot_src, dest / "protein.pdb")

    # CHARMM-GUI upload variant: protein + ligand in one file, cleaned TER
    complex_text = complex_src.read_text()
    protein, ligand = split_pdb_lines(complex_text)
    cleaned = protein + ["TER"] + ligand
    write_pdb(dest / "complex_clean.pdb", cleaned)


def write_manifest() -> None:
    manifest_path = OUT / "manifest.csv"
    rows = []
    for system in SYSTEMS:
        rows.append(
            {
                "system_id": system,
                "run_order": 1 if system == "S_Warfarin_ref" else "",
                "complex_pdb": f"systems/{system}/charmm_gui/complex.pdb",
                "protein_pdb": f"systems/{system}/charmm_gui/protein.pdb",
                "ligand_pdb": f"systems/{system}/charmm_gui/ligand.pdb",
                "ligand_sdf": f"systems/{system}/ligand_params/ligand.sdf",
                "cgenff_risk": CGENFF_RISK[system],
                "pilot_ns": 20,
                "production_ns": 100,
            }
        )
    rows[0]["run_order"] = 1
    order = 2
    for r in rows[1:]:
        r["run_order"] = order
        order += 1
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    for system in SYSTEMS:
        yaml_name, key = SMILES_SOURCE[system]
        smiles = load_smiles(yaml_name, key)
        copy_charmm_gui_pdbs(system)
        params = OUT / "systems" / system / "ligand_params"
        smiles_to_sdf_mol2(smiles, params / "ligand.sdf", params / "ligand.mol2")
        write_cgenff_notes(system, smiles, yaml_name, key)
        print(f"Prepared {system}")
    write_manifest()
    print(f"Wrote manifest -> {OUT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
