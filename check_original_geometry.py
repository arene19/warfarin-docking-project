#check_original_geometry.py
"""Recover the 3D-embedded chirality of the flat RL_Gen_37 docking run.

RL_Gen_37 was docked from a flat (stereo-undefined) SMILES. The distance
geometry / embedding step still produced a concrete 3D structure, which means
the benzhydryl carbon ended up as a specific R or S configuration. This utility
loads that original conformer and uses RDKit's AssignStereochemistryFrom3D to
report which hand was actually generated.

Raw .pdbqt files carry no bond orders, so (when available) a matching template
(a sibling .sdf or an explicit --template-smiles) is used to restore correct
connectivity before CIP perception. Without a template the script still runs
using proximity-perceived bonds, but may report extra artifact centers.
"""

import os
import sys
import glob
import argparse

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

LIGAND = "RL_Gen_37"

# Where the original (flat) 3D structure might live, in priority order.
SEARCH_DIRS = [
    "results/docked_poses/VKORC1_Human",
    "results/docked_poses",
    "results/pdbqt_ligands",
    "results/ligands",
    "results",
    "output",
    ".",
]


def find_structure():
    """Locate the flat RL_Gen_37 .pdb/.pdbqt (excluding _iso* variants)."""
    found = []
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for ext in ("pdbqt", "pdb"):
            for path in sorted(glob.glob(os.path.join(d, f"{LIGAND}*.{ext}"))):
                stem = os.path.basename(path)
                if f"{LIGAND}_iso" in stem:  # reject enumerated isomers
                    continue
                found.append(path)
    seen, ordered = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def find_template(smiles_arg):
    """Build a bond-order template: explicit SMILES wins, else a sibling .sdf."""
    if smiles_arg:
        tmpl = Chem.MolFromSmiles(smiles_arg)
        if tmpl is not None:
            return tmpl, f"--template-smiles ({smiles_arg})"
        print(f"  WARNING: could not parse --template-smiles: {smiles_arg}")

    for d in SEARCH_DIRS:
        sdf = os.path.join(d, f"{LIGAND}.sdf")
        if os.path.exists(sdf):
            ref = Chem.MolFromMolFile(sdf, sanitize=True, removeHs=True)
            if ref is not None:
                # Strip stereo/coords down to a clean connectivity template.
                flat = Chem.MolFromSmiles(Chem.MolToSmiles(ref))
                if flat is not None:
                    return flat, sdf
    return None, None


def load_pdb(path):
    return Chem.MolFromPDBFile(path, sanitize=True, removeHs=False)


def load_pdbqt(path):
    """Text-parse a raw .pdbqt: keep the first MODEL's ATOM/HETATM records,
    drop AutoDock-only columns by truncating to the standard PDB coordinate
    width, then let RDKit infer bonds by proximity."""
    pdb_lines = []
    with open(path, "r") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "ENDMDL":
                break  # only analyze the first (best) pose
            if rec in ("ATOM", "HETATM"):
                pdb_lines.append(line[:66].rstrip())
    if not pdb_lines:
        return None
    block = "\n".join(pdb_lines) + "\nEND\n"

    mol = Chem.MolFromPDBBlock(
        block, sanitize=True, removeHs=False, proximityBonding=True
    )
    if mol is None:
        mol = Chem.MolFromPDBBlock(
            block, sanitize=False, removeHs=False, proximityBonding=True
        )
        if mol is not None:
            Chem.SanitizeMol(mol, catchErrors=True)
    return mol


def load_structure(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdb":
        return load_pdb(path)
    if ext == ".pdbqt":
        return load_pdbqt(path)
    raise ValueError(f"Unsupported extension for {path} (expected .pdb/.pdbqt)")


def apply_template(mol, template):
    """Restore correct bond orders from a template, preserving 3D coords."""
    try:
        heavy = Chem.RemoveHs(mol)
    except Exception:
        heavy = mol
    return AllChem.AssignBondOrdersFromTemplate(template, heavy)


def report_chirality(mol):
    Chem.AssignStereochemistryFrom3D(mol)
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )

    if not centers:
        print("No chiral centers were perceived in this structure.")
        return

    print(f"Detected {len(centers)} chiral center(s):")
    for atom_idx, label in centers:
        atom = mol.GetAtomWithIdx(atom_idx)
        neighbors = ", ".join(sorted(n.GetSymbol() for n in atom.GetNeighbors()))
        # Benzhydryl-amine carbon: sp3 C bonded to N + two (aromatic) carbons.
        syms = sorted(n.GetSymbol() for n in atom.GetNeighbors())
        is_benzhydryl = atom.GetSymbol() == "C" and syms.count("C") == 2 and "N" in syms
        tag = "  <-- benzhydryl carbon" if is_benzhydryl else ""
        print(
            f"  - Atom idx {atom_idx:>3} ({atom.GetSymbol()}): "
            f"configuration = {label}  | neighbors: [{neighbors}]{tag}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Report the 3D-embedded chirality of the flat RL_Gen_37 run."
    )
    parser.add_argument(
        "--file",
        help="Explicit path to the original .pdb/.pdbqt conformer "
        "(overrides auto-detection).",
    )
    parser.add_argument(
        "--template-smiles",
        help="Flat SMILES of the ligand, used to restore bond orders from the "
        "bond-order-free .pdbqt (improves CIP accuracy).",
    )
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(f"ERROR: provided --file does not exist: {args.file}")
            sys.exit(1)
        candidates = [args.file]
    else:
        candidates = find_structure()
        if not candidates:
            print(
                f"ERROR: could not find an original 3D structure for {LIGAND}.\n"
                f"Searched these directories for {LIGAND}*.pdb / {LIGAND}*.pdbqt:\n"
                + "\n".join(f"  - {d}" for d in SEARCH_DIRS)
                + "\n\nIf your docked conformer lives elsewhere, pass it directly:\n"
                f"  python check_original_geometry.py --file /path/to/{LIGAND}.pdbqt"
            )
            sys.exit(1)

    template, template_src = find_template(args.template_smiles)
    if template is not None:
        print(f"Using bond-order template: {template_src}")
    else:
        print("No bond-order template found; using proximity-perceived bonds "
              "(extra artifact centers may appear).")

    print(f"\nFound {len(candidates)} candidate structure(s) for {LIGAND}:")
    for c in candidates:
        print(f"  - {c}")

    for path in candidates:
        print(f"\n=== Analyzing: {path} ===")
        try:
            mol = load_structure(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed to parse ({type(exc).__name__}): {exc}")
            continue

        if mol is None:
            print("  RDKit could not build a molecule from this file. "
                  "Trying the next candidate...")
            continue

        if template is not None:
            try:
                mol = apply_template(mol, template)
                print("  Bond orders restored from template.")
            except Exception as exc:  # noqa: BLE001
                print(f"  Template match failed ({type(exc).__name__}); "
                      "falling back to proximity bonds.")

        try:
            report_chirality(mol)
        except Exception as exc:  # noqa: BLE001
            print(f"  Stereo perception failed ({type(exc).__name__}): {exc}")
            continue

        return  # stop at first successfully analyzed structure

    print("\nERROR: none of the candidate structures could be analyzed. "
          "Adjust the path with --file (and/or supply --template-smiles).")
    sys.exit(1)


if __name__ == "__main__":
    main()
