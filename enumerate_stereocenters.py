#enumerate_stereocenters.py
"""Expand ligands with undefined stereochemistry into explicit stereoisomers.

Loads config.yaml, finds every ligand whose SMILES has undefined chiral
centers (or other unassigned stereo), enumerates the explicit isomers with
RDKit, and replaces the flat entry with suffixed entries (_isoA, _isoB, ...).
Ligands that are already fully defined are left untouched. The receptors and
docking_params sections are preserved exactly.
"""

import os
import string

import yaml
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

RDLogger.DisableLog("rdApp.*")

from config_utils import MASTER_CONFIG, load_config, save_config, sync_active_config


def get_smiles(entry):
    """Support both 'name: SMILES' and 'name: {smiles: ..., ...}' formats."""
    if isinstance(entry, dict):
        return entry.get("smiles")
    return entry


def make_entry(template, smiles):
    """Rebuild a ligand value in the same format as the original entry."""
    if isinstance(template, dict):
        new = dict(template)
        new["smiles"] = smiles
        return new
    return smiles


def count_undefined_centers(mol):
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    return sum(1 for _, label in centers if label == "?")


def suffix(index):
    """0 -> 'A', 1 -> 'B', ... 26 -> 'AA' (graceful overflow)."""
    letters = string.ascii_uppercase
    if index < len(letters):
        return letters[index]
    first, second = divmod(index, len(letters))
    return letters[first - 1] + letters[second]


def main():
    if not os.path.exists(MASTER_CONFIG):
        raise FileNotFoundError(f"Could not find {MASTER_CONFIG}")

    config = load_config(MASTER_CONFIG)

    ligands = config.get("ligands") or {}
    new_ligands = {}

    opts = StereoEnumerationOptions(onlyUnassigned=True, tryEmbedding=False)

    for name, entry in ligands.items():
        smiles = get_smiles(entry)
        mol = Chem.MolFromSmiles(smiles) if smiles else None

        if mol is None:
            print(f"  [skip] {name}: invalid/empty SMILES, left unchanged.")
            new_ligands[name] = entry
            continue

        undefined = count_undefined_centers(mol)
        if undefined == 0:
            # Fully defined (e.g. refs with @/@@) -> leave completely alone.
            new_ligands[name] = entry
            continue

        isomers = list(EnumerateStereoisomers(mol, options=opts))
        if len(isomers) <= 1:
            new_ligands[name] = entry
            continue

        iso_smiles = [Chem.MolToSmiles(iso, isomericSmiles=True) for iso in isomers]
        print(
            f"  [enumerate] {name}: {undefined} undefined center(s) -> "
            f"{len(iso_smiles)} isomers"
        )
        for i, iso in enumerate(iso_smiles):
            iso_name = f"{name}_iso{suffix(i)}"
            new_ligands[iso_name] = make_entry(entry, iso)

    config["ligands"] = new_ligands

    save_config(MASTER_CONFIG, config)
    sync_active_config()

    print(
        f"\nDone. Ligands: {len(ligands)} -> {len(new_ligands)}. "
        f"Synced {MASTER_CONFIG} -> config.yaml (active ligands only)."
    )


if __name__ == "__main__":
    main()
