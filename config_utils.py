"""
YAML config helpers for ligand/receptor dictionaries.

Supports both flat SMILES strings and {smiles, active} dict entries used by
config_master.yaml and the Streamlit app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from rdkit import Chem

MASTER_CONFIG = "config_master.yaml"
ACTIVE_CONFIG = "config.yaml"


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(path: str | Path, config: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_smiles(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        return entry.get("smiles")
    if isinstance(entry, str):
        return entry
    return None


def is_active(entry: Any, default: bool = True) -> bool:
    if isinstance(entry, dict):
        return bool(entry.get("active", default))
    return default


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def ligands_to_smiles_dict(
    ligands: Dict[str, Any], *, active_only: bool = False
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, entry in (ligands or {}).items():
        if active_only and not is_active(entry):
            continue
        smi = get_smiles(entry)
        if smi:
            out[name] = smi
    return out


def active_ligand_names(ligands: Dict[str, Any]) -> list[str]:
    return [name for name, entry in (ligands or {}).items() if is_active(entry)]


def sync_active_config(
    master_path: str | Path = MASTER_CONFIG,
    active_path: str | Path = ACTIVE_CONFIG,
) -> None:
    """
    Write config.yaml as flat SMILES for active ligands only, preserving
    receptors and docking_params from the master file.
    """
    master = load_config(master_path)
    active_ligands = ligands_to_smiles_dict(master.get("ligands", {}), active_only=True)
    flat = {
        k: v
        for k, v in master.items()
        if k != "ligands"
    }
    flat["ligands"] = active_ligands
    save_config(active_path, flat)


def resolve_config_paths(
    master_path: Optional[str] = None,
    active_path: Optional[str] = None,
) -> Tuple[Path, Path]:
    return Path(master_path or MASTER_CONFIG), Path(active_path or ACTIVE_CONFIG)
