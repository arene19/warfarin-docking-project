"""
Calculates localized structural search grids.
Supports automatic pocket mapping and manual grid overrides.
"""

import numpy as np
from Bio.PDB import PDBParser

def auto_box_from_native_ligand(pdb_path: str, ligand_resname: str, padding: float = 5.0, force_size: list = None, force_center: list = None) -> dict:
    """
    Calculates search boundaries based on reference ligand coordinates.
    Applies overrides immediately if manual settings are provided.
    """
    if force_center is not None and force_size is not None:
        return {
            "center": (float(force_center[0]), float(force_center[1]), float(force_center[2])),
            "size": (float(force_size[0]), float(force_size[1]), float(force_size[2]))
        }

    if not ligand_resname:
        raise ValueError(f"Dynamic grid search requires a target residue name: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == ligand_resname:
                    for atom in residue:
                        coords.append(atom.get_coord())
                        
    if not coords:
        raise ValueError(f"Reference ligand code '{ligand_resname}' not found in {pdb_path}.")
        
    coords = np.array(coords)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    center = coords.mean(axis=0)
    
    size = np.array(force_size) if force_size is not None else (max_coords - min_coords) + (2 * padding)
    
    return {
        "center": (float(center[0]), float(center[1]), float(center[2])),
        "size": (float(size[0]), float(size[1]), float(size[2]))
    }