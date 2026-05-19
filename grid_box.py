# grid_box.py
import numpy as np
from Bio.PDB import PDBParser

def auto_box_from_native_ligand(pdb_path: str, ligand_resname: str, padding: float = 5.0, force_size: list = None, force_center: list = None) -> dict:
    """
    Automatically detects the geometric box center and required axis dimensions.
    Can be bypassed entirely if explicit coordinates are provided.
    """
    # >>> 1. THE ULTIMATE OVERRIDE <<<
    # If the config file has both center and size, skip opening the PDB entirely!
    if force_center is not None and force_size is not None:
        print("  [OVERRIDE] Using explicit manual grid coordinates from config.")
        return {
            "center": (float(force_center[0]), float(force_center[1]), float(force_center[2])),
            "size": (float(force_size[0]), float(force_size[1]), float(force_size[2]))
        }

    # >>> 2. DYNAMIC CALCULATION <<<
    if not ligand_resname:
        raise ValueError(f"No ligand_resname or manual coordinates provided for {pdb_path}")

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
        raise ValueError(f"CRITICAL Error: Native tracking ligand code '{ligand_resname}' was not detected inside {pdb_path}.")
        
    coords = np.array(coords)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    center = coords.mean(axis=0)
    
    size = np.array(force_size) if force_size is not None else (max_coords - min_coords) + (2 * padding)
    
    print(f"  [AUTO-GRID] Bound ligand {ligand_resname} localized at Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"  [AUTO-GRID] Bound box size computed: ({size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f})")
    
    return {
        "center": (float(center[0]), float(center[1]), float(center[2])),
        "size": (float(size[0]), float(size[1]), float(size[2]))
    }
def write_vina_config(receptor_pdbqt: str, ligand_pdbqt: str,
                      output_pdbqt: str, box_params: dict,
                      config_path: str = "config.txt") -> str:
    """Writes standard AutoDock Vina local grid configuration configuration files."""
    config_content = f"""# AutoDock Vina Execution Target Config
receptor = {receptor_pdbqt}
ligand   = {ligand_pdbqt}
out      = {output_pdbqt}
log      = {output_pdbqt.replace('.pdbqt', '.log')}

center_x = {box_params['center'][0]:.3f}
center_y = {box_params['center'][1]:.3f}
center_z = {box_params['center'][2]:.3f}

size_x = {box_params['size'][0]:.1f}
size_y = {box_params['size'][1]:.1f}
size_z = {box_params['size'][2]:.1f}
"""
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path
# -------------------------------------------------------
# Pre-defined binding boxes (from literature/crystal structures)
# -------------------------------------------------------
BINDING_BOXES ={
"VKORC1_Human": {
        "center": (-9.83, 26.82, 55.76),
        "size":   (20.0, 20.0, 20.0),
        "exhaustiveness": 32,
        "num_modes": 20

    },
    "Factor_Xa": {
        "center": (10.43, 46.56, 60.76),
        "size":   (20.12, 24.21, 22.30),
        "exhaustiveness": 32,
        "num_modes": 20
    },
        "Thrombin": {
        "center": (-23.89, -23.63, -3.45),
        "size":   (25.0, 25.0, 25.0),
        "exhaustiveness": 32,
        "num_modes": 20
    },
    "CYP2C9": {
        "center": (-18.11, 85.71, 39.91),
        "size":   (21.84, 28.74, 22.27),
        "exhaustiveness": 32,
        "num_modes": 20
    },
    "HSA": {
        "center": (32.24, 14.10, 9.29), # Replace with your intended coordinates
        "size":   (15.0, 15.0, 15.0),
        "exhaustiveness": 32,
        "num_modes": 20
    }

}
def compute_box_from_residues(pdb_path: str, residue_numbers: list,
                               chain_id: str = "A",
                               padding: float = 5.0) -> dict:
    """
    Auto-compute bounding box from known binding site residues.
    
    Args:
        pdb_path: Path to protein PDB file
        residue_numbers: List of residue numbers defining binding site
        chain_id: Chain identifier
        padding: Extra space around residues (Angstroms)
    
    Returns:
        dict with 'center' and 'size' tuples
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    coords = []
    
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_id()[1] in residue_numbers:
                    for atom in residue:
                        coords.append(atom.get_coord())
    
    if not coords:
        raise ValueError(f"No atoms found for residues {residue_numbers}")
    
    coords = np.array(coords)
    
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    center = (coords.mean(axis=0)).tolist()
    size = (max_coords - min_coords + 2 * padding).tolist()
    
    print(f"  Binding box center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"  Binding box size:   ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")
    
    return {
        "center": tuple(center),
        "size": tuple(size)
    }


def write_vina_config(receptor_pdbqt: str, ligand_pdbqt: str,
                      output_pdbqt: str, box_params: dict,
                      config_path: str = "config.txt") -> str:
    """Write AutoDock Vina configuration file."""
    
    config = f"""# AutoDock Vina Configuration
# Generated for warfarin docking study

receptor = {receptor_pdbqt}
ligand   = {ligand_pdbqt}
out      = {output_pdbqt}
log      = {output_pdbqt.replace('.pdbqt', '.log')}

# Search space (Angstroms)
center_x = {box_params['center'][0]:.3f}
center_y = {box_params['center'][1]:.3f}
center_z = {box_params['center'][2]:.3f}

size_x = {box_params['size'][0]:.1f}
size_y = {box_params['size'][1]:.1f}
size_z = {box_params['size'][2]:.1f}

# Docking parameters
exhaustiveness = {box_params.get('exhaustiveness', 32)}
num_modes      = {box_params.get('num_modes', 20)}
energy_range   = 3
cpu            = 0

# Scoring function
scoring = vina
"""
    
    with open(config_path, "w") as f:
        f.write(config)
    
    print(f"  Vina config written: {config_path}")
    return config_path


if __name__ == "__main__":
    box = compute_box_from_residues(
        pdb_path="proteins/clean/6WV3_chainA.pdb", 
        residue_numbers=[55, 132, 135, 139],
        chain_id="A",
        padding=5.0
    )
    print("\n--- NEW 6WV3 COORDINATES ---")
    print(box)
