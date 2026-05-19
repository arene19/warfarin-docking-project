# protein_preparation.py
"""
Download and prepare target proteins for docking.
Handles: PDB download, chain selection, water/ligand removal,
         hydrogen addition, and PDBQT conversion.
"""

import os
import requests
import subprocess
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select


# -------------------------------------------------------
# Target protein definitions


def download_pdb(pdb_id: str, output_dir: str = "proteins/raw/") -> str:
    """Download PDB structure from RCSB."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(output_path):
        print(f"  [CACHED] {pdb_id}.pdb")
        return output_path
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id} from RCSB...")
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(output_path, "w") as f:
        f.write(response.text)
    
    print(f"  Saved: {output_path}")
    return output_path


class ChainSelect(Select):
    """BioPython selector: keep only specified chain."""
    
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
    
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    
    def accept_residue(self, residue):
        # Remove water molecules (HOH/WAT)
        if residue.get_resname() in ["HOH", "WAT", "H2O"]:
            return False
        # Remove common co-solvents and ions
        if residue.get_resname() in ["SO4", "GOL", "PEG", "EDO", "PO4"]:
            return False
        return True


def clean_protein(pdb_path: str, chain_id: str, 
                  output_dir: str = "proteins/clean/") -> str:
    """
    Clean protein: keep single chain, remove waters/heterogens.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_id = Path(pdb_path).stem
    output_path = os.path.join(output_dir, f"{pdb_id}_chain{chain_id}.pdb")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, ChainSelect(chain_id))
    
    print(f"  Cleaned PDB saved: {output_path}")
    return output_path


def add_hydrogens_pdbfixer(input_pdb: str, output_pdb: str, 
                            pH: float = 7.4) -> str:
    """
    Add missing hydrogens and residues using PDBFixer.
    Mimics physiological pH 7.4 conditions.
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        
        fixer = PDBFixer(filename=input_pdb)
        
        # Find and add missing residues
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        
        # Remove heterogens (keep only protein)
        fixer.removeHeterogens(keepWater=False)
        
        # Add missing atoms and hydrogens
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH)
        
        with open(output_pdb, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
        print(f"  Protonated PDB saved: {output_pdb}")
        
    except ImportError:
        print("  [WARNING] PDBFixer not installed. Using OpenBabel fallback.")
        cmd = f"obabel {input_pdb} -O {output_pdb} -h --pH {pH}"
        subprocess.run(cmd, shell=True, check=True)
    
    return output_pdb


def protein_to_pdbqt(protonated_pdb: str, 
                      output_dir: str = "pdbqt_receptors/") -> str:
    """
    Convert protonated PDB to PDBQT for AutoDock Vina.
    Uses MGLTools prepare_receptor script.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_name = Path(protonated_pdb).stem
    output_path = os.path.join(output_dir, f"{pdb_name}.pdbqt")
    
    # Method 1: MGLTools (preferred)
    mgl_script = "prepare_receptor"  # must be in PATH
    cmd = (f"{mgl_script} -r {protonated_pdb} -o {output_path} "
           f"-A hydrogens -U nphs_lps_waters_deleteAltB")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        # Method 2: OpenBabel fallback
        print(f"  [Fallback] Using OpenBabel for PDBQT conversion")
        cmd = f"obabel {protonated_pdb} -O {output_path} -xr"
        subprocess.run(cmd, shell=True, check=True)
    
    print(f"  Receptor PDBQT: {output_path}")
    return output_path


def prepare_all_targets(targets: dict = None) -> dict:
    """Full pipeline: download -> clean -> protonate -> PDBQT."""
    # Handle the absence of the old global TARGETS dictionary gracefully
    if targets is None:
        print("[WARNING] No targets dictionary provided to prepare_all_targets.")
        return {}

    prepared = {}

    for name, info in targets.items():
        print(f"\n{'='*50}")
        print(f"  Preparing: {name} ({info['pdb_id']})")
        print(f"  {info['description']}")
        print(f"{'='*50}")

        try:
            # 1. Download
            raw_pdb = download_pdb(info["pdb_id"])

            # 2. Clean (single chain, no waters)
            clean_pdb = clean_protein(raw_pdb, info["chain"])

            # 3. Add hydrogens at pH 7.4
            protonated_pdb = os.path.join("proteins/protonated", f"{info['pdb_id']}_chain{info['chain']}_protonated.pdb")
            os.makedirs(os.path.dirname(protonated_pdb), exist_ok=True)
            add_hydrogens_pdbfixer(clean_pdb, protonated_pdb)

            # 4. Convert to PDBQT
            receptor_pdbqt = protein_to_pdbqt(protonated_pdb)

            prepared[name] = {
                "pdbqt": receptor_pdbqt,
                "binding_residues": info.get("binding_residues", []),
                "pdb_id": info["pdb_id"]
            }

        except Exception as e:
            print(f"  [ERROR] Preparation failed for {name}: {e}")

    return prepared

if __name__ == "__main__":
    # For standalone testing of this module without config.yaml, 
    # we pass a manual test target dictionary.
    test_targets = {
        "VKORC1_Human": {
            "pdb_id": "6WV3",
            "chain": "A",
            "description": "Human VKORC1 bound to Warfarin (True Target)"
        }
    }
    prepared_targets = prepare_all_targets(test_targets)
    print(f"\nSuccessfully verified {len(prepared_targets)} target(s) standalone.")
