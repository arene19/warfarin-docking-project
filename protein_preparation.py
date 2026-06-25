"""
Handles target receptor cleaning, downloading, and formatting.
"""

import os
import glob
import shutil
import requests
import subprocess
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select

from pipeline_utils import clean_subprocess_env

def run_system_command(cmd: str, **kwargs):
    """
    Runs a command in a sanitized system environment to prevent 
    virtual-environment library pollution (specifically OpenBabel clashes).
    """
    return subprocess.run(cmd, shell=True, env=clean_subprocess_env(), **kwargs)

def download_pdb(pdb_id: str, output_dir: str = "proteins/raw/") -> str:
    """Downloads structural coordinates from the RCSB Protein Data Bank."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(output_path):
        return output_path
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(output_path, "w") as f:
        f.write(response.text)
        
    return output_path

class ChainSelect(Select):
    """Filters structures to keep only specified chains and remove non-receptor elements."""
    
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
    
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    
    def accept_residue(self, residue):
        # Remove solvent structures
        if residue.get_resname() in ["HOH", "WAT", "H2O"]:
            return False
        # Remove common co-solvents
        if residue.get_resname() in ["SO4", "GOL", "PEG", "EDO", "PO4"]:
            return False
        return True

def clean_protein(pdb_path: str, chain_id: str, output_dir: str = "proteins/clean/") -> str:
    """Saves a cleaned structural variant retaining specified chains only."""
    os.makedirs(output_dir, exist_ok=True)
    pdb_id = Path(pdb_path).stem
    output_path = os.path.join(output_dir, f"{pdb_id}_chain{chain_id}.pdb")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, ChainSelect(chain_id))
    
    return output_path

def add_hydrogens_pdbfixer(input_pdb: str, output_pdb: str, pH: float = 7.4) -> str:
    """Applies protonation patterns using PDBFixer or falls back to OpenBabel."""
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        
        fixer = PDBFixer(filename=input_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH)
        
        with open(output_pdb, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
            
    except ImportError:
        print("  [WARNING] PDBFixer is unavailable. Defaulting to OpenBabel protonation.")
        cmd = f"obabel {input_pdb} -O {output_pdb} -h --pH {pH}"
        run_system_command(cmd, check=True)
        
    return output_pdb

def protein_to_pdbqt(protonated_pdb: str, output_dir: str = "pdbqt_receptors/") -> str:
    """Prepares a rigid receptor PDBQT using a single, standardized toolchain.

    Tool preference (matches the flexible-receptor path, which uses Meeko):
      1. Meeko 'mk_prepare_receptor.py'  (canonical AutoDock toolchain)
      2. ADFRSuite/MGLTools 'prepare_receptor'
      3. OpenBabel '-xr'                 (last-resort fallback)
    Each stage degrades gracefully so a missing tool never hard-fails the run.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdb_name = Path(protonated_pdb).stem
    output_path = os.path.join(output_dir, f"{pdb_name}.pdbqt")
    output_basename = os.path.join(output_dir, pdb_name)

    # 1. Preferred: Meeko mk_prepare_receptor (same toolchain as flexible prep).
    #    -p/--write_pdbqt is required or the tool prepares but writes no file.
    cmd = f"mk_prepare_receptor.py -i {protonated_pdb} -o {output_basename} -p"
    result = run_system_command(cmd, capture_output=True, text=True)
    produced = output_path if os.path.exists(output_path) else _first_match(output_basename)
    if result.returncode == 0 and produced:
        if produced != output_path:
            shutil.move(produced, output_path)
        return output_path

    # 2. ADFRSuite/MGLTools prepare_receptor
    print("  [Receptor Prep] Meeko unavailable; trying ADFR prepare_receptor...")
    cmd = (f"prepare_receptor -r {protonated_pdb} -o {output_path} "
           f"-A hydrogens -U nphs_lps_waters_deleteAltB")
    result = run_system_command(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(output_path):
        return output_path

    # 3. OpenBabel last-resort fallback
    print("  [Receptor Prep] prepare_receptor unavailable. Defaulting to OpenBabel...")
    cmd = f"obabel {protonated_pdb} -O {output_path} -xr"
    run_system_command(cmd, check=True)
    return output_path


def _first_match(basename: str):
    """Returns the first PDBQT produced for a given output basename, or None."""
    matches = sorted(glob.glob(f"{basename}*.pdbqt"))
    return matches[0] if matches else None

def strip_disulfide_bonds(pdb_path: str):
    """Removes SSBOND text headers so the pipeline knows the bonds are broken."""
    with open(pdb_path, 'r') as f:
        lines = [l for l in f.readlines() if not l.startswith("SSBOND")]
    with open(pdb_path, 'w') as f:
        f.writelines(lines)

def minimize_sterics(pdb_path: str):
    """Uses OpenBabel to physically push clashing unbonded sulfurs apart."""
    print("  [PHYSICS] Running energy minimization to open reduced pocket...")
    temp_out = pdb_path.replace(".pdb", "_relaxed.pdb")
    
    # Run a fast 150-step Universal Force Field (UFF) minimization
    cmd = f"obminimize -ff UFF -n 150 {pdb_path} > {temp_out}"
    run_system_command(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Overwrite the unrelaxed file with the relaxed one
    os.replace(temp_out, pdb_path)