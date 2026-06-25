"""
Translates coordinate structures from SDF formatting to standard PDBQT inputs.
Supports both Meeko and native OpenBabel fallback routines.
"""

import os
import subprocess
from pathlib import Path
from rdkit import Chem

def sdf_to_pdbqt_meeko(sdf_path: str, output_dir: str = "pdbqt_ligands/") -> str:
    """Converts a single SDF file to PDBQT format using Meeko with an OpenBabel fallback."""
    os.makedirs(output_dir, exist_ok=True)
    mol_name = Path(sdf_path).stem
    output_path = os.path.join(output_dir, f"{mol_name}.pdbqt")
    
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol = next(supplier)
        if mol is None:
            raise ValueError(f"RDKit unable to read molecule from {sdf_path}")
        
        preparator = MoleculePreparation(
            merge_these_atom_types=["H"],
            hydrate=False,
            rigid_macrocycles=False
        )
        
        mol_setups = preparator.prepare(mol)
        
        for setup in mol_setups:
            pdbqt_string = PDBQTWriterLegacy.write_string(setup)[0]
            with open(output_path, "w") as f:
                f.write(pdbqt_string)
            return output_path
            
    except Exception as e:
        print(f"  [Meeko Fallback] Direct Meeko translation failed for {mol_name} ({e}). Trying OpenBabel...")
        cmd = f"obabel {sdf_path} -O {output_path} -m"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        raise RuntimeError(f"All structural conversion pipelines failed for {mol_name}")

def batch_convert_ligands(sdf_dir: str = "ligands/", pdbqt_dir: str = "pdbqt_ligands/") -> dict:
    """Converts directories containing SDF files to target formats."""
    sdf_files = list(Path(sdf_dir).glob("*.sdf"))
    results = {}
    
    for sdf_file in sdf_files:
        try:
            # Skip if PDBQT is already present
            expected_pdbqt = os.path.join(pdbqt_dir, f"{sdf_file.stem}.pdbqt")
            if os.path.exists(expected_pdbqt):
                results[sdf_file.stem] = expected_pdbqt
                continue
                
            pdbqt_path = sdf_to_pdbqt_meeko(str(sdf_file), pdbqt_dir)
            if pdbqt_path:
                results[sdf_file.stem] = pdbqt_path
        except Exception as e:
            print(f"  [ERROR] conversion failed for {sdf_file.stem}: {e}")
            
    return results