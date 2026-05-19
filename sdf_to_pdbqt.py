import os
from pathlib import Path
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem

def sdf_to_pdbqt_meeko(sdf_path: str, output_dir: str = "pdbqt_ligands/") -> str:
    os.makedirs(output_dir, exist_ok=True)
    mol_name = Path(sdf_path).stem
    output_path = os.path.join(output_dir, f"{mol_name}.pdbqt")
    
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next(supplier)
    if mol is None:
        raise ValueError(f"Could not read molecule from {sdf_path}")
    
    preparator = MoleculePreparation(
        merge_these_atom_types=["H"],
        hydrate=False,
        rigid_macrocycles=False
    )
    
    # In older/specific versions of Meeko, prepare returns a list of setups
    mol_setups = preparator.prepare(mol)
    
    try:
        # Loop through setups and use the first valid one
        for setup in mol_setups:
            # We use the Legacy writer or the setup's own string writer
            pdbqt_string = PDBQTWriterLegacy.write_string(setup)[0]
            with open(output_path, "w") as f:
                f.write(pdbqt_string)
            return output_path
    except Exception as e:
        # Fallback for even older versions if the above fails
        try:
            pdbqt_string, is_ok, error_msg = mol_setups[0].write_pdbqt_string()
            if is_ok:
                with open(output_path, "w") as f:
                    f.write(pdbqt_string)
                return output_path
        except:
            raise RuntimeError(f"Meeko writing failed for {mol_name}: {e}")

def batch_convert_ligands(sdf_dir: str = "ligands/", pdbqt_dir: str = "pdbqt_ligands/") -> dict:
    sdf_files = list(Path(sdf_dir).glob("*.sdf"))
    results = {}
    print(f"\nConverting {len(sdf_files)} SDF files to PDBQT...")
    for sdf_file in sdf_files:
        try:
            pdbqt_path = sdf_to_pdbqt_meeko(str(sdf_file), pdbqt_dir)
            if pdbqt_path:
                results[sdf_file.stem] = pdbqt_path
        except Exception as e:
            print(f"  [ERROR] {sdf_file.stem}: {e}")
    print(f"Converted: {len(results)}/{len(sdf_files)} ligands")
    return results

if __name__ == "__main__":
    batch_convert_ligands()
