#!/usr/bin/env python3
import os
import glob
from rdkit import Chem

def verify_3d_chirality(sdf_path):
    """Loads a 3D molecule and calculates stereocenters directly from spatial coordinates."""
    # Read the SDF file preserving all hydrogrens and 3D positions
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next(suppl)
    
    if mol is None:
        return "❌ Error: Could not parse file format."
    
    # CRITICAL step: Force RDKit to compute R/S labels based on 3D coordinates
    Chem.AssignStereochemistryFrom3D(mol)
    
    # Locate all perceived stereocenters
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    
    if not chiral_centers:
        return "⚠️ No chiral centers detected in this 3D structure."
    
    results = []
    for center in chiral_centers:
        atom_idx, label = center
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        results.append(f"Atom {symbol}(Index {atom_idx}) -> 3D Perceived Configuration: {label}")
        
    return " | ".join(results)

def main():
    target_dir = "results/ligands"
    print("=" * 70)
    print("         LIGAND 3D GEOMETRY CHIRALITY VERIFICATION DIAGNOSTIC")
    print("=" * 70)
    
    if not os.path.exists(target_dir):
        print(f"❌ Error: Target directory '{target_dir}' does not exist.")
        print("Please run this script from the root folder where your pipeline executes.")
        return
        
    sdf_files = glob.glob(os.path.join(target_dir, "*.sdf"))
    if not sdf_files:
        print(f"⚠️ No .sdf files found in '{target_dir}'. Please run Phase 1 of your pipeline first.")
        return
        
    for sdf_path in sorted(sdf_files):
        filename = os.path.basename(sdf_path)
        # Focus on testing references and core assets
        if "ref" in filename or "_S" in filename or "_R" in filename:
            print(f"\n📁 Analyzing: {filename}")
            try:
                outcome = verify_3d_chirality(sdf_path)
                print(f"  {outcome}")
            except Exception as e:
                print(f"  ❌ Analytical crash: {e}")
                
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

