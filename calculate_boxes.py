import os
from protein_preparation import TARGETS
from grid_box import compute_box_from_residues

print("="*50)
print("  CALCULATING EXACT GRID BOXES")
print("="*50)

# The targets giving 0.00 kcal/mol
targets_to_fix = ["Factor_Xa", "Thrombin", "CYP2C9"]

for target_name in targets_to_fix:
    info = TARGETS[target_name]
    # Point to the clean PDB file we generated in Phase 2
    pdb_path = f"proteins/clean/{info['pdb_id']}_chain{info['chain']}.pdb"
    
    if not os.path.exists(pdb_path):
        print(f"\n[ERROR] Missing {pdb_path}. Did you delete it?")
        continue
        
    print(f"\n[TARGET] {target_name} ({info['pdb_id']})")
    
    try:
        # Auto-compute the box based on the known binding residues
        box = compute_box_from_residues(
            pdb_path=pdb_path,
            residue_numbers=info["binding_residues"],
            chain_id=info["chain"],
            padding=5.0
        )
        
        # Print the exact dictionary format so you can copy-paste it
        print("  PASTE THIS INTO grid_box.py:")
        print(f"    \"{target_name}\": {{")
        print(f"        \"center\": ({box['center'][0]:.2f}, {box['center'][1]:.2f}, {box['center'][2]:.2f}),")
        print(f"        \"size\":   ({box['size'][0]:.2f}, {box['size'][1]:.2f}, {box['size'][2]:.2f}),")
        print(f"        \"exhaustiveness\": 32,")
        print(f"        \"num_modes\": 20")
        print(f"    }},")
    except Exception as e:
        print(f"  [ERROR] {e}")
