# test_flex_docking.py
from docking_engine import run_vina_docking

# 1. Define your paths (Make sure these files exist in your directory!)
receptor = "receptors/VKORC1_Human_chainA_protonated.pdbqt"
ligand = "ligands/p_nitro_S.pdbqt"
output = "results/test_p_nitro_S_docked.pdbqt"

# 2. Match your specific grid box parameters from your config
box_parameters = {
    "center": [32.24, 14.10, 9.29],  # Replace with your actual X, Y, Z grid center
    "size": [15.0, 15.0, 15.0]       # Replace with your actual box dimensions
}

print("🚀 Starting single flexible docking test...")

# 3. Call the docking function directly
result = run_vina_docking(
    receptor_pdbqt=receptor,
    ligand_pdbqt=ligand,
    output_pdbqt=output,
    box_params=box_parameters,
    exhaustiveness=8,  # Lowered to 8 just for a quick test run!
    target_name="VKORC1_Human"
)

# 4. Check if it worked
if result.success:
    print("\n✅ TEST SUCCESSFUL!")
    print(f"Best Binding Affinity: {result.best_affinity} kcal/mol")
    print(f"Output saved to: {result.docked_pdbqt}")
else:
    print("\n❌ TEST FAILED!")
    print(f"Error Message: {result.error}")
