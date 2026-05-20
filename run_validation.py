import os
import subprocess

def run_command(command, step_name):
    print(f"\n--- Starting {step_name} ---")
    try:
        # Run the command and print the output in real-time
        process = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(process.stdout)
        print(f"--- {step_name} Completed Successfully ---\n")
    except subprocess.CalledProcessError as e:
        print(f"Error during {step_name}:")
        print(e.stderr)
        exit(1)

def main():
    # File names
    empty_protein_pdb = "6WV3_empty.pdb"
    protein_pdbqt = "6WV3_empty.pdbqt"
    
    pubchem_sdf = "pubchem_warfarin.sdf" # Make sure your downloaded file is named this
# Change this line inside run_validation.py:
    ligand_pdbqt = "native_warfarin_AnswerKey.pdbqt"
    docked_output = "pubchem_warfarin_docked.pdbqt"

    # Grid box coordinates for VKORC1 (from your previous successful run)
    center_x, center_y, center_z = -9.83, 26.82, 55.76
    size_x, size_y, size_z = 15, 15, 15

    # ---------------------------------------------------------
    # STEP 1: Prepare the Protein (Add hydrogens & convert to PDBQT)
    # Using OpenBabel: -xr makes it a rigid receptor, -p adds hydrogens at pH 7.4
    # ---------------------------------------------------------
    if not os.path.exists(protein_pdbqt):
        prep_protein_cmd = f"obabel {empty_protein_pdb} -O {protein_pdbqt} -xr -p 7.4"
        run_command(prep_protein_cmd, "Protein Preparation")
    else:
        print(f"Found existing {protein_pdbqt}, skipping preparation.")

    # ---------------------------------------------------------
    # STEP 2: Prepare the Ligand (Convert SDF to PDBQT, add charges/torsions)
    # Using OpenBabel: -h adds hydrogens, --gen3d ensures 3D coords
    # ---------------------------------------------------------
    if not os.path.exists(ligand_pdbqt):
        if not os.path.exists(pubchem_sdf):
            print(f"ERROR: Cannot find {pubchem_sdf}. Please download the 3D SDF from PubChem and rename it!")
            exit(1)
        
        prep_ligand_cmd = f"obabel {pubchem_sdf} -O {ligand_pdbqt} -h --gen3d"
        run_command(prep_ligand_cmd, "Ligand Preparation")
    else:
        print(f"Found existing {ligand_pdbqt}, skipping preparation.")

    # ---------------------------------------------------------
    # STEP 3: Run AutoDock Vina
    # ---------------------------------------------------------
    vina_cmd = (
        f"vina --receptor {protein_pdbqt} --ligand {ligand_pdbqt} "
        f"--center_x {center_x} --center_y {center_y} --center_z {center_z} "
        f"--size_x {size_x} --size_y {size_y} --size_z {size_z} "
        f"--out {docked_output}"
    )
    run_command(vina_cmd, "AutoDock Vina Simulation")

    print("===================================================")
    print("VALIDATION PIPELINE COMPLETE!")
    print(f"Your final docking results are saved in: {docked_output}")
    print("===================================================")
    print("Next step: Open ChimeraX and run the RMSD command to compare")
    print(f"native_warfarin_AnswerKey.pdb and {docked_output}")

if __name__ == "__main__":
    main()

