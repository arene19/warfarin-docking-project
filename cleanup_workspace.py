import os
import shutil

def cleanup_workspace():
    # 1. Define the new organization folders
    keep_folder = "cleanup_keep"
    unsure_folder = "cleanup_unsure"
    
    for folder in [keep_folder, unsure_folder]:
        os.makedirs(folder, exist_ok=True)

    # 2. Define the exact files/folders to target
    files_to_delete = [
        "VKORC1_Reduced_chainA_protonated.pdbqt.pdbqt",
        "6WV3.pdb", "6WV3_empty.pdb", "6WV3_empty.pdbqt",
        "native_ligand.pdb",
        "reference_warfarin.pdb", "reference_warfarin.pdbqt", 
        "reference_warfarin_chainA.pdb", "reference_warfarin_chainA.pdbqt",
        "full_project_code.txt", "project_context.txt",
        "docking_results.csv" # Old output format
    ]

    files_to_keep = [
        "publication_admet_table16.png",
        "publication_heatmap16.png",
        "publication_residue_interactions16.png",
        "publication_stereoselectivity16.png"
    ]

    folders_unsure = [
        "old_file_backup", "ligands_archive", "scripts_archive", 
        "inputs", "outputs", "random", 
        "vkor_docking", "validated_system", "scenes"
    ]

    print(f"🧹 Starting workspace cleanup...\n")

    # 3. DELETE FILES
    for file in files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"  [DELETED] {file}")
            except Exception as e:
                print(f"  [ERROR] Could not delete {file}: {e}")

    # 4. DELETE __pycache__ folders if they exist at root
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")
        print(f"  [DELETED] __pycache__ directory")

    # 5. MOVE GOOD FILES TO 'KEEP'
    for file in files_to_keep:
        if os.path.exists(file):
            try:
                shutil.move(file, os.path.join(keep_folder, file))
                print(f"  [MOVED TO KEEP] {file}")
            except Exception as e:
                print(f"  [ERROR] Could not move {file}: {e}")

    # 6. MOVE UNKNOWN FOLDERS TO 'UNSURE'
    for item in folders_unsure:
        if os.path.exists(item):
            try:
                shutil.move(item, os.path.join(unsure_folder, item))
                print(f"  [MOVED TO UNSURE] {item}")
            except Exception as e:
                print(f"  [ERROR] Could not move {item}: {e}")

    print(f"\n✅ Cleanup complete! Your core .py files, configs, and venv were untouched.")

if __name__ == "__main__":
    cleanup_workspace()