# main_pipeline.py
import os
import yaml
import time
import shutil
import pandas as pd
from pathlib import Path

from ligand_library      import prepare_ligand_library, generate_3d_conformers
from sdf_to_pdbqt        import batch_convert_ligands
from protein_preparation import download_pdb, clean_protein, add_hydrogens_pdbfixer, protein_to_pdbqt
from grid_box            import auto_box_from_native_ligand
from docking_engine      import virtual_screening
from analysis            import compute_admet_profile, set_publication_style, plot_binding_affinities, generate_summary_report

def setup_directories():
    """Generates an organized data architecture tree for tracking outputs."""
    dirs = [
        "ligands/", "pdbqt_ligands/", "proteins/raw/", "proteins/clean/",
        "proteins/protonated/", "pdbqt_receptors/", "results/docked_poses/",
        "results/screening/", "figures/", "logs/"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_pipeline(config_file="config.yaml"):
    start_time = time.time()
    setup_directories()
    
    # 1. Load runtime parameters from configuration file
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration profile setup missing: create {config_file} to proceed.")
        
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"\n{'#'*70}\n# RUNNING STUDY: {config['project_name']}\n{'#'*70}")

    # 2. Ligand Chemistry Libraries Formulation
    print("\n>>> PHASE 1: GENERATING AND CONVERTING LIGAND LIBRARY...")
    df_ligands = prepare_ligand_library(config['ligands'])
    
    for _, row in df_ligands.iterrows():
        try:
            generate_3d_conformers(row["Mol"], row["Name"])
        except Exception as e:
            print(f"  [ERROR] Conformer calculation initialization failed for {row['Name']}: {e}")
            
    batch_convert_ligands(sdf_dir="ligands/", pdbqt_dir="pdbqt_ligands/")
    
    # ADMET Profile Formulation
    admet_df = compute_admet_profile(config['ligands'])
    admet_df.to_csv("results/admet_profile.csv", index=False)

    # 3. Protein Handling & Screen Processing Loop
    screening_results = {}
    
    for target_name, target_info in config['receptors'].items():
        print(f"\n>>> PHASE 2 & 3: PREPARING TARGET & DOCKING -> {target_name}")
        
        # Structure acquisition
        raw_pdb = download_pdb(target_info['pdb_id'])
        clean_pdb = clean_protein(raw_pdb, target_info['chain'])
        
        protonated_pdb = os.path.join("proteins/protonated", f"{target_info['pdb_id']}_chain{target_info['chain']}_protonated.pdb")
        add_hydrogens_pdbfixer(clean_pdb, protonated_pdb)
        receptor_pdbqt = protein_to_pdbqt(protonated_pdb)
        
        # Automated grid localization
        box_params = auto_box_from_native_ligand(
            raw_pdb, 
            target_info['native_ligand_resname'], 
            padding=target_info.get('padding', 5.0),
            force_size=target_info.get('force_size'),
            force_center=target_info.get('force_center')    
        )
        
        # Append docking speed parameters 
        box_params['exhaustiveness'] = config['docking_params'].get('exhaustiveness', 32)
        box_params['num_modes'] = config['docking_params'].get('num_modes', 20)
        
        target_output_dir = os.path.join("results/docked_poses", target_name)
        os.makedirs(target_output_dir, exist_ok=True)
        
        # Execute Virtual Screen Process
        df_target_results = virtual_screening(
            ligand_pdbqt_dir="pdbqt_ligands",
            receptor_pdbqt=receptor_pdbqt,
            target_name=target_name,
            box_params=box_params,
            output_dir=target_output_dir,
            exhaustiveness=box_params['exhaustiveness']
        )
        screening_results[target_name] = df_target_results

    # 4. Post-Run Graphical Data Figure Generation
    print("\n>>> PHASE 4: ANALYSIS AND GRAPHICAL PLOTTING...")
    set_publication_style()
    
    for target_name, res_df in screening_results.items():
        plot_binding_affinities(res_df, target_name=target_name, output_path=f"figures/{target_name}_affinities.png")
        
    generate_summary_report(screening_results, admet_df, "results/docking_summary_report.txt")
    print(f"\n{'='*70}\n[SUCCESS] Pipeline executed in {(time.time() - start_time)/60:.2f} minutes.\n{'='*70}")

if __name__ == "__main__":
    run_pipeline()
