"""
Main virtual screening orchestration script.
Provides command-line options, clean folder isolation, and dynamic RMSD baseline selection.
Includes diagnostic debugging prints for the PLIP pipeline.
Modified to allow gradual, non-destructive library building.
"""

import os
import glob
import yaml
import time
import argparse
import shutil
import subprocess
import pandas as pd
from pathlib import Path

from config_utils import active_ligand_names, ligands_to_smiles_dict
from pipeline_utils import clean_subprocess_env
from ligand_library import prepare_ligand_library, generate_3d_conformers
from sdf_to_pdbqt import batch_convert_ligands
from protein_preparation import download_pdb, clean_protein, add_hydrogens_pdbfixer, protein_to_pdbqt
from grid_box import auto_box_from_native_ligand
from docking_engine import virtual_screening
from analysis import (
    compute_admet_profile, 
    set_publication_style, 
    plot_binding_affinities, 
    generate_summary_report,
    calculate_docking_rmsd,
    run_plip_analysis,
    generate_interaction_heatmap,
    generate_chemical_space_map
)

def auto_configure_openbabel_paths():
    """
    Locates and sets BABEL_LIBDIR and BABEL_DATADIR programmatically 
    to prevent plugin loading failures on WSL/Ubuntu environments.
    """
    possible_libdirs = [
        "/usr/lib/x86_64-linux-gnu/openbabel/3.1.1",
        "/usr/lib/x86_64-linux-gnu/openbabel/3.1.0",
        "/usr/lib/openbabel/3.1.1",
        "/usr/lib/openbabel/3.1.0",
        "/usr/local/lib/openbabel/3.1.1",
        "/usr/local/lib/openbabel/3.1.0"
    ]
    for libdir in possible_libdirs:
        if os.path.exists(libdir):
            os.environ["BABEL_LIBDIR"] = libdir
            datadir = libdir.replace("lib/x86_64-linux-gnu", "share").replace("lib", "share")
            if os.path.exists(datadir):
                os.environ["BABEL_DATADIR"] = datadir
            else:
                parent_datadir = os.path.dirname(datadir)
                if os.path.exists(parent_datadir):
                    os.environ["BABEL_DATADIR"] = parent_datadir
                break

def setup_directories():
    """Generates an organized data architecture tree for tracking outputs."""
    dirs = [
        "ligands/", "pdbqt_ligands/", "proteins/raw/", "proteins/clean/",
        "proteins/protonated/", "proteins/custom/", "pdbqt_receptors/", 
        "results/docked_poses/", "results/screening/", "figures/", "logs/",
        "results/ligands/", "results/pdbqt_ligands/", "results/pdbqt_receptors/",
        "results/plots/"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_clean_system_command(cmd: str, **kwargs):
    """
    Runs a command in a sanitized system environment to prevent 
    virtual-environment library pollution (specifically OpenBabel clashes).
    """
    return subprocess.run(cmd, shell=True, env=clean_subprocess_env(), **kwargs)

def parse_arguments():
    """Defines command line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(description="Automated Molecular Docking Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the pipeline configuration YAML file."
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Validate directories, verify configurations, and calculate grid boxes without execution."
    )
    parser.add_argument(
        "--plip",
        action="store_true",
        help="Enable PLIP interaction profiling and the aggregated interaction heatmap (disabled by default)."
    )
    parser.add_argument(
        "--report-all",
        action="store_true",
        help="Include the full docked history in the summary report and interaction heatmap. "
             "By default the report is scoped to the ligands docked in this run."
    )
    return parser.parse_args()

def run_pipeline():
    auto_configure_openbabel_paths()
    
    # profile_csv = "results/interaction_profile.csv"
    # if os.path.exists(profile_csv):
    #     try:
    #         os.remove(profile_csv)
    #     except OSError:
    #         pass

    args = parse_arguments()
    start_time = time.time()
    setup_directories()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration profile missing: {args.config}")
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"\n{'#'*70}\n# STUDY: {config['project_name']}\n{'#'*70}")

    ligand_dict = config.get('ligands', {})
    active_names = set(active_ligand_names(ligand_dict))
    flat_ligands = ligands_to_smiles_dict(ligand_dict, active_only=True)
    receptors_dict = config.get('receptors', {})
    
    if args.dry_run:
        print("\n>>> [DRY-RUN MODE ACTIVATED] — Validating setup details...")
        errors = 0
        
        # Verify ligands
        if not ligand_dict:
            print("  [ERROR] No ligands defined in config.")
            errors += 1
        else:
            print(f"  [OK] {len(flat_ligands)} active ligand(s) detected in config.")
            
        # Verify receptors and pre-calculate grid boxes
        if not receptors_dict:
            print("  [ERROR] No receptors defined in config.")
            errors += 1
        else:
            print(f"  [OK] {len(receptors_dict)} receptor(s) detected in config.")
            for name, info in receptors_dict.items():
                print(f"\n  Calculating grid box coordinates for: {name}")
                base_id = info.get('base_pdb_id', info['pdb_id'])
                
                expected_pdbqt = os.path.join(
                    "pdbqt_receptors", 
                    f"{name}_chain{info['chain']}_protonated.pdbqt"
                )
                if os.path.exists(expected_pdbqt):
                    print(f"    [CACHE] Pre-compiled PDBQT detected: {expected_pdbqt}")
                
                try:
                    raw_path = os.path.join("proteins/raw", f"{base_id}.pdb")
                    if not os.path.exists(raw_path):
                        print(f"    [DOWNLOAD] Pulling structure reference {base_id} for grid coordinates...")
                        raw_path = download_pdb(base_id)
                        
                    box_params = auto_box_from_native_ligand(
                        raw_path,
                        info.get('native_ligand_resname'),
                        padding=info.get('padding', 5.0),
                        force_size=info.get('force_size'),
                        force_center=info.get('force_center')
                    )
                    print(f"    [GRID] Center: {box_params['center']}")
                    print(f"    [GRID] Size:   {box_params['size']}")
                except Exception as e:
                    print(f"    [ERROR] Grid calculation failed for {name}: {e}")
                    errors += 1
                    
        print(f"\n{'='*70}\nDRY-RUN COMPLETED with {errors} blocking error(s).\n{'='*70}")
        return

    print("\n>>> PHASE 1: GENERATING AND CONVERTING LIGAND LIBRARY...")
    df_ligands = prepare_ligand_library(ligand_dict)
    df_ligands = df_ligands[df_ligands["Name"].isin(active_names)].reset_index(drop=True)
    needed_ligands = list(flat_ligands.keys())

    # Number of conformers embedded per ligand is now a config knob.
    num_confs = int(config.get("docking_params", {}).get("num_conformers", 20))

    # Conformer generation only. PDBQT conversion is handled by a single,
    # canonical Meeko pass below (batch_convert_ligands) — the previous
    # per-ligand OpenBabel pre-pass has been removed to avoid a second,
    # divergent conversion route and an extra round of hydrogen handling.
    for _, row in df_ligands.iterrows():
        name = row['Name']
        sdf_path = f"results/ligands/{name}.sdf"
        pdbqt_path = f"pdbqt_ligands/{name}.pdbqt"

        if os.path.exists(pdbqt_path):
            print(f"  [CACHE] Confirmed PDBQT target for {name}. Skipping.")
            continue

        if os.path.exists(sdf_path):
            continue

        try:
            print(f"  [COMPUTING] Generating coordinates for: {name}")
            generate_3d_conformers(row["Mol"], name, output_dir="results/ligands", num_confs=num_confs)
        except Exception as e:
            print(f"  [ERROR] Conformer mapping failed for {name}: {e}")

    print("  [INFO] Synchronizing conversions to PDBQT format (Meeko)...")
    try:
        batch_convert_ligands(sdf_dir="results/ligands/", pdbqt_dir="pdbqt_ligands/")
    except Exception as e:
        print(f"  [CRITICAL ERROR] Batch translation failed: {e}")

    # Mirror the converted PDBQT ligands into the results tree for downstream tools.
    os.makedirs("results/pdbqt_ligands", exist_ok=True)
    for pq in glob.glob("pdbqt_ligands/*.pdbqt"):
        dest = os.path.join("results/pdbqt_ligands", os.path.basename(pq))
        try:
            if not os.path.exists(dest) or os.path.getmtime(pq) > os.path.getmtime(dest):
                shutil.copy2(pq, dest)
        except OSError:
            pass

    # Compute ADMET Profiles (Accumulative, cached by SMILES)
    print("  [INFO] Computing ADMET profiles...")
    admet_csv_path = "results/admet_profile.csv"

    admet_df_old = None
    cached_smiles = {}
    if os.path.exists(admet_csv_path):
        try:
            admet_df_old = pd.read_csv(admet_csv_path)
            if "SMILES" in admet_df_old.columns:
                cached_smiles = dict(zip(admet_df_old["Name"].astype(str), admet_df_old["SMILES"].astype(str)))
        except Exception:
            admet_df_old = None

    # Only (re)compute ligands that are new or whose SMILES changed since last time.
    to_compute = {n: s for n, s in flat_ligands.items() if cached_smiles.get(str(n)) != str(s)}
    if to_compute:
        print(f"  [INFO] ADMET cache: computing {len(to_compute)} new/changed ligand(s); "
              f"reusing {len(ligand_dict) - len(to_compute)} cached.")
        admet_df_new = compute_admet_profile(to_compute)
    else:
        print("  [INFO] ADMET cache: all ligands up to date; nothing to recompute.")
        admet_df_new = pd.DataFrame()

    if admet_df_old is not None and not admet_df_old.empty:
        if not admet_df_new.empty:
            admet_df_old = admet_df_old[~admet_df_old["Name"].isin(admet_df_new["Name"])]
            admet_df = pd.concat([admet_df_old, admet_df_new], ignore_index=True)
        else:
            admet_df = admet_df_old
    else:
        admet_df = admet_df_new

    admet_df.to_csv(admet_csv_path, index=False)

    # PHASE 2 & 3: Receptor Preparation & Virtual Screening
    screening_results = {}
    rmsd_results = {}
    
    for target_name, target_info in receptors_dict.items():
        print(f"\n>>> PHASE 2 & 3: PREPARING TARGET & DOCKING -> {target_name}")
        
        expected_pdbqt = os.path.join(
            "pdbqt_receptors", 
            f"{target_name}_chain{target_info['chain']}_protonated.pdbqt"
        )
        
        protonated_pdb = os.path.join(
            "proteins/protonated", 
            f"{target_name}_chain{target_info['chain']}_protonated.pdb"
        )
        
        pdb_code = target_info['pdb_id'] 
        raw_pdb = os.path.join("proteins/raw", f"{pdb_code}.pdb")
        
        custom_pdb_path = os.path.join("proteins/custom", f"{target_name}.pdb")
        custom_source_found = os.path.exists(custom_pdb_path)
        
        try:
            if os.path.exists(expected_pdbqt):
                print(f"  [CACHE] Pre-compiled PDBQT found: {expected_pdbqt}. Bypassing preparation.")
                receptor_pdbqt = expected_pdbqt
                # Only re-download the raw structure if the grid-box step actually needs it.
                if not os.path.exists(raw_pdb):
                    raw_pdb = download_pdb(pdb_code)
            else:
                if not os.path.exists(raw_pdb):
                    raw_pdb = download_pdb(pdb_code)
                
                if custom_source_found:
                    print(f"  [CUSTOM SOURCE] Found manual custom coordinate file: {custom_pdb_path}")
                    print(f"  [CUSTOM SOURCE] Preserving custom coordinates; bypassing automated cleaning.")
                    receptor_pdbqt = protein_to_pdbqt(custom_pdb_path)
                    shutil.copy2(custom_pdb_path, protonated_pdb)
                else:
                    print(f"  [AUTOMATED PREPARATION] Compiling standard model for: {target_name}")
                    clean_pdb = clean_protein(raw_pdb, target_info['chain'])
                    add_hydrogens_pdbfixer(clean_pdb, protonated_pdb)
                    receptor_pdbqt = protein_to_pdbqt(protonated_pdb)
            
            # --- SAFE CUSTOM COORDINATE HANDLING & BYPASS ---
            if target_info.get('force_center') and target_info.get('force_size'):
                print(f"  [GRID] Using manually forced box coordinates for {target_name}. Bypassing native ligand search.")
                box_params = {
                    "center": target_info['force_center'],
                    "size": target_info['force_size']
                }
            else:
                box_params = auto_box_from_native_ligand(
                    raw_pdb, 
                    target_info['native_ligand_resname'], 
                    padding=target_info.get('padding', 5.0),
                    force_size=target_info.get('force_size'),
                    force_center=target_info.get('force_center')    
                )
            
            dock_cfg = config['docking_params']
            box_params['exhaustiveness'] = dock_cfg.get('exhaustiveness', 32)
            box_params['num_modes'] = dock_cfg.get('num_modes', 20)

            target_output_dir = os.path.join("results/docked_poses", target_name)
            os.makedirs(target_output_dir, exist_ok=True)
            
            # Run Virtual Screening
            df_target_results = virtual_screening(
                ligand_pdbqt_dir="pdbqt_ligands",
                receptor_pdbqt=receptor_pdbqt,
                target_name=target_name,
                box_params=box_params,
                output_dir=target_output_dir,
                exhaustiveness=box_params['exhaustiveness'],
                receptor_pdb=protonated_pdb if os.path.exists(protonated_pdb) else raw_pdb,
                flex_res_list=target_info.get('flexible_residues', []),
                needed_ligands=needed_ligands,
                num_modes=box_params['num_modes'],
                min_rmsd=dock_cfg.get('min_rmsd', 1.0),
                n_cpu=dock_cfg.get('n_cpu'),
                dock_timeout=dock_cfg.get('dock_timeout_s', 900)
            )
            screening_results[target_name] = df_target_results

            # virtual_screening returns the full accumulated history; restrict the
            # per-ligand post-processing (PLIP, RMSD) to only the ligands docked in
            # THIS run so we don't reprofile every historical pose on disk.
            run_results = df_target_results[df_target_results["ligand_name"].isin(needed_ligands)]

            # --- PROCESS ISOLATED PLIP INTEGRATION LOOP ---
            if not args.plip:
                print("  [INFO] PLIP interaction profiling disabled (pass --plip to enable).")
            else:
                print(f"  [INFO] Running PLIP interaction profiling on {len(run_results)} ligand(s) from this run...")
            for idx, row in (run_results.iterrows() if args.plip else []):
                docked_file = row['docked_pdbqt']
                
                if os.path.exists(docked_file):
                    receptor_pdb = protonated_pdb if os.path.exists(protonated_pdb) else raw_pdb
                    
                    try:
                        clean_env = clean_subprocess_env()
                        
                        cmd = [
                            "python", "-c",
                            f"import sys; sys.path.append('.'); from analysis import run_plip_analysis; run_plip_analysis('{receptor_pdb}', '{docked_file}', 'results/interaction_profile.csv')"
                        ]
                        
                        result = subprocess.run(cmd, env=clean_env, capture_output=True, text=True, timeout=30)
                        
                        if result.returncode != 0:
                            # DIAGNOSTIC 2: If PLIP background crash happens, capture and print the C++/Python error traceback
                            print(f"  [WARNING] PLIP analysis failed for {row['ligand_name']} (exit code {result.returncode})")
                            print(f"  [PLIP DEBUG] Subprocess stderr traceback:\n{result.stderr.strip()}")
                            try:
                                from analysis import write_interaction_record
                                fallback_record = {
                                    "Receptor": target_name,
                                    "Ligand": row['ligand_name'],
                                    "Num_H_Bonds": 0,
                                    "H_Bond_Residues": "None (C++ Abort)",
                                    "Num_Hydrophobic": 0,
                                    "Hydrophobic_Residues": "None (C++ Abort)",
                                    "Num_Pi_Stacking": 0,
                                    "Pi_Stacking_Residues": "None (C++ Abort)"
                                }
                                write_interaction_record(fallback_record, "results/interaction_profile.csv")
                            except Exception:
                                pass
                        else:
                            output_lines = result.stdout.strip().split("\n")
                            for line in output_lines:
                                if "[PLIP]" in line:
                                    print(line)
                                    
                    except subprocess.TimeoutExpired:
                        print(f"  [WARNING] PLIP analysis timed out for {row['ligand_name']}")
                    except Exception as plip_err:
                        print(f"  [WARNING] Skipping PLIP profile for {row['ligand_name']}: {plip_err}")
            
            # Automated RMSD Validation of reference ligands
            for idx, row in run_results.iterrows():
                ligand_name = row['ligand_name']
                if ligand_name.endswith("_ref") and target_info.get('native_ligand_resname'):
                    docked_file = row['docked_pdbqt']
                    if os.path.exists(docked_file):
                        ref_smiles = flat_ligands.get(ligand_name)
                        
                        rmsd_baseline_pdb = raw_pdb
                        
                        # --- EXPLICIT LOG DETAILS WRITTEN HERE AS REQUESTED ---
                        print(f"  [RMSD BASELINE] Calculating heavy-atom RMSD for '{ligand_name}' "
                              f"against native ligand '{target_info['native_ligand_resname']}' "
                              f"from crystal reference file '{os.path.basename(rmsd_baseline_pdb)}'")
                        
                        rmsd_val = calculate_docking_rmsd(
                            docked_pdbqt_path=docked_file,
                            native_pdb_path=rmsd_baseline_pdb,
                            resname=target_info['native_ligand_resname'],
                            ref_smiles=ref_smiles
                        )
                        if rmsd_val is not None:
                            rmsd_results[f"{target_name}_{ligand_name}"] = rmsd_val
                            print(f"  [VALIDATION] Heavy-atom RMSD for {ligand_name}: {rmsd_val:.2f} Å")
                        
        except Exception as e:
            print(f"  [ERROR] Execution failed for receptor {target_name}: {e}")
            continue

    # PHASE 4: Analysis and Plotting
    print("\n>>> PHASE 4: ANALYSIS AND GRAPHICAL PLOTTING...")
    set_publication_style()

    # Scope the report/plots to this run's ligands unless the user asked for the full history.
    report_scope = None if args.report_all else needed_ligands
    if args.report_all:
        print("  [REPORT] Including full docked history (--report-all).")
    else:
        print(f"  [REPORT] Scoping summary report & plots to this run's {len(needed_ligands)} ligand(s).")

    for target_name, res_df in screening_results.items():
        try:
            plot_df = res_df if report_scope is None else res_df[res_df["ligand_name"].isin(report_scope)]
            plot_binding_affinities(
                plot_df,
                target_name=target_name, 
                output_path=f"figures/{target_name}_affinities.png"
            )
        except Exception as e:
            print(f"  [ERROR] Could not generate plots for {target_name}: {e}")

    generate_summary_report(
        screening_results, 
        admet_df, 
        rmsd_results=rmsd_results,
        output_path="results/docking_summary_report.txt",
        include_ligands=report_scope
    )

    # PHASE 5: Aggregated PLIP Interaction Heatmap & Chemical Space Map Generation
    print("\n>>> PHASE 5: INTERACTION HEATMAP & CHEMICAL SPACE GENERATION...")
    if args.plip:
        try:
            generate_interaction_heatmap("results/interaction_profile.csv", "results/plots/", include_ligands=report_scope)
        except Exception as e:
            print(f"  [ERROR] Could not generate interaction heatmaps: {e}")
    else:
        print("  [INFO] Skipping interaction heatmap (PLIP disabled; pass --plip to enable).")

    try:
        # Pass the human VKORC1 target results for chemical space mapping
        target_df = screening_results.get("VKORC1_Human")
        if target_df is None and screening_results:
            target_df = list(screening_results.values())[0]
        if target_df is not None:
            generate_chemical_space_map(target_df, flat_ligands, "results/plots/")
        else:
            print("  [WARNING] No active VKORC1 results for Figure 5. Skipping.")
    except Exception as e:
        print(f"  [ERROR] Could not generate chemical space map: {e}")

    print(f"\n{'='*70}\n[SUCCESS] Pipeline completed in {(time.time() - start_time)/60:.2f} minutes.\n{'='*70}")

if __name__ == "__main__":
    run_pipeline()