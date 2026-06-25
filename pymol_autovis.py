#!/usr/bin/env python3
"""
pymol_autovis.py (UCSF ChimeraX 1.11.1/1.12 Native)
======================================================================
Automated Publication-Grade Structural Visualization Engine.
Generates headless images (.png), saved sessions (.cxs), and
universal, platform-independent command scripts (.cxc) for docking poses.
======================================================================
"""

import os
import sys
import yaml
import argparse
import tempfile
import subprocess
import shutil
import glob
from pathlib import Path

def setup_arguments():
    """Defines command-line entry points for path overrides."""
    parser = argparse.ArgumentParser(
        description="Automate publication-quality ChimeraX structural renders from docking runs."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to pipeline config file (default: config.yaml)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/visualizations", 
        help="Target folder for rendered images and sessions"
    )
    parser.add_argument(
        "--clean_dir", 
        type=str, 
        default="proteins/protonated", 
        help="Folder containing protonated receptor structures"
    )
    parser.add_argument(
        "--raw_dir", 
        type=str, 
        default="proteins/raw", 
        help="Folder containing raw crystal structures with native ligands"
    )
    parser.add_argument(
        "--dock_dir", 
        type=str, 
        default="results/docked_poses", 
        help="Folder containing target docked poses"
    )
    parser.add_argument(
        "--ligands",
        type=str,
        default="",
        help="Comma-separated ligand/compound names to render. Empty = every docked pose found."
    )
    return parser.parse_args()


def parse_ligand_selection(raw: str) -> set:
    """Parses a comma-separated ligand selection string into a clean set (empty = all)."""
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}

def sanitize_environment():
    """
    Sanitizes active environment variables to prevent venv interpreter
    collisions, while forcing headless software OpenGL rasterization 
    and removing WSL GPU-accelerated library paths to prevent headless crashes.
    """
    clean_env = os.environ.copy()
    
    # 1. Extract active project virtualenv paths
    venv_path = clean_env.get("VIRTUAL_ENV")
    clean_env.pop("VIRTUAL_ENV", None)
    clean_env.pop("PYTHONPATH", None)

    # 2. Filter out venv bin folders from the execution PATH
    path_dirs = clean_env.get("PATH", "").split(os.pathsep)
    sanitized_dirs = []
    
    for directory in path_dirs:
        if venv_path and venv_path in directory:
            continue
        if "venv/bin" in directory or "venv/sbin" in directory:
            continue
        # Strip out WSL GPU-accelerated paths from PATH to force native binaries
        if "/usr/lib/wsl" in directory:
            continue
        sanitized_dirs.append(directory)

    clean_env["PATH"] = os.pathsep.join(sanitized_dirs)
    
    # 3. Filter out WSL GPU-accelerated driver directories from LD_LIBRARY_PATH
    if "LD_LIBRARY_PATH" in clean_env:
        ld_paths = clean_env["LD_LIBRARY_PATH"].split(os.pathsep)
        sanitized_ld = [p for p in ld_paths if "/usr/lib/wsl" not in p]
        # Prepend standard Ubuntu x86 Mesa path for absolute stability
        clean_env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.pathsep.join(sanitized_ld)
    else:
        # If not set, explicitly point it to standard system libraries to bypass driver searches
        clean_env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/lib"
    
    # 4. Force headless CPU-based software OpenGL rasterization (Mesa OSMesa/llvmpipe)
    clean_env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    clean_env["GALLIUM_DRIVER"] = "llvmpipe"
    
    return clean_env

def extract_ligand_only(input_path: str, output_path: str) -> bool:
    """
    Reads a multi-model Vina output file, extracts the first model,
    and strips out any flexible amino acid residues, leaving ONLY 
    the docked organic ligand coordinates.
    """
    amino_acids = {
        "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
        "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
    }
    try:
        with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
            in_first_model = False
            for line in f_in:
                if line.startswith("MODEL"):
                    if "1" in line:
                        in_first_model = True
                    continue
                if line.startswith("ENDMDL"):
                    break
                
                if in_first_model:
                    # Skip flexible residue header structures
                    if line.startswith(("BEGIN_RES", "END_RES")):
                        continue
                    
                    # Skip standard amino acid coordinates
                    if line.startswith(("ATOM", "HETATM")):
                        res_name = line[17:20].strip()
                        if res_name in amino_acids:
                            continue
                            
                    f_out.write(line)
        return True
    except Exception as e:
        print(f"  [Warning] Failed to extract clean ligand: {e}")
        return False

def convert_pdbqt_to_pdb(pdbqt_path: str, pdb_path: str) -> str:
    """
    Attempts to translate a clean PDBQT ligand file into a clean PDB file 
    using OpenBabel on-the-fly. Falls back to original path if OpenBabel is missing.
    """
    try:
        cmd = ["obabel", pdbqt_path, "-O", pdb_path]
        clean_env = sanitize_environment()
        
        result = subprocess.run(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            env=clean_env,
            check=True
        )
        if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0:
            return pdb_path
    except Exception:
        pass
    return pdbqt_path

def generate_headless_cxc(
    receptor_pdb: str,
    docked_pose_path: str,
    output_png: str,
    output_cxs: str
) -> str:
    """
    Writes a headless ChimeraX Command (.cxc) script that simply opens the
    protein and ligand, centers the view, and saves the output.
    """
    # Force absolute path resolution to bypass temporary /tmp/ directory offsets
    receptor_pdb = os.path.abspath(receptor_pdb)
    docked_pose_path = os.path.abspath(docked_pose_path)
    output_png = os.path.abspath(output_png)
    output_cxs = os.path.abspath(output_cxs)

    cxc_lines = [
        f"open \"{receptor_pdb}\"",
        f"open \"{docked_pose_path}\"",
        "view",
        f"save \"{output_cxs}\"",
        f"save \"{output_png}\" width 1200 height 900 supersample 3",
        "exit"
    ]

    # Write transient script
    temp_cxc = tempfile.NamedTemporaryFile(suffix=".cxc", delete=False, mode="w")
    temp_cxc.write("\n".join(cxc_lines))
    temp_cxc.close()
    return temp_cxc.name

def generate_universal_cxc(
    receptor_pdb_rel: str,
    docked_pose_rel: str,
    output_cxc_path: str,
    raw_crystal_rel: str = None,
    native_resname: str = None
):
    """
    Writes a platform-independent, version-safe ChimeraX script using relative paths.
    This file has fully configured styling (shadows, silhouettes, cyan side-chains, magenta ligand,
    and yellow H-bonds) and can be opened in any ChimeraX GUI (Windows/Linux/Mac) with zero commands.
    """
    cxc_lines = [
        "# Load models (Model #1 = Protein, Model #2 = Clean Ligand)",
        f"open \"{receptor_pdb_rel}\"",
        f"open \"{docked_pose_rel}\"",
        "",
        "# Graphical styling & Quality",
        "set bgColor white",
        "lighting soft",
        "graphics silhouettes true width 1.5",
        "camera ortho",
        "",
        "# Style protein ribbon",
        "hide #1 atoms",
        "show #1 cartoons",
        "color #1 lightgrey",
        "",
        "# Style docked ligand",
        "show #2",
        "style #2 stick",
        "size #2 stickRadius 0.25",
        "color #2 magenta",
        "color #2 byhetero",
        ""
    ]

    if raw_crystal_rel and native_resname:
        cxc_lines.extend([
            "# Load native crystal reference as Model #3",
            f"open \"{raw_crystal_rel}\"",
            "hide #3 atoms",
            f"show #3:{native_resname}",
            f"style #3:{native_resname} stick",
            f"size #3:{native_resname} stickRadius 0.10",
            f"color #3:{native_resname} green",
            f"color #3:{native_resname} byhetero",
            ""
        ])

    cxc_lines.extend([
        "# Isolate and style active-site binding pocket residues",
        "show #1 within 5.0 of #2 & sideonly",
        "color #1 within 5.0 of #2 & sideonly cyan",
        "color #1 within 5.0 of #2 & sideonly byhetero",
        "hide #1 within 5.0 of #2 & mainchain",
        "hide H",
        "",
        "# Map and color polar contacts (H-bonds)",
        "hbonds #1 restrict #2 color #f1c40f lineWidth 2.5",
        "",
        "# Center view to the ligand",
        "view #2"
    ])

    with open(output_cxc_path, "w") as f:
        f.write("\n".join(cxc_lines))

def check_chimerax_installation():
    """Verifies ChimeraX is present on system PATH using a safe PATH search."""
    clean_path = "/usr/bin:/usr/local/bin:/bin:" + os.environ.get("PATH", "")
    for bin_name in ["chimerax", "ucsf-chimerax"]:
        pymol_bin_path = shutil.which(bin_name, path=clean_path)
        if pymol_bin_path:
            return pymol_bin_path
    return None

def main():
    chimerax_bin = check_chimerax_installation()

    # Verify ChimeraX executable on system PATH before execution
    if not chimerax_bin:
        print("[CRITICAL ERROR] ChimeraX executable could not be verified on your system PATH.")
        print("Please ensure ChimeraX is installed globally (e.g. 'sudo apt install chimerax').")
        sys.exit(1)

    # Load configuration
    if not os.path.exists(args.config):
        print(f"[ERROR] Pipeline config file missing: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    receptors_meta = config.get("receptors", {})
    if not receptors_meta:
        print("[ERROR] No receptors matched inside config file.")
        sys.exit(1)

    selected_ligands = parse_ligand_selection(getattr(args, "ligands", ""))

    print(f"\n{'='*70}\n  LAUNCHING CHIMERAX AUTOMATED VISUALIZATION MODULE\n{'='*70}\n")
    if selected_ligands:
        print(f"  [SELECTION] Rendering only {len(selected_ligands)} selected compound(s): "
              f"{', '.join(sorted(selected_ligands))}")
    else:
        print("  [SELECTION] No ligand filter supplied — rendering every docked pose found.")

    total_runs = 0
    success_runs = 0

    # Process each target in the config
    for target_name, target_info in receptors_meta.items():
        print(f"🧬 Processing Visualizations for target: {target_name}")

        # Locate clean/protonated receptor structure
        expected_receptors = [
            os.path.join(args.clean_dir, f"{target_name}_chain{target_info['chain']}_protonated.pdb"),
            os.path.join(args.clean_dir, f"{target_name}.pdb")
        ]
        
        receptor_pdb = None
        for r_path in expected_receptors:
            if os.path.exists(r_path):
                receptor_pdb = r_path
                break

        if not receptor_pdb:
            print(f"  [SKIP] No protonated receptor PDB found for {target_name}. Skipping target.")
            continue

        # Crystal Reference paths for native ligand extraction
        raw_crystal_pdb = os.path.join(args.raw_dir, f"{target_info['pdb_id']}.pdb")
        native_resname = target_info.get("native_ligand_resname")
        flexible_residues = target_info.get("flexible_residues", [])

        # Create output target visualizations directory
        target_vis_dir = os.path.join(args.output_dir, target_name)
        os.makedirs(target_vis_dir, exist_ok=True)

        # Scan for docked poses
        target_dock_folder = os.path.join(args.dock_dir, target_name)
        if not os.path.exists(target_dock_folder):
            print(f"  [INFO] No docked poses folder found at: {target_dock_folder}. Skipping target.")
            continue

        docked_files = glob.glob(os.path.join(target_dock_folder, "*_docked.pdbqt")) + \
                       glob.glob(os.path.join(target_dock_folder, "*_docked.pdb"))

        if not docked_files:
            print(f"  [INFO] No docking poses parsed in: {target_dock_folder}. Skipping.")
            continue

        print(f"  Found {len(docked_files)} docked conformations to visualize.")

        for dock_path in docked_files:
            pose_stem = Path(dock_path).stem
            # Clean compound prefix stem
            compound_name = pose_stem.replace(f"_{target_name}_docked", "")

            # Honor the explicit ligand selection (skip unselected compounds entirely)
            if selected_ligands and compound_name not in selected_ligands:
                continue

            total_runs += 1
            print(f"  -> Rendering compound conformation: {compound_name}")

            output_cxs = os.path.join(target_vis_dir, f"{compound_name}_ligand_only.cxs")
            output_png = os.path.join(target_vis_dir, f"{compound_name}_ligand_only.png")
            output_cxc_universal = os.path.join(target_vis_dir, f"{compound_name}_load.cxc")

            # Check if poses are empty/corrupted
            if os.path.getsize(dock_path) == 0:
                print(f"     [WARNING] Empty pose file encountered at: {dock_path}. Skipping.")
                continue

            # Create a clean single-model ligand-only variant of the docked pose to prevent overlaps
            single_pose_raw = os.path.join(target_vis_dir, f"{compound_name}_temp.pdbqt")
            if not extract_ligand_only(dock_path, single_pose_raw):
                single_pose_raw = dock_path

            # Establish permanent clean ligand paths
            output_ligand_pdb = os.path.join(target_vis_dir, f"{compound_name}_ligand_only.pdb")
            output_ligand_pdbqt = os.path.join(target_vis_dir, f"{compound_name}_ligand_only.pdbqt")

            # Try to convert to clean PDB first using OpenBabel
            active_pose_path = convert_pdbqt_to_pdb(single_pose_raw, output_ligand_pdb)
            
            # If conversion failed or bypassed, save the clean ligand as .pdbqt permanently
            if active_pose_path == single_pose_raw:
                shutil.copy2(single_pose_raw, output_ligand_pdbqt)
                active_pose_path = output_ligand_pdbqt

            # Prepare platform-independent relative paths for Windows/Linux GUI loading
            receptor_rel = f"../../../proteins/protonated/{target_name}_chain{target_info['chain']}_protonated.pdb"
            if not os.path.exists(os.path.join(target_vis_dir, receptor_rel)):
                receptor_rel = f"../../../proteins/protonated/{target_name}.pdb"
                
            ligand_rel = f"./{compound_name}_ligand_only.pdb" if active_pose_path == output_ligand_pdb else f"./{compound_name}_ligand_only.pdbqt"
            raw_crystal_rel = f"../../../proteins/raw/{target_info['pdb_id']}.pdb" if os.path.exists(raw_crystal_pdb) else None

            # Generate the permanent, universal .cxc script (immune to version mismatches)
            generate_universal_cxc(
                receptor_pdb_rel=receptor_rel,
                docked_pose_rel=ligand_rel,
                output_cxc_path=output_cxc_universal,
                raw_crystal_rel=raw_crystal_rel,
                native_resname=native_resname
            )

            # Generate headless CXC script for offscreen rendering
            cxc_headless_script = generate_headless_cxc(
                receptor_pdb=receptor_pdb,
                docked_pose_path=active_pose_path,
                output_png=output_png,
                output_cxs=output_cxs
            )

            # Run ChimeraX headlessly inside native OSMesa offscreen mode
            try:
                clean_env = sanitize_environment()
                chimerax_cmd = [chimerax_bin, "--offscreen", "--exit", cxc_headless_script]

                result = subprocess.run(
                    chimerax_cmd,
                    capture_output=True,
                    text=True,
                    env=clean_env,
                    check=True
                )
                if os.path.exists(output_png) and os.path.getsize(output_png) > 0:
                    success_runs += 1
                    print(f"     [RENDER SUCCESS] Session and high-res image compiled successfully.")
                else:
                    print(f"     [RENDER ERROR] ChimeraX processed successfully, but expected output was missing.")
                    if result.stdout:
                        print(f"     [ChimeraX STDOUT]:\n{result.stdout.strip()}")
                    if result.stderr:
                        print(f"     [ChimeraX STDERR]:\n{result.stderr.strip()}")
            except subprocess.CalledProcessError as err:
                print(f"     [RENDER ERROR] ChimeraX visualization script failed on call.")
                if err.stdout:
                    print(f"     [ChimeraX STDOUT]:\n{err.stdout.strip()}")
                if err.stderr:
                    print(f"     [ChimeraX STDERR]:\n{err.stderr.strip()}")
            finally:
                # Clean up temporary script
                if os.path.exists(cxc_headless_script):
                    os.remove(cxc_headless_script)
                # Clean up temporary on-the-fly single pose files
                if single_pose_raw != dock_path and "temp" in single_pose_raw and os.path.exists(single_pose_raw):
                    os.remove(single_pose_raw)

    print(f"\n{'='*70}\n  [VISUALIZATION SUCCESS] Processed {success_runs}/{total_runs} visualization outputs.\n{'='*70}\n")

if __name__ == "__main__":
    args = setup_arguments()
    main()