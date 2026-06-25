"""
analysis.py

Comprehensive analysis suite for the docking pipeline. Handles:
- Physicochemical property calculation (ADMET/Lipinski rule of 5 properties)
- Graphical plotting of binding affinities configured for publication quality
- Validation of docked ligand conformations using RMSD calculations
- PLIP structural interaction profiling featuring Excel CSV sanitization
- Dynamic clustermaps/heatmaps of PLIP contacts with automatic structural scaling
- High-resolution Chemical Space Maps (PCA projections) of evaluated compound libraries
"""

import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
import warnings
warnings.filterwarnings("ignore")

# Dynamic dependencies load to avoid execution-blocking failures
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, AllChem
    from rdkit.Chem import rdFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARNING] RDKit module is not installed. ADMET, RMSD, and Chemical Space features will run using basic fallbacks.")

try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("[WARNING] adjustText module is not installed. Scatter plot labels may overlap.")


def run_system_command(cmd: str, **kwargs):
    """
    Runs a command in a sanitized system environment to prevent 
    virtual-environment library pollution (specifically OpenBabel clashes).
    """
    clean_env = os.environ.copy()
    if "LD_LIBRARY_PATH" in clean_env:
        del clean_env["LD_LIBRARY_PATH"]
    if "PYTHONPATH" in clean_env:
        del clean_env["PYTHONPATH"]
    clean_env["PATH"] = "/usr/bin:/usr/local/bin:/bin:" + clean_env.get("PATH", "")
    return subprocess.run(cmd, shell=True, env=clean_env, **kwargs)


def set_publication_style():
    """Applies clean, standard formatting for publication-grade figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 1.2,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'png'
    })
    sns.set_style("ticks")


def coordinate_fallback_rmsd(mol1, mol2) -> float:
    """
    Calculates the RMSD between two molecules by matching heavy atoms of the 
    same element based on spatial proximity, bypassing topological match requirements.
    """
    try:
        conf1 = mol1.GetConformer(0)
        conf2 = mol2.GetConformer(0)
    except Exception:
        return None

    atoms1 = [(atom.GetSymbol(), conf1.GetAtomPosition(atom.GetIdx())) 
              for atom in mol1.GetAtoms() if atom.GetSymbol() != 'H']
    atoms2 = [(atom.GetSymbol(), conf2.GetAtomPosition(atom.GetIdx())) 
              for atom in mol2.GetAtoms() if atom.GetSymbol() != 'H']
              
    if not atoms1 or not atoms2:
        return None
        
    squared_distances = []
    used_indices2 = set()
    
    for symbol1, pos1 in atoms1:
        best_dist_sq = float('inf')
        best_idx2 = -1
        
        for idx2, (symbol2, pos2) in enumerate(atoms2):
            if idx2 in used_indices2:
                continue
            if symbol1 != symbol2:
                continue
                
            dist_sq = (pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx2 = idx2
                
        if best_idx2 != -1:
            used_indices2.add(best_idx2)
            squared_distances.append(best_dist_sq)
            
    if not squared_distances:
        return None
        
    return float(np.sqrt(np.mean(squared_distances)))


def calculate_docking_rmsd(docked_pdbqt_path: str, native_pdb_path: str, resname: str, ref_smiles: str = None) -> float:
    """
    Extracts the native ligand from the crystal structure, converts the first docked pose,
    and calculates the heavy-atom RMSD using a hierarchical fallback scheme.
    """
    temp_native_pdb = "temp_native_ligand.pdb"
    temp_docked_pdb = "temp_docked_pose.pdb"
    
    try:
        # Extract native ligand from the crystal structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("native", native_pdb_path)
        
        class LigandSelect(Select):
            def accept_residue(self, residue):
                return residue.get_resname() == resname
                
        io = PDBIO()
        io.set_structure(structure)
        io.save(temp_native_pdb, LigandSelect())
        
        # Pure-Python conversion of the first model (top pose) of PDBQT to standard PDB
        # Automatically extracts and writes standard single-letter elements to columns 77-78 to prevent RDKit warnings
        with open(docked_pdbqt_path, "r") as f_in, open(temp_docked_pdb, "w") as f_out:
            for line in f_in:
                if line.startswith("ENDMDL"):
                    break
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    element_symbol = line[76:78].strip()
                    if element_symbol:
                        clean_element = element_symbol[0].upper()
                        if clean_element not in ["C", "O", "N", "S", "H", "P", "F", "B"]:
                            clean_element = "C"
                    else:
                        clean_element = "C"
                    pdb_line = f"{line[:66]:<76}{clean_element:>2}\n"
                    f_out.write(pdb_line)
                elif line.startswith("TER") or line.startswith("CONECT"):
                    f_out.write(line)
        
        # Load both structures into RDKit without hydrogens
        native_mol = Chem.MolFromPDBFile(temp_native_pdb, removeHs=True)
        docked_mol = Chem.MolFromPDBFile(temp_docked_pdb, removeHs=True)
        
        if native_mol is None or docked_mol is None:
            return None
            
        # Attempt 1: Standard symmetry-aware BestRMS with SMILES template support if available
        if ref_smiles:
            try:
                template = Chem.MolFromSmiles(ref_smiles)
                if template is not None:
                    template = Chem.RemoveHs(template)
                    native_assigned = Chem.AssignBondOrdersFromTemplate(template, native_mol)
                    docked_assigned = Chem.AssignBondOrdersFromTemplate(template, docked_mol)
                    rmsd = AllChem.GetBestRMS(docked_assigned, native_assigned)
                    return float(rmsd)
            except Exception:
                pass

        # Attempt 2: Direct GetBestRMS (no template assignment)
        try:
            rmsd = AllChem.GetBestRMS(docked_mol, native_mol)
            return float(rmsd)
        except Exception:
            pass

        # Attempt 3: Spatial proximity fallback (completely topology-independent)
        rmsd_fallback = coordinate_fallback_rmsd(docked_mol, native_mol)
        return rmsd_fallback
        
    except Exception as e:
        print(f"  [RMSD WARNING] RMSD calculation failed: {e}")
        return None
        
    finally:
        # Clean up temporary files
        for path in [temp_native_pdb, temp_docked_pdb]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def compute_admet_profile(smiles_dict: dict) -> pd.DataFrame:
    """Computes basic ADMET properties and Lipinski criteria."""
    records = []
    for name, value in smiles_dict.items():
        if value.startswith("/") or value.endswith(".pdbqt"):
            continue

        mol = Chem.MolFromSmiles(value)
        if mol is None:
            continue
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rb = Descriptors.NumRotatableBonds(mol)
        
        qed_score = QED.qed(mol)
        bbb = "Yes" if (mw < 450 and logp < 3 and tpsa < 90) else "No"
        gi_absorb = "High" if tpsa < 140 and rb < 10 else "Low"
        egan_pass = bool(logp <= 5.88 and tpsa <= 131.6)
        
        records.append({
            "Name": name,
            "SMILES": value,
            "MW": round(mw, 2),
            "LogP": round(logp, 3),
            "HBD": hbd,
            "HBA": hba,
            "TPSA": round(tpsa, 2),
            "RotBonds": rb,
            "QED": round(qed_score, 4),
            "BBB_Penetrant": bbb,
            "GI_Absorption": gi_absorb,
            "Egan_Pass": egan_pass,
            "Lipinski": bool(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
        })
    
    return pd.DataFrame(records)


def plot_binding_affinities(results_df: pd.DataFrame, target_name: str = "Target", output_path: str = "figures/affinities.png"):
    """Generates standard horizontal bar plots for publication figures."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df = results_df.sort_values("best_affinity")
    
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("viridis", len(df))
    
    bars = ax.barh(
        df["ligand_name"], 
        df["best_affinity"].abs(), 
        color=colors, 
        edgecolor="black", 
        linewidth=0.8, 
        height=0.6
    )
    
    for bar, val in zip(bars, df["best_affinity"]):
        ax.text(
            bar.get_width() + 0.1, 
            bar.get_y() + bar.get_height() / 2, 
            f"{abs(val):.2f}", 
            va="center", 
            ha="left", 
            fontsize=9, 
            fontweight="semibold"
        )
        
    ax.set_xlabel("Binding Affinity |ΔG| (kcal/mol)", fontweight="medium")
    ax.set_ylabel("Evaluated Compounds", fontweight="medium")
    ax.set_title(f"Target Selection Affinity Profiling: {target_name}", fontweight="bold", pad=12)
    
    sns.despine(ax=ax, top=True, right=True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="grey")
    ax.set_axisbelow(True)
    
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_summary_report(screening_results: dict, admet_df: pd.DataFrame, rmsd_results: dict = None, output_path: str = "results/summary_report.txt", include_ligands=None):
    """Generates and writes a summary report containing screening results, ADMET properties, and RMSD scores.

    If include_ligands is provided (an iterable of ligand names), the report is restricted to those
    ligands only. This does not modify any on-disk data — it only scopes the rendered report.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rmsd_results = rmsd_results or {}
    include_set = set(include_ligands) if include_ligands else None

    if include_set is not None:
        scoped_results = {}
        for target_name, df in screening_results.items():
            if df is not None and not df.empty and "ligand_name" in df.columns:
                scoped_results[target_name] = df[df["ligand_name"].astype(str).isin(include_set)]
            else:
                scoped_results[target_name] = df
        screening_results = scoped_results
        if admet_df is not None and not admet_df.empty and "Name" in admet_df.columns:
            admet_df = admet_df[admet_df["Name"].astype(str).isin(include_set)]
    
    lines = [
        "=" * 70,
        "  WARFARIN DERIVATIVE DOCKING STUDY — SUMMARY REPORT",
        "  Blood Coagulation Target Analysis",
        "=" * 70,
        ""
    ]
    
    for target_name, df in screening_results.items():
        lines.append(f"\n[TARGET] {target_name}")
        lines.append("-" * 50)
        
        df_sorted = df.sort_values("best_affinity")

        for _, row in df_sorted.iterrows():
            ligand_name = row['ligand_name']
            affinity = row['best_affinity']

            # A docking is "failed" when it errored or produced no finite affinity.
            row_success = bool(row.get("success", True)) if "success" in df_sorted.columns else True
            if (not row_success) or (not np.isfinite(affinity)):
                err = str(row.get("error", "") if "error" in df_sorted.columns else "")
                detail = f" ({err[:40]})" if err else ""
                lines.append(f"  [FAIL] {ligand_name:<25} docking failed{detail}")
                continue

            rmsd_key = f"{target_name}_{ligand_name}"
            rmsd_str = f" | RMSD: {rmsd_results[rmsd_key]:.2f} Å" if rmsd_key in rmsd_results else ""

            lines.append(
                f"        {ligand_name:<25} "
                f"dG = {affinity:.2f} kcal/mol{rmsd_str}"
            )

        # Best binder is chosen only among successful (finite-affinity) dockings.
        finite_df = df_sorted[np.isfinite(df_sorted["best_affinity"])] if not df_sorted.empty else df_sorted
        if not finite_df.empty:
            best = finite_df.iloc[0]
            lines.append(f"\n  => Best binder: {best['ligand_name']} "
                         f"({best['best_affinity']:.2f} kcal/mol)")
        elif not df_sorted.empty:
            lines.append("\n  => No successful dockings for this target.")
    
    lines.extend([
        "\n" + "=" * 70,
        "  ADMET PROFILE",
        "=" * 70
    ])
    
    for _, row in admet_df.iterrows():
        lines.append(
            f"\n  {row['Name']:<25} | "
            f"MW={row['MW']:.0f}  LogP={row['LogP']:.2f}  "
            f"QED={row['QED']:.3f}  "
            f"Lipinski={'PASS' if row['Lipinski'] else 'FAIL'}"
        )
    
    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
        
    print(report)
    return report


def sanitize_interaction_field(value) -> str:
    """
    Cleans interaction data to ensure Excel CSV compatibility.
    """
    if value is None:
        return "None"
        
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            cleaned_item = str(item).replace("\n", "").replace("\r", "").strip()
            if cleaned_item:
                cleaned_items.append(cleaned_item)
        return ", ".join(cleaned_items) if cleaned_items else "None"
        
    val_str = str(value).replace("\n", " ").replace("\r", " ").strip()
    val_str = " ".join(val_str.split())
    
    return val_str if val_str else "None"


# --- NEW: PLIP INTEGRATION WITH LOCALIZED IMPORTING ---
def run_plip_analysis(receptor_pdb: str, docked_ligand_pdbqt: str, output_csv: str = "results/interaction_profile.csv") -> dict:
    """
    Uses PLIP to analyze hydrogen bonds, hydrophobic contacts, and pi-stacking 
    interactions between the receptor and the top docked ligand pose.
    Bypasses OpenBabel completely using a pure-Python parser.
    Imports PLIP locally to protect multi-threaded servers from SWIG crashes.
    """
    # --- FIXED: LOCAL IMPORT TO PREVENT THREADING IMPORT CRASHES ---
    try:
        from plip.structure.preparation import PDBComplex
    except ImportError:
        print("  [WARNING] PLIP Python module is not installed. Skipping interaction profiling.")
        return None

    # Reconstruct original ligand name safely
    stem = Path(docked_ligand_pdbqt).stem
    if "_docked" in stem:
        ligand_name = stem.split("_docked")[0]
        for target_suffix in ["_VKORC1_Human", "_VKORC1_Reduced", "_HSA"]:
            if ligand_name.endswith(target_suffix):
                ligand_name = ligand_name[:-len(target_suffix)]
    else:
        ligand_name = stem.split('_')[0]

    target_name = Path(receptor_pdb).stem
    temp_ligand_pdb = f"temp_{ligand_name}_ligand.pdb"
    temp_complex_pdb = f"temp_{ligand_name}_{target_name}_complex.pdb"
    
    try:
        # 1. Convert ONLY the first model (top pose) of PDBQT to standard PDB in pure Python
        # Automatically extracts and writes standard single-letter elements to columns 77-78 to prevent RDKit warnings
        with open(docked_ligand_pdbqt, "r") as f_in, open(temp_ligand_pdb, "w") as f_out:
            for line in f_in:
                if line.startswith("ENDMDL"):
                    break
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    element_symbol = line[76:78].strip()
                    if element_symbol:
                        clean_element = element_symbol[0].upper()
                        if clean_element not in ["C", "O", "N", "S", "H", "P", "F", "B"]:
                            clean_element = "C"
                    else:
                        clean_element = "C"
                    pdb_line = f"{line[:66]:<76}{clean_element:>2}\n"
                    f_out.write(pdb_line)
                elif line.startswith("TER") or line.startswith("CONECT"):
                    f_out.write(line)
        
        # 2. Merge receptor and ligand into a single complex PDB file
        with open(receptor_pdb, 'r') as r_file, open(temp_ligand_pdb, 'r') as l_file, open(temp_complex_pdb, 'w') as out_file:
            for line in r_file:
                if not line.startswith(("END", "CONECT")):
                    out_file.write(line)
            for line in l_file:
                if line.startswith(("ATOM", "HETATM")):
                    out_file.write(line)
                    
        # 3. Load the complex into PLIP
        atoms = PDBComplex()
        atoms.load_pdb(temp_complex_pdb)
        atoms.analyze()
        
        # 4. Parse interactions
        interaction_data = {
            "Receptor": target_name,
            "Ligand": ligand_name,
            "Num_H_Bonds": 0,
            "H_Bond_Residues": "None",
            "Num_Hydrophobic": 0,
            "Hydrophobic_Residues": "None",
            "Num_Pi_Stacking": 0,
            "Pi_Stacking_Residues": "None"
        }
        
        if atoms.interaction_sets:
            ligand_key = list(atoms.interaction_sets.keys())[0]
            interactions = atoms.interaction_sets[ligand_key]
            
            def get_res_info(intxn):
                rname = getattr(intxn, "restype", None) or getattr(intxn, "resname", None) or ""
                rnr = getattr(intxn, "resnr", None) or getattr(intxn, "resnum", None) or ""
                
                # VKORC1 residue offset correction
                if "VKORC1" in target_name and rnr:
                    try:
                        adjusted_rnr = int(rnr) - 137
                        if adjusted_rnr > 0:
                            rnr = str(adjusted_rnr)
                    except (ValueError, TypeError):
                        pass
                
                dist = getattr(intxn, "distance_ad", None) or getattr(intxn, "distance", None)
                res_str = f"{rname}{rnr}".strip()
                if dist is not None:
                    res_str += f"({dist:.2f}Å)"
                return res_str
            
            hbonds = []
            hydrophobics = []
            pistackings = []
            
            for intxn in getattr(interactions, "all_itypes", []):
                class_name = intxn.__class__.__name__.lower()
                if "hbond" in class_name:
                    hbonds.append(intxn)
                elif "hydroph" in class_name:
                    hydrophobics.append(intxn)
                elif "pistack" in class_name:
                    pistackings.append(intxn)
            
            hb_residues = sorted(list(set(get_res_info(hb) for hb in hbonds if get_res_info(hb))))
            interaction_data["Num_H_Bonds"] = len(hbonds)
            interaction_data["H_Bond_Residues"] = sanitize_interaction_field(hb_residues)
            
            hp_residues = sorted(list(set(get_res_info(hp) for hp in hydrophobics if get_res_info(hp))))
            interaction_data["Num_Hydrophobic"] = len(hydrophobics)
            interaction_data["Hydrophobic_Residues"] = sanitize_interaction_field(hp_residues)
            
            pi_residues = sorted(list(set(get_res_info(pi) for pi in pistackings if get_res_info(pi))))
            interaction_data["Num_Pi_Stacking"] = len(pistackings)
            interaction_data["Pi_Stacking_Residues"] = sanitize_interaction_field(pi_residues)

        # 5. Append/update data in output CSV file
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_new = pd.DataFrame([interaction_data])
        if os.path.exists(output_csv):
            df_old = pd.read_csv(output_csv)
            df_old = df_old[~((df_old['Receptor'] == target_name) & (df_old['Ligand'] == ligand_name))]
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
            
        df_all.to_csv(output_csv, index=False)
        print(f"  [PLIP] Logged interaction profile for {ligand_name} -> {output_csv}")
        return interaction_data
        
    except Exception as e:
        print(f"  [PLIP WARNING] PLIP analysis skipped for {ligand_name}: {e}")
        return None
        
    finally:
        # Clean up temporary files
        for path in [temp_ligand_pdb, temp_complex_pdb]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def write_interaction_record(row_data: dict, output_csv: str):
    """Safely appends or updates interaction tracking CSV files."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_new = pd.DataFrame([row_data])
    if not os.path.exists(output_csv):
        df_new.to_csv(output_csv, index=False)
    else:
        df_existing = pd.read_csv(output_csv)
        if not df_existing.empty:
            df_existing = df_existing[~((df_existing["Receptor"] == row_data["Receptor"]) & 
                                         (df_existing["Ligand"] == row_data["Ligand"]))]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_csv, index=False)


# --- NEW: INTERACTION HEATMAP & CHEMICAL SPACE GENERATION FUNCTIONS RESTORED ---
def generate_interaction_heatmap(csv_path: str, output_dir: str = "results/plots/", include_ligands=None):
    """
    Parses the PLIP interaction CSV to dynamically generate publication-ready heatmaps.

    If include_ligands is provided, the heatmap is restricted to those ligands only (the on-disk
    PLIP CSV is left untouched as full history).
    """
    if not os.path.exists(csv_path):
        print(f"  [Heatmap Warning] File not found: {csv_path}. Skipping.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [Heatmap Warning] Error reading {csv_path}: {e}.")
        return

    if df.empty:
        return

    if include_ligands and "Ligand" in df.columns:
        df = df[df["Ligand"].astype(str).isin(set(include_ligands))]
        if df.empty:
            print("  [Heatmap Warning] No interaction rows match the selected ligands. Skipping.")
            return

    os.makedirs(output_dir, exist_ok=True)
    residue_cols = [col for col in df.columns if col.endswith('_Residues')]
    if not residue_cols:
        return

    def parse_residues(cell_value):
        if pd.isna(cell_value): return []
        val_str = str(cell_value).strip()
        if val_str.lower() in ["none", ""] or "c++ abort" in val_str.lower(): return []
        parsed = []
        for token in val_str.split(","):
            token = token.strip()
            if not token or token.lower() == "none": continue
            res_id = token.split("(")[0].strip()
            if res_id: parsed.append(res_id)
        return parsed

    def get_residue_sort_key(res_name: str) -> tuple:
        match = re.search(r'\d+', res_name)
        if match: return (int(match.group()), res_name)
        return (9999, res_name)

    unique_receptors = df['Receptor'].unique()
    
    for receptor in unique_receptors:
        sub_df = df[df['Receptor'] == receptor]
        matrix_data = defaultdict(lambda: defaultdict(int))
        
        all_discovered_ligands = sub_df['Ligand'].unique()
        for ligand in all_discovered_ligands:
            matrix_data[ligand] = defaultdict(int)
        
        for _, row in sub_df.iterrows():
            ligand = row['Ligand']
            for col in residue_cols:
                for res in parse_residues(row[col]):
                    matrix_data[ligand][res] += 1

        matrix_df = pd.DataFrame.from_dict(matrix_data, orient='index').fillna(0).astype(int)
        
        if matrix_df.empty or matrix_df.shape[1] == 0:
            continue

        sorted_cols = sorted(matrix_df.columns, key=get_residue_sort_key)
        matrix_df = matrix_df[sorted_cols]

        n_rows, n_cols = matrix_df.shape

        fig_width = max(8, n_cols * 0.6)
        fig_height = max(5, n_rows * 0.5)
        figsize = (fig_width, fig_height)
        output_path = os.path.join(output_dir, f"{receptor}_interaction_heatmap.png")

        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            sns.heatmap(
                matrix_df,
                cmap="Blues",
                annot=True,
                fmt="d",
                linewidths=1.0,
                linecolor='lightgrey',
                square=True,
                cbar_kws={
                    'shrink': 0.6,
                    'label': 'Total Contacts',
                    'pad': 0.03
                },
                ax=ax
            )
            
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10, fontweight='medium')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, fontweight='medium')
            
            ax.set_ylabel("Ligand Name", fontweight='bold', labelpad=10)
            ax.set_xlabel("Receptor Residues", fontweight='bold', labelpad=10)
            ax.set_title(f"PLIP Interaction Matrix: {receptor}", fontweight="bold", fontsize=14, pad=20)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
            plt.close()
            
            print(f"  [HEATMAP] Successfully generated clean heatmap for {receptor} at: {output_path}")
        except Exception as e:
            print(f"  [HEATMAP] ERROR: Could not generate plot for {receptor}. Reason: {e}")


def generate_chemical_space_map(df: pd.DataFrame, smiles_dict: dict, output_dir: str = "results/plots/"):
    """
    Constructs a 2D Chemical Space representation of the compound library.
    Generates 2048-bit Morgan Fingerprints (Radius 2) via RDKit, followed
    by dimensionality reduction (PCA) color-coded by ΔG.
    
    Features adjusted text to automatically repel clustered labels to prevent overlap.
    """
    if not RDKIT_AVAILABLE:
        print("  [Chemical Space Warning] RDKit is not installed or available. Bypassing map generation.")
        return

    print(">>> Generating Figure 5: Chemical Space (SAR) Map...")
    
    try:
        mf_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    except Exception as e:
        try:
            mf_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        except NameError:
            print("  [Chemical Space Warning] Could not load rdFingerprintGenerator. Please update RDKit.")
            return
    
    fps = []
    valid_ligands = []
    
    for ligand_name, smiles in smiles_dict.items():
        if not isinstance(smiles, str) or smiles.startswith("/") or smiles.endswith(".pdbqt"):
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = mf_gen.GetFingerprintAsNumPy(mol)
            fps.append(fp)
            valid_ligands.append(ligand_name)
        else:
            print(f"  [WARNING] Could not parse SMILES for {ligand_name}. Skipping.")
            
    if len(valid_ligands) < 4:
        print("  [WARNING] Not enough valid ligands for chemical space mapping (Need >= 4). Skipping Figure 5.")
        return

    print("  [SAR] Running PCA dimensionality reduction...")
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  [Chemical Space Error] scikit-learn is missing. Cannot perform PCA. Skipping.")
        return

    X = np.array(fps)
    pca = PCA(n_components=2, random_state=42)
    reduced_coords = pca.fit_transform(X)
    
    plot_df = pd.DataFrame({
        'Ligand': valid_ligands,
        'PC1': reduced_coords[:, 0],
        'PC2': reduced_coords[:, 1]
    })
    
    target_col = "VKORC1_Human" if "VKORC1_Human" in df.columns else "best_affinity"
    
    if target_col in df.columns:
        affinity_df = df[['Ligand', target_col]] if 'Ligand' in df.columns else df[['Compound', target_col]].rename(columns={'Compound': 'Ligand'})
        plot_df = plot_df.merge(affinity_df, on='Ligand', how='left')
    else:
        print("  [WARNING] Binding affinity data not found. Dots will not be color-coded.")
        plot_df[target_col] = 0

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    
    scatter = sns.scatterplot(
        data=plot_df, 
        x='PC1', 
        y='PC2', 
        hue=target_col,
        palette='viridis_r', 
        s=100,
        edgecolor='black',
        linewidth=0.8,
        ax=ax
    )
    
    x_range = plot_df['PC1'].max() - plot_df['PC1'].min()
    y_range = plot_df['PC2'].max() - plot_df['PC2'].min()
    x_range = x_range if x_range > 1e-5 else 1.0
    y_range = y_range if y_range > 1e-5 else 1.0
    
    jitter_x = x_range * 0.015
    jitter_y = y_range * 0.015

    np.random.seed(42)
    texts = []
    for i in range(plot_df.shape[0]):
        dx = np.random.uniform(-jitter_x, jitter_x)
        dy = np.random.uniform(-jitter_y, jitter_y)
        texts.append(
            ax.text(
                plot_df['PC1'].iloc[i] + dx, 
                plot_df['PC2'].iloc[i] + dy, 
                plot_df['Ligand'].iloc[i], 
                size=9, 
                color='black', 
                weight='medium'
            )
        )

    if ADJUST_TEXT_AVAILABLE:
        try:
            adjust_text(
                texts, 
                x=plot_df['PC1'].values,
                y=plot_df['PC2'].values,
                expand=(1.5, 1.8),     
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6)
            )
        except TypeError:
            adjust_text(
                texts,
                expand_points=(2.0, 2.0),
                force_text=(0.8, 1.2),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6)
            )

    ax.set_title("Chemical Space Map (PCA of Morgan Fingerprints)", pad=15, fontweight="bold")
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontweight="semibold")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight="semibold")
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="ΔG (kcal/mol)", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "publication_chemical_space.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"  [FIGURE SAVED] Saved adjusted Figure 5 to '{out_path}'")