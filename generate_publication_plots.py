"""
generate_publication_plots.py
======================================================================
Automated molecular docking data visualization engine.
Generates up to five publication-quality figures (Heatmap, 
Stereoselectivity, ADMET table, Residue-level Fingerprint, and SAR Map) 
matching Nature/Science minimalist styles.
======================================================================
"""

import os
import re
import time
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import your updated SAR mapping function
from analysis import generate_chemical_space_map

# ======================================================================
# GLOBAL STYLE CONFIGURATION (Nature/Science Publication Standard)
# ======================================================================
def set_academic_style():
    """Sets standard matplotlib parameters for academic publishing."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'png'
    })
    sns.set_theme(style="ticks")

# ======================================================================
# DATA INGESTION & ROBUST FALLBACK ENGINE
# ======================================================================
def load_docking_results() -> pd.DataFrame:
    """
    Dynamically loads and validates combined docking results. If the file is 
    missing or malformed, it automatically merges individual target files.
    """
    required_cols = ["VKORC1_Human", "VKORC1_Reduced"]
    paths = ["results/docking_results.csv", "docking_results.csv", "results/screening_results.csv"]
    
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                # Verify that the file actually contains the target columns
                if any(col in df.columns for col in required_cols):
                    print(f"  [LOAD] Successfully loaded valid combined results from: {p}")
                    return df
            except Exception:
                pass
            
    # Fallback: Merge target screening files dynamically from the project tree
    print("  [REBUILD] Combined docking file missing or malformed. Merging target files...")
    target_mappings = {
        "VKORC1_Human": [
            "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv",
            "results/screening/VKORC1_Human_screening_results.csv"
        ],
        "VKORC1_Reduced": [
            "results/docked_poses/VKORC1_Reduced/VKORC1_Reduced_screening_results.csv",
            "results/screening/VKORC1_Reduced_screening_results.csv"
        ],
        "HSA": [
            "results/docked_poses/HSA/HSA_screening_results.csv",
            "results/screening/HSA_screening_results.csv"
        ]
    }
    
    merged_df = None
    for target_name, file_options in target_mappings.items():
        target_df = None
        for file_path in file_options:
            if os.path.exists(file_path):
                target_df = pd.read_csv(file_path)
                break
        
        if target_df is not None:
            # Clean up columns for merging
            ligand_col = "ligand_name" if "ligand_name" in target_df.columns else "Ligand"
            target_df = target_df[[ligand_col, "best_affinity"]].rename(
                columns={"best_affinity": target_name, ligand_col: "Ligand"}
            )
            if merged_df is None:
                merged_df = target_df
            else:
                merged_df = pd.merge(merged_df, target_df, on="Ligand", how="outer")
                
    if merged_df is not None:
        os.makedirs("results", exist_ok=True)
        merged_df.to_csv("results/docking_results.csv", index=False)
        merged_df.to_csv("docking_results.csv", index=False)
        print("  [SUCCESS] Dynamically compiled and cached 'results/docking_results.csv'.")
        return merged_df
        
    raise FileNotFoundError(
        "CRITICAL ERROR: No docking result files or target-specific screening CSVs detected on disk."
    )

def load_admet_results() -> pd.DataFrame:
    """Loads ADMET profiling data dynamically."""
    paths = [
        "results/admet_profiles.csv", "admet_profiles.csv",
        "results/admet_profile.csv", "admet_profile.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"  [LOAD] Found ADMET results file: {p}")
            return pd.read_csv(p)
    
    print("  [WARNING] No ADMET profile CSV detected. Figure 3 will be skipped.")
    return pd.DataFrame()

def load_interaction_results() -> pd.DataFrame:
    """Loads interaction profiling data dynamically with a graceful fallback."""
    paths = [
        "results/interaction_profile.csv", 
        "interaction_profile_.csv",
        "results/interaction_profile_full16.csv", 
        "results/interaction_profile_full16.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"  [LOAD] Found interaction profile file: {p}")
            return pd.read_csv(p)
            
    # Graceful fallback so the rest of the script continues running
    print("  [WARNING] No interaction profile CSV detected. Figure 4 will be skipped.")
    return pd.DataFrame()

# ======================================================================
# LIGAND SELECTION FILTERING
# ======================================================================
def parse_ligand_selection(raw: str):
    """Parses a comma-separated ligand selection string into a clean set (empty = all)."""
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def filter_by_ligands(df: pd.DataFrame, selected: set, name_cols) -> pd.DataFrame:
    """Restricts a dataframe to the selected ligand names using the first matching name column.
    An empty selection is a no-op (keeps all rows) for backward compatibility."""
    if not selected or df is None or df.empty:
        return df
    for col in name_cols:
        if col in df.columns:
            return df[df[col].astype(str).isin(selected)].copy()
    return df


# ======================================================================
# GRAPHICS ENGINE FUNCTIONS
# ======================================================================

def generate_heatmap_figure(df: pd.DataFrame, out_dir: str, suffix: str):
    """FIGURE 1: Target Affinity Heatmap"""
    if df.empty: return
    print(">>> Generating Figure 1: Target Affinity Heatmap...")
    # Determine columns dynamically
    ligand_col = "Ligand" if "Ligand" in df.columns else "Compound"
    receptor_cols = [col for col in df.columns if col != ligand_col]
    
    # Pivot and sort by human VKORC1 (ascending order represents strongest negative ΔG binding)
    if "VKORC1_Human" in df.columns:
        df_sorted = df.sort_values(by="VKORC1_Human", ascending=True)
    else:
        df_sorted = df
        
    heatmap_data = df_sorted.set_index(ligand_col)[receptor_cols]
    
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Minimalist sequential colormap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="Blues_r",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={'label': 'Binding Affinity (ΔG, kcal/mol)'},
        ax=ax,
        annot_kws={"size": 8, "weight": "medium"}
    )
    
    ax.set_title("Molecular Docking Target Affinity Heatmap", pad=15, fontweight="bold")
    ax.set_xlabel("Biological Targets", fontweight="semibold")
    ax.set_ylabel("Evaluated Compounds", fontweight="semibold")
    plt.xticks(rotation=15, ha="right")
    
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"publication_heatmap_{suffix}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [FIGURE SAVED] Saved Figure 1 to '{out_path}'")

def generate_stereoselectivity_figure(df: pd.DataFrame, out_dir: str, suffix: str):
    """FIGURE 2: Enantiomer Stereoselectivity Analysis"""
    if df.empty: return
    print(">>> Generating Figure 2: Enantiomer Stereoselectivity Analysis...")
    ligand_col = "Ligand" if "Ligand" in df.columns else "Compound"
    
    if "VKORC1_Human" not in df.columns:
        print("  [WARNING] 'VKORC1_Human' target not found. Skipping Stereoselectivity figure.")
        return

    # Identify pairs of enantiomers dynamically
    base_enantiomers = set()
    for name in df[ligand_col].unique():
        if name.endswith("_S") or name.endswith("_R"):
            base_enantiomers.add(name[:-2])
            
    paired_data = []
    for base in sorted(base_enantiomers):
        s_name = f"{base}_S"
        r_name = f"{base}_R"
        if s_name in df[ligand_col].values and r_name in df[ligand_col].values:
            s_val = df.loc[df[ligand_col] == s_name, "VKORC1_Human"].values[0]
            r_val = df.loc[df[ligand_col] == r_name, "VKORC1_Human"].values[0]
            paired_data.append({
                "Compound": base,
                "S Enantiomer": s_val,
                "R Enantiomer": r_val
            })
            
    if not paired_data:
        print("  [WARNING] No paired enantiomers found. Skipping Stereoselectivity figure.")
        return
        
    pair_df = pd.DataFrame(paired_data)
    
    # Plotting setup
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pair_df))
    width = 0.35
    
    # Muted nature/academic palette
    rects1 = ax.bar(x - width/2, pair_df["S Enantiomer"], width, label="S Enantiomer", color="#3F5D7D", edgecolor="black", linewidth=0.5)
    rects2 = ax.bar(x + width/2, pair_df["R Enantiomer"], width, label="R Enantiomer", color="#A5B8C4", edgecolor="black", linewidth=0.5)
    
    ax.set_ylabel("Binding Affinity (ΔG, kcal/mol)", fontweight="semibold")
    ax.set_title("Stereoselective Binding Analysis (VKORC1_Human)", pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_df["Compound"], rotation=15, ha="right")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    
    # Nature style: Despine top and right axes
    sns.despine(ax=ax, top=True, right=True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"publication_stereoselectivity_{suffix}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [FIGURE SAVED] Saved Figure 2 to '{out_path}'")

def generate_admet_table_figure(df: pd.DataFrame, out_dir: str, suffix: str):
    """FIGURE 3: Clean ADMET Data Table"""
    if df.empty: return
    print(">>> Generating Figure 3: ADMET Data Table Graphic...")
    # Dynamic column mapping
    name_col = "Name" if "Name" in df.columns else ("Compound" if "Compound" in df.columns else df.columns[0])
    lipinski_col = "Lipinski" if "Lipinski" in df.columns else "Lipinski_Pass"
    
    if lipinski_col not in df.columns or "MW" not in df.columns:
        print("  [WARNING] Required ADMET columns missing. Skipping Figure 3.")
        return
        
    table_data = df[[name_col, "MW", "LogP", "QED", lipinski_col]].copy()
    table_data.columns = ["Compound", "MW (g/mol)", "LogP", "QED Score", "Lipinski Pass"]
    
    # Sort for consistent display
    table_data = table_data.sort_values(by="Compound").reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(8.5, len(table_data) * 0.35 + 1.0))
    ax.axis("off")
    ax.axis("tight")
    
    # Styled table plotting
    tbl = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc="center",
        cellLoc="center"
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)  # Scale cell heights for legibility
    
    # Apply elegant structural biology style (Zebra striping and header coloring)
    for i, key in enumerate(tbl.get_celld().keys()):
        cell = tbl.get_celld()[key]
        row, col = key
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#D3D3D3")
        
        if row == 0:
            # Header Row Styling
            cell.set_facecolor("#3F5D7D")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            # Alternating Zebra striping
            if row % 2 == 0:
                cell.set_facecolor("#F4F6F8")
            else:
                cell.set_facecolor("white")
                
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"publication_admet_table_{suffix}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [FIGURE SAVED] Saved Figure 3 to '{out_path}'")

def generate_fingerprint_figure(df: pd.DataFrame, out_dir: str, suffix: str):
    """FIGURE 4: Residue-Level Interaction Fingerprint"""
    if df.empty: return
    print(">>> Generating Figure 4: Residue-Level Interaction Fingerprint...")
    
    # Filter for the target receptor
    target_receptor = "VKORC1_Human_chainA_protonated"
    if "Receptor" not in df.columns:
        print("  [WARNING] Could not find 'Receptor' column. Skipping Figure 4.")
        return

    df_filtered = df[df["Receptor"] == target_receptor]
    
    if df_filtered.empty:
        # Fallback check for receptor without "_chainA_protonated" suffix
        target_receptor = "VKORC1_Human"
        df_filtered = df[df["Receptor"] == target_receptor]
        if df_filtered.empty:
            print(f"  [WARNING] Target receptor '{target_receptor}' not found in interaction profile. Skipping Figure 4.")
            return
        
    def parse_residues(res_str):
        """Extracts residue indices cleanly ignoring distance tags."""
        if pd.isna(res_str) or not isinstance(res_str, str):
            return {}
        # Extracts amino-acid name + residue number using regex (e.g., ASN80)
        matches = re.findall(r"([A-Z]{3}\d+)", res_str, re.IGNORECASE)
        counts = {}
        for m in matches:
            counts[m.upper()] = counts.get(m.upper(), 0) + 1
        return counts

    # Parse and construct the interaction dataset
    residue_data = []
    for _, row in df_filtered.iterrows():
        ligand = row["Ligand"]
        hb_counts = parse_residues(row.get("H_Bond_Residues", ""))
        hp_counts = parse_residues(row.get("Hydrophobic_Residues", ""))
        
        # Merge residues dynamically
        all_res = set(list(hb_counts.keys()) + list(hp_counts.keys()))
        for r in all_res:
            residue_data.append({
                "Ligand": ligand,
                "Residue": r,
                "Hydrogen Bonds": hb_counts.get(r, 0),
                "Hydrophobic Contacts": hp_counts.get(r, 0)
            })
            
    if not residue_data:
        print("  [WARNING] No residue interactions extracted. Skipping Figure 4.")
        return
        
    rf_df = pd.DataFrame(residue_data)
    
    # Find the critical residues based on highest overall contact frequency
    rf_df["Total_Contacts"] = rf_df["Hydrogen Bonds"] + rf_df["Hydrophobic Contacts"]
    top_residues = rf_df.groupby("Residue")["Total_Contacts"].sum().sort_values(ascending=False).index.tolist()
    
    # Restructure data to show interactions stacked/grouped per key residue
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Group by residue first, then sum the total H-bonds vs Hydrophobic contacts across all ligands
    summary_df = rf_df.groupby("Residue")[["Hydrogen Bonds", "Hydrophobic Contacts"]].sum().loc[top_residues]
    
    x = np.arange(len(summary_df))
    width = 0.6
    
    ax.bar(x, summary_df["Hydrogen Bonds"], width, label="Hydrogen Bonds", color="#4F81BD", edgecolor="black", linewidth=0.5)
    ax.bar(x, summary_df["Hydrophobic Contacts"], width, bottom=summary_df["Hydrogen Bonds"], label="Hydrophobic Contacts", color="#D99694", edgecolor="black", linewidth=0.5)
    
    ax.set_ylabel("Total Interaction Count", fontweight="semibold")
    ax.set_xlabel("Active-Site Residues", fontweight="semibold")
    ax.set_title("Residue-Specific Interaction Fingerprint (VKORC1_Human)", pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df.index, rotation=45)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    
    sns.despine(ax=ax, top=True, right=True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"publication_residue_interactions_{suffix}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [FIGURE SAVED] Saved Figure 4 to '{out_path}'")

# ======================================================================
# MAIN EXECUTION ROUTINE
# ======================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate publication-quality docking figures.")
    parser.add_argument(
        "--ligands",
        type=str,
        default="",
        help="Comma-separated ligand names to include. Empty = all ligands found on disk."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the pipeline config file (used for the SAR map and run suffix)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    selected_ligands = parse_ligand_selection(args.ligands)

    set_academic_style()
    print(f"\n{'='*70}\n  LAUNCHING VISUALIZATION ENGINE\n{'='*70}\n")
    if selected_ligands:
        print(f"  [SELECTION] Restricting figures to {len(selected_ligands)} ligand(s): "
              f"{', '.join(sorted(selected_ligands))}")
    else:
        print("  [SELECTION] No ligand filter supplied — including all available ligands.")
    
    # 1. Prepare Target Directory
    target_dir = "Publication Figures"
    os.makedirs(target_dir, exist_ok=True)
    
    # 2. Extract configuration dictionary and run ID suffix
    exh_val = "Unknown"
    config_dict = {}
    try:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
            exh_val = config_dict.get("docking_params", {}).get("exhaustiveness", "Unknown")
    except Exception as e:
        print(f"  [WARNING] Could not load {args.config} fully: {e}")
    
    # Format: exh32_May30_14h30m
    timestamp = time.strftime("%b%d_%Hh%Mm")
    file_suffix = f"exh{exh_val}_{timestamp}"
    
    print(f"  [CONFIG] Using output classifier suffix: '{file_suffix}'")

    try:
        # Load and validate datasets dynamically
        docking_df = load_docking_results()
        admet_df = load_admet_results()
        interaction_df = load_interaction_results()

        # Restrict every dataset to the selected ligands (no-op when selection is empty)
        docking_df = filter_by_ligands(docking_df, selected_ligands, ["Ligand", "Compound"])
        admet_df = filter_by_ligands(admet_df, selected_ligands, ["Name", "Compound"])
        interaction_df = filter_by_ligands(interaction_df, selected_ligands, ["Ligand", "Compound"])
        
        # Execute plotting routines
        generate_heatmap_figure(docking_df, target_dir, file_suffix)
        generate_stereoselectivity_figure(docking_df, target_dir, file_suffix)
        generate_admet_table_figure(admet_df, target_dir, file_suffix)
        generate_fingerprint_figure(interaction_df, target_dir, file_suffix)
        
        # Execute the new Chemical Space Map
        ligands_dict = config_dict.get("ligands", {})
        if selected_ligands and ligands_dict:
            ligands_dict = {k: v for k, v in ligands_dict.items() if k in selected_ligands}
        if ligands_dict:
            # Call the function from analysis.py
            generate_chemical_space_map(docking_df, ligands_dict, target_dir)
        else:
            print("  [WARNING] No ligands found in config.yaml. Skipping Figure 5 (SAR Map).")
        
        print(f"\n{'='*70}\n  [SUCCESS] All figures rendered successfully and saved to '{target_dir}/'.\n{'='*70}\n")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Visualizer execution failed: {e}\n")

if __name__ == "__main__":
    main()