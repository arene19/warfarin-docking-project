# analysis.py
"""
Comprehensive analysis of docking results:
- Binding affinity comparison
- Protein-Ligand Interaction Fingerprints (PLIF)
- ADMET property prediction
- Statistical ranking
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import warnings
warnings.filterwarnings("ignore")

# analysis.py

def set_publication_style():
    """
    Sets global matplotlib runtime parameters to meet formatting criteria 
    for academic publishers (Elsevier, ACS, Springer).
    """
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
        'figure.dpi': 300,          # Clear vector graphics standard resolution
        'savefig.bbox': 'tight',    # Prevents cropping out labels or legends
        'savefig.format': 'png'
    })
    sns.set_style("ticks")          # Clean white backdrop with standard axis ticks

# -------------------------------------------------------
# Binding Free Energy Analysis
# -------------------------------------------------------

def binding_free_energy_analysis(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived thermodynamic quantities from docking scores.
    
    dG (kcal/mol) -> Ki (inhibition constant) at 298K
    dG = RT * ln(Ki)  =>  Ki = exp(dG / RT)
    """
    R = 1.987e-3   # kcal / (mol * K)
    T = 298.0      # Kelvin (room temperature)
    
    df = results_df.copy()
    
    # Compute Ki from binding affinity
    df["Ki_nM"] = df["best_affinity"].apply(
        lambda dg: np.exp(dg / (R * T)) * 1e9  # convert M to nM
    )
    
    # Ligand Efficiency: dG / heavy atom count
    # Note: requires SMILES or MW data
    df["Rank"] = df["best_affinity"].rank(ascending=True).astype(int)
    
    return df.sort_values("best_affinity")

def compute_admet_profile(smiles_dict: dict) -> pd.DataFrame:
    records = []
    for name, value in smiles_dict.items():
        # Skip if the value is a path
        if value.startswith("/") or value.endswith(".pdbqt"):
            continue

        mol = Chem.MolFromSmiles(value)
        if mol is None:
            continue
        
        # ... rest of the existing descriptor logic ...
        
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = Descriptors.NumHDonors(mol)
        hba  = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rb   = Descriptors.NumRotatableBonds(mol)
        
        # QED: Quantitative Estimate of Drug-likeness (0-1)
        qed_score = QED.qed(mol)
        
        # Blood-Brain Barrier penetration (simplified model)
        bbb = "Yes" if (mw < 450 and logp < 3 and tpsa < 90) else "No"
        
        # GI absorption
        gi_absorb = "High" if tpsa < 140 and rb < 10 else "Low"
        
        # Egan's egg model for passive absorption
        egan_pass = (logp <= 5.88 and tpsa <= 131.6)
        
        records.append({
            "Name": name,
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
            "Lipinski": (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
        })
    
    return pd.DataFrame(records)


# -------------------------------------------------------
# Protein-Ligand Interaction Fingerprints
# -------------------------------------------------------

def compute_plif(topology_file: str, trajectory_file: str,
                 ligand_selection: str = "resname LIG") -> object:
    """
    Compute protein-ligand interaction fingerprints using ProLIF.
    Works on docked poses or MD trajectories.
    """
    try:
        import prolif as plf
        import MDAnalysis as mda
        
        u = mda.Universe(topology_file, trajectory_file)
        
        ligand   = u.select_atoms(ligand_selection)
        protein  = u.select_atoms("protein")
        
        # Compute fingerprints
        fp = plf.Fingerprint(
            interactions=[
                "HBDonor", "HBAcceptor",
                "Hydrophobic", "PiStacking",
                "CationPi", "Anionic", "Cationic",
                "XBDonor", "XBAcceptor"
            ]
        )
        
        fp.run(u.trajectory, ligand, protein)
        
        return fp
        
    except ImportError:
        print("[WARNING] ProLIF/MDAnalysis not available for PLIF computation.")
        return None


def parse_interactions_from_pdbqt(docked_pdbqt: str, 
                                   receptor_pdb: str) -> dict:
    """
    Parse key interactions (H-bonds, hydrophobic) from docked pose.
    Simple distance-based approach without ProLIF.
    """
    from Bio.PDB import PDBParser
    
    # Load docked ligand atoms
    ligand_atoms = []
    with open(docked_pdbqt) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_type = line[77:79].strip()
                ligand_atoms.append((x, y, z, atom_type))
    
    if not ligand_atoms:
        return {}
    
    # Load receptor
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", receptor_pdb)
    
    interactions = {
        "hydrogen_bonds": [],
        "hydrophobic": [],
        "all_contacts": []
    }
    
    # H-bond donors/acceptors (N, O atoms)
    hbond_types = {"N", "O"}
    # Hydrophobic atoms (C atoms in nonpolar context)
    hydrophobic_types = {"C"}
    
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                res_num  = residue.get_id()[1]
                
                for atom in residue:
                    ax, ay, az = atom.get_coord()
                    element = atom.element if atom.element else atom.get_name()[0]
                    
                    for lx, ly, lz, lat in ligand_atoms:
                        dist = np.sqrt(
                            (ax - lx)**2 + (ay - ly)**2 + (az - lz)**2
                        )
                        
                        # H-bond: donor-acceptor distance < 3.5 A
                        if dist < 3.5 and element in hbond_types and lat in hbond_types:
                            interactions["hydrogen_bonds"].append({
                                "residue": f"{res_name}{res_num}",
                                "atom": atom.get_name(),
                                "distance": round(dist, 2)
                            })
                        
                        # Hydrophobic: C-C contact < 4.5 A
                        if dist < 4.5 and element in hydrophobic_types and lat in hydrophobic_types:
                            interactions["hydrophobic"].append({
                                "residue": f"{res_name}{res_num}",
                                "distance": round(dist, 2)
                            })
                        
                        # Any contact < 5 A
                        if dist < 5.0:
                            interactions["all_contacts"].append({
                                "residue": f"{res_name}{res_num}",
                                "distance": round(dist, 2)
                            })
    
    # Deduplicate
    for key in interactions:
        seen = set()
        unique = []
        for item in interactions[key]:
            sig = item["residue"]
            if sig not in seen:
                seen.add(sig)
                unique.append(item)
        interactions[key] = unique
    
    return interactions


# -------------------------------------------------------
# Visualization & Reporting
# -------------------------------------------------------

def plot_binding_affinities(results_df: pd.DataFrame, target_name: str = "Target Target", output_path: str = "figures/affinities.png"):
    """Generates crisp standard bar plots suited for thesis publication entries."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df = results_df.sort_values("best_affinity")
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Clean academic palette (single tone or muted gradient sequence)
    colors = sns.color_palette("viridis", len(df))
    
    bars = ax.barh(
        df["ligand_name"], 
        df["best_affinity"].abs(), 
        color=colors, 
        edgecolor="black", 
        linewidth=0.8, 
        height=0.6
    )
    
    # Precise textual placement indicators next to bars
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
        
    ax.set_xlabel(r"Binding Affinity |$\Delta$G| (kcal/mol)", fontweight="medium")
    ax.set_ylabel("Evaluated Compounds", fontweight="medium")
    ax.set_title(f"Target Selection Affinity Profiling: {target_name}", fontweight="bold", pad=12)
    
    # Modern clean layout structure stripping outer grid clutter
    sns.despine(ax=ax, top=True, right=True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="grey")
    ax.set_axisbelow(True)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [FIGURE SAVED] Academic format graphic produced -> {output_path}")

def plot_admet_radar(admet_df: pd.DataFrame, 
                     ligand_names: list = None,
                     output_path: str = "figures/admet_radar.png"):
    """
    Radar/spider chart comparing ADMET properties of derivatives.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    if ligand_names:
        df = admet_df[admet_df["Name"].isin(ligand_names)].copy()
    else:
        df = admet_df.copy()
    
    # Normalize key properties 0-1
    props = ["MW", "LogP", "HBD", "HBA", "TPSA", "QED"]
    norm_df = df[props].copy()
    
    for col in props:
        col_max = norm_df[col].max()
        col_min = norm_df[col].min()
        if col_max != col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 0.5
    
    # Setup radar
    N = len(props)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")
    
    colors = plt.cm.cool(np.linspace(0, 1, len(df)))
    
    for idx, (_, row) in enumerate(df.iterrows()):
        values = norm_df.iloc[idx].tolist()
        values += values[:1]
        
        ax.plot(angles, values, color=colors[idx], linewidth=2, 
                label=row["Name"])
        ax.fill(angles, values, color=colors[idx], alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(props, color="#E6EDF3", fontsize=11)
    ax.set_yticklabels([])
    ax.grid(color="#30363D", linewidth=0.5)
    ax.spines["polar"].set_color("#30363D")
    
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        facecolor="#21262D", edgecolor="#30363D",
        labelcolor="#E6EDF3", fontsize=9
    )
    ax.set_title("ADMET Property Comparison (Normalized)", 
                 color="#E6EDF3", fontsize=13, fontweight="bold", pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0D1117")
    plt.close()
    print(f"ADMET radar saved: {output_path}")


def generate_summary_report(screening_results: dict,
                             admet_df: pd.DataFrame,
                             output_path: str = "results/summary_report.txt"):
    """Generate a text-based docking summary report."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
        
        for i, row in df_sorted.iterrows():
            status = "[BEST]" if i == df_sorted.index[0] else "      "
            lines.append(
                f"  {status} {row['ligand_name']:<25} "
                f"dG = {row['best_affinity']:.2f} kcal/mol"
            )
        
        best = df_sorted.iloc[0]
        lines.append(f"\n  => Best binder: {best['ligand_name']} "
                     f"({best['best_affinity']:.2f} kcal/mol)")
    
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
    print(f"\nReport saved: {output_path}")
    
    return report

