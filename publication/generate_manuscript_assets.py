#!/usr/bin/env python3
"""
Generate manuscript figures, tables, and supplementary assets from project data.

Usage (from repository root):
    python publication/generate_manuscript_assets.py
    python publication/generate_manuscript_assets.py --skip-gnn-predictions
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "publication" / "output"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
QC_DIR = OUT / "internal_qc"

TASKS = [
    "VKORC1_pXC50",
    "Factor_XIIa_pXC50",
    "Factor_Xa_pXC50",
    "Thrombin_pXC50",
    "CYP2C9_pXC50",
    "HSA_pXC50",
]
TASK_SHORT = ["VKORC1", "Factor XIIa", "Factor Xa", "Thrombin", "CYP2C9", "HSA"]


def first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    if HAS_SEABORN:
        sns.set_theme(style="ticks")


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)


def write_qc_table(df: pd.DataFrame, stem: str, markdown_title: str = "") -> None:
    """Write internal QC tables (reference ligands) outside manuscript table set."""
    df.to_csv(QC_DIR / f"{stem}.csv", index=False)
    md_lines = [markdown_title, ""] if markdown_title else []
    try:
        md_lines.append(df.to_markdown(index=False))
    except ImportError:
        md_lines.append("| " + " | ".join(df.columns) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        for _, row in df.iterrows():
            md_lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    (QC_DIR / f"{stem}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"  [qc] {stem}.csv / .md → internal_qc/")


def save_fig(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close()
    print(f"  [fig] {name}.png / .pdf")


def display_label(name: str) -> str:
    """Convert snake_case identifiers to readable axis/legend labels."""
    overrides = {
        "Unique_scaffolds": "Unique scaffolds",
        "Num_H_Bonds": "H-bonds",
        "Num_Hydrophobic": "Hydrophobic",
        "Num_Pi_Stacking": "Pi stacking",
        "Docking_score": "Docking score",
        "Pred_VKORC1": "REINVENT4 predicted VKORC1",
    }
    if name in overrides:
        return overrides[name]
    return name.replace("_", " ")


def ligand_smiles_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for config_path in (ROOT / "config_master.yaml", ROOT / "config.yaml"):
        if not config_path.exists():
            continue
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for name, entry in (cfg.get("ligands") or {}).items():
            if isinstance(entry, dict):
                smi = entry.get("smiles")
            else:
                smi = entry
            if smi:
                mapping[name] = smi

    ligands_dir = ROOT / "results" / "ligands"
    if ligands_dir.exists():
        try:
            from rdkit import Chem

            for sdf in ligands_dir.glob("*.sdf"):
                name = sdf.stem
                if name in mapping:
                    continue
                mol = Chem.MolFromMolFile(str(sdf), removeHs=False)
                if mol:
                    mapping[name] = Chem.MolToSmiles(mol)
        except ImportError:
            pass
    return mapping


def screening_csv() -> Path:
    p = first_existing(
        [
            ROOT / "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv",
            ROOT / "results/screening/VKORC1_Human_screening_results.csv",
            ROOT / "deposition/package/docking/VKORC1_Human_screening_results.csv",
        ]
    )
    if p is None:
        raise FileNotFoundError(
            "VKORC1 screening results CSV not found under results/ or deposition/package/. "
            "Run flexible docking or scripts/bootstrap_results_from_deposition.py first."
        )
    return p


def interaction_csv() -> Path:
    p = ROOT / "results/interaction_profile.csv"
    if not p.exists():
        raise FileNotFoundError("results/interaction_profile.csv not found.")
    return p


def admet_csv() -> Path:
    p = ROOT / "results/admet_profile.csv"
    if not p.exists():
        raise FileNotFoundError("results/admet_profile.csv not found.")
    return p


def load_train_smiles() -> set[str]:
    split_path = ROOT / "publication/data/gnn_scaffold_split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Frozen scaffold split required: {split_path}")
    with open(split_path, encoding="utf-8") as f:
        meta = json.load(f)
    raw = meta.get("train_smiles", [])
    try:
        from rdkit import Chem

        canon: set[str] = set()
        for smi in raw:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canon.add(Chem.MolToSmiles(mol))
            else:
                canon.add(smi)
        return canon
    except ImportError:
        return set(raw)


def sync_flexible_redock_spotcheck() -> Path:
    """Sync screening affinities into an output copy; canonical QC CSV is not mutated."""
    canonical = ROOT / "publication/data/flexible_redock_spotcheck.csv"
    out_path = OUT / "data" / "flexible_redock_spotcheck.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not canonical.exists():
        print("  [warn] flexible_redock_spotcheck.csv missing; skipping sync.")
        return canonical if canonical.exists() else out_path
    df = pd.read_csv(canonical)
    dock = pd.read_csv(screening_csv())
    aff_map = dock.set_index("ligand_name")["best_affinity"].astype(float).to_dict()
    updated = 0
    drop_rows: list[int] = []
    for idx, row in df.iterrows():
        name = str(row["ligand_name"])
        if not name.startswith("RL_Gen"):
            continue
        if name not in aff_map:
            continue
        new_train = float(aff_map[name])
        redock = float(row["affinity_redock_kcal_mol"])
        new_delta = abs(new_train - redock)
        if new_delta > 1.0:
            print(
                f"  [warn] {name}: screening={new_train:.3f} vs redock={redock:.3f} "
                f"(|Δ|={new_delta:.2f}); excluding from spot-check (stale re-dock)"
            )
            drop_rows.append(idx)
            continue
        old_train = float(row["affinity_training_kcal_mol"])
        if abs(new_train - old_train) > 1e-6:
            updated += 1
        df.at[idx, "affinity_training_kcal_mol"] = new_train
        df.at[idx, "delta_kcal_mol"] = new_delta
    if drop_rows:
        df = df.drop(index=drop_rows)
    df.to_csv(out_path, index=False)
    if updated:
        print(f"  [ok] flexible_redock_spotcheck synced → {out_path.relative_to(ROOT)} ({updated} rows)")
    return out_path


def eval_report() -> Dict[str, Any]:
    pub = ROOT / "publication/data/gnn_evaluation_report.json"
    res = ROOT / "results/gnn_evaluation_report.json"
    if pub.exists() and res.exists():
        pub_body = pub.read_text(encoding="utf-8")
        res_body = res.read_text(encoding="utf-8")
        if pub_body != res_body:
            raise RuntimeError(
                "Conflicting gnn_evaluation_report.json in publication/data/ vs results/. "
                "Re-run gnn_baseline_evaluation.py or remove the stale copy."
            )
    p = pub if pub.exists() else res
    if p is None or not p.exists():
        raise FileNotFoundError(
            "Run `python gnn_baseline_evaluation.py` first to create gnn_evaluation_report.json"
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_table(df: pd.DataFrame, stem: str, markdown_title: str = "") -> None:
    df.to_csv(TAB_DIR / f"{stem}.csv", index=False)
    md_lines = [markdown_title, ""] if markdown_title else []
    try:
        md_lines.append(df.to_markdown(index=False))
    except ImportError:
        md_lines.append("| " + " | ".join(df.columns) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        for _, row in df.iterrows():
            md_lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    (TAB_DIR / f"{stem}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"  [table] {stem}.csv / .md")


def table1_gnn_metrics() -> pd.DataFrame:
    report = eval_report()
    sp = report.get("vkorc1_spearman", {})
    rows = []
    for col, short in zip(TASKS, TASK_SHORT):
        mt = report["multitask_gnn"]["per_task"][col]
        rf = report["morgan_rf"]["per_task"][col]
        st = report.get("vkorc1_single_task_gnn", {}).get("per_task", {}).get(col, {})
        rows.append(
            {
                "Task": short,
                "n_test": mt["n"],
                "GAT_RMSE": round(mt["rmse"], 2),
                "GAT_R2": round(mt["r2"], 2),
                "GAT_Spearman_rho": round(sp.get("multitask_gat", np.nan), 2)
                if col == "VKORC1_pXC50"
                else np.nan,
                "RF_R2": round(rf["r2"], 2) if mt["n"] else np.nan,
                "RF_Spearman_rho": round(sp.get("morgan_rf", np.nan), 2)
                if col == "VKORC1_pXC50"
                else np.nan,
                "VKORC1_only_GAT_R2": round(st.get("r2", np.nan), 2) if st else np.nan,
            }
        )
    g = report["multitask_gnn"]["global"]
    rows.append(
        {
            "Task": "Pooled (all tasks)",
            "n_test": sum(report["multitask_gnn"]["per_task"][c]["n"] for c in TASKS),
            "GAT_RMSE": round(g["rmse"], 2),
            "GAT_R2": round(g["r2"], 2),
            "GAT_Spearman_rho": np.nan,
            "RF_R2": np.nan,
            "RF_Spearman_rho": np.nan,
            "VKORC1_only_GAT_R2": np.nan,
        }
    )
    return pd.DataFrame(rows)


def table_s1_full_metrics() -> pd.DataFrame:
    report = eval_report()
    gat_sp = report.get("per_task_spearman", {}).get("multitask_gat", {})
    rf_sp = report.get("per_task_spearman", {}).get("morgan_rf", {})
    rows = []
    for col, short in zip(TASKS, TASK_SHORT):
        mt = report["multitask_gnn"]["per_task"][col]
        rf = report["morgan_rf"]["per_task"][col]
        st_r2 = np.nan
        if col == "VKORC1_pXC50":
            st = report.get("vkorc1_single_task_gnn", {}).get("per_task", {}).get(col, {})
            st_r2 = st.get("r2", np.nan)
        rows.append(
            {
                "Task": short,
                "n_test": mt["n"],
                "n_train_labels": report["label_counts"]["train"][col],
                "GAT_RMSE": round(mt["rmse"], 3),
                "GAT_MAE": round(mt["mae"], 3),
                "GAT_R2": round(mt["r2"], 3),
                "GAT_Spearman_rho": round(gat_sp.get(col, np.nan), 3),
                "RF_RMSE": round(rf["rmse"], 3),
                "RF_R2": round(rf["r2"], 3),
                "RF_Spearman_rho": round(rf_sp.get(col, rf.get("spearman_rho", np.nan)), 3),
                "VKORC1_only_GAT_R2": round(st_r2, 3) if col == "VKORC1_pXC50" else np.nan,
            }
        )
    g = report["multitask_gnn"]["global"]
    rows.append(
        {
            "Task": "Pooled",
            "n_test": sum(report["multitask_gnn"]["per_task"][c]["n"] for c in TASKS),
            "n_train_labels": np.nan,
            "GAT_RMSE": round(g["rmse"], 3),
            "GAT_MAE": round(g["mae"], 3),
            "GAT_R2": round(g["r2"], 3),
            "GAT_Spearman_rho": np.nan,
            "RF_RMSE": np.nan,
            "RF_R2": np.nan,
            "RF_Spearman_rho": np.nan,
            "VKORC1_only_GAT_R2": np.nan,
        }
    )
    return pd.DataFrame(rows)


def table2_docking_leaderboard(n: int = 15) -> pd.DataFrame:
    df = pd.read_csv(screening_csv())
    df = df[df["ligand_name"].astype(str).str.startswith("RL_Gen")].copy()
    df = df.sort_values("best_affinity").head(n).copy()
    df["pXC50_Vina_derived"] = (-df["best_affinity"] / 1.36).round(3)
    df = df.rename(columns={"ligand_name": "Ligand", "best_affinity": "dG_kcal_mol"})
    return df[["Ligand", "dG_kcal_mol", "pXC50_Vina_derived", "success"]]


MANDATORY_REFS = [
    "S_Warfarin_ref",
    "R_Warfarin_ref",
    "p_nitro_R",
    "p_nitro_S",
    "BENZ_R",
    "BENZ_S",
    "dimethoxy_23_R",
    "dimethoxy_23_S",
    "m_nitro_R",
    "m_nitro_S",
]


def table2_combined_leaderboard(top_n: int = 10) -> pd.DataFrame:
    """Top overall dockers plus mandatory reference ligands (Table 2 main text)."""
    df = pd.read_csv(screening_csv()).copy()
    df = df.sort_values("best_affinity")
    top = df.head(top_n).copy()
    refs = df[df["ligand_name"].isin(MANDATORY_REFS)].copy()
    combined = pd.concat([top, refs]).drop_duplicates(subset="ligand_name")
    combined = combined.sort_values("best_affinity")
    combined["Class"] = np.where(
        combined["ligand_name"].astype(str).str.startswith("RL_Gen"), "RL_Gen", "Reference"
    )
    combined["pXC50_Vina_derived"] = (-combined["best_affinity"] / 1.36).round(3)
    combined = combined.rename(
        columns={"ligand_name": "Ligand", "best_affinity": "dG_kcal_mol"}
    )
    return combined[["Ligand", "Class", "dG_kcal_mol", "pXC50_Vina_derived", "success"]]


RL_ADMET_TOP_N = 6


def _rl_gen_top_ligands(n: int = RL_ADMET_TOP_N) -> list[str]:
    dock = pd.read_csv(screening_csv())
    rl = dock[dock["ligand_name"].astype(str).str.startswith("RL_Gen")].copy()
    rl = rl.sort_values("best_affinity")
    return rl.head(n)["ligand_name"].astype(str).tolist()


def table3_admet_rl_gen_top(n: int = RL_ADMET_TOP_N) -> pd.DataFrame:
    """ADMET + interaction summary for top RL_Gen hits by VKORC1 docking (Table 3)."""
    dock = pd.read_csv(screening_csv()).set_index("ligand_name")
    admet = pd.read_csv(admet_csv()).set_index("Name")
    ip = pd.read_csv(interaction_csv())
    ip_vkor = ip[ip["Receptor"].astype(str).str.contains("VKORC1_Human", na=False)]
    manifest = ROOT / "md_poses/md_pose_manifest.csv"
    hsa_map: Dict[str, float] = {}
    if manifest.exists():
        man = pd.read_csv(manifest)
        rl_man = man[man["ligand_name"].astype(str).str.startswith("RL_Gen")]
        hsa_map = dict(zip(rl_man["ligand_name"], rl_man["HSA_dG_kcal_mol"]))

    rows = []
    for lig in _rl_gen_top_ligands(n):
        row: Dict[str, Any] = {"Ligand": lig}
        if lig in dock.index:
            row["VKOR_dG_kcal_mol"] = round(float(dock.loc[lig, "best_affinity"]), 3)
        if lig in admet.index:
            a = admet.loc[lig]
            row["QED"] = round(float(a["QED"]), 3)
            row["TPSA"] = round(float(a["TPSA"]), 1)
            row["Lipinski"] = a["Lipinski"]
            row["MW"] = round(float(a["MW"]), 1)
        if lig in ip_vkor.set_index("Ligand").index:
            row["VKOR_H_bonds"] = int(ip_vkor[ip_vkor["Ligand"] == lig]["Num_H_Bonds"].iloc[0])
            row["VKOR_hydrophobic"] = int(
                ip_vkor[ip_vkor["Ligand"] == lig]["Num_Hydrophobic"].iloc[0]
            )
        if lig in hsa_map:
            row["HSA_dG_kcal_mol"] = round(float(hsa_map[lig]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def table_s_stereoselectivity() -> pd.DataFrame:
    """ΔΔG (S − R) for reference enantiomer pairs."""
    df = pd.read_csv(screening_csv()).set_index("ligand_name")
    pairs = [
        ("Warfarin", "R_Warfarin_ref", "S_Warfarin_ref"),
        ("p_nitro", "p_nitro_R", "p_nitro_S"),
        ("m_nitro", "m_nitro_R", "m_nitro_S"),
        ("m_bromo", "m_bromo_R", "m_bromo_S"),
        ("BENZ", "BENZ_R", "BENZ_S"),
        ("dimethoxy_23", "dimethoxy_23_R", "dimethoxy_23_S"),
    ]
    rows = []
    for scaffold, r_name, s_name in pairs:
        if r_name not in df.index or s_name not in df.index:
            continue
        dg_r = float(df.loc[r_name, "best_affinity"])
        dg_s = float(df.loc[s_name, "best_affinity"])
        rows.append(
            {
                "Scaffold": scaffold,
                "dG_R_kcal_mol": round(dg_r, 3),
                "dG_S_kcal_mol": round(dg_s, 3),
                "delta_dG_S_minus_R": round(dg_s - dg_r, 3),
                "Preferred_enantiomer": "S" if dg_s < dg_r else "R",
            }
        )
    return pd.DataFrame(rows)


def table_docking_enrichment_summary() -> pd.DataFrame:
    """Parse archived retrospective enrichment metrics from validation/."""
    summary_path = ROOT / "validation/enrichment_summary.txt"
    if not summary_path.exists():
        return pd.DataFrame()
    text = summary_path.read_text(encoding="utf-8")
    rows: list[dict[str, str]] = []
    patterns = [
        ("ROC-AUC", r"ROC-AUC:\s*([\d.]+)"),
        ("EF@1%", r"Enrichment factor @ 1%:\s*([\d.]+)"),
        ("EF@5%", r"Enrichment factor @ 5%:\s*([\d.]+)"),
        ("EF@10%", r"Enrichment factor @ 10%:\s*([\d.]+)"),
        ("Actives", r"Actives:\s*(\d+)"),
        ("Decoys", r"Decoys:\s*(\d+)"),
        ("Successfully docked", r"Successfully docked:\s*(\d+)\s*/\s*(\d+)"),
    ]
    for label, pat in patterns:
        match = re.search(pat, text)
        if not match:
            continue
        value = " / ".join(match.groups()) if len(match.groups()) > 1 else match.group(1)
        rows.append({"Metric": label, "Value": value})
    return pd.DataFrame(rows)


def archive_docking_validation_qc() -> None:
    """Copy docking_validation.py outputs into internal_qc/ for deposition."""
    validation = ROOT / "validation"
    if not validation.exists():
        print("  [skip] docking validation QC — validation/ missing")
        return
    copies = {
        "enrichment_summary.txt": "docking_enrichment_summary.txt",
        "seed_reproducibility_summary.txt": "docking_seed_reproducibility_summary.txt",
        "enrichment_roc.png": "figure_docking_enrichment_roc.png",
        "enrichment_scores.csv": "docking_enrichment_scores.csv",
        "seed_reproducibility.csv": "docking_seed_reproducibility.csv",
    }
    copied = 0
    for src_name, dst_name in copies.items():
        src = validation / src_name
        if not src.exists():
            continue
        dst = QC_DIR / dst_name
        if src_name.endswith(".csv"):
            df = pd.read_csv(src)
            if "docked_pdbqt" in df.columns:
                df["docked_pdbqt"] = df["docked_pdbqt"].apply(
                    lambda x: Path(str(x)).name if pd.notna(x) and str(x).strip() else x
                )
            df.to_csv(dst, index=False)
        else:
            shutil.copy2(src, dst)
        copied += 1
    script_dst = QC_DIR / "run_docking_validation.sh"
    script_dst.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")/../..\"\n"
        "python docking_validation.py\n",
        encoding="utf-8",
    )
    script_dst.chmod(0o755)
    copied += 1
    if copied:
        print(f"  [qc] archived {copied} docking-validation files → internal_qc/")


def archive_receptor_validation_report() -> None:
    """Run receptor_validation.py and save clean JSON to internal_qc/."""
    import subprocess

    cmd = [
        sys.executable,
        str(ROOT / "receptor_validation.py"),
        "--target",
        "VKORC1_Human",
        "--check-pose",
        "S_Warfarin_ref",
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    start = proc.stdout.find("{")
    if start < 0:
        print("  [warn] receptor validation JSON missing; skipping receptor_validation_report.json")
        return
    (QC_DIR / "receptor_validation_report.json").write_text(proc.stdout[start:], encoding="utf-8")
    print("  [qc] receptor_validation_report.json → internal_qc/")


def write_internal_qc_readme() -> None:
    readme = QC_DIR / "README.md"
    readme.write_text(
        "# Internal QC artifacts (not cited in main manuscript)\n\n"
        "| File | Description |\n"
        "|------|-------------|\n"
        "| `table_s_stereoselectivity.csv` | Reference enantiomer ΔΔG (S − R) for six scaffolds "
        "including m_bromo |\n"
        "| `table2_vkorc1_combined_leaderboard.csv` | RL_Gen + reference flexible-docking ranks |\n"
        "| `table_docking_enrichment_summary.csv` | Retrospective VKORC1 enrichment (ROC-AUC, EF) |\n"
        "| `docking_enrichment_summary.txt` | Full enrichment protocol summary |\n"
        "| `figure_docking_enrichment_roc.png` | ROC curve from `docking_validation.py` |\n"
        "| `docking_seed_reproducibility_summary.txt` | Multi-seed reference-ligand reproducibility |\n"
        "| `run_docking_validation.sh` | Rerun `docking_validation.py` (requires full repo + Vina) |\n"
        "| `receptor_validation_report.json` | VKORC1_Human setup checks + S-warfarin re-dock RMSD |\n\n"
        "Regenerate: `python docking_validation.py` then `bash publication/build_all.sh`.\n",
        encoding="utf-8",
    )
    print("  [qc] README.md → internal_qc/")


def table_s2_rl_gen_docking() -> pd.DataFrame:
    df = pd.read_csv(screening_csv())
    rl = df[df["ligand_name"].astype(str).str.startswith("RL_Gen")].copy()
    rl = rl.sort_values("best_affinity")
    rl["pXC50_Vina_derived"] = (-rl["best_affinity"] / 1.36).round(3)
    rl = rl.rename(
        columns={"ligand_name": "Ligand", "best_affinity": "dG_kcal_mol", "runtime_seconds": "runtime_s"}
    )
    return rl[["Ligand", "dG_kcal_mol", "pXC50_Vina_derived", "success", "runtime_s"]]


def table3_selectivity(run_gnn: bool = True) -> pd.DataFrame:
    df = pd.read_csv(screening_csv())
    top = df[df["ligand_name"].astype(str).str.startswith("RL_Gen")].copy()
    top = top.sort_values("best_affinity").head(5)
    names = top["ligand_name"].tolist()
    smap = ligand_smiles_map()

    rows = []
    predictor = None
    if run_gnn:
        try:
            import torch
            from gnn_model import DEFAULT_GNN_CONFIG, load_checkpoint, smiles_to_graph
            from torch_geometric.data import Batch

            ckpt = ROOT / "coagulation_admet_gnn.pth"
            if ckpt.exists():
                predictor, _ = load_checkpoint(ckpt, torch.device("cpu"))
                predictor.eval()
        except Exception as exc:
            print(f"  [warn] GNN predictions skipped: {exc}")

    for _, r in top.iterrows():
        name = r["ligand_name"]
        smi = smap.get(name, "")
        row: Dict[str, Any] = {
            "Ligand": name,
            "dG_kcal_mol": round(float(r["best_affinity"]), 3),
            "SMILES": smi,
        }
        if predictor is not None and smi:
            import torch
            from gnn_model import smiles_to_graph
            from torch_geometric.data import Batch

            g = smiles_to_graph(smi, [0.0] * 6, [0.0] * 6)
            if g is not None:
                with torch.no_grad():
                    pred = predictor(Batch.from_data_list([g])).squeeze().tolist()
                for short, val in zip(TASK_SHORT, pred):
                    row[f"Pred_{short}"] = round(val, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def table_s5_chembl_only_vkorc1() -> pd.DataFrame:
    """Supplementary VKORC1 benchmark excluding RL Vina pseudo-labels from test."""
    path = ROOT / "publication/data/gnn_vkorc1_chembl_only_benchmark.json"
    if path.exists():
        block = json.loads(path.read_text(encoding="utf-8"))
    else:
        report_path = ROOT / "publication/data/gnn_evaluation_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                "Missing gnn_vkorc1_chembl_only_benchmark.json — run gnn_baseline_evaluation.py first."
            )
        block = json.loads(report_path.read_text(encoding="utf-8")).get("vkorc1_chembl_only_test", {})
    rows = []
    for model, key in (("Multi-task GAT", "multitask_gat"), ("Morgan FP + RF", "morgan_rf")):
        stats = block.get(key, {})
        if not stats:
            continue
        rows.append(
            {
                "Model": model,
                "n_test": stats.get("n"),
                "RMSE": round(stats["rmse"], 3) if stats.get("rmse") == stats.get("rmse") else None,
                "MAE": round(stats["mae"], 3) if stats.get("mae") == stats.get("mae") else None,
                "R2": round(stats["r2"], 3) if stats.get("r2") == stats.get("r2") else None,
                "Spearman_rho": round(stats["spearman_rho"], 3)
                if stats.get("spearman_rho") == stats.get("spearman_rho")
                else None,
            }
        )
    if not rows:
        raise ValueError("ChEMBL-only VKORC1 benchmark JSON contains no model metrics.")
    return pd.DataFrame(rows)


def table_s3_reinvent_scoring() -> pd.DataFrame:
    rows = [
        {"Component": "Synthetic Accessibility (SA Score)", "Weight": 1.5, "Role": "Penalize difficult syntheses", "_key": "SAScore"},
        {"Component": "QED", "Weight": 1.0, "Role": "Drug-likeness", "_key": "QED"},
        {"Component": "Molecular Weight", "Weight": 0.5, "Role": "Prefer 150–500 Da", "_key": "MolecularWeight"},
        {"Component": "GNN VKORC1 (ExternalProcess)", "Weight": 2.5, "Role": "Custom gnn_predict.py ranker", "_key": "ExternalProcess"},
    ]
    toml_path = ROOT / "configs" / "coumarin_rl.toml"
    if not toml_path.exists():
        toml_path = ROOT / "REINVENT4" / "coumarin_rl.toml"
    if toml_path.exists():
        import tomllib

        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)
        components = toml_data.get("stage", [{}])[0].get("scoring", {}).get("component", [])
        weight_by_key: Dict[str, float] = {}
        for comp in components:
            for key, block in comp.items():
                if isinstance(block, dict) and "weight" in block:
                    weight_by_key[key] = float(block["weight"])
        for i, row in enumerate(rows):
            w = weight_by_key.get(row["_key"])
            if w is not None:
                rows[i]["Weight"] = w
    return pd.DataFrame([{k: v for k, v in r.items() if k != "_key"} for r in rows])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure1_training_curves() -> None:
    hist_path = first_existing(
        [
            ROOT / "publication/data/gnn_training_history.json",
            ROOT / "results/gnn_training_history.json",
        ]
    )
    if hist_path is None:
        print("  [skip] Figure 1 — run `python dynamic_gnn_pipeline.py` to create training history.")
        return

    with open(hist_path, encoding="utf-8") as f:
        data = json.load(f)
    hist = pd.DataFrame(data["epochs"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(hist["epoch"], hist["train_loss"], label="Train masked MSE", color="#2166ac")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="Val masked MSE", color="#b2182b")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Masked MSE")
    axes[0].set_title("Multi-task GAT training")
    axes[0].legend(frameon=False)

    axes[1].plot(hist["epoch"], hist["val_r2"], color="#4daf4a", label="Val R²")
    ax2 = axes[1].twinx()
    ax2.plot(hist["epoch"], hist["learning_rate"], color="#984ea3", alpha=0.6, label="Learning rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation R²")
    ax2.set_ylabel("Learning rate")
    axes[1].set_title("Validation R² and LR schedule")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right")

    fig.tight_layout()
    save_fig("figure1_gnn_training_curves")


def figure6_interaction_heatmap() -> None:
    try:
        ipath = interaction_csv()
    except FileNotFoundError:
        print("  [skip] Figure 2 — interaction_profile.csv not found.")
        return

    df = pd.read_csv(ipath)
    df = df[df["Receptor"].astype(str).str.contains("VKORC1_Human", na=False)]
    dock = pd.read_csv(screening_csv())
    top_rl = dock[dock["ligand_name"].astype(str).str.startswith("RL_Gen")]
    keep = top_rl.sort_values("best_affinity").head(12)["ligand_name"].tolist()
    keep = [x for x in keep if x in set(df["Ligand"])]
    sub = df[df["Ligand"].isin(keep)].copy()
    if sub.empty:
        print("  [skip] Figure 2 — no overlapping ligands.")
        return

    sub = sub.set_index("Ligand")
    mat = sub[["Num_H_Bonds", "Num_Hydrophobic", "Num_Pi_Stacking"]].astype(float)
    aff = dock.set_index("ligand_name").loc[mat.index, "best_affinity"]
    mat["DeltaG_kcal_mol"] = aff.values

    col_labels = ["H-bonds", "Hydrophobic", "Pi stacking", "ΔG (kcal/mol)"]
    mat.columns = col_labels

    annot = np.empty(mat.shape, dtype=object)
    for i in range(len(mat.index)):
        for j, col in enumerate(mat.columns):
            val = mat.iloc[i, j]
            if col == "ΔG (kcal/mol)":
                annot[i, j] = f"{val:.2f}"
            else:
                annot[i, j] = f"{val:.0f}"

    # Color scale: counts use raw values; ΔG uses |ΔG| so stronger binders read warmer
    plot_mat = mat.copy()
    plot_mat["ΔG (kcal/mol)"] = plot_mat["ΔG (kcal/mol)"].abs()

    plt.figure(figsize=(6.5, max(4, 0.35 * len(mat))))
    if HAS_SEABORN:
        sns.heatmap(
            plot_mat,
            annot=annot,
            fmt="",
            cmap="YlOrRd",
            cbar_kws={"label": "Count / |ΔG| (kcal/mol)"},
        )
    else:
        plt.imshow(plot_mat.values, aspect="auto", cmap="YlOrRd")
        plt.colorbar(label="Count / |ΔG| (kcal/mol)")
        plt.xticks(range(len(plot_mat.columns)), plot_mat.columns, rotation=45, ha="right")
        plt.yticks(range(len(plot_mat.index)), plot_mat.index)
        for i in range(len(plot_mat.index)):
            for j in range(len(plot_mat.columns)):
                plt.text(j, i, annot[i, j], ha="center", va="center", fontsize=8)
    plt.title("VKORC1 interaction fingerprint (top RL_Gen leads)")
    plt.xlabel("Interaction type")
    plt.ylabel("Ligand")
    save_fig("figure6_interaction_heatmap")


def figure7_reinvent_distribution() -> None:
    gen_path = first_existing(
        [
            ROOT / "REINVENT4/coumarin_generation_1.csv",
            ROOT / "coumarin_generation_1.csv",
            ROOT / "deposition/package/reinvent/coumarin_generation_1.csv",
        ]
    )
    master_path = ROOT / "data/coagulation_admet_multi_task.csv"
    if gen_path is None or not master_path.exists():
        print("  [skip] Figure 7 — REINVENT CSV or master dataset missing.")
        return

    gen = pd.read_csv(gen_path)
    raw_col = "Pred_VKORC1_pXC50 (raw)"
    if raw_col not in gen.columns:
        print("  [skip] Figure 7 — Pred_VKORC1_pXC50 column missing in REINVENT output.")
        return
    gen_scores = gen[raw_col].dropna()
    gen_scores = gen_scores[(gen_scores > 3) & (gen_scores < 12)]

    master = pd.read_csv(master_path)
    train_smiles = load_train_smiles()
    train_rows = master[master["canonical_smiles"].isin(train_smiles)]
    train_scores = train_rows["VKORC1_pXC50"].dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    train_label = f"Train-split VKORC1 labels (n={len(train_scores)})"
    gen_label = f"REINVENT4 predicted VKORC1 (n={len(gen_scores)})"
    if HAS_SEABORN:
        sns.kdeplot(
            train_scores, ax=ax, label=train_label, fill=True, alpha=0.35, color="#2166ac", linewidth=1.5
        )
        sns.kdeplot(
            gen_scores, ax=ax, label=gen_label, fill=True, alpha=0.35, color="#ff7f00", linewidth=1.5
        )
    else:
        ax.hist(train_scores, bins=25, density=True, alpha=0.45, label=train_label, color="#2166ac")
        ax.hist(gen_scores, bins=25, density=True, alpha=0.45, label=gen_label, color="#ff7f00")
    if len(train_scores) > 0:
        median = float(train_scores.median())
        p90 = float(train_scores.quantile(0.9))
        ax.axvline(median, color="#2166ac", linestyle="--", linewidth=1.2, label=f"Train median ({median:.2f})")
        ax.axvline(p90, color="#2166ac", linestyle=":", linewidth=1.2, label=f"Train 90th pct ({p90:.2f})")
    ax.set_xlabel("VKORC1 pXC50 (experimental / predicted)")
    ax.set_ylabel("Density")
    ax.set_title("REINVENT generation vs. train-split VKORC1 labels")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=True,
        facecolor="white",
        edgecolor="0.75",
        fontsize=8,
    )
    fig.subplots_adjust(right=0.72)
    save_fig("figure7_reinvent_vs_training_distribution")


def figure3_scaffold_split() -> None:
    try:
        from gnn_model import generate_scaffold, load_multitask_data, resolve_scaffold_split
    except ImportError:
        print("  [skip] Figure 3 — torch/torch_geometric not available.")
        return

    csv_path = ROOT / "data/coagulation_admet_multi_task.csv"
    if not csv_path.exists():
        return
    split_path = ROOT / "publication/data/gnn_scaffold_split.json"
    if not split_path.exists():
        print("  [skip] Figure 3 — gnn_scaffold_split.json missing.")
        return
    split_from = str(split_path)
    dataset = load_multitask_data(str(csv_path))
    train, val, test, _ = resolve_scaffold_split(dataset, split_from=split_from)
    splits = {"Train": train, "Validation": val, "Test": test}
    rows = []
    for split_name, data_list in splits.items():
        scaffolds = [generate_scaffold(d.smiles) for d in data_list]
        rows.append({"Split": split_name, "Compounds": len(data_list), "Unique_scaffolds": len(set(scaffolds))})
    df = pd.DataFrame(rows)
    write_table(df, "table_s6_scaffold_split_stats", "## Supplementary Table S6. Scaffold split statistics")

    plt.figure(figsize=(5, 4))
    x = np.arange(len(df))
    w = 0.35
    plt.bar(x - w / 2, df["Compounds"], width=w, label="Compounds", color="#2166ac")
    plt.bar(x + w / 2, df["Unique_scaffolds"], width=w, label="Unique scaffolds", color="#92c5de")
    plt.xticks(x, df["Split"])
    plt.ylabel("Count")
    plt.title("Murcko scaffold split statistics")
    plt.legend(frameon=False)
    save_fig("figure3_scaffold_split")


def figure2_baseline_comparison() -> None:
    """GAT vs Morgan FP+RF vs VKORC1-only GAT — test R² by endpoint."""
    report = eval_report()
    tasks, gat_r2, rf_r2 = [], [], []
    for col, short in zip(TASKS, TASK_SHORT):
        mt = report["multitask_gnn"]["per_task"][col]
        rf = report["morgan_rf"]["per_task"][col]
        if mt["n"] == 0:
            continue
        tasks.append(short)
        gat_r2.append(mt["r2"])
        rf_r2.append(rf["r2"])

    if not tasks:
        print("  [skip] Figure 2 — no evaluation metrics found.")
        return

    x = np.arange(len(tasks))
    width = 0.25
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width, gat_r2, width, label="Multi-task GAT", color="#2166ac")
    plt.bar(x, rf_r2, width, label="Morgan FP + RF", color="#4daf4a")
    if "VKORC1" in tasks:
        vi = tasks.index("VKORC1")
        st_val = report.get("vkorc1_single_task_gnn", {}).get("per_task", {}).get("VKORC1_pXC50", {})
        if st_val.get("n", 0) > 0:
            plt.bar(vi + width, [st_val["r2"]], width, label="VKORC1-only GAT", color="#984ea3")

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, tasks, rotation=25, ha="right")
    plt.ylabel("Test R²")
    plt.title("Model comparison (Murcko scaffold hold-out)")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    save_fig("figure2_baseline_comparison")


def figure4_morgan_rf_analysis() -> None:
    """Three-panel Morgan RF baseline: test R², Spearman ρ, VKORC1 scatter."""
    try:
        from morgan_fp_baseline import TARGET_COLS
    except ImportError:
        print("  [skip] Figure 4 — morgan_fp_baseline not available.")
        return

    pred_path = ROOT / "publication/data/morgan_rf/morgan_rf_test_predictions.json"
    metrics_path = ROOT / "publication/data/morgan_rf/morgan_rf_metrics.json"
    if not pred_path.exists() or not metrics_path.exists():
        print("  [skip] Figure 4 — Morgan RF prediction/metric files missing.")
        return

    import json

    with open(pred_path, encoding="utf-8") as f:
        preds = json.load(f)
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)["per_task"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    ax = axes[0]
    tasks, r2_vals, ns = [], [], []
    for col in TARGET_COLS:
        m = metrics.get(col, {})
        if m.get("n_test", 0) == 0:
            continue
        tasks.append(m.get("task", col))
        r2_vals.append(m["r2"])
        ns.append(m["n_test"])
    colors = ["#4daf4a" if r >= 0 else "#e41a1c" for r in r2_vals]
    bars = ax.bar(tasks, r2_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, zorder=3)
    y_top = max(r2_vals) if r2_vals else 1.0
    y_bot = min(r2_vals) if r2_vals else 0.0
    ax.set_ylim((y_bot - 0.07) if y_bot < 0 else -0.01, y_top + 0.10)
    ax.set_ylabel("Test R²")
    ax.set_title("A. Morgan FP + RF (ECFP4, 2048-bit)")
    ax.tick_params(axis="x", rotation=30)
    for bar, n in zip(bars, ns):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + (0.04 if h >= 0 else -0.04),
            f"n={n}",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=7,
            clip_on=False,
        )

    ax = axes[1]
    task_rho = [metrics[col]["task"] for col in TARGET_COLS if metrics.get(col, {}).get("n_test", 0) > 0]
    rhos = [metrics[col].get("spearman_rho", np.nan) for col in TARGET_COLS if metrics.get(col, {}).get("n_test", 0) > 0]
    ax.bar(task_rho, rhos, color="#377eb8", edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("B. Rank correlation (test set)")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[2]
    vk = preds.get("VKORC1_pXC50")
    if vk:
        yt = np.array(vk["y_true"])
        yp = np.array(vk["y_pred"])
        novel = np.array(vk.get("scaffold_novel", [True] * len(yt)))
        ax.scatter(yt[novel], yp[novel], alpha=0.75, c="#4daf4a", edgecolors="k", linewidths=0.4, label="Novel scaffold", s=45)
        if (~novel).any():
            ax.scatter(yt[~novel], yp[~novel], alpha=0.75, c="#ff7f00", marker="s", label="Shared scaffold", s=45)
        lo, hi = min(yt.min(), yp.min()) - 0.3, max(yt.max(), yp.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        rho = metrics["VKORC1_pXC50"].get("spearman_rho", np.nan)
        r2 = metrics["VKORC1_pXC50"].get("r2", np.nan)
        ax.set_xlabel("Observed pXC50")
        ax.set_ylabel("Morgan RF predicted pXC50")
        ax.set_title(f"C. VKORC1 (R²={r2:.2f}, ρ={rho:.2f})")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.set_visible(False)

    fig.suptitle("Morgan ECFP4 + Random Forest baseline (Murcko test set)", fontsize=11)
    fig.tight_layout()
    save_fig("figure4_morgan_rf_analysis")


def figure_morgan_rf_standalone() -> None:
    """Single-row summary: R² + Spearman + top-task scatter (thrombin + VKORC1)."""
    pred_path = ROOT / "publication/data/morgan_rf/morgan_rf_test_predictions.json"
    metrics_path = ROOT / "publication/data/morgan_rf/morgan_rf_metrics.json"
    if not pred_path.exists() or not metrics_path.exists():
        print("  [skip] Morgan standalone — run `python morgan_fp_baseline.py` first.")
        return

    import json
    from morgan_fp_baseline import TARGET_COLS

    with open(pred_path, encoding="utf-8") as f:
        preds = json.load(f)
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)["per_task"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    tasks = [metrics[c]["task"] for c in TARGET_COLS if metrics.get(c, {}).get("n_test", 0) > 0]
    r2s = [metrics[c]["r2"] for c in TARGET_COLS if metrics.get(c, {}).get("n_test", 0) > 0]
    rhos = [metrics[c]["spearman_rho"] for c in TARGET_COLS if metrics.get(c, {}).get("n_test", 0) > 0]

    axes[0].barh(tasks, r2s, color="#4daf4a", edgecolor="k", linewidth=0.4)
    axes[0].axvline(0, color="k", linewidth=0.8)
    axes[0].set_xlabel("Test R²")
    axes[0].set_title("Morgan RF — R² by endpoint")

    axes[1].barh(tasks, rhos, color="#377eb8", edgecolor="k", linewidth=0.4)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Spearman ρ")
    axes[1].set_title("Morgan RF — rank correlation")

    # Best-supported task scatter (thrombin if available, else VKORC1)
    scatter_col = "Thrombin_pXC50" if "Thrombin_pXC50" in preds else "VKORC1_pXC50"
    pt = preds[scatter_col]
    yt, yp = np.array(pt["y_true"]), np.array(pt["y_pred"])
    axes[2].scatter(yt, yp, alpha=0.35, s=20, c="#4daf4a", edgecolors="none")
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    axes[2].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[2].set_xlabel("Observed pXC50")
    axes[2].set_ylabel("Predicted pXC50")
    axes[2].set_title(f"{metrics[scatter_col]['task']} (n={len(yt)}, R²={metrics[scatter_col]['r2']:.2f})")

    fig.suptitle("Morgan ECFP4 + Random Forest (Murcko test set)", fontsize=11)
    fig.tight_layout()
    save_fig("figure_morgan_rf_standalone")


def figure5_flexible_redock_spotcheck(spotcheck_path: Optional[Path] = None) -> None:
    path = spotcheck_path or (OUT / "data" / "flexible_redock_spotcheck.csv")
    if not path.exists():
        path = ROOT / "publication/data/flexible_redock_spotcheck.csv"
    if not path.exists():
        print("  [skip] Figure 5 — flexible_redock_spotcheck.csv missing.")
        return
    df = pd.read_csv(path)
    if "category" in df.columns:
        rl = df[df["category"] == "RL_Gen"].copy()
        ref = df[df["category"] == "reference"].copy()
    else:
        rl = df[df["ligand_name"].astype(str).str.startswith("RL_Gen")].copy()
        ref = df[~df["ligand_name"].astype(str).str.startswith("RL_Gen")].copy()
    if rl.empty and ref.empty:
        print("  [skip] Figure 5 — no spot-check rows.")
        return

    def _plot_panel(ax, subdf: pd.DataFrame, title: str, color: str) -> None:
        sub = subdf.copy()
        sub["abs_delta"] = sub["delta_kcal_mol"].abs()
        sub = sub.sort_values("abs_delta")
        ax.barh(sub["ligand_name"], sub["abs_delta"], color=color)
        max_delta = float(sub["abs_delta"].max())
        worst = sub.loc[sub["abs_delta"].idxmax(), "ligand_name"]
        ax.axvline(
            max_delta,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Max |Δ| = {max_delta:.3f} ({worst})",
        )
        ax.set_xlabel("|Δ affinity| (stored vs. flexible re-dock, kcal/mol)")
        ax.set_ylabel("Ligand")
        ax.set_title(title)
        ax.legend(frameon=False, loc="lower right", fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    if not rl.empty:
        _plot_panel(axes[0], rl, f"RL_Gen ligands (n={len(rl)})", "#b2182b")
    else:
        axes[0].set_visible(False)
    if not ref.empty:
        _plot_panel(axes[1], ref, f"Reference ligands (n={len(ref)})", "#2166ac")
    else:
        axes[1].set_visible(False)
    fig.suptitle("Flexible docking reproducibility spot-check", fontsize=11)
    fig.tight_layout()
    save_fig("figure5_flexible_redock_spotcheck")


def _read_gromacs_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return time (ps) and values from a GROMACS .xvg file."""
    xs, ys = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
    return np.array(xs), np.array(ys)


def _load_md_results_summary() -> dict:
    """Load multi-system MD metrics from publication/data/md_results_summary.json."""
    path = ROOT / "publication/data/md_results_summary.json"
    if not path.exists():
        return {"systems": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _md_system_by_label(summary: dict, label: str) -> dict:
    for sys in summary.get("systems", []):
        if sys.get("manuscript_label") == label or sys.get("run_id") == label:
            return sys
    return {}


def table5_md_summary() -> pd.DataFrame:
    """Membrane MD summary for completed systems only."""
    aff_map: Dict[str, float] = {}
    try:
        dock = pd.read_csv(screening_csv())
        aff_map = dock.set_index("ligand_name")["best_affinity"].to_dict()
    except FileNotFoundError:
        pass

    summary = _load_md_results_summary()
    rows: List[dict] = []
    for sys in summary.get("systems", []):
        flat_key = sys.get("docking_flat_ligand")
        iso_key = sys.get("docking_isoA_ligand")
        rows.append(
            {
                "Ligand": sys["manuscript_label"],
                "VKOR_DeltaG_flat_kcal_mol": aff_map.get(flat_key, np.nan) if flat_key else np.nan,
                "VKOR_DeltaG_isoA_kcal_mol": aff_map.get(iso_key, np.nan) if iso_key else np.nan,
                "Production_ns": sys.get("production_ns"),
                "Temperature_K": sys.get("temperature_K"),
                "Pressure_bar": sys.get("pressure_bar"),
                "Protein_RMSD_mean_A": sys.get("protein_rmsd_mean_A"),
                "Protein_RMSD_last25pct_A": sys.get("protein_rmsd_last25pct_A"),
                "Ligand_RMSD_2nd_half_mean_A": sys.get("ligand_rmsd_2nd_half_mean_A"),
                "Ligand_RMSD_max_A": sys.get("ligand_rmsd_max_A"),
                "ASN80_Hbond_occupancy_pct": sys.get("hbond_ASN80_occupancy_pct"),
                "Status": sys.get("status", "Complete"),
            }
        )

    return pd.DataFrame(rows)


def table_s7_md_hbond_occupancy() -> pd.DataFrame:
    """Per-residue ligand–protein H-bond occupancy across completed MD systems."""
    summary = _load_md_results_summary()
    residue_order = ["ASN80", "SER81", "TYR139", "THR138", "PHE55", "TRP59", "VAL134"]
    sys_labels = [s["manuscript_label"] for s in summary.get("systems", [])]
    columns = ["Residue_PLIP", *sys_labels, "GROMACS_resid_note"]

    rows: List[dict] = []
    for plip in residue_order:
        row: dict = {"Residue_PLIP": plip, "GROMACS_resid_note": ""}
        resid_notes: List[str] = []
        for sys in summary.get("systems", []):
            label = sys["manuscript_label"]
            match = next((r for r in sys.get("hbond_residues", []) if r["plip_label"] == plip), None)
            row[label] = match["occupancy_pct"] if match else np.nan
            if match:
                resid_notes.append(f"{label}: {match['gmx_resid']}")
        row["GROMACS_resid_note"] = "; ".join(resid_notes)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def figure8_md_rmsd() -> None:
    """Three-system protein/ligand RMSD time series (100 ns RL + 20 ns warfarin ref)."""
    summary = _load_md_results_summary()
    panels = [
        ("RL_Gen_37", "RL_Gen_37_isoA", 100.0),
        ("RL_Gen_29_isoA", "RL_Gen_29_isoA", 100.0),
        ("S_Warfarin_ref", "S-warfarin (ref)", 20.0),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    any_panel = False

    for ax, (run_id, title, ns_max) in zip(axes, panels):
        run_dir = ROOT / f"md_gromacs/runs/{run_id}/analysis"
        prot = run_dir / "rmsd_protein.xvg"
        lig = run_dir / "rmsd_ligand.xvg"
        if prot.exists() and lig.exists():
            tp, rp = _read_gromacs_xvg(prot)
            tl, rl = _read_gromacs_xvg(lig)
            ax.plot(tp / 1000.0, rp, label="Protein backbone", color="#2166ac", linewidth=1.0)
            ax.plot(tl / 1000.0, rl, label="Ligand (fit protein)", color="#b2182b", linewidth=1.0)
            any_panel = True
        else:
            sys = _md_system_by_label(summary, title)
            if sys:
                ax.bar(
                    ["Protein\n(mean)", "Protein\n(last 25%)", "Ligand\n(2nd half)", "Ligand\n(max)"],
                    [
                        sys.get("protein_rmsd_mean_A", np.nan),
                        sys.get("protein_rmsd_last25pct_A", np.nan),
                        sys.get("ligand_rmsd_2nd_half_mean_A", np.nan),
                        sys.get("ligand_rmsd_max_A", np.nan),
                    ],
                    color="#b2182b",
                    edgecolor="black",
                    linewidth=0.5,
                )
                any_panel = True
        ax.set_ylabel("RMSD (Å)")
        ax.set_xlabel("Time (ns)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_xlim(0, ns_max)

    if not any_panel:
        print("  [warn] No MD RMSD data; skipping figure8_md_rmsd_rl_gen_37")
        plt.close(fig)
        return

    fig.suptitle("Membrane MD — comparative RMSD", y=1.01)
    fig.tight_layout()
    save_fig("figure8_md_rmsd_rl_gen_37")


def figure9_md_hbond_occupancy() -> None:
    """Heatmap of binding-site H-bond occupancy across completed MD systems."""
    df = table_s7_md_hbond_occupancy()
    if df.empty:
        print("  [warn] No H-bond occupancy data; skipping figure9_md_hbond_occupancy")
        return

    summary = _load_md_results_summary()
    sys_labels = [s["manuscript_label"] for s in summary.get("systems", [])]
    plot_df = df.set_index("Residue_PLIP")[sys_labels].astype(float)
    fig, ax = plt.subplots(figsize=(6, 4))
    if HAS_SEABORN:
        sns.heatmap(plot_df, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=100, ax=ax, cbar_kws={"label": "Occupancy (%)"})
    else:
        im = ax.imshow(plot_df.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
        ax.set_xticks(range(len(sys_labels)))
        ax.set_xticklabels(sys_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(plot_df.index)))
        ax.set_yticklabels(plot_df.index)
        fig.colorbar(im, ax=ax, label="Occupancy (%)")
    ax.set_title("Binding-site H-bond occupancy")
    ax.set_xlabel("System")
    ax.set_ylabel("Residue (PLIP label)")
    save_fig("figure9_md_hbond_occupancy")


def cleanup_stale_assets() -> None:
    stale_tables = (
        "table2_vkorc1_combined_leaderboard.csv",
        "table2_vkorc1_combined_leaderboard.md",
        "table_s_stereoselectivity.csv",
        "table_s_stereoselectivity.md",
        "table3_admet_md_candidates.csv",
        "table3_admet_md_candidates.md",
        "table3_top5_rl_gen_selectivity.csv",
        "table3_top5_rl_gen_selectivity.md",
        "table_s_scaffold_split_stats.csv",
        "table_s_scaffold_split_stats.md",
        "table_s1_scaffold_split_stats.csv",
        "table_s1_scaffold_split_stats.md",
    )
    stale_figures = (
        "figure_s1_scaffold_split",
        "figure_s2_baseline_comparison",
        "figure_s2_gnn_training_curves",
        "figure_s3_flexible_redock_spotcheck",
        "figure_s4_md_rmsd_rl_gen_37",
        "figure2_interaction_heatmap",
        "figure3_reinvent_vs_training_distribution",
        "figure_morgan_rf_standalone",
    )
    for stale in stale_tables:
        path = TAB_DIR / stale
        if path.exists():
            path.unlink()
            print(f"  [cleanup] removed legacy table {stale}")
    for stem in stale_figures:
        for ext in ("png", "pdf"):
            path = FIG_DIR / f"{stem}.{ext}"
            if path.exists():
                path.unlink()
                print(f"  [cleanup] removed legacy figure {path.name}")


def write_manifest() -> None:
    manifest = {
        "figures": sorted(p.name for p in FIG_DIR.glob("*")),
        "tables": sorted(p.name for p in TAB_DIR.glob("*")),
        "internal_qc": sorted(p.name for p in QC_DIR.glob("*")),
        "note": "Manuscript tables/figures only in tables/ and figures/. Reference-ligand QC in internal_qc/. "
        "table5_md_rl_gen_37_summary and table_s4_md_rl_gen_37_summary are intentional mirrors — "
        "edit table5_md_summary() only and regenerate via build_all.sh.",
    }
    with open(OUT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [ok] manifest → {OUT / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript figures and tables.")
    parser.add_argument("--skip-gnn-predictions", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    set_style()
    ensure_dirs()
    spotcheck_path = sync_flexible_redock_spotcheck()
    print("Generating publication assets…")

    md_table = table5_md_summary()
    hbond_table = table_s7_md_hbond_occupancy()
    write_table(table1_gnn_metrics(), "table1_gnn_test_metrics", "## Table 1. GNN test metrics by endpoint")
    write_table(table_s1_full_metrics(), "table_s1_full_gnn_metrics", "## Supplementary Table S1")
    write_table(
        table2_docking_leaderboard(),
        "table2_vkorc1_docking_leaderboard",
        "## Table 2. VKORC1 flexible docking leaderboard (top RL_Gen leads)",
    )
    write_table(
        table3_admet_rl_gen_top(),
        "table3_admet_rl_gen_top",
        "## Table 3. ADMET and interaction profile for top RL_Gen leads",
    )
    write_table(
        table3_selectivity(run_gnn=not args.skip_gnn_predictions),
        "table4_top5_rl_gen_selectivity",
        "## Table 4. Multi-task GNN predictions for top docked RL_Gen leads",
    )
    write_table(table_s2_rl_gen_docking(), "table_s2_rl_gen_docking_full", "## Supplementary Table S2. RL_Gen docking affinities")
    write_table(table_s3_reinvent_scoring(), "table_s3_reinvent_scoring", "## Supplementary Table S3. REINVENT scoring weights")
    write_table(
        table_s5_chembl_only_vkorc1(),
        "table_s5_vkorc1_chembl_only_benchmark",
        "## Supplementary Table S5. VKORC1 test metrics excluding RL-library SMILES overlaps",
    )
    write_table(
        md_table,
        "table_s4_md_rl_gen_37_summary",
        "## Supplementary Table S4. Membrane MD summary (mirrors Table 5)",
    )
    write_table(
        hbond_table,
        "table_s7_md_hbond_occupancy",
        "## Supplementary Table S7. Binding-site H-bond occupancy (% frames)",
    )
    write_qc_table(
        table2_combined_leaderboard(),
        "table2_vkorc1_combined_leaderboard",
        "## Internal QC — combined RL + reference docking (not in manuscript)",
    )
    write_qc_table(
        table_s_stereoselectivity(),
        "table_s_stereoselectivity",
        "## Internal QC — reference enantiomer ΔΔG (not in manuscript)",
    )
    enrich = table_docking_enrichment_summary()
    if not enrich.empty:
        write_qc_table(
            enrich,
            "table_docking_enrichment_summary",
            "## Internal QC — retrospective VKORC1 docking enrichment",
        )
    archive_docking_validation_qc()
    archive_receptor_validation_report()
    write_internal_qc_readme()
    write_table(
        md_table,
        "table5_md_rl_gen_37_summary",
        "## Table 5. Membrane MD trajectory metrics (three completed systems)",
    )

    cleanup_stale_assets()

    figure1_training_curves()
    figure2_baseline_comparison()
    figure3_scaffold_split()
    figure4_morgan_rf_analysis()
    figure5_flexible_redock_spotcheck(spotcheck_path)
    figure6_interaction_heatmap()
    figure7_reinvent_distribution()
    figure8_md_rmsd()
    figure9_md_hbond_occupancy()
    write_manifest()
    print("\nDone. Outputs in publication/output/")


if __name__ == "__main__":
    main()
