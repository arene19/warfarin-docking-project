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


def save_fig(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close()
    print(f"  [fig] {name}.png / .pdf")


def ligand_smiles_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    config_path = ROOT / "config.yaml"
    if config_path.exists():
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
        ]
    )
    if p is None:
        raise FileNotFoundError("VKORC1 screening results CSV not found.")
    return p


def eval_report() -> Dict[str, Any]:
    p = first_existing(
        [
            ROOT / "results/gnn_evaluation_report.json",
            ROOT / "publication/data/gnn_evaluation_report.json",
        ]
    )
    if p is None:
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
                "RF_R2": round(rf["r2"], 2) if mt["n"] else np.nan,
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
            "RF_R2": np.nan,
            "VKORC1_only_GAT_R2": np.nan,
        }
    )
    return pd.DataFrame(rows)


def table_s1_full_metrics() -> pd.DataFrame:
    report = eval_report()
    rows = []
    for col, short in zip(TASKS, TASK_SHORT):
        mt = report["multitask_gnn"]["per_task"][col]
        rf = report["morgan_rf"]["per_task"][col]
        st_r2 = np.nan
        if col == "VKORC1_pXC50":
            st = report.get("vkorc1_single_task_gnn", {}).get("per_task", {}).get(col, {})
            st_r2 = st.get("r2", np.nan)
        sp = report.get("vkorc1_spearman", {})
        rows.append(
            {
                "Task": short,
                "n_test": mt["n"],
                "n_train_labels": report["label_counts"]["train"][col],
                "GAT_RMSE": round(mt["rmse"], 3),
                "GAT_MAE": round(mt["mae"], 3),
                "GAT_R2": round(mt["r2"], 3),
                "GAT_Spearman_rho": round(sp.get("multitask_gat", np.nan), 3)
                if col == "VKORC1_pXC50"
                else np.nan,
                "RF_RMSE": round(rf["rmse"], 3),
                "RF_R2": round(rf["r2"], 3),
                "RF_Spearman_rho": round(sp.get("morgan_rf", np.nan), 3)
                if col == "VKORC1_pXC50"
                else np.nan,
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
    df["pXC50_derived"] = (-df["best_affinity"] / 1.36).round(3)
    df = df.rename(columns={"ligand_name": "Ligand", "best_affinity": "dG_kcal_mol"})
    return df[["Ligand", "dG_kcal_mol", "pXC50_derived", "success"]]


def table_s2_rl_gen_docking() -> pd.DataFrame:
    df = pd.read_csv(screening_csv())
    rl = df[df["ligand_name"].astype(str).str.startswith("RL_Gen")].copy()
    rl = rl.sort_values("best_affinity")
    rl["pXC50_derived"] = (-rl["best_affinity"] / 1.36).round(3)
    rl = rl.rename(
        columns={"ligand_name": "Ligand", "best_affinity": "dG_kcal_mol", "runtime_seconds": "runtime_s"}
    )
    return rl[["Ligand", "dG_kcal_mol", "pXC50_derived", "success", "runtime_s"]]


def table3_selectivity(run_gnn: bool = True) -> pd.DataFrame:
    df = pd.read_csv(screening_csv())
    top = df[df["ligand_name"].astype(str).str.match(r"^RL_Gen_\d+$", na=False)]
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


def table_s3_reinvent_scoring() -> pd.DataFrame:
    rows = [
        {"Component": "Synthetic Accessibility (SA Score)", "Weight": 1.5, "Role": "Penalize difficult syntheses"},
        {"Component": "QED", "Weight": 1.0, "Role": "Drug-likeness"},
        {"Component": "Molecular Weight", "Weight": 0.5, "Role": "Prefer 150–500 Da"},
        {"Component": "GNN VKORC1 (ExternalProcess)", "Weight": 2.5, "Role": "Custom gnn_predict.py ranker"},
    ]
    toml_path = ROOT / "REINVENT4" / "coumarin_rl.toml"
    if toml_path.exists():
        text = toml_path.read_text(encoding="utf-8")
        for i, comp in enumerate(rows):
            m = re.search(rf'\[\[stage\.scoring\.component\]\].*?name\s*=\s*"{comp["Component"].split("(")[0].strip()}",', text, re.S)
            if not m and "GNN" in comp["Component"]:
                m = re.search(r"ExternalProcess.*?weight\s*=\s*([\d.]+)", text, re.S)
                if m:
                    rows[i]["Weight"] = float(m.group(1))
    return pd.DataFrame(rows)


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


def figure2_interaction_heatmap() -> None:
    ipath = ROOT / "results/interaction_profile.csv"
    if not ipath.exists():
        print("  [skip] Figure 2 — interaction_profile.csv not found.")
        return

    df = pd.read_csv(ipath)
    df = df[df["Receptor"].astype(str).str.contains("VKORC1_Human", na=False)]
    dock = pd.read_csv(screening_csv())
    top_rl = dock[dock["ligand_name"].astype(str).str.startswith("RL_Gen")]
    top_rl = top_rl.sort_values("best_affinity").head(12)["ligand_name"].tolist()
    refs = [
        "S_Warfarin_ref",
        "R_Warfarin_ref",
        "RL_Gen_37",
        "RL_Gen_22",
        "RL_Gen_29",
        "RL_Gen_07",
        "RL_Gen_26",
    ]
    keep = [x for x in refs + top_rl if x in set(df["Ligand"])]
    keep = list(dict.fromkeys(keep))[:15]
    sub = df[df["Ligand"].isin(keep)].copy()
    if sub.empty:
        print("  [skip] Figure 2 — no overlapping ligands.")
        return

    sub = sub.set_index("Ligand")
    mat = sub[["Num_H_Bonds", "Num_Hydrophobic", "Num_Pi_Stacking"]].astype(float)
    aff = dock.set_index("ligand_name").loc[mat.index, "best_affinity"]
    mat["Docking_score"] = -aff.values

    plt.figure(figsize=(6, max(4, 0.35 * len(mat))))
    if HAS_SEABORN:
        sns.heatmap(mat, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Count / score"})
    else:
        plt.imshow(mat.values, aspect="auto", cmap="YlOrRd")
        plt.colorbar(label="Count / score")
        plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
        plt.yticks(range(len(mat.index)), mat.index)
        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                plt.text(j, i, f"{mat.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
    plt.title("VKORC1 interaction fingerprint (top RL_Gen + references)")
    plt.xlabel("Interaction type")
    plt.ylabel("Ligand")
    save_fig("figure2_interaction_heatmap")


def figure3_reinvent_distribution() -> None:
    gen_path = first_existing(
        [
            ROOT / "REINVENT4/coumarin_generation_1.csv",
            ROOT / "coumarin_generation_1.csv",
        ]
    )
    master_path = ROOT / "data/coagulation_admet_multi_task.csv"
    if gen_path is None or not master_path.exists():
        print("  [skip] Figure 3 — REINVENT CSV or master dataset missing.")
        return

    gen = pd.read_csv(gen_path)
    raw_col = "Pred_VKORC1_pXC50 (raw)"
    if raw_col not in gen.columns:
        print("  [skip] Figure 3 — Pred_VKORC1_pXC50 column missing in REINVENT output.")
        return
    gen_scores = gen[raw_col].dropna()
    gen_scores = gen_scores[(gen_scores > 3) & (gen_scores < 12)]

    master = pd.read_csv(master_path)
    train_scores = master["VKORC1_pXC50"].dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    train_label = f"Training VKORC1 labels (n={len(train_scores)})"
    gen_label = f"REINVENT4 Pred_VKORC1 (n={len(gen_scores)})"
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
    ax.set_xlabel("VKORC1 pXC50 (experimental / predicted)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 3. REINVENT generation vs. training VKORC1 label distribution")
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
    save_fig("figure3_reinvent_vs_training_distribution")


def figure_s1_scaffold_split() -> None:
    try:
        from gnn_model import generate_scaffold, load_multitask_data, scaffold_split
    except ImportError:
        print("  [skip] Figure S1 — torch/torch_geometric not available.")
        return

    csv_path = ROOT / "data/coagulation_admet_multi_task.csv"
    if not csv_path.exists():
        return
    dataset = load_multitask_data(str(csv_path))
    train, val, test, _ = scaffold_split(dataset)
    splits = {"Train": train, "Validation": val, "Test": test}
    rows = []
    for split_name, data_list in splits.items():
        scaffolds = [generate_scaffold(d.smiles) for d in data_list]
        rows.append({"Split": split_name, "Compounds": len(data_list), "Unique_scaffolds": len(set(scaffolds))})
    df = pd.DataFrame(rows)
    write_table(df, "table_s_scaffold_split_stats", "Scaffold split statistics")

    plt.figure(figsize=(5, 4))
    x = np.arange(len(df))
    w = 0.35
    plt.bar(x - w / 2, df["Compounds"], width=w, label="Compounds", color="#2166ac")
    plt.bar(x + w / 2, df["Unique_scaffolds"], width=w, label="Unique scaffolds", color="#92c5de")
    plt.xticks(x, df["Split"])
    plt.ylabel("Count")
    plt.title("Murcko scaffold split statistics")
    plt.legend(frameon=False)
    save_fig("figure_s1_scaffold_split")


def figure_s2_baseline_comparison() -> None:
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
        print("  [skip] Figure S2 — no evaluation metrics found.")
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
    plt.title("Figure S2. Model comparison (Murcko scaffold hold-out)")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    save_fig("figure_s2_baseline_comparison")


def figure4_morgan_rf_analysis() -> None:
    """2×2 panel: Murcko split context + Morgan RF R², Spearman, VKORC1 scatter."""
    try:
        from morgan_fp_baseline import TARGET_COLS, TASK_SHORT, load_or_train
    except ImportError:
        print("  [skip] Figure 4 — morgan_fp_baseline not available.")
        return

    pred_path = ROOT / "publication/data/morgan_rf/morgan_rf_test_predictions.json"
    if pred_path.exists():
        import json

        with open(pred_path, encoding="utf-8") as f:
            preds = json.load(f)
        metrics_path = ROOT / "publication/data/morgan_rf/morgan_rf_metrics.json"
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)["per_task"]
        from gnn_model import load_multitask_data
        from morgan_fp_baseline import scaffold_split_summary

        split_df = scaffold_split_summary(load_multitask_data(str(ROOT / "data/coagulation_admet_multi_task.csv")))
    else:
        _, _, metrics, _, bundle = load_or_train(str(ROOT / "data/coagulation_admet_multi_task.csv"))
        preds = bundle["predictions"]
        split_df = bundle["split"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))

    # A — Murcko scaffold split (evaluation context)
    ax = axes[0, 0]
    x = np.arange(len(split_df))
    w = 0.35
    ax.bar(x - w / 2, split_df["Compounds"], width=w, label="Compounds", color="#bdbdbd")
    ax.bar(x + w / 2, split_df["Unique_scaffolds"], width=w, label="Unique scaffolds", color="#2166ac")
    ax.set_xticks(x, split_df["Split"])
    ax.set_ylabel("Count")
    ax.set_title("A. Murcko scaffold split", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.margins(y=0.08)

    # B — Morgan RF test R² (standalone)
    ax = axes[0, 1]
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
    # Tight y-limits: small pad above for n-labels; below zero only if a bar is negative
    y_max = y_top + 0.10
    y_min = (y_bot - 0.07) if y_bot < 0 else -0.01
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Test R²")
    ax.set_title("B. Morgan FP + RF (ECFP4, 2048-bit)", pad=10)
    ax.tick_params(axis="x", rotation=30)
    for bar, n in zip(bars, ns):
        h = bar.get_height()
        if h >= 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.04,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=False,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.04,
                f"n={n}",
                ha="center",
                va="top",
                fontsize=7,
                clip_on=False,
            )

    # C — Spearman rank correlation
    ax = axes[1, 0]
    rhos = [metrics[col].get("spearman_rho", np.nan) for col in TARGET_COLS if metrics.get(col, {}).get("n_test", 0) > 0]
    task_rho = [metrics[col]["task"] for col in TARGET_COLS if metrics.get(col, {}).get("n_test", 0) > 0]
    ax.bar(task_rho, rhos, color="#377eb8", edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("C. Rank correlation (test set)")
    ax.tick_params(axis="x", rotation=30)

    # D — VKORC1 predicted vs observed
    ax = axes[1, 1]
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
        ax.set_title(f"D. VKORC1 (R²={r2:.2f}, ρ={rho:.2f})")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.set_visible(False)

    fig.suptitle(
        "Figure 4. Morgan fingerprint baseline under Murcko scaffold split",
        fontsize=12,
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Morgan RF metrics are computed on the scaffold-disjoint test split (panel A).",
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
    )
    fig.subplots_adjust(bottom=0.08, top=0.93, hspace=0.38, wspace=0.28)
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


def figure_s3_flexible_spotcheck() -> None:
    path = ROOT / "publication/data/flexible_redock_spotcheck.csv"
    if not path.exists():
        print("  [skip] Figure S3 — flexible_redock_spotcheck.csv missing.")
        return
    df = pd.read_csv(path)
    df = df.sort_values("delta_kcal_mol")
    colors = df["category"].map({"reference": "#2166ac", "RL_Gen": "#b2182b"}).fillna("#666666")

    plt.figure(figsize=(8, 5))
    plt.barh(df["ligand_name"], df["delta_kcal_mol"], color=colors)
    plt.axvline(0.52, color="black", linestyle="--", linewidth=1, label="Max |Δ| = 0.52 kcal/mol")
    plt.xlabel("|Δ affinity| (training vs. flexible re-dock, kcal/mol)")
    plt.ylabel("Ligand")
    plt.title("Flexible docking reproducibility spot-check (n=15)")
    plt.legend(frameon=False)
    save_fig("figure_s3_flexible_redock_spotcheck")


def write_manifest() -> None:
    manifest = {
        "figures": sorted(p.name for p in FIG_DIR.glob("*")),
        "tables": sorted(p.name for p in TAB_DIR.glob("*")),
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
    print("Generating publication assets…")

    write_table(table1_gnn_metrics(), "table1_gnn_test_metrics", "## Table 1. GNN test metrics by endpoint")
    write_table(table_s1_full_metrics(), "table_s1_full_gnn_metrics", "## Supplementary Table S1")
    write_table(table2_docking_leaderboard(), "table2_vkorc1_docking_leaderboard", "## Table 2. Top VKORC1 docking affinities")
    write_table(table_s2_rl_gen_docking(), "table_s2_rl_gen_docking_full", "## Supplementary Table S2. RL_Gen docking affinities")
    write_table(
        table3_selectivity(run_gnn=not args.skip_gnn_predictions),
        "table3_top5_rl_gen_selectivity",
        "## Table 3. Multi-task GNN predictions for top docked RL_Gen leads",
    )
    write_table(table_s3_reinvent_scoring(), "table_s3_reinvent_scoring", "## Supplementary Table S3. REINVENT scoring weights")

    figure1_training_curves()
    figure2_interaction_heatmap()
    figure3_reinvent_distribution()
    figure_s1_scaffold_split()
    figure_s2_baseline_comparison()
    figure4_morgan_rf_analysis()
    figure_morgan_rf_standalone()
    figure_s3_flexible_spotcheck()
    write_manifest()
    print("\nDone. Outputs in publication/output/")


if __name__ == "__main__":
    main()
