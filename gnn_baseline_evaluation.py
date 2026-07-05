#gnn_baseline_evaluation.py
"""Evaluate multi-task GNN (checkpoint), Morgan FP + RF, and VKORC1-only GNN baselines."""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from morgan_fp_baseline import TARGET_COLS, data_list_to_frame, train_and_evaluate, save_artifacts
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

from gnn_model import (
    DEFAULT_GNN_CONFIG,
    DynamicMultiTaskGNN,
    inference_loop,
    load_checkpoint,
    load_multitask_data,
    resolve_scaffold_split,
    set_seed,
    train_loop,
    validation_loss,
)

GNN_CONFIG = DEFAULT_GNN_CONFIG.copy()


@torch.no_grad()
def per_task_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device, num_tasks: int = 6
) -> Dict[str, Dict[str, float]]:
    model.eval()
    y_true = [[] for _ in range(num_tasks)]
    y_pred = [[] for _ in range(num_tasks)]

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        y = batch.y.view(out.shape).cpu().numpy()
        pred = out.cpu().numpy()
        mask = batch.mask.view(out.shape).cpu().numpy()
        for i in range(num_tasks):
            valid = mask[:, i] == 1.0
            if valid.any():
                y_true[i].extend(y[valid, i].tolist())
                y_pred[i].extend(pred[valid, i].tolist())

    results: Dict[str, Dict[str, float]] = {}
    for i, col in enumerate(TARGET_COLS[:num_tasks]):
        if not y_true[i]:
            results[col] = {"n": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            continue
        yt = np.array(y_true[i])
        yp = np.array(y_pred[i])
        mse = mean_squared_error(yt, yp)
        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = float("nan")
        results[col] = {
            "n": int(len(yt)),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2),
        }
    return results


@torch.no_grad()
def global_metrics_from_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    rmse, mae, r2 = inference_loop(model, loader, device)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _rf_metrics_for_report(raw: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        col: {
            "n": int(v["n_test"]),
            "rmse": v["rmse"],
            "mae": v["mae"],
            "r2": v["r2"],
            "spearman_rho": v.get("spearman_rho", float("nan")),
        }
        for col, v in raw.items()
    }


def audit_vkorc1_label_sources(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rl_smiles: set[str],
) -> Dict[str, Any]:
    """Summarize ChEMBL vs active-learning VKORC1 labels for supplementary transparency."""
    def _classify(df: pd.DataFrame, split: str) -> Dict[str, Any]:
        labeled = df[df["VKORC1_pXC50"].notna()]
        rl_mask = labeled["canonical_smiles"].isin(rl_smiles)
        overlap_count = int(rl_mask.sum())
        return {
            "split": split,
            "total_vkorc1_labels": int(len(labeled)),
            "chembl_or_non_rl_labels": int((~rl_mask).sum()),
            "rl_library_smiles_overlaps": overlap_count,
            "rl_pseudo_labels": overlap_count,  # deprecated alias; SMILES overlap, not label provenance
        }

    return {
        "rl_smiles_in_screening": len(rl_smiles),
        "splits": [_classify(train_df, "train"), _classify(test_df, "test")],
        "note": (
            "rl_library_smiles_overlaps counts test/train VKORC1 labels whose canonical SMILES "
            "appear in the RL screening library (config + SDF resolution), not necessarily "
            "Vina-derived pseudo-labels. Table S5 excludes these SMILES from the held-out test set."
        ),
    }


@torch.no_grad()
def vkorc1_chembl_only_test_metrics(
    model: nn.Module,
    test_data: List[Any],
    rl_smiles: set[str],
    device: torch.device,
) -> Dict[str, float]:
    """VKORC1 test metrics on scaffold hold-out compounds excluding RL pseudo-labels."""
    yt, yp = [], []
    model.eval()
    for data in test_data:
        if data.smiles in rl_smiles:
            continue
        if data.mask[0, 0].item() != 1.0:
            continue
        loader = DataLoader([data], batch_size=1, shuffle=False)
        for batch_item in loader:
            batch_item = batch_item.to(device)
            pred = model(batch_item)[0, 0].item()
            yt.append(data.y[0, 0].item())
            yp.append(pred)
    if len(yt) < 3:
        return {
            "n": len(yt),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "spearman_rho": float("nan"),
        }
    yt_arr = np.array(yt)
    yp_arr = np.array(yp)
    mse = mean_squared_error(yt_arr, yp_arr)
    try:
        r2 = r2_score(yt_arr, yp_arr)
    except Exception:
        r2 = float("nan")
    try:
        rho = float(spearmanr(yt_arr, yp_arr).statistic)
    except Exception:
        rho = float("nan")
    return {
        "n": int(len(yt)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(yt_arr, yp_arr)),
        "r2": float(r2),
        "spearman_rho": rho,
    }


def vkorc1_chembl_only_rf_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rl_smiles: set[str],
    seed: int = 42,
) -> Dict[str, float]:
    """Morgan RF VKORC1 metrics on test rows excluding RL pseudo-label SMILES."""
    from morgan_fp_baseline import RF_KWARGS, build_fingerprint_matrix
    from sklearn.ensemble import RandomForestRegressor

    tr = train_df.dropna(subset=["VKORC1_pXC50"])
    te = test_df.dropna(subset=["VKORC1_pXC50"])
    te = te[~te["canonical_smiles"].isin(rl_smiles)]
    if len(tr) < 5 or len(te) < 3:
        return {
            "n": len(te),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "spearman_rho": float("nan"),
        }
    x_train, tr_idx = build_fingerprint_matrix(tr["canonical_smiles"].tolist())
    x_test, te_idx = build_fingerprint_matrix(te["canonical_smiles"].tolist())
    y_train = tr.iloc[tr_idx]["VKORC1_pXC50"].values
    y_test = te.iloc[te_idx]["VKORC1_pXC50"].values
    rf = RandomForestRegressor(**{**RF_KWARGS, "random_state": seed})
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    try:
        r2 = r2_score(y_test, pred)
    except Exception:
        r2 = float("nan")
    try:
        rho = float(spearmanr(y_test, pred).statistic)
    except Exception:
        rho = float("nan")
    return {
        "n": int(len(y_test)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2),
        "spearman_rho": rho,
    }


def load_rl_gen_canonical_smiles() -> set[str]:
    """Canonical SMILES for RL_Gen ligands from screening CSV (config + SDF fallback)."""
    from config_utils import canonicalize_smiles, get_smiles, load_config

    paths = [
        "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv",
        "deposition/package/docking/VKORC1_Human_screening_results.csv",
    ]
    csv_path = next((p for p in paths if os.path.exists(p)), None)
    if not csv_path:
        raise FileNotFoundError(
            "RL screening CSV required for label audit. Expected under results/docked_poses/ "
            "or deposition/package/docking/."
        )

    df = pd.read_csv(csv_path)
    rl_names = df[df["ligand_name"].astype(str).str.startswith("RL_Gen_")]["ligand_name"].tolist()
    if not rl_names:
        raise ValueError("No RL_Gen ligands found in screening CSV.")

    config = load_config("config_master.yaml") if os.path.exists("config_master.yaml") else {}
    ligands = config.get("ligands", {}) or {}
    ligands_dir = os.path.join("results", "ligands")
    out: set[str] = set()
    unresolved: list[str] = []

    for name in rl_names:
        entry = ligands.get(name, {})
        smi = get_smiles(entry) if entry else None
        if not smi and os.path.exists(os.path.join(ligands_dir, f"{name}.sdf")):
            try:
                from rdkit import Chem

                mol = Chem.MolFromMolFile(os.path.join(ligands_dir, f"{name}.sdf"), sanitize=True)
                if mol is not None:
                    smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = None
        if smi:
            canon = canonicalize_smiles(smi) or smi
            out.add(canon)
        else:
            unresolved.append(name)

    if unresolved:
        print(f"  Warning: {len(unresolved)} RL_Gen names unresolved to SMILES (config/SDF): {unresolved[:5]}...")
    print(f"  RL library canonical SMILES for audit: {len(out)}")
    return out


def train_single_task_vkorc1(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    checkpoint_path: str = "vkorc1_single_task_gnn.pth",
) -> nn.Module:
    """VKORC1-only GNN with identical architecture width but one output head."""
    model = DynamicMultiTaskGNN(
        config=GNN_CONFIG, num_node_features=10, num_tasks=1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    best_val = float("inf")
    stalls = 0
    warmup = 15
    patience = 25
    max_epochs = 150

    for epoch in range(1, max_epochs + 1):
        train_loop(model, train_loader, optimizer, device)
        val_loss = validation_loss(model, val_loader, device)
        scheduler.step(val_loss)

        if epoch > warmup:
            if val_loss < best_val:
                best_val = val_loss
                stalls = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                stalls += 1

        if epoch > warmup and stalls >= patience:
            print(f"  VKORC1-only GNN early stop at epoch {epoch}")
            break

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def vkorc1_only_graphs(data_list: List[Any]) -> List[Any]:
    """Keep only VKORC1 label/mask; clone graphs so the shared split is not mutated."""
    out = []
    for d in data_list:
        copy = d.clone()
        y0 = copy.y[0, 0].item()
        m0 = copy.mask[0, 0].item()
        copy.y = torch.tensor([[y0]], dtype=torch.float)
        copy.mask = torch.tensor([[m0]], dtype=torch.float)
        out.append(copy)
    return out


def label_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {col: int(df[col].notna().sum()) for col in TARGET_COLS}


@torch.no_grad()
def spearman_for_task(
    model: nn.Module, loader: DataLoader, device: torch.device, task_idx: int = 0
) -> float:
    model.eval()
    yt, yp = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        y = batch.y.view(out.shape).cpu().numpy()
        m = batch.mask.view(out.shape).cpu().numpy()
        pred = out.cpu().numpy()
        for i in range(len(y)):
            if m[i, task_idx] == 1.0:
                yt.append(y[i, task_idx])
                yp.append(pred[i, task_idx])
    if len(yt) < 3:
        return float("nan")
    return float(spearmanr(yt, yp).statistic)


def print_metrics_table(title: str, metrics: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}")
    print(f"{'Task':<22} {'N':>5} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 55)
    for col in TARGET_COLS:
        s = metrics.get(col, {})
        n = s.get("n", 0)
        if n == 0:
            print(f"{col:<22} {n:>5} {'—':>8} {'—':>8} {'—':>8}")
        else:
            print(
                f"{col:<22} {n:>5} {s['rmse']:>8.3f} {s['mae']:>8.3f} {s['r2']:>8.3f}"
            )


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GNN and Morgan RF baselines.")
    parser.add_argument("--data", default=os.path.join("data", "coagulation_admet_multi_task.csv"))
    parser.add_argument(
        "--split-from",
        default=os.path.join("publication", "data", "gnn_scaffold_split.json"),
        help="Frozen scaffold split JSON (required).",
    )
    parser.add_argument(
        "--retrain-vkorc1-only",
        action="store_true",
        help="Retrain VKORC1-only GAT even if frozen checkpoint exists.",
    )
    parser.add_argument(
        "--retrain-morgan",
        action="store_true",
        help="Retrain Morgan FP + RF even if frozen models exist.",
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.split_from or not os.path.exists(args.split_from):
        raise FileNotFoundError(
            f"Frozen scaffold split required: {args.split_from}. "
            "Run dynamic_gnn_pipeline.py with --split-from or create gnn_scaffold_split.json."
        )
    split_from = args.split_from
    dataset = load_multitask_data(args.data)
    train_data, val_data, test_data, _ = resolve_scaffold_split(
        dataset, seed=42, split_from=split_from
    )
    print(f"Split sizes — train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    train_df = data_list_to_frame(train_data)
    val_df = data_list_to_frame(val_data)
    test_df = data_list_to_frame(test_data)
    print("\nLabeled compounds per split (train / val / test):")
    for col in TARGET_COLS:
        print(
            f"  {col}: {train_df[col].notna().sum()} / "
            f"{val_df[col].notna().sum()} / {test_df[col].notna().sum()}"
        )

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # --- Multi-task GNN (existing checkpoint) ---
    ckpt = "coagulation_admet_gnn.pth"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    mt_model, _ = load_checkpoint(ckpt, device)
    mt_model.eval()

    mt_global = global_metrics_from_loader(mt_model, test_loader, device)
    mt_per_task = per_task_metrics(mt_model, test_loader, device, num_tasks=6)
    print_metrics_table("Multi-task GAT (checkpoint)", mt_per_task)
    print(
        f"  Global (masked pool): RMSE={mt_global['rmse']:.3f}, "
        f"MAE={mt_global['mae']:.3f}, R²={mt_global['r2']:.3f}"
    )

    # --- Morgan FP + Random Forest ---
    morgan_dir = Path("publication/data/morgan_rf")
    morgan_model = morgan_dir / "morgan_rf_models.joblib"
    if morgan_model.exists() and not args.retrain_morgan:
        print("\nLoading frozen Morgan FP + Random Forest baselines...")
        with open(morgan_dir / "morgan_rf_metrics.json", encoding="utf-8") as f:
            rf_metrics = json.load(f)["per_task"]
        import joblib
        from morgan_fp_baseline import collect_test_predictions

        models = joblib.load(morgan_model)
        preds = collect_test_predictions(train_df, test_df, models)
        pred_path = morgan_dir / "morgan_rf_test_predictions.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(preds), f, indent=2)
        print(f"  Refreshed Morgan test predictions → {pred_path}")
    else:
        print("\nTraining Morgan FP + Random Forest baselines...")
        rf_raw, rf_models = train_and_evaluate(train_df, test_df)
        save_artifacts(rf_raw, rf_models)
        rf_metrics = _rf_metrics_for_report(rf_raw)
    print_metrics_table("Morgan FP + Random Forest", rf_metrics)

    gat_spearman = {
        col: spearman_for_task(mt_model, test_loader, device, task_idx=i)
        for i, col in enumerate(TARGET_COLS)
    }

    # --- VKORC1-only single-task GNN ---
    st_ckpt = os.path.join("publication", "data", "vkorc1_single_task_gnn.pth")
    os.makedirs(os.path.dirname(st_ckpt), exist_ok=True)
    st_train = vkorc1_only_graphs([d for d in train_data])
    st_val = vkorc1_only_graphs([d for d in val_data])
    st_test = vkorc1_only_graphs([d for d in test_data])
    st_train_loader = DataLoader(st_train, batch_size=64, shuffle=True)
    st_val_loader = DataLoader(st_val, batch_size=64, shuffle=False)
    st_test_loader = DataLoader(st_test, batch_size=64, shuffle=False)

    if os.path.exists(st_ckpt) and not args.retrain_vkorc1_only:
        print(f"\nLoading frozen VKORC1-only single-task GAT from {st_ckpt}...")
        st_model = DynamicMultiTaskGNN(
            config=GNN_CONFIG, num_node_features=10, num_tasks=1
        ).to(device)
        st_model.load_state_dict(torch.load(st_ckpt, map_location=device, weights_only=True))
    else:
        print("\nTraining VKORC1-only single-task GAT baseline...")
        st_model = train_single_task_vkorc1(
            st_train_loader, st_val_loader, device, checkpoint_path=st_ckpt
        )
    st_per_task = per_task_metrics(st_model, st_test_loader, device, num_tasks=1)
    st_global = global_metrics_from_loader(st_model, st_test_loader, device)
    print_metrics_table("VKORC1-only single-task GAT", st_per_task)
    print(
        f"  Global (VKORC1 test): RMSE={st_global['rmse']:.3f}, "
        f"MAE={st_global['mae']:.3f}, R²={st_global['r2']:.3f}"
    )

    rl_smiles = load_rl_gen_canonical_smiles()
    label_audit = audit_vkorc1_label_sources(train_df, test_df, rl_smiles)
    chembl_only_gat = vkorc1_chembl_only_test_metrics(mt_model, test_data, rl_smiles, device)
    chembl_only_rf = vkorc1_chembl_only_rf_metrics(train_df, test_df, rl_smiles)
    print(
        f"\nVKORC1 ChEMBL-only test (excl. RL pseudo-labels): "
        f"GAT n={chembl_only_gat['n']} R²={chembl_only_gat['r2']:.3f} "
        f"ρ={chembl_only_gat.get('spearman_rho', float('nan')):.3f}; "
        f"RF R²={chembl_only_rf['r2']:.3f} ρ={chembl_only_rf.get('spearman_rho', float('nan')):.3f}"
    )

    try:
        report = to_jsonable(
        {
            "split_sizes": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
            "split_from": split_from,
            "label_counts": {
                "train": label_counts(train_df),
                "val": label_counts(val_df),
                "test": label_counts(test_df),
            },
            "vkorc1_label_audit": label_audit,
            "vkorc1_chembl_only_test": {
                "description": "VKORC1 test compounds excluding RL Vina pseudo-label SMILES",
                "multitask_gat": chembl_only_gat,
                "morgan_rf": chembl_only_rf,
            },
            "multitask_gnn": {"global": mt_global, "per_task": mt_per_task},
            "morgan_rf": {"per_task": rf_metrics},
            "vkorc1_single_task_gnn": {
                "global": st_global,
                "per_task": st_per_task,
            },
            "per_task_spearman": {
                "multitask_gat": gat_spearman,
                "morgan_rf": {col: rf_metrics[col].get("spearman_rho", float("nan")) for col in TARGET_COLS},
            },
            "vkorc1_spearman": {
                "multitask_gat": gat_spearman.get("VKORC1_pXC50", float("nan")),
                "morgan_rf": rf_metrics.get("VKORC1_pXC50", {}).get("spearman_rho", float("nan")),
                "vkorc1_only_gat": spearman_for_task(st_model, st_test_loader, device, task_idx=0),
            },
        }
    )
    except Exception as exc:
        import traceback
        print(f"Report build failed: {exc}")
        traceback.print_exc()
        raise

    out_path = "results/gnn_evaluation_report.json"
    pub_path = "publication/data/gnn_evaluation_report.json"
    os.makedirs("results", exist_ok=True)
    os.makedirs("publication/data", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(pub_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    audit_path = "publication/data/gnn_vkorc1_label_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(label_audit), f, indent=2)
    chembl_path = "publication/data/gnn_vkorc1_chembl_only_benchmark.json"
    with open(chembl_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(report["vkorc1_chembl_only_test"]), f, indent=2)
    print(f"\nSaved report to {out_path} and {pub_path}")
    print(f"Saved VKORC1 label audit to {audit_path}")
    print(f"Saved ChEMBL-only VKORC1 benchmark to {chembl_path}")


if __name__ == "__main__":
    main()
