#gnn_baseline_evaluation.py
"""Evaluate multi-task GNN (checkpoint), Morgan FP + RF, and VKORC1-only GNN baselines."""
import json
import os
import random
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
    scaffold_split,
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


def global_masked_metrics(per_task: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Pool all valid task predictions for aggregate RMSE/MAE/R²."""
    yt, yp = [], []
    for stats in per_task.values():
        if stats["n"] == 0:
            continue
        # Reconstruct not stored — compute from per-task arrays in caller instead
        pass
    return {}


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
        }
        for col, v in raw.items()
    }


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
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csv_path = os.path.join("data", "coagulation_admet_multi_task.csv")
    dataset = load_multitask_data(csv_path)
    train_data, val_data, test_data, _ = scaffold_split(dataset, seed=42)
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
    print("\nTraining Morgan FP + Random Forest baselines...")
    rf_raw, rf_models = train_and_evaluate(train_df, test_df)
    save_artifacts(rf_raw, rf_models)
    rf_metrics = _rf_metrics_for_report(rf_raw)
    print_metrics_table("Morgan FP + Random Forest", rf_metrics)

    # --- VKORC1-only single-task GNN ---
    print("\nTraining VKORC1-only single-task GAT baseline...")
    st_train = vkorc1_only_graphs([d for d in train_data])
    st_val = vkorc1_only_graphs([d for d in val_data])
    st_test = vkorc1_only_graphs([d for d in test_data])
    st_train_loader = DataLoader(st_train, batch_size=64, shuffle=True)
    st_val_loader = DataLoader(st_val, batch_size=64, shuffle=False)
    st_test_loader = DataLoader(st_test, batch_size=64, shuffle=False)

    st_model = train_single_task_vkorc1(st_train_loader, st_val_loader, device)
    st_per_task = per_task_metrics(st_model, st_test_loader, device, num_tasks=1)
    st_global = global_metrics_from_loader(st_model, st_test_loader, device)
    print_metrics_table("VKORC1-only single-task GAT", st_per_task)
    print(
        f"  Global (VKORC1 test): RMSE={st_global['rmse']:.3f}, "
        f"MAE={st_global['mae']:.3f}, R²={st_global['r2']:.3f}"
    )

    try:
        report = to_jsonable(
        {
            "split_sizes": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
            "label_counts": {
                "train": label_counts(train_df),
                "val": label_counts(val_df),
                "test": label_counts(test_df),
            },
            "multitask_gnn": {"global": mt_global, "per_task": mt_per_task},
            "morgan_rf": {"per_task": rf_metrics},
            "vkorc1_single_task_gnn": {
                "global": st_global,
                "per_task": st_per_task,
            },
            "vkorc1_spearman": {
                "multitask_gat": spearman_for_task(mt_model, test_loader, device, task_idx=0),
                "morgan_rf": rf_raw.get("VKORC1_pXC50", {}).get("spearman_rho", float("nan")),
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
    print(f"\nSaved report to {out_path} and {pub_path}")


if __name__ == "__main__":
    main()
