#morgan_fp_baseline.py
"""Morgan fingerprint (ECFP4) + Random Forest baseline for multi-task pXC50 prediction."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gnn_model import generate_scaffold, load_multitask_data, resolve_scaffold_split, TARGET_COLS
TASK_SHORT = ["VKORC1", "Factor XIIa", "Factor Xa", "Thrombin", "CYP2C9", "HSA"]

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
RF_KWARGS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


def smiles_to_morgan(
    smiles: str, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_BITS
) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_fingerprint_matrix(smiles_list: List[str]) -> Tuple[np.ndarray, List[int]]:
    """Return (n_valid x n_bits) matrix and indices of valid rows in smiles_list."""
    rows, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_morgan(smi)
        if fp is not None:
            rows.append(fp)
            valid_idx.append(i)
    if not rows:
        return np.empty((0, MORGAN_BITS)), []
    return np.vstack(rows), valid_idx


def data_list_to_frame(data_list) -> pd.DataFrame:
    rows = []
    for d in data_list:
        row = {"canonical_smiles": d.smiles}
        y = d.y.squeeze(0).tolist()
        m = d.mask.squeeze(0).tolist()
        for i, col in enumerate(TARGET_COLS):
            row[col] = y[i] if m[i] == 1.0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, RandomForestRegressor]]:
    """Train one RF per task; return metrics dict and fitted models."""
    metrics: Dict[str, Dict[str, float]] = {}
    models: Dict[str, RandomForestRegressor] = {}

    for col, short in zip(TARGET_COLS, TASK_SHORT):
        tr = train_df.dropna(subset=[col])
        te = test_df.dropna(subset=[col])
        if len(tr) < 5 or len(te) == 0:
            metrics[col] = {
                "task": short,
                "n_test": len(te),
                "rmse": float("nan"),
                "mae": float("nan"),
                "r2": float("nan"),
                "spearman_rho": float("nan"),
            }
            continue

        x_train, tr_idx = build_fingerprint_matrix(tr["canonical_smiles"].tolist())
        x_test, te_idx = build_fingerprint_matrix(te["canonical_smiles"].tolist())
        y_train = tr.iloc[tr_idx][col].values
        y_test = te.iloc[te_idx][col].values

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

        metrics[col] = {
            "task": short,
            "n_test": int(len(y_test)),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2),
            "spearman_rho": rho,
        }
        models[col] = rf

    return metrics, models


def collect_test_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models: Dict[str, RandomForestRegressor],
) -> Dict[str, Dict[str, list]]:
    """Return per-task test labels and predictions for plotting."""
    train_scaffolds = {
        generate_scaffold(s) for s in train_df["canonical_smiles"] if generate_scaffold(s)
    }
    out: Dict[str, Dict[str, list]] = {}
    for col, short in zip(TARGET_COLS, TASK_SHORT):
        if col not in models:
            continue
        te = test_df.dropna(subset=[col])
        if te.empty:
            continue
        x_test, te_idx = build_fingerprint_matrix(te["canonical_smiles"].tolist())
        y_test = te.iloc[te_idx][col].values
        pred = models[col].predict(x_test)
        smiles = te.iloc[te_idx]["canonical_smiles"].tolist()
        scaffolds = [generate_scaffold(s) for s in smiles]
        novel = [s not in train_scaffolds for s in scaffolds]
        out[col] = {
            "task": short,
            "y_true": y_test.tolist(),
            "y_pred": pred.tolist(),
            "smiles": smiles,
            "scaffold_novel": novel,
        }
    return out


def scaffold_split_summary(dataset, split_from: Optional[str] = None) -> pd.DataFrame:
    train, val, test, _ = resolve_scaffold_split(dataset, split_from=split_from)
    splits = {"Train": train, "Validation": val, "Test": test}
    rows = []
    for name, data_list in splits.items():
        scaffolds = [generate_scaffold(d.smiles) for d in data_list]
        rows.append(
            {
                "Split": name,
                "Compounds": len(data_list),
                "Unique_scaffolds": len(set(scaffolds)),
            }
        )
    return pd.DataFrame(rows)


def load_or_train(
    csv_path: str = "data/coagulation_admet_multi_task.csv",
    model_dir: str | Path = "publication/data/morgan_rf",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, RandomForestRegressor], Dict]:
    """Load saved Morgan RF models or train fresh; return frames, metrics, models, predictions."""
    model_dir = Path(model_dir)
    dataset = load_multitask_data(csv_path)
    split_path = Path("publication/data/gnn_scaffold_split.json")
    split_from = str(split_path) if split_path.exists() else None
    train_data, _, test_data, _ = resolve_scaffold_split(dataset, split_from=split_from)
    train_df = data_list_to_frame(train_data)
    test_df = data_list_to_frame(test_data)

    model_path = model_dir / "morgan_rf_models.joblib"
    if model_path.exists():
        models = joblib.load(model_path)
        metrics_path = model_dir / "morgan_rf_metrics.json"
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)["per_task"]
    else:
        metrics, models = train_and_evaluate(train_df, test_df)
        save_artifacts(metrics, models, model_dir)

    preds = collect_test_predictions(train_df, test_df, models)
    pred_path = model_dir / "morgan_rf_test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

    split_df = scaffold_split_summary(dataset, split_from=split_from)
    return train_df, test_df, metrics, models, {"predictions": preds, "split": split_df}


def save_artifacts(
    metrics: Dict[str, Dict[str, float]],
    models: Dict[str, RandomForestRegressor],
    out_dir: str | Path = "publication/data/morgan_rf",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "morgan_rf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "fingerprint": {
                    "type": "Morgan",
                    "radius": MORGAN_RADIUS,
                    "n_bits": MORGAN_BITS,
                },
                "model": "RandomForestRegressor",
                "hyperparameters": RF_KWARGS,
                "per_task": metrics,
            },
            f,
            indent=2,
        )

    joblib.dump(models, out / "morgan_rf_models.joblib")
    print(f"Saved Morgan RF models → {out / 'morgan_rf_models.joblib'}")
    print(f"Saved metrics → {out / 'morgan_rf_metrics.json'}")


def main() -> None:
    csv_path = os.path.join("data", "coagulation_admet_multi_task.csv")
    dataset = load_multitask_data(csv_path)
    split_path = Path("publication/data/gnn_scaffold_split.json")
    split_from = str(split_path) if split_path.exists() else None
    train_data, _, test_data, _ = resolve_scaffold_split(dataset, split_from=split_from)
    train_df = data_list_to_frame(train_data)
    test_df = data_list_to_frame(test_data)

    print("Training Morgan FP + Random Forest baselines (ECFP4, radius=2, 2048 bits)...")
    metrics, models = train_and_evaluate(train_df, test_df)
    save_artifacts(metrics, models)
    preds = collect_test_predictions(train_df, test_df, models)
    with open("publication/data/morgan_rf/morgan_rf_test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

    print(f"\n{'Task':<14} {'N':>5} {'RMSE':>8} {'R²':>8} {'Spearman':>10}")
    print("-" * 50)
    for col in TARGET_COLS:
        s = metrics[col]
        print(
            f"{s['task']:<14} {s['n_test']:>5} {s['rmse']:>8.3f} "
            f"{s['r2']:>8.3f} {s.get('spearman_rho', float('nan')):>10.3f}"
        )


if __name__ == "__main__":
    main()
