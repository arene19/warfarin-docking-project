#!/usr/bin/env python3
"""Finalize gnn_evaluation_report.json with Spearman block (fast path)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gnn_baseline_evaluation import spearman_for_task, to_jsonable
from gnn_model import load_checkpoint, load_multitask_data, scaffold_split


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_multitask_data(str(ROOT / "data/coagulation_admet_multi_task.csv"))
    _, _, test_data, _ = scaffold_split(dataset, seed=42)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    mt_model, _ = load_checkpoint(ROOT / "coagulation_admet_gnn.pth", device)
    mt_model.eval()

    pub_path = ROOT / "publication/data/gnn_evaluation_report.json"
    with open(pub_path, encoding="utf-8") as f:
        report = json.load(f)

    morgan_path = ROOT / "publication/data/morgan_rf/morgan_rf_metrics.json"
    morgan_rho = float("nan")
    if morgan_path.exists():
        with open(morgan_path, encoding="utf-8") as f:
            morgan_rho = (
                json.load(f).get("per_task", {}).get("VKORC1_pXC50", {}).get("spearman_rho", float("nan"))
            )

    st_rho = float("nan")
    st_ckpt = ROOT / "vkorc1_single_task_gnn.pth"
    if st_ckpt.exists():
        from gnn_baseline_evaluation import vkorc1_only_graphs, GNN_CONFIG, DynamicMultiTaskGNN

        st_model = DynamicMultiTaskGNN(config=GNN_CONFIG, num_node_features=10, num_tasks=1).to(device)
        st_model.load_state_dict(torch.load(st_ckpt, map_location=device, weights_only=False))
        st_model.eval()
        st_test = vkorc1_only_graphs(list(test_data))
        st_loader = DataLoader(st_test, batch_size=64, shuffle=False)
        st_rho = spearman_for_task(st_model, st_loader, device, task_idx=0)

    report["vkorc1_spearman"] = to_jsonable(
        {
            "multitask_gat": spearman_for_task(mt_model, test_loader, device, task_idx=0),
            "morgan_rf": morgan_rho,
            "vkorc1_only_gat": st_rho,
        }
    )

    report = to_jsonable(report)
    for path in (ROOT / "results/gnn_evaluation_report.json", pub_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(f"Updated {pub_path} with vkorc1_spearman: {report['vkorc1_spearman']}")


if __name__ == "__main__":
    main()
