#!/usr/bin/env python3
"""Score REINVENT-generated SMILES with the multi-task GNN and export top leads."""
from __future__ import annotations

import os

import pandas as pd
import torch
from rdkit import RDLogger

from gnn_model import load_checkpoint, smiles_to_inference_graph

RDLogger.DisableLog("rdApp.*")

TARGET_LABELS = ["VKORC1", "FXIIa", "FXa", "Thrombin", "CYP2C9", "HSA"]


def main() -> None:
    print("=" * 50)
    print("   AI LEAD FILTERING: GNN INFERENCE PIPELINE")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "coagulation_admet_gnn.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Run dynamic_gnn_pipeline.py first.")

    model, _ = load_checkpoint(model_path, device)
    model.eval()
    print("Trained GNN model loaded.")

    csv_path = os.path.join("data", "ai_generated_drugs.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df_ai = pd.read_csv(csv_path)
    smiles_col = "SMILES" if "SMILES" in df_ai.columns else "smiles"
    raw_smiles = df_ai[smiles_col].dropna().unique().tolist()
    generated_smiles = [s for s in raw_smiles if "*" not in s and "|" not in s]
    print(f"Loaded {len(generated_smiles)} clean AI molecules.")

    results = []
    for smiles in generated_smiles:
        graph_data = smiles_to_inference_graph(smiles)
        if graph_data is None:
            continue
        graph_data = graph_data.to(device)
        graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            preds = model(graph_data).cpu().numpy().flatten()
        record = {"SMILES": smiles}
        for label, score in zip(TARGET_LABELS, preds):
            record[f"Pred_{label}_pXC50"] = round(float(score), 3)
        results.append(record)

    df_results = pd.DataFrame(results).sort_values(by="Pred_VKORC1_pXC50", ascending=False)
    top_50 = df_results.head(50)
    out_path = os.path.join("data", "top_50_ai_leads.csv")
    top_50.to_csv(out_path, index=False)
    print(f"Top 50 leads saved to {out_path}")


if __name__ == "__main__":
    main()
