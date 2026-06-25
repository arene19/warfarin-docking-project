#!/usr/bin/env python3
"""REINVENT4 ExternalProcess endpoint for multi-task GNN VKORC1 scoring."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

import torch
from rdkit import RDLogger

from gnn_model import load_checkpoint, smiles_to_inference_graph

RDLogger.DisableLog("rdApp.*")


def main() -> None:
    parser = argparse.ArgumentParser(description="REINVENT4 ExternalProcess GNN scoring endpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument(
        "--target", default="Pred_VKORC1_pXC50", help="JSON payload key for returned scores."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    lines = [line.strip() for line in sys.stdin.read().splitlines()]
    smiles_list = [line for line in lines if line]
    if smiles_list and smiles_list[0].lower() == "smiles":
        smiles_list = smiles_list[1:]

    scores: List[float] = []
    for smiles in smiles_list:
        graph_data = smiles_to_inference_graph(smiles) if isinstance(smiles, str) else None
        if graph_data is None:
            scores.append(float("nan"))
            continue

        graph_data = graph_data.to(device)
        graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            preds = model(graph_data).cpu().numpy().flatten()
        scores.append(float(preds[0]))

    output = {"version": 1, "payload": {args.target: scores}}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
