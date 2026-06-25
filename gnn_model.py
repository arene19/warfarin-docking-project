"""
Shared multi-task GNN model, featurization, data loading, and checkpoint I/O.

Used by dynamic_gnn_pipeline.py, gnn_predict.py, gnn_baseline_evaluation.py,
morgan_fp_baseline.py, and screening utilities.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, global_add_pool, global_mean_pool

TARGET_COLS = [
    "VKORC1_pXC50",
    "Factor_XIIa_pXC50",
    "Factor_Xa_pXC50",
    "Thrombin_pXC50",
    "CYP2C9_pXC50",
    "HSA_pXC50",
]

TASK_SHORT = ["VKORC1", "Factor XIIa", "Factor Xa", "Thrombin", "CYP2C9", "HSA"]

DEFAULT_GNN_CONFIG: Dict[str, Any] = {
    "conv_type": "GAT",
    "hidden_dims": [64, 128, 128, 64],
    "dropout": 0.1,
    "pool_type": "mean",
    "mlp_hidden": 128,
}

DEFAULT_SEED = 42
NUM_NODE_FEATURES = 10
NUM_TASKS = 6


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atom_features(atom: Chem.Atom) -> List[float]:
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        float(atom.GetHybridization().real),
        int(atom.GetIsAromatic()),
        atom.GetMass(),
        atom.GetValence(Chem.ValenceType.IMPLICIT),
        int(atom.GetValence(Chem.ValenceType.EXPLICIT)),
        int(atom.GetTotalNumHs()),
        int(atom.IsInRing()),
    ]


def smiles_to_graph(smiles: str, y_vals: List[float], mask_vals: List[float]) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_indices: List[List[int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor([y_vals], dtype=torch.float)
    mask = torch.tensor([mask_vals], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y, mask=mask, smiles=smiles)


def smiles_to_inference_graph(smiles: str) -> Optional[Data]:
    return smiles_to_graph(smiles, [0.0] * NUM_TASKS, [0.0] * NUM_TASKS)


def load_multitask_data(csv_path: str) -> List[Data]:
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    dataset: List[Data] = []
    valid_count = 0

    for _, row in df.iterrows():
        smiles = row["canonical_smiles"]
        y_vals: List[float] = []
        mask_vals: List[float] = []
        for col in TARGET_COLS:
            val = row[col]
            if pd.isna(val):
                y_vals.append(0.0)
                mask_vals.append(0.0)
            else:
                y_vals.append(float(val))
                mask_vals.append(1.0)

        graph_data = smiles_to_graph(smiles, y_vals, mask_vals)
        if graph_data is not None:
            dataset.append(graph_data)
            valid_count += 1

    print(f"Successfully featurized {valid_count} molecules.")
    return dataset


def generate_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        Chem.RemoveStereochemistry(mol)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(
    dataset: List[Data],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Data], List[Data], List[Data], Dict[str, Any]]:
    print("Performing Bemis-Murcko Scaffold Split...")
    scaffolds: Dict[str, List[Data]] = {}
    for data in dataset:
        scaffold = generate_scaffold(data.smiles)
        scaffolds.setdefault(scaffold, []).append(data)

    sorted_scaffolds = sorted(scaffolds.values(), key=len, reverse=True)
    train_cutoff = int(frac_train * len(dataset))
    val_cutoff = int((frac_train + frac_val) * len(dataset))

    train_set: List[Data] = []
    val_set: List[Data] = []
    test_set: List[Data] = []
    for group in sorted_scaffolds:
        if len(train_set) < train_cutoff:
            train_set.extend(group)
        elif len(train_set) + len(val_set) < val_cutoff:
            val_set.extend(group)
        else:
            test_set.extend(group)

    if not train_set or not val_set or not test_set:
        print("Warning: Scaffold split yielded empty sets. Falling back to seeded random split.")
        rng = random.Random(seed)
        shuffled = dataset.copy()
        rng.shuffle(shuffled)
        train_set = shuffled[:train_cutoff]
        val_set = shuffled[train_cutoff:val_cutoff]
        test_set = shuffled[val_cutoff:]

    split_meta = {
        "seed": seed,
        "frac_train": frac_train,
        "frac_val": frac_val,
        "train_smiles": [d.smiles for d in train_set],
        "val_smiles": [d.smiles for d in val_set],
        "test_smiles": [d.smiles for d in test_set],
        "train_n": len(train_set),
        "val_n": len(val_set),
        "test_n": len(test_set),
    }
    return train_set, val_set, test_set, split_meta


class DynamicMultiTaskGNN(nn.Module):
    """Graph attention network for masked multi-task pXC50 regression."""

    def __init__(
        self,
        config: Dict[str, Any],
        num_node_features: int = NUM_NODE_FEATURES,
        num_tasks: int = NUM_TASKS,
    ):
        super().__init__()
        self.config = config
        self.convs = nn.ModuleList()

        layer_dims = [num_node_features] + config.get("hidden_dims", [64, 128, 128, 64])
        conv_type = config.get("conv_type", "GAT")

        for i in range(len(layer_dims) - 1):
            if conv_type == "GCN":
                self.convs.append(GCNConv(layer_dims[i], layer_dims[i + 1]))
            elif conv_type == "GAT":
                self.convs.append(GATConv(layer_dims[i], layer_dims[i + 1]))
            else:
                raise ValueError("Unsupported conv_type. Use GCN or GAT.")

        self.dropout = config.get("dropout", DEFAULT_GNN_CONFIG["dropout"])
        self.pool_type = config.get("pool_type", "mean")
        mlp_hidden = config.get("mlp_hidden", 128)
        self.fc1 = nn.Linear(layer_dims[-1], mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, num_tasks)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pool_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool_type == "add":
            x = global_add_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    masked_sq_diff = (diff**2) * mask
    mask_sum = mask.sum()
    if mask_sum > 0:
        return masked_sq_diff.sum() / mask_sum
    return torch.tensor(0.0, device=pred.device, requires_grad=True)


def train_loop(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        y = batch.y.view(out.shape)
        mask = batch.mask.view(out.shape)
        loss = masked_mse_loss(out, y, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def validation_loss(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        y = batch.y.view(out.shape)
        mask = batch.mask.view(out.shape)
        loss = masked_mse_loss(out, y, mask)
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    if total_graphs == 0:
        return 0.0
    return total_loss / total_graphs


@torch.no_grad()
def inference_loop(
    model: nn.Module, loader, device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    y_true_all, y_pred_all, mask_all = [], [], []

    if len(loader.dataset) == 0:
        return 0.0, 0.0, 0.0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        y_true_all.append(batch.y.view(out.shape).cpu().numpy())
        y_pred_all.append(out.cpu().numpy())
        mask_all.append(batch.mask.view(out.shape).cpu().numpy())

    y_true_flat = np.concatenate(y_true_all).flatten()
    y_pred_flat = np.concatenate(y_pred_all).flatten()
    mask_flat = np.concatenate(mask_all).flatten()
    valid_indices = mask_flat == 1.0
    y_true_valid = y_true_flat[valid_indices]
    y_pred_valid = y_pred_flat[valid_indices]

    if len(y_true_valid) == 0:
        return 0.0, 0.0, 0.0

    mse = mean_squared_error(y_true_valid, y_pred_valid)
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    try:
        r2 = r2_score(y_true_valid, y_pred_valid)
    except Exception:
        r2 = 0.0
    return float(np.sqrt(mse)), float(mae), float(r2)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "format_version": 2,
        "state_dict": model.state_dict(),
        "config": config or DEFAULT_GNN_CONFIG.copy(),
        "target_cols": TARGET_COLS,
        "num_node_features": NUM_NODE_FEATURES,
        "num_tasks": NUM_TASKS,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    num_tasks: int = NUM_TASKS,
) -> Tuple[DynamicMultiTaskGNN, Dict[str, Any]]:
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        config = raw.get("config", DEFAULT_GNN_CONFIG.copy())
        metadata = raw.get("metadata", {})
        model = DynamicMultiTaskGNN(
            config=config,
            num_node_features=raw.get("num_node_features", NUM_NODE_FEATURES),
            num_tasks=raw.get("num_tasks", num_tasks),
        ).to(device)
        model.load_state_dict(raw["state_dict"])
        return model, metadata

    config = DEFAULT_GNN_CONFIG.copy()
    model = DynamicMultiTaskGNN(
        config=config, num_node_features=NUM_NODE_FEATURES, num_tasks=num_tasks
    ).to(device)
    model.load_state_dict(raw)
    return model, {"legacy_state_dict_only": True}


def migrate_legacy_checkpoint(
    legacy_path: str | Path,
    output_path: Optional[str | Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Wrap a weights-only .pth file in the v2 checkpoint format."""
    device = torch.device("cpu")
    model, _ = load_checkpoint(legacy_path, device)
    out = Path(output_path or legacy_path)
    save_checkpoint(out, model, config=DEFAULT_GNN_CONFIG.copy(), metadata=metadata or {})
    return out


def save_split_metadata(path: str | Path, split_meta: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
