#!/usr/bin/env python3
"""Train the multi-task GAT on coagulation/ADMET labels with Murcko scaffold split."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import torch
from torch_geometric.loader import DataLoader

from gnn_model import (
    DEFAULT_GNN_CONFIG,
    DEFAULT_SEED,
    DynamicMultiTaskGNN,
    inference_loop,
    load_checkpoint,
    load_multitask_data,
    save_checkpoint,
    save_split_metadata,
    resolve_scaffold_split,
    set_seed,
    train_loop,
    validation_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-task GNN for coagulation/ADMET.")
    p.add_argument("--data", default=os.path.join("data", "coagulation_admet_multi_task.csv"))
    p.add_argument("--checkpoint", default="coagulation_admet_gnn.pth")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--history", default=os.path.join("publication", "data", "gnn_training_history.json"))
    p.add_argument("--split-meta", default=os.path.join("publication", "data", "gnn_scaffold_split.json"))
    p.add_argument(
        "--split-from",
        default=os.path.join("publication", "data", "gnn_scaffold_split.json"),
        help="Load train/val/test SMILES from this JSON (default if file exists).",
    )
    p.add_argument(
        "--recompute-split",
        action="store_true",
        help="Ignore frozen split JSON and recompute Murcko partition.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware routed to: {device}")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Could not find {args.data}")

    dataset = load_multitask_data(args.data)
    if args.recompute_split:
        split_from = None
    else:
        split_from = args.split_from if args.split_from and os.path.exists(args.split_from) else None
    train_data, val_data, test_data, split_meta = resolve_scaffold_split(
        dataset, seed=args.seed, split_from=split_from
    )
    if split_from is None:
        split_meta["data_path"] = os.path.relpath(os.path.abspath(args.data), os.getcwd())
        split_meta["created_at"] = datetime.now(timezone.utc).isoformat()
        save_split_metadata(args.split_meta, split_meta)
    print(
        f"Dataset split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    config = DEFAULT_GNN_CONFIG.copy()
    model = DynamicMultiTaskGNN(config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    early_stopping_patience = 25
    warmup_epochs = 15
    best_val_loss = float("inf")
    epochs_no_improve = 0
    training_history = []

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_loop(model, train_loader, optimizer, device)
        val_loss = validation_loss(model, val_loader, device)
        val_rmse, val_mae, val_r2 = inference_loop(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_rmse": float(val_rmse),
                "val_mae": float(val_mae),
                "val_r2": float(val_r2),
                "learning_rate": float(lr),
            }
        )

        prev_lr = lr
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"Epoch {epoch:03d} | Learning rate reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        if epoch > warmup_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(
                    args.checkpoint,
                    model,
                    config=config,
                    metadata={
                        "seed": args.seed,
                        "best_val_loss": float(best_val_loss),
                        "epoch": epoch,
                        "data_path": os.path.relpath(os.path.abspath(args.data), os.getcwd()),
                        "split_from": split_from or args.split_meta,
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(
                    f"Epoch {epoch:03d} | New best Val Loss: {best_val_loss:.4f} "
                    f"-> checkpoint saved to {args.checkpoint}"
                )
            else:
                epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            tag = " [warmup]" if epoch <= warmup_epochs else ""
            print(
                f"Epoch {epoch:03d}{tag} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val R2: {val_r2:.4f}"
            )

        if epoch > warmup_epochs and epochs_no_improve >= early_stopping_patience:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {early_stopping_patience} epochs)."
            )
            break

    if os.path.exists(args.checkpoint):
        model, _ = load_checkpoint(args.checkpoint, device)
        print(f"\nLoaded best model (Val Loss: {best_val_loss:.4f}) for final evaluation.")

    test_rmse, test_mae, test_r2 = inference_loop(model, test_loader, device)
    print("\n--- Final Test Set Evaluation (Valid Assays Only) ---")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test R2:   {test_r2:.4f}")

    os.makedirs(os.path.dirname(args.history), exist_ok=True)
    with open(args.history, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "best_val_loss": float(best_val_loss),
                "test_rmse": float(test_rmse),
                "test_mae": float(test_mae),
                "test_r2": float(test_r2),
                "note": "Pooled masked metrics; per-task metrics from gnn_baseline_evaluation.py",
                "epochs": training_history,
            },
            f,
            indent=2,
        )
    print(f"Training history saved to {args.history}")


if __name__ == "__main__":
    main()
