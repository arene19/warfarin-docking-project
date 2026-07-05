#!/usr/bin/env python3
"""Freeze the master multi-task CSV for publication builds and deposition."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIVE_MASTER = ROOT / "data/coagulation_admet_multi_task.csv"
FROZEN_MASTER = ROOT / "publication/data/coagulation_admet_multi_task_publication.csv"
SPLIT_PATH = ROOT / "publication/data/gnn_scaffold_split.json"
META_PATH = ROOT / "publication/data/master_snapshot_meta.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not LIVE_MASTER.exists():
        raise FileNotFoundError(f"Live master not found: {LIVE_MASTER}")
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Frozen split required: {SPLIT_PATH}")

    df = pd.read_csv(LIVE_MASTER)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split_smiles = set(split.get("train_smiles", [])) | set(split.get("val_smiles", [])) | set(
        split.get("test_smiles", [])
    )
    in_split = df["canonical_smiles"].isin(split_smiles)
    vk = df["VKORC1_pXC50"].notna()

    FROZEN_MASTER.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FROZEN_MASTER, index=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(LIVE_MASTER.relative_to(ROOT)),
        "frozen_path": str(FROZEN_MASTER.relative_to(ROOT)),
        "sha256": _sha256(FROZEN_MASTER),
        "n_compounds_total": int(len(df)),
        "n_compounds_in_frozen_split": int(in_split.sum()),
        "n_compounds_outside_split": int((~in_split).sum()),
        "n_vkorc1_labels_total": int(vk.sum()),
        "n_vkorc1_labels_in_split": int((vk & in_split).sum()),
        "n_vkorc1_labels_outside_split": int((vk & ~in_split).sum()),
        "split_sizes": {
            "train": int(split.get("train_n", 0)),
            "val": int(split.get("val_n", 0)),
            "test": int(split.get("test_n", 0)),
        },
        "note": (
            "GNN benchmarks (Table 1, S5) use gnn_scaffold_split.json on the in-split partition only. "
            "Compounds outside the split are retained in the frozen master for active-learning "
            "transparency but are not assigned to train/val/test benchmarks."
        ),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Frozen {len(df)} compounds -> {FROZEN_MASTER}")
    print(f"  in split: {meta['n_compounds_in_frozen_split']}, outside: {meta['n_compounds_outside_split']}")
    print(f"  VKORC1 labels: {meta['n_vkorc1_labels_total']} "
          f"({meta['n_vkorc1_labels_in_split']} in split, {meta['n_vkorc1_labels_outside_split']} outside)")
    print(f"  metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
