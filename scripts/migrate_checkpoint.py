#!/usr/bin/env python3
"""Wrap legacy weights-only checkpoint in v2 format with publication metadata."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gnn_model import DEFAULT_SEED, migrate_legacy_checkpoint, save_checkpoint, load_checkpoint


def publication_metadata(existing: dict | None = None) -> dict:
    """Merge frozen-master / split provenance into checkpoint metadata."""
    meta = dict(existing or {})
    snap_path = ROOT / "publication/data/master_snapshot_meta.json"
    frozen = ROOT / "publication/data/coagulation_admet_multi_task_publication.csv"
    split_path = ROOT / "publication/data/gnn_scaffold_split.json"

    meta.update(
        {
            "stamped_at": datetime.now(timezone.utc).isoformat(),
            "seed": DEFAULT_SEED,
            "benchmark_data_path": str(frozen.relative_to(ROOT)),
            "live_data_path": "data/coagulation_admet_multi_task.csv",
            "split_from": str(split_path.relative_to(ROOT)),
            "note": (
                "Weights trained on live master (June 2026); benchmarks regenerated with "
                "frozen publication master + gnn_scaffold_split.json (Table 1/S5)."
            ),
        }
    )
    if snap_path.exists():
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        meta["publication_master_sha256"] = snap.get("sha256")
        meta["n_compounds_total"] = snap.get("n_compounds_total")
        meta["n_compounds_in_frozen_split"] = snap.get("n_compounds_in_frozen_split")
        meta["n_compounds_outside_split"] = snap.get("n_compounds_outside_split")
    return meta


def stamp_existing(ckpt_path: Path, force: bool = False) -> bool:
    """Update metadata on an existing v2 checkpoint; returns True if written."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or raw.get("format_version") != 2:
        return False

    old_meta = raw.get("metadata") or {}
    new_meta = publication_metadata(old_meta)
    if not force and old_meta.get("publication_master_sha256") == new_meta.get("publication_master_sha256"):
        if old_meta.get("split_from") == new_meta.get("split_from"):
            print(f"Checkpoint metadata current ({ckpt_path}); skipping stamp.")
            return False

    import torch as th

    model, _ = load_checkpoint(ckpt_path, th.device("cpu"))
    save_checkpoint(
        ckpt_path,
        model,
        config=raw.get("config"),
        metadata=new_meta,
    )
    print(f"Stamped publication metadata -> {ckpt_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="coagulation_admet_gnn.pth")
    p.add_argument("--output", default="coagulation_admet_gnn.pth")
    p.add_argument("--force", action="store_true", help="Re-wrap legacy or re-stamp metadata")
    args = p.parse_args()

    ckpt_path = ROOT / args.input
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and raw.get("format_version") == 2:
        if args.force or not (raw.get("metadata") or {}).get("publication_master_sha256"):
            stamp_existing(ckpt_path, force=args.force)
        else:
            stamp_existing(ckpt_path, force=False)
        return

    metadata = publication_metadata(
        {
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "legacy_state_dict_only": True,
        }
    )
    out = migrate_legacy_checkpoint(args.input, args.output, metadata=metadata)
    print(f"Migrated checkpoint -> {out}")


if __name__ == "__main__":
    main()
