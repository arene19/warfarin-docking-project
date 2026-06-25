#!/usr/bin/env python3
"""Wrap legacy weights-only checkpoint in v2 format with metadata."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gnn_model import DEFAULT_SEED, migrate_legacy_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="coagulation_admet_gnn.pth")
    p.add_argument("--output", default="coagulation_admet_gnn.pth")
    args = p.parse_args()

    metadata = {
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "seed": DEFAULT_SEED,
        "data_path": "data/coagulation_admet_multi_task.csv",
        "note": "Migrated from legacy state_dict-only checkpoint",
    }
    out = migrate_legacy_checkpoint(args.input, args.output, metadata=metadata)
    print(f"Migrated checkpoint -> {out}")


if __name__ == "__main__":
    main()
