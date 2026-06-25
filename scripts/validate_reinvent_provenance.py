#!/usr/bin/env python3
"""Validate REINVENT config vs archived generation CSVs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    prov_path = ROOT / "publication/data/reinvent_provenance.json"
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)

    ok = True
    canonical = ROOT / prov["canonical_generation_csv"]
    if not canonical.exists():
        print(f"MISSING canonical CSV: {canonical}")
        ok = False
    else:
        header = canonical.read_text(encoding="utf-8").splitlines()[0]
        has_gnn = "Pred_VKORC1_pXC50" in header
        print(f"Canonical CSV: {canonical} — Pred_VKORC1_pXC50: {has_gnn}")
        ok = ok and has_gnn

    toml = ROOT / prov["canonical_reinvent_config"]
    if toml.exists():
        text = toml.read_text(encoding="utf-8")
        print(f"Config TOML GNN block: {'ExternalProcess' in text and 'gnn_predict.py' in text}")
    else:
        print(f"MISSING config: {toml}")
        ok = False

    pilot = ROOT / "coumarin_rl.json"
    if pilot.exists():
        data = json.loads(pilot.read_text(encoding="utf-8"))
        comps = data.get("stage", [{}])[0].get("scoring", {}).get("component", [])
        has_ext = any("ExternalProcess" in str(c) for c in comps)
        print(f"Root coumarin_rl.json has GNN ExternalProcess: {has_ext} (expected False — pilot run)")

    print("VALIDATION:", "PASS" if ok else "FAIL — see messages above")


if __name__ == "__main__":
    main()
