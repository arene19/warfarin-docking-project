#!/usr/bin/env python3
"""Validate REINVENT config vs archived generation CSVs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    prov_path = ROOT / "publication/data/reinvent_provenance.json"
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)

    ok = True
    canonical = ROOT / prov["canonical_generation_csv"]
    if not canonical.exists():
        fallback = ROOT / "deposition/package/reinvent/coumarin_generation_1.csv"
        if fallback.exists():
            canonical = fallback
            print(f"Using deposition fallback CSV: {fallback}")
        else:
            print(f"MISSING canonical CSV: {ROOT / prov['canonical_generation_csv']}")
            ok = False
    if canonical.exists():
        lines = canonical.read_text(encoding="utf-8").splitlines()
        header = lines[0] if lines else ""
        row_count = max(0, len(lines) - 1)
        has_gnn = "Pred_VKORC1_pXC50" in header
        digest = sha256(canonical)
        print(f"Canonical CSV: {canonical} — rows={row_count}, Pred_VKORC1_pXC50={has_gnn}, sha256={digest[:16]}…")
        ok = ok and has_gnn and row_count > 0

        dep_copy = ROOT / "deposition/package/reinvent/coumarin_generation_1.csv"
        if dep_copy.exists() and dep_copy.resolve() != canonical.resolve():
            if sha256(dep_copy) != digest:
                print("WARN: deposition/package REINVENT CSV checksum differs from canonical")
                ok = False
            else:
                print("Deposition REINVENT CSV checksum matches canonical")

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
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
