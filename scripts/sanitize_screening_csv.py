#!/usr/bin/env python3
"""Sanitize screening CSV all_affinities column (remove np.float64 repr strings)."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv"


def clean_affinities_cell(cell: object) -> str:
    if pd.isna(cell):
        return "[]"
    text = str(cell)
    if "np.float64" in text:
        text = text.replace("np.float64(", "").replace(")", "")
    try:
        vals = ast.literal_eval(text)
        return json.dumps([float(v) for v in vals])
    except (ValueError, SyntaxError, TypeError):
        nums = re.findall(r"-?\d+\.?\d*", text)
        return json.dumps([float(n) for n in nums]) if nums else text


def sanitize(path: Path, inplace: bool = True) -> int:
    df = pd.read_csv(path)
    if "all_affinities" not in df.columns:
        return 0
    before = df["all_affinities"].astype(str).str.contains("np.float64", na=False).sum()
    df["all_affinities"] = df["all_affinities"].map(clean_affinities_cell)
    out = path if inplace else path.with_suffix(".sanitized.csv")
    df.to_csv(out, index=False)
    return int(before)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize screening CSV all_affinities column.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT), help="Path to screening CSV")
    parser.add_argument("--no-inplace", action="store_true", help="Write .sanitized.csv sibling")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)
    n = sanitize(path, inplace=not args.no_inplace)
    print(f"Sanitized {path} ({n} np.float64 rows cleaned)")


if __name__ == "__main__":
    main()
