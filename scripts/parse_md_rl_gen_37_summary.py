#!/usr/bin/env python3
"""Backward-compatible wrapper: refresh RL_Gen_37 stub via parse_md_results_summary.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    script = ROOT / "scripts/parse_md_results_summary.py"
    subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
