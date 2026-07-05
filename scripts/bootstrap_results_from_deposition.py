#!/usr/bin/env python3
"""Restore repository inputs from a deposition package for asset regeneration on a fresh clone."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEP = ROOT / "deposition" / "package"


def copy_if(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def resolve_package_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        return p
    if (ROOT / "docking").exists() and (ROOT / "metrics").exists():
        return ROOT
    return DEFAULT_DEP


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap results/ and publication/data from deposition.")
    parser.add_argument(
        "--package-dir",
        default=None,
        help="Deposition root (default: deposition/package/ or cwd if unzipped Zenodo layout)",
    )
    args = parser.parse_args()

    dep = resolve_package_dir(args.package_dir)
    if not dep.exists():
        raise FileNotFoundError(
            f"Deposition package not found: {dep}. "
            "Run prepare_deposition.py or pass --package-dir to the unzipped Zenodo folder."
        )

    mappings = [
        (dep / "docking/VKORC1_Human_screening_results.csv", ROOT / "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv"),
        (dep / "docking/interaction_profile.csv", ROOT / "results/interaction_profile.csv"),
        (dep / "docking/admet_profile.csv", ROOT / "results/admet_profile.csv"),
        (dep / "docking/flexible_redock_spotcheck.csv", ROOT / "publication/data/flexible_redock_spotcheck.csv"),
        (dep / "docking/poses/VKORC1_Human", ROOT / "results/docked_poses/VKORC1_Human"),
        (dep / "reinvent/coumarin_generation_1.csv", ROOT / "REINVENT4/coumarin_generation_1.csv"),
        (dep / "data/coagulation_admet_multi_task.csv", ROOT / "data/coagulation_admet_multi_task.csv"),
        (dep / "models/coagulation_admet_gnn.pth", ROOT / "coagulation_admet_gnn.pth"),
        (dep / "models/vkorc1_single_task_gnn.pth", ROOT / "publication/data/vkorc1_single_task_gnn.pth"),
        (dep / "models/morgan_rf", ROOT / "publication/data/morgan_rf"),
        (dep / "metrics/gnn_evaluation_report.json", ROOT / "publication/data/gnn_evaluation_report.json"),
        (dep / "metrics/gnn_training_history.json", ROOT / "publication/data/gnn_training_history.json"),
        (dep / "metrics/gnn_scaffold_split.json", ROOT / "publication/data/gnn_scaffold_split.json"),
        (dep / "metrics/gnn_vkorc1_label_audit.json", ROOT / "publication/data/gnn_vkorc1_label_audit.json"),
        (dep / "metrics/gnn_vkorc1_chembl_only_benchmark.json", ROOT / "publication/data/gnn_vkorc1_chembl_only_benchmark.json"),
        (dep / "metrics/LABEL_PROVENANCE.md", ROOT / "publication/data/LABEL_PROVENANCE.md"),
        (dep / "data/ligands_sdf", ROOT / "results/ligands"),
        (dep / "config/config_master.yaml", ROOT / "config_master.yaml"),
        (dep / "scripts/gnn_model.py", ROOT / "gnn_model.py"),
        (dep / "scripts/config_utils.py", ROOT / "config_utils.py"),
        (dep / "scripts/morgan_fp_baseline.py", ROOT / "morgan_fp_baseline.py"),
        (dep / "scripts/gnn_baseline_evaluation.py", ROOT / "gnn_baseline_evaluation.py"),
        (dep / "scripts/gnn_predict.py", ROOT / "gnn_predict.py"),
    ]
    n = 0
    for src, dst in mappings:
        if copy_if(src, dst):
            print(f"  [ok] {dst.relative_to(ROOT)}")
            n += 1
        else:
            print(f"  [skip] missing {src.relative_to(dep)}")
    print(f"Bootstrap complete ({n} paths restored from {dep}).")


if __name__ == "__main__":
    main()
