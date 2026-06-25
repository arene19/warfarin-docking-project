#!/usr/bin/env python3
"""Bundle docking, interaction, and REINVENT outputs for Zenodo deposition."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deposition" / "package"


def copy_if_exists(src: Path, dst: Path) -> bool:
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


def main() -> None:
    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "missing": [],
    }

    candidates = [
        (ROOT / "data" / "coagulation_admet_multi_task.csv", OUT / "data/coagulation_admet_multi_task.csv"),
        (ROOT / "coagulation_admet_gnn.pth", OUT / "models/coagulation_admet_gnn.pth"),
        (ROOT / "publication/data/gnn_evaluation_report.json", OUT / "metrics/gnn_evaluation_report.json"),
        (ROOT / "publication/data/gnn_training_history.json", OUT / "metrics/gnn_training_history.json"),
        (ROOT / "publication/data/gnn_scaffold_split.json", OUT / "metrics/gnn_scaffold_split.json"),
        (ROOT / "publication/data/flexible_redock_spotcheck.csv", OUT / "docking/flexible_redock_spotcheck.csv"),
        (ROOT / "publication/data/reinvent_provenance.json", OUT / "reinvent/reinvent_provenance.json"),
        (ROOT / "configs/coumarin_rl.toml", OUT / "reinvent/coumarin_rl.toml"),
        (ROOT / "REINVENT4/coumarin_generation_1.csv", OUT / "reinvent/coumarin_generation_1.csv"),
        (
            ROOT / "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv",
            OUT / "docking/VKORC1_Human_screening_results.csv",
        ),
        (ROOT / "results/interaction_profile.csv", OUT / "docking/interaction_profile.csv"),
        (ROOT / "results/admet_profile.csv", OUT / "docking/admet_profile.csv"),
        (ROOT / "publication/output", OUT / "publication_output"),
    ]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for src, dst in candidates:
        rel = str(dst.relative_to(OUT))
        if copy_if_exists(src, dst):
            manifest["files"].append(rel)
        else:
            manifest["missing"].append(str(src.relative_to(ROOT)))

    readme = OUT / "README_DEPOSITION.txt"
    readme.write_text(
        "VKORC1 coumarin discovery — supplementary data package\n\n"
        "Upload this folder to Zenodo and cite the DOI in the manuscript "
        "Data Availability section.\n\n"
        f"Generated: {manifest['created_at']}\n"
        f"Included files: {len(manifest['files'])}\n"
        f"Missing (regenerate locally): {len(manifest['missing'])}\n",
        encoding="utf-8",
    )
    manifest["files"].append("README_DEPOSITION.txt")

    with open(OUT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(ROOT / "deposition" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Deposition package -> {OUT}")
    print(f"  included: {len(manifest['files'])} files")
    if manifest["missing"]:
        print(f"  missing: {len(manifest['missing'])} (see manifest.json)")


if __name__ == "__main__":
    main()
