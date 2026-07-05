#!/usr/bin/env python3
"""Bundle docking, interaction, and REINVENT outputs for Zenodo deposition."""
from __future__ import annotations

import ast
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deposition" / "package"

CASE_STUDY_LIGANDS = [
    "RL_Gen_37",
    "RL_Gen_22",
    "RL_Gen_29",
    "RL_Gen_29_isoA",
    "RL_Gen_29_isoB",
    "RL_Gen_45",
    "RL_Gen_49",
    "RL_Gen_07",
    "RL_Gen_26",
    "RL_Gen_39",
]


def sanitize_screening_csv(src: Path, dst: Path) -> None:
    """Copy screening CSV with clean all_affinities (no np.float64 repr strings)."""
    import pandas as pd

    df = pd.read_csv(src)
    if "all_affinities" not in df.columns:
        shutil.copy2(src, dst)
        return

    def _clean(cell: object) -> str:
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

    df["all_affinities"] = df["all_affinities"].map(_clean)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


def prune_publication_tables(tables_dir: Path) -> None:
    """Keep only manuscript tables listed in publication/output/manifest.json."""
    manifest_path = ROOT / "publication/output/manifest.json"
    if not manifest_path.exists() or not tables_dir.exists():
        return
    allowed = set(json.loads(manifest_path.read_text(encoding="utf-8")).get("tables", []))
    for path in list(tables_dir.iterdir()):
        if path.is_file() and path.name not in allowed:
            path.unlink()
            print(f"  [prune] removed stale table {path.name}")


def copy_md_analysis_bundle(manifest: dict) -> None:
    """Copy MD analysis summaries and key .xvg (not multi-GB trajectories)."""
    md_systems = ["RL_Gen_37", "RL_Gen_29_isoA", "S_Warfarin_ref"]
    analysis_globs = (
        "hbond_summary.json",
        "hbond_residue_map.json",
        "rmsd_protein.xvg",
        "rmsd_ligand.xvg",
        "temperature.xvg",
        "pressure.xvg",
    )
    for sys_id in md_systems:
        src_dir = ROOT / "md_gromacs/runs" / sys_id / "analysis"
        if not src_dir.exists():
            manifest["missing"].append(f"md_gromacs/runs/{sys_id}/analysis")
            continue
        dst_dir = OUT / "md_analysis" / sys_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied_any = False
        for name in analysis_globs:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)
                manifest["files"].append(f"md_analysis/{sys_id}/{name}")
                copied_any = True
        if not copied_any:
            manifest["missing"].append(f"md_gromacs/runs/{sys_id}/analysis (no files)")


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


def copy_case_study_poses(manifest: dict) -> None:
    pose_src = ROOT / "results/docked_poses/VKORC1_Human"
    pose_dst = OUT / "docking/poses/VKORC1_Human"
    pose_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in CASE_STUDY_LIGANDS:
        matches = list(pose_src.glob(f"{name}_VKORC1_Human_docked.pdbqt"))
        for src in matches:
            dst = pose_dst / src.name
            shutil.copy2(src, dst)
            rel = str(dst.relative_to(OUT))
            if rel not in manifest["files"]:
                manifest["files"].append(rel)
            copied += 1
    if copied:
        print(f"  [ok] case-study docked poses: {copied} files")


def ensure_reinvent_canonical() -> None:
    """Ensure REINVENT4/coumarin_generation_1.csv exists for provenance validation."""
    canonical = ROOT / "REINVENT4" / "coumarin_generation_1.csv"
    if canonical.exists():
        return
    for fallback in (ROOT / "coumarin_generation_1.csv",):
        if fallback.exists():
            canonical.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback, canonical)
            print(f"  [ok] restored canonical REINVENT CSV → {canonical}")
            return
    print(
        "  [warn] REINVENT4/coumarin_generation_1.csv missing — "
        "run scripts/bootstrap_results_from_deposition.py or place CSV manually."
    )


def main() -> None:
    ensure_reinvent_canonical()

    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "missing": [],
    }

    candidates = [
        (ROOT / "publication/data/coagulation_admet_multi_task_publication.csv", OUT / "data/coagulation_admet_multi_task.csv"),
        (ROOT / "publication/data/master_snapshot_meta.json", OUT / "metrics/master_snapshot_meta.json"),
        (ROOT / "publication/data/MASTER_SNAPSHOT.md", OUT / "metrics/MASTER_SNAPSHOT.md"),
        (ROOT / "publication/data/md_rl_gen_37_summary.json", OUT / "metrics/md_rl_gen_37_summary.json"),
        (ROOT / "publication/data/md_results_summary.json", OUT / "metrics/md_results_summary.json"),
        (ROOT / "publication/data/md/MD_RESULTS_SUMMARY.json", OUT / "metrics/MD_RESULTS_SUMMARY.json"),
        (ROOT / "coagulation_admet_gnn.pth", OUT / "models/coagulation_admet_gnn.pth"),
        (ROOT / "publication/data/vkorc1_single_task_gnn.pth", OUT / "models/vkorc1_single_task_gnn.pth"),
        (ROOT / "publication/data/morgan_rf", OUT / "models/morgan_rf"),
        (ROOT / "publication/data/gnn_evaluation_report.json", OUT / "metrics/gnn_evaluation_report.json"),
        (ROOT / "publication/data/gnn_training_history.json", OUT / "metrics/gnn_training_history.json"),
        (ROOT / "publication/data/gnn_scaffold_split.json", OUT / "metrics/gnn_scaffold_split.json"),
        (ROOT / "publication/data/gnn_vkorc1_label_audit.json", OUT / "metrics/gnn_vkorc1_label_audit.json"),
        (ROOT / "publication/data/gnn_vkorc1_chembl_only_benchmark.json", OUT / "metrics/gnn_vkorc1_chembl_only_benchmark.json"),
        (ROOT / "publication/data/LABEL_PROVENANCE.md", OUT / "metrics/LABEL_PROVENANCE.md"),
        (ROOT / "publication/data/flexible_redock_spotcheck.csv", OUT / "docking/flexible_redock_spotcheck.csv"),
        (ROOT / "publication/data/reinvent_provenance.json", OUT / "reinvent/reinvent_provenance.json"),
        (ROOT / "configs/coumarin_rl.toml", OUT / "reinvent/coumarin_rl.toml"),
        (ROOT / "REINVENT4/coumarin_rl.toml", OUT / "reinvent/coumarin_rl_run.toml"),
        (ROOT / "REINVENT4/coumarin_generation_1.csv", OUT / "reinvent/coumarin_generation_1.csv"),
        (ROOT / "REINVENT4/coumarin_agent.chkpt", OUT / "reinvent/coumarin_agent.chkpt"),
        (ROOT / "config_master.yaml", OUT / "config/config_master.yaml"),
        (ROOT / "gnn_predict.py", OUT / "scripts/gnn_predict.py"),
        (ROOT / "gnn_model.py", OUT / "scripts/gnn_model.py"),
        (ROOT / "config_utils.py", OUT / "scripts/config_utils.py"),
        (ROOT / "morgan_fp_baseline.py", OUT / "scripts/morgan_fp_baseline.py"),
        (ROOT / "gnn_baseline_evaluation.py", OUT / "scripts/gnn_baseline_evaluation.py"),
        (ROOT / "active_learning_merger.py", OUT / "scripts/active_learning_merger.py"),
        (ROOT / "dynamic_gnn_pipeline.py", OUT / "scripts/dynamic_gnn_pipeline.py"),
        (ROOT / "docking_engine.py", OUT / "scripts/docking_engine.py"),
        (ROOT / "protein_preparation.py", OUT / "scripts/protein_preparation.py"),
        (ROOT / "main_pipeline.py", OUT / "scripts/main_pipeline.py"),
        (ROOT / "pipeline_utils.py", OUT / "scripts/pipeline_utils.py"),
        (ROOT / "grid_box.py", OUT / "scripts/grid_box.py"),
        (ROOT / "scripts/bootstrap_results_from_deposition.py", OUT / "scripts/bootstrap_results_from_deposition.py"),
        (ROOT / "scripts/regenerate_flex_redock_spotcheck.py", OUT / "scripts/regenerate_flex_redock_spotcheck.py"),
        (ROOT / "scripts/run_flex_redock_spotcheck.py", OUT / "scripts/run_flex_redock_spotcheck.py"),
        (ROOT / "scripts/parse_md_rl_gen_37_summary.py", OUT / "scripts/parse_md_rl_gen_37_summary.py"),
        (ROOT / "scripts/parse_md_results_summary.py", OUT / "scripts/parse_md_results_summary.py"),
        (ROOT / "scripts/sanitize_screening_csv.py", OUT / "scripts/sanitize_screening_csv.py"),
        (ROOT / "scripts/freeze_publication_master.py", OUT / "scripts/freeze_publication_master.py"),
        (ROOT / "docking_validation.py", OUT / "scripts/docking_validation.py"),
        (ROOT / "ligand_library.py", OUT / "scripts/ligand_library.py"),
        (ROOT / "sdf_to_pdbqt.py", OUT / "scripts/sdf_to_pdbqt.py"),
        (ROOT / "config.yaml", OUT / "config/config.yaml"),
        (ROOT / "ENVIRONMENT.md", OUT / "ENVIRONMENT.md"),
        (ROOT / "environment/reinvent4-environment.yml", OUT / "environment/reinvent4-environment.yml"),
        (ROOT / "publication/data/reinvent_prior_provenance.json", OUT / "reinvent/reinvent_prior_provenance.json"),
        (ROOT / "REPRODUCE_REINVENT.md", OUT / "REPRODUCE_REINVENT.md"),
        (ROOT / "RELEASE.md", OUT / "RELEASE.md"),
        (ROOT / "results/ligands", OUT / "data/ligands_sdf"),
        (
            ROOT / "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv",
            OUT / "docking/VKORC1_Human_screening_results.csv",
        ),
        (ROOT / "results/interaction_profile.csv", OUT / "docking/interaction_profile.csv"),
        (ROOT / "results/admet_profile.csv", OUT / "docking/admet_profile.csv"),
        (ROOT / "md_poses", OUT / "md_poses"),
        (ROOT / "publication/output/figures", OUT / "publication_output/figures"),
        (ROOT / "publication/output/tables", OUT / "publication_output/tables"),
        (ROOT / "vkorc1_integrated_workflow_manuscript.md", OUT / "manuscript/vkorc1_integrated_workflow_manuscript.md"),
        (ROOT / "publication/output/manifest.json", OUT / "publication_output/manifest.json"),
        (ROOT / "publication/output/internal_qc", OUT / "internal_qc"),
    ]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for src, dst in candidates:
        rel = str(dst.relative_to(OUT))
        if src == ROOT / "results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv":
            if src.exists():
                sanitize_screening_csv(src, dst)
                manifest["files"].append(rel)
            else:
                manifest["missing"].append(str(src.relative_to(ROOT)))
            continue
        if copy_if_exists(src, dst):
            manifest["files"].append(rel)
        else:
            manifest["missing"].append(str(src.relative_to(ROOT)))

    prune_publication_tables(OUT / "publication_output/tables")

    reproduce = OUT / "scripts/reproduce_docking.sh"
    reproduce.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")/..\"\n"
        "python scripts/main_pipeline.py --config config/config_master.yaml\n",
        encoding="utf-8",
    )
    reproduce.chmod(0o755)
    manifest["files"].append("scripts/reproduce_docking.sh")

    copy_case_study_poses(manifest)
    copy_md_analysis_bundle(manifest)

    readme = OUT / "README_DEPOSITION.txt"
    readme.write_text(
        "VKORC1 de novo ligand discovery — supplementary data package\n\n"
        "Upload this folder to Zenodo and cite the DOI in the manuscript "
        "Data Availability section.\n\n"
        "Manuscript-facing assets: publication_output/figures and publication_output/tables.\n"
        "Reference-ligand QC tables: internal_qc/ (not cited in manuscript).\n"
        "Case-study docked poses: docking/poses/VKORC1_Human/ (top RL_Gen ligands).\n"
        "RL ligand structures: data/ligands_sdf/; config: config/config_master.yaml.\n"
        "Pipeline scripts: scripts/ (gnn_predict.py, active_learning_merger.py, "
        "main_pipeline.py, docking_engine.py, bootstrap_results_from_deposition.py).\n"
        "Minimal docking reproduction: bash scripts/reproduce_docking.sh (requires Vina, Meeko, RDKit).\n"
        "REINVENT: reinvent/coumarin_rl.toml (portable) and reinvent/coumarin_rl_run.toml (machine-local run snapshot).\n"
        "REINVENT reproduction: REPRODUCE_REINVENT.md (prior model not bundled; see reinvent_prior_provenance.json).\n"
        "Environment pinning: ENVIRONMENT.md.\n"
        "GitHub/Zenodo sync: RELEASE.md.\n"
        "MD starting structures: md_poses/; analysis summaries (RMSD, H-bond, T/P .xvg) in md_analysis/ "
        "(full production .xtc trajectories are not bundled due to size; available locally under md_gromacs/runs/).\n\n"
        "interaction_profile.csv mixes VKORC1_Reduced (reference ligands, rows 1–25) "
        "and VKORC1_Human (RL_Gen ligands, row 66+); filter by Receptor column for reanalysis.\n\n"
        f"Generated: {manifest['created_at']}\n"
        f"Included bundles: {len(manifest['files'])}\n"
        f"Missing (regenerate locally): {len(manifest['missing'])}\n",
        encoding="utf-8",
    )
    manifest["files"].append("README_DEPOSITION.txt")

    with open(OUT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(ROOT / "deposition" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Deposition package -> {OUT}")
    print(f"  included: {len(manifest['files'])} bundles/files")
    if manifest["missing"]:
        print(f"  missing: {len(manifest['missing'])} (see manifest.json)")


if __name__ == "__main__":
    main()
