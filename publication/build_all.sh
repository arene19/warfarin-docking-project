#!/usr/bin/env bash
# Regenerate all manuscript assets from repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

ALLOW_STALE="${ALLOW_STALE_CHECKPOINT:-0}"
SCREENING="results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv"
PUBLICATION_MASTER="publication/data/coagulation_admet_multi_task_publication.csv"
CKPT="coagulation_admet_gnn.pth"

if [[ ! -f "$PUBLICATION_MASTER" ]]; then
  echo "ERROR: frozen publication master missing at $PUBLICATION_MASTER"
  echo "Run: $PYTHON scripts/freeze_publication_master.py"
  exit 1
fi

if [[ "$ALLOW_STALE" != "1" && -f "$SCREENING" && -f "$CKPT" ]]; then
  if [[ "$SCREENING" -nt "$PUBLICATION_MASTER" || "$SCREENING" -nt "$CKPT" ]]; then
    echo "ERROR: screening CSV is newer than frozen publication master or GNN checkpoint."
    echo "Optional: RUN_ACTIVE_LEARNING_MERGE=1 $PYTHON active_learning_merger.py"
    echo "Then: $PYTHON scripts/freeze_publication_master.py"
    echo "Or set ALLOW_STALE_CHECKPOINT=1 to skip this guard."
    exit 1
  fi
fi

echo "==> Migrate checkpoint metadata (if needed)"
"$PYTHON" scripts/migrate_checkpoint.py

if [[ "${RUN_ACTIVE_LEARNING_MERGE:-0}" == "1" ]]; then
  if [[ -f results/docked_poses/VKORC1_Human/VKORC1_Human_screening_results.csv ]]; then
    echo "==> Active-learning merge (mutates data/coagulation_admet_multi_task.csv)"
    "$PYTHON" active_learning_merger.py
    echo "    Re-freeze after merge: $PYTHON scripts/freeze_publication_master.py"
  else
    echo "==> RUN_ACTIVE_LEARNING_MERGE=1 but screening CSV not found; skipping merge"
  fi
else
  echo "==> Using frozen publication master (skip merge; set RUN_ACTIVE_LEARNING_MERGE=1 to mutate live master)"
fi

if [[ -f "$SCREENING" ]]; then
  echo "==> Sanitize workspace screening CSV (all_affinities)"
  "$PYTHON" scripts/sanitize_screening_csv.py "$SCREENING"
fi

echo "==> GNN baseline evaluation (frozen master + split)"
"$PYTHON" gnn_baseline_evaluation.py --split-from publication/data/gnn_scaffold_split.json --data "$PUBLICATION_MASTER"

echo "==> Parse MD results summary"
"$PYTHON" scripts/parse_md_results_summary.py

echo "==> Manuscript figures and tables"
"$PYTHON" publication/generate_manuscript_assets.py

echo "==> REINVENT provenance check"
"$PYTHON" scripts/validate_reinvent_provenance.py

echo "==> Deposition package"
"$PYTHON" deposition/prepare_deposition.py

echo "==> Word export"
if [[ -x .docx_venv/bin/python ]]; then
  .docx_venv/bin/python publication/md_to_docx.py
elif "$PYTHON" -c "import docx" 2>/dev/null; then
  "$PYTHON" publication/md_to_docx.py
else
  echo "    Skip docx — pip install python-docx"
fi

echo "Done. See publication/output/ and deposition/package/"
