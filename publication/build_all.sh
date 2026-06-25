#!/usr/bin/env bash
# Regenerate all manuscript assets from repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Migrate checkpoint metadata (if needed)"
python scripts/migrate_checkpoint.py

echo "==> GNN baseline evaluation"
python gnn_baseline_evaluation.py

echo "==> Manuscript figures and tables"
python publication/generate_manuscript_assets.py

echo "==> REINVENT provenance check"
python scripts/validate_reinvent_provenance.py

echo "==> Deposition package"
python deposition/prepare_deposition.py

echo "==> Word export"
if [[ -x .docx_venv/bin/python ]]; then
  .docx_venv/bin/python publication/md_to_docx.py
elif python -c "import docx" 2>/dev/null; then
  python publication/md_to_docx.py
else
  echo "    Skip docx — pip install python-docx"
fi

echo "Done. See publication/output/ and deposition/package/"
