#!/usr/bin/env bash
# Verify a CHARMM-GUI GROMACS export is ready for run_one_complex.sh.
# Usage: validate_charmm_export.sh SYSTEM_ID [export_dir]
set -euo pipefail

REPO="${REPO:-${HOME}/warfarin_project}"
SYSTEM_ID="${1:?Usage: $0 SYSTEM_ID [export_dir]}"
EXPORT_DIR="${2:-${REPO}/md_gromacs/charmm_gui_export/${SYSTEM_ID}}"

echo "=== CHARMM-GUI export validation: ${SYSTEM_ID} ==="
echo "Directory: ${EXPORT_DIR}"

fail=0
for f in topol.top step5_input.gro; do
  if [[ -f "${EXPORT_DIR}/${f}" ]]; then
    echo "[OK] ${f}"
  else
    echo "[FAIL] Missing ${f}" >&2
    fail=1
  fi
done

if [[ -d "${EXPORT_DIR}/toppar" ]] || compgen -G "${EXPORT_DIR}/*.itp" >/dev/null; then
  echo "[OK] topology includes / itp files present"
else
  echo "[WARN] No toppar/ or *.itp found — verify unpack path"
fi

MANIFEST="${REPO}/md_gromacs/systems/${SYSTEM_ID}/charmm_gui/complex_clean.pdb"
if [[ -f "${MANIFEST}" ]]; then
  echo "[OK] Source complex: ${MANIFEST}"
else
  echo "[WARN] No local complex_clean.pdb for ${SYSTEM_ID} (check manifest)"
fi

if [[ "${fail}" -eq 0 ]]; then
  echo
  echo "Export looks ready. Pilot command:"
  echo "  cd ${REPO}/md_gromacs/scripts"
  echo "  ./run_one_complex.sh ${SYSTEM_ID} ${EXPORT_DIR} pilot"
else
  exit 1
fi
