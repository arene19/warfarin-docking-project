#!/usr/bin/env bash
# Run all WSL-side setup steps after Windows driver + Ubuntu user exist.
# Usage: bash setup_pc_all.sh [repo_path]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-${HOME}/warfarin_project}"

echo "=== Warfarin MD PC setup (WSL phases 1–5) ==="
echo "Repo: ${REPO}"
echo

run() {
  echo "--- $1 ---"
  bash "${SCRIPT_DIR}/$1"
}

run setup_ubuntu_base.sh
run setup_cuda.sh
REPO="${REPO}" bash "${SCRIPT_DIR}/setup_clone_repo.sh"
run install_gromacs_cuda.sh

echo
echo "--- Pre-flight checklist ---"
bash "${REPO}/md_gromacs/scripts/pc_checklist.sh" "${REPO}"

echo
echo "=== WSL setup complete ==="
echo "Next (browser): see md_gromacs/CHARMM_GUI_CHECKLIST.md"
echo "  1. Register at https://www.charmm-gui.org/"
echo "  2. Membrane-build first system (S_Warfarin_ref pilot or RL_Gen_37)"
echo "  3. Unpack export to md_gromacs/charmm_gui_export/<SYSTEM>/"
echo "  4. bash md_gromacs/scripts/run_one_complex.sh <SYSTEM> <export_dir> pilot"
