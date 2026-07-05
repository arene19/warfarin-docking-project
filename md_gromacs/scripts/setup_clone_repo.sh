#!/usr/bin/env bash
# Phase 3 — clone warfarin-docking-project and verify MD inputs.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/arene19/warfarin-docking-project.git}"
REPO="${REPO:-${HOME}/warfarin_project}"

echo "=== Clone / verify repo (Phase 3) ==="
echo "Target: ${REPO}"

if [[ -d "${REPO}/.git" ]]; then
  echo "[OK] Repo exists — pulling latest main..."
  git -C "${REPO}" pull origin main
else
  echo "Cloning ${REPO_URL} ..."
  git clone "${REPO_URL}" "${REPO}"
fi

fail=0
check() {
  if [[ -e "$1" ]]; then
    echo "[OK] $1"
  else
    echo "[FAIL] Missing: $1" >&2
    fail=1
  fi
}

check "${REPO}/md_poses/complexes"
check "${REPO}/md_poses/md_pose_manifest.csv"
check "${REPO}/md_gromacs/manifest.csv"
check "${REPO}/md_gromacs/scripts/pc_checklist.sh"

N=$(find "${REPO}/md_poses/complexes" -name '*.pdb' 2>/dev/null | wc -l)
EXPECTED=$(tail -n +2 "${REPO}/md_gromacs/manifest.csv" | wc -l)
if [[ "${N}" -ge "${EXPECTED}" ]]; then
  echo "[OK] md_poses complexes: ${N} PDB files (manifest lists ${EXPECTED} systems)"
else
  echo "[FAIL] Expected >= ${EXPECTED} complex PDBs, found ${N}" >&2
  fail=1
fi

N_SYS=$(find "${REPO}/md_gromacs/systems" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [[ "${N_SYS}" -ge "${EXPECTED}" ]]; then
  echo "[OK] md_gromacs/systems: ${N_SYS} folders"
else
  echo "[FAIL] Expected >= ${EXPECTED} system folders, found ${N_SYS}" >&2
  fail=1
fi

if [[ "${fail}" -eq 0 ]]; then
  echo
  echo "All repo checks passed."
  echo "Next: bash install_gromacs_cuda.sh"
else
  exit 1
fi
