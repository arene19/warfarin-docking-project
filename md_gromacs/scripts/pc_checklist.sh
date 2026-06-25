#!/usr/bin/env bash
# PC pre-flight checklist before membrane MD (run on RTX 4060 Ti WSL2)

set -euo pipefail

REPO="${1:-${HOME}/warfarin_project}"
echo "=== MD PC checklist ==="
echo "Repo: ${REPO}"
echo "Date: $(date -Iseconds)"
echo

fail=0

check() {
  if "$@"; then
    echo "[OK] $*"
  else
    echo "[FAIL] $*"
    fail=1
  fi
}

check test -d "${REPO}"
check test -d "${REPO}/md_poses/complexes"
check test -f "${REPO}/md_poses/md_pose_manifest.csv"
check test -d "${REPO}/md_gromacs"

N=$(find "${REPO}/md_poses/complexes" -name '*.pdb' 2>/dev/null | wc -l)
if [[ "${N}" -ge 6 ]]; then
  echo "[OK] md_poses complexes: ${N} PDB files"
else
  echo "[FAIL] Expected 6 complex PDBs, found ${N}"
  fail=1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "[OK] nvidia-smi"
else
  echo "[FAIL] nvidia-smi not found"
  fail=1
fi

if command -v gmx >/dev/null 2>&1; then
  gmx -version 2>&1 | head -5
  if gmx -version 2>&1 | grep -qi cuda; then
    echo "[OK] GROMACS with CUDA"
  else
    echo "[WARN] GROMACS found but CUDA not detected in version string"
  fi
else
  echo "[FAIL] gmx not in PATH — install GROMACS 2023+ with GPU support"
  fail=1
fi

echo
echo "Suggested pilot (EM only, after CHARMM-GUI export):"
echo "  cd ${REPO}/md_gromacs/runs/S_Warfarin_ref"
echo "  gmx grompp -f ../mdp/em.mdp -c /path/to/step5_input.gro -p /path/to/topol.top -maxwarn 2"
echo "  gmx mdrun -v -nb gpu -pme gpu -bonded gpu -update gpu"

if [[ "${fail}" -eq 0 ]]; then
  echo
  echo "All critical checks passed."
else
  echo
  echo "Fix failures before starting production MD."
  exit 1
fi
