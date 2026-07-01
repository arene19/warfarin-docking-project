#!/usr/bin/env bash
# Run GROMACS equilibration + production for one VKORC1–ligand membrane system.
# Usage: ./run_one_complex.sh SYSTEM_ID /path/to/charmm_gui_gromacs_dir [pilot|production]
#
# SYSTEM_ID: e.g. S_Warfarin_ref (see md_gromacs/manifest.csv)
# charmm_gui_gromacs_dir: unpacked CHARMM-GUI GROMACS folder containing topol.top, step5_input.gro
#
# Expects GROMACS 2023.x or 2024.x with CUDA. Do NOT run on laptop unless testing EM only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MDG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MDG_ROOT}/.." && pwd)"
MDP_DIR="${MDG_ROOT}/mdp"

SYSTEM_ID="${1:?Usage: $0 SYSTEM_ID CHARMM_GUI_GRO_DIR [pilot|production]}"
CHARMM_DIR="${2:?Usage: $0 SYSTEM_ID CHARMM_GUI_GRO_DIR [pilot|production]}"
MODE="${3:-pilot}"

# Resolve before cd to RUN_ROOT (relative paths are otherwise wrong)
CHARMM_DIR="$(cd "${CHARMM_DIR}" && pwd)"

if [[ ! -d "${CHARMM_DIR}" ]]; then
  echo "ERROR: CHARMM-GUI GROMACS directory not found: ${CHARMM_DIR}" >&2
  exit 1
fi

for f in topol.top step5_input.gro index.ndx; do
  if [[ ! -f "${CHARMM_DIR}/${f}" ]]; then
    echo "ERROR: Missing ${f} in ${CHARMM_DIR}" >&2
    echo "       Unpack the CHARMM-GUI GROMACS download and point to that folder." >&2
    exit 1
  fi
done

RUN_ROOT="${MDG_ROOT}/runs/${SYSTEM_ID}"
mkdir -p "${RUN_ROOT}"
LOG="${RUN_ROOT}/run.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== MD run: ${SYSTEM_ID} mode=${MODE} ==="
echo "Started: $(date -Iseconds)"
echo "CHARMM dir: ${CHARMM_DIR}"
echo "Run dir: ${RUN_ROOT}"

GMX="${GMX:-gmx}"
NTMPI="${NTMPI:-1}"
NTOMP="${NTOMP:-8}"
GPU_ID="${GPU_ID:-0}"

run_gmx() {
  local deffnm="$1"
  local subcmd="$2"
  shift 2
  echo "--- gmx ${subcmd} $* (deffnm=${deffnm}) ---"
  if [[ "${subcmd}" == "grompp" ]]; then
    if ! "${GMX}" grompp "$@" -o "${deffnm}.tpr"; then
      echo "ERROR: gmx grompp failed for ${deffnm}" >&2
      exit 1
    fi
  elif [[ "${deffnm}" == "em" ]]; then
    # Steep minimization runs on CPU only in GROMACS 2024
    if ! "${GMX}" mdrun "$@" -deffnm "${deffnm}"; then
      echo "ERROR: gmx mdrun failed for ${deffnm}" >&2
      exit 1
    fi
  else
    if ! "${GMX}" "${subcmd}" "$@" -deffnm "${deffnm}"; then
      echo "ERROR: gmx ${subcmd} failed for ${deffnm}" >&2
      exit 1
    fi
  fi
}

cd "${RUN_ROOT}"

# Link CHARMM-GUI inputs (read-only reference)
ln -sf "${CHARMM_DIR}/topol.top" ./topol.top
ln -sf "${CHARMM_DIR}/step5_input.gro" ./step5_input.gro
ln -sf "${CHARMM_DIR}/index.ndx" ./index.ndx
ln -sfn "${CHARMM_DIR}/toppar" ./toppar
shopt -s nullglob
for inc in "${CHARMM_DIR}"/*.itp; do
  [[ -e "$inc" ]] && ln -sf "$inc" .
done
shopt -u nullglob

# Energy minimization
if [[ ! -f em.gro ]]; then
  run_gmx em grompp -f "${MDP_DIR}/em.mdp" -c step5_input.gro -p topol.top -maxwarn 2
  run_gmx em mdrun -v -ntmpi "${NTMPI}" -ntomp "${NTOMP}"
else
  echo "Skipping EM (em.gro exists)"
fi

# CHARMM-GUI step6.1–6.6 equilibration (POSRES via export mdps + index.ndx)
GPU_FLAGS=(-nb gpu -pme gpu -bonded gpu -update gpu -gpu_id "${GPU_ID}")
for inc in 1 2 3 4 5 6; do
  deffnm="step6.${inc}_equilibration"
  mdp="${CHARMM_DIR}/step6.${inc}_equilibration.mdp"
  if [[ ! -f "${mdp}" ]]; then
    echo "ERROR: Missing ${mdp}" >&2
    exit 1
  fi
  if [[ -f "${deffnm}.gro" ]]; then
    echo "Skipping ${deffnm} (${deffnm}.gro exists)"
    continue
  fi
  if [[ "${inc}" -eq 1 ]]; then
    run_gmx "${deffnm}" grompp -f "${mdp}" -c em.gro -r step5_input.gro -p topol.top -n index.ndx -maxwarn 2
  else
    prev="step6.$((inc - 1))_equilibration"
    run_gmx "${deffnm}" grompp -f "${mdp}" -c "${prev}.gro" -r step5_input.gro -t "${prev}.cpt" -p topol.top -n index.ndx -maxwarn 2
  fi
  run_gmx "${deffnm}" mdrun -v -ntmpi "${NTMPI}" -ntomp "${NTOMP}" "${GPU_FLAGS[@]}"
done

if [[ "${MODE}" == "pilot" ]]; then
  MDP="${MDP_DIR}/md_20ns_pilot.mdp"
  PREFIX="md_20ns"
elif [[ "${MODE}" == "production" ]]; then
  MDP="${MDP_DIR}/md_100ns_production.mdp"
  PREFIX="md_100ns"
else
  echo "ERROR: MODE must be pilot or production" >&2
  exit 1
fi

if [[ -f "${PREFIX}.cpt" ]]; then
  echo "Restarting ${PREFIX} from checkpoint"
  if [[ -f "${PREFIX}.gro" ]]; then
    run_gmx "${PREFIX}" grompp -f "${MDP}" -c "${PREFIX}.gro" -t "${PREFIX}.cpt" -p topol.top -n index.ndx -maxwarn 2
  elif [[ -f "${PREFIX}.tpr" ]]; then
    echo "Using existing ${PREFIX}.tpr (checkpoint resume without .gro)"
  else
    echo "ERROR: ${PREFIX}.cpt exists but no ${PREFIX}.gro or ${PREFIX}.tpr" >&2
    exit 1
  fi
else
  run_gmx "${PREFIX}" grompp -f "${MDP}" -c step6.6_equilibration.gro -t step6.6_equilibration.cpt -p topol.top -n index.ndx -maxwarn 2
fi

MDRUN_FLAGS=(-v -ntmpi "${NTMPI}" -ntomp "${NTOMP}" -nb gpu -pme gpu -bonded gpu -update gpu -gpu_id "${GPU_ID}")
if [[ -f "${PREFIX}.cpt" ]]; then
  echo "Restarting ${PREFIX} mdrun from checkpoint"
  run_gmx "${PREFIX}" mdrun "${MDRUN_FLAGS[@]}" -cpi "${PREFIX}.cpt" -append
else
  run_gmx "${PREFIX}" mdrun "${MDRUN_FLAGS[@]}"
fi

echo "Finished: $(date -Iseconds)"
echo "Log: ${LOG}"
echo "Next: python3 ${SCRIPT_DIR}/analyze_md.py --system ${SYSTEM_ID} --run-dir ${RUN_ROOT}"
