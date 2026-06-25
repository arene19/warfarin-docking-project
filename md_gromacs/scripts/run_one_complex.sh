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

if [[ ! -d "${CHARMM_DIR}" ]]; then
  echo "ERROR: CHARMM-GUI GROMACS directory not found: ${CHARMM_DIR}" >&2
  exit 1
fi

for f in topol.top step5_input.gro; do
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
  shift
  echo "--- gmx $* (deffnm=${deffnm}) ---"
  if ! "${GMX}" "$@" -deffnm "${deffnm}"; then
    echo "ERROR: gmx failed for ${deffnm}" >&2
    exit 1
  fi
}

cd "${RUN_ROOT}"

# Link CHARMM-GUI inputs (read-only reference)
ln -sf "${CHARMM_DIR}/topol.top" ./topol.top
ln -sf "${CHARMM_DIR}/step5_input.gro" ./step5_input.gro
for inc in "${CHARMM_DIR}"/*.itp "${CHARMM_DIR}"/toppar/*.itp 2>/dev/null; do
  [[ -e "$inc" ]] && ln -sf "$inc" .
done

# Energy minimization
if [[ ! -f em.gro ]]; then
  run_gmx em grompp -f "${MDP_DIR}/em.mdp" -c step5_input.gro -p topol.top -maxwarn 2
  run_gmx em mdrun -v -ntmpi "${NTMPI}" -ntomp "${NTOMP}" -nb gpu -pme gpu -bonded gpu -update gpu -gpu_id "${GPU_ID}"
else
  echo "Skipping EM (em.gro exists)"
fi

# NVT
if [[ ! -f nvt.gro ]]; then
  run_gmx nvt grompp -f "${MDP_DIR}/nvt_posres.mdp" -c em.gro -r em.gro -p topol.top -maxwarn 2
  run_gmx nvt mdrun -v -ntmpi "${NTMPI}" -ntomp "${NTOMP}" -nb gpu -pme gpu -bonded gpu -update gpu -gpu_id "${GPU_ID}"
else
  echo "Skipping NVT (nvt.gro exists)"
fi

# NPT with restraints
if [[ ! -f npt_posres.gro ]]; then
  run_gmx npt_posres grompp -f "${MDP_DIR}/npt_posres.mdp" -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -maxwarn 2
  run_gmx npt_posres mdrun -v -ntmpi "${NTMPI}" -ntomp "${NTOMP}" -nb gpu -pme gpu -bonded gpu -update gpu -gpu_id "${GPU_ID}"
else
  echo "Skipping NPT posres (npt_posres.gro exists)"
fi

# NPT release
if [[ ! -f npt.gro ]]; then
  run_gmx npt grompp -f "${MDP_DIR}/npt_equil.mdp" -c npt_posres.gro -r npt_posres.gro -t npt_posres.cpt -p topol.top -maxwarn 2
  run_gmx npt mdrun -v -ntmpi "${NTMPI}" -ntomp "${NTOMP}" -nb gpu -pme gpu -bonded gpu -update gpu -gpu_id "${GPU_ID}"
else
  echo "Skipping NPT equil (npt.gro exists)"
fi

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
  run_gmx "${PREFIX}" grompp -f "${MDP}" -c "${PREFIX}.gro" -t "${PREFIX}.cpt" -p topol.top -maxwarn 2
else
  run_gmx "${PREFIX}" grompp -f "${MDP}" -c npt.gro -t npt.cpt -p topol.top -maxwarn 2
fi

run_gmx "${PREFIX}" mdrun -v -ntmpi "${NTMPI}" -ntomp "${NTOMP}" -nb gpu -pme gpu -bonded gpu -update gpu -gpu_id "${GPU_ID}" -cpi "${PREFIX}.cpt" -append 2>/dev/null \
  || run_gmx "${PREFIX}" mdrun -v -ntmpi "${NTMPI}" -ntomp "${NTOMP}" -nb gpu -pme gpu -bonded gpu -update gpu -gpu_id "${GPU_ID}"

echo "Finished: $(date -Iseconds)"
echo "Log: ${LOG}"
echo "Next: python3 ${SCRIPT_DIR}/analyze_md.py --system ${SYSTEM_ID} --run-dir ${RUN_ROOT}"
