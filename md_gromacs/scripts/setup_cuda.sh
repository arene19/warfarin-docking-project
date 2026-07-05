#!/usr/bin/env bash
# Phase 2 — verify GPU passthrough and install CUDA 12.x toolkit in WSL2.
# Uses NVIDIA's WSL-Ubuntu repo (NOT apt nvidia-cuda-toolkit, which is CUDA 11.5).
set -euo pipefail

echo "=== CUDA / GPU setup (Phase 2) ==="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    export PATH="/usr/lib/wsl/lib:${PATH}"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[OK] GPU detected:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "[FAIL] nvidia-smi not found in WSL." >&2
  echo "       Install the latest NVIDIA driver on Windows (not in WSL), reboot, then retry." >&2
  echo "       https://www.nvidia.com/Download/index.aspx" >&2
  exit 1
fi

need_install=0
if command -v nvcc >/dev/null 2>&1; then
  NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
  MAJOR=$(echo "${NVCC_VER}" | cut -d. -f1)
  if [[ "${MAJOR}" -ge 12 ]]; then
    echo "[OK] nvcc already CUDA ${NVCC_VER}:"
    nvcc --version | head -1
    exit 0
  fi
  echo "[WARN] nvcc is CUDA ${NVCC_VER} (< 12). Upgrading to CUDA 12.x for GROMACS + RTX 4060 Ti..."
  need_install=1
else
  need_install=1
fi

if [[ "${need_install}" -eq 1 ]]; then
  echo "Removing old apt CUDA toolkit (11.5) if present..."
  sudo apt remove -y nvidia-cuda-toolkit 2>/dev/null || true

  echo "Installing CUDA 12.x from NVIDIA WSL-Ubuntu repository..."
  TMP=$(mktemp -d)
  cd "${TMP}"

  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update

  # Prefer latest 12.x meta-package available in repo
  TOOLKIT_PKG=$(apt-cache search '^cuda-toolkit-12-' 2>/dev/null | awk '{print $1}' | sort -V | tail -1)
  if [[ -z "${TOOLKIT_PKG}" ]]; then
    TOOLKIT_PKG="cuda-toolkit-12-6"
  fi
  echo "Installing ${TOOLKIT_PKG} ..."
  sudo apt-get -y install "${TOOLKIT_PKG}"

  cd - >/dev/null
  rm -rf "${TMP}"
fi

# NVIDIA WSL toolkit installs to /usr/local/cuda-12.x
CUDA12=$(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -1 || true)
if [[ -n "${CUDA12}" ]]; then
  sudo ln -sfn "${CUDA12}" /usr/local/cuda
fi

BASHRC="${HOME}/.bashrc"
PATH_LINE='export PATH=/usr/local/cuda/bin:$PATH'
LD_LINE='export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}'
for line in "${PATH_LINE}" "${LD_LINE}"; do
  if ! grep -qF "${line}" "${BASHRC}" 2>/dev/null; then
    echo "${line}" >> "${BASHRC}"
  fi
done
export PATH="/usr/local/cuda/bin:${PATH}"

if command -v nvcc >/dev/null 2>&1; then
  echo "[OK] nvcc installed:"
  nvcc --version | head -1
else
  echo "[FAIL] nvcc still not on PATH after install." >&2
  echo "       Run: source ~/.bashrc && nvcc --version" >&2
  exit 1
fi

echo
echo "Do NOT install Linux NVIDIA drivers inside WSL — Windows driver provides GPU access."
