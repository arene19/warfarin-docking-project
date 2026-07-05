#!/usr/bin/env bash
# Phase 4 — build GROMACS 2024.3 with CUDA support into $HOME/gromacs.
set -euo pipefail

GMX_VERSION="${GMX_VERSION:-2024.3}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${HOME}/gromacs}"
BUILD_DIR="${BUILD_DIR:-${HOME}/src/gromacs-${GMX_VERSION}}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

echo "=== GROMACS ${GMX_VERSION} CUDA install (Phase 4) ==="

if command -v gmx >/dev/null 2>&1 && gmx -version 2>&1 | grep -qi cuda; then
  echo "[OK] GROMACS with CUDA already installed:"
  gmx -version 2>&1 | head -5
  exit 0
fi

if ! command -v nvcc >/dev/null 2>&1; then
  if [[ -d "${CUDA_ROOT}/bin" ]]; then
    export PATH="${CUDA_ROOT}/bin:${PATH}"
  fi
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[FAIL] nvcc not found. Run setup_cuda.sh first." >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "[FAIL] cmake not found. Run setup_ubuntu_base.sh first." >&2
  exit 1
fi

mkdir -p "${HOME}/src"
cd "${HOME}/src"

TARBALL="gromacs-${GMX_VERSION}.tar.gz"
if [[ ! -f "${TARBALL}" ]]; then
  wget "https://ftp.gromacs.org/gromacs/${TARBALL}"
fi

if [[ ! -d "gromacs-${GMX_VERSION}" ]]; then
  tar xzf "${TARBALL}"
fi

mkdir -p "${BUILD_DIR}/build"
cd "${BUILD_DIR}/build"

CMAKE_CUDA=""
if [[ -d "${CUDA_ROOT}" ]]; then
  CMAKE_CUDA="-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT}"
fi

cmake .. \
  -DGMX_GPU=CUDA \
  ${CMAKE_CUDA} \
  -DGMX_MPI=OFF \
  -DGMX_OPENMP=ON \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"

make -j"$(nproc)"
make install

BASHRC="${HOME}/.bashrc"
PATH_LINE="export PATH=${INSTALL_PREFIX}/bin:\$PATH"
if ! grep -qF "${INSTALL_PREFIX}/bin" "${BASHRC}" 2>/dev/null; then
  echo "${PATH_LINE}" >> "${BASHRC}"
fi
export PATH="${INSTALL_PREFIX}/bin:${PATH}"

echo
echo "Verifying GROMACS CUDA build..."
gmx -version 2>&1 | head -8
if gmx -version 2>&1 | grep -qi cuda; then
  echo "[OK] GROMACS ${GMX_VERSION} with CUDA installed to ${INSTALL_PREFIX}"
else
  echo "[FAIL] GROMACS built but CUDA not reported in version string." >&2
  exit 1
fi
