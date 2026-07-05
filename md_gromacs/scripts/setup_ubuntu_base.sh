#!/usr/bin/env bash
# Phase 1 — base Ubuntu packages inside WSL2 (run after first Ubuntu login).
set -euo pipefail

echo "=== Ubuntu base packages (Phase 1) ==="

sudo apt update
sudo apt upgrade -y
sudo apt install -y \
  build-essential \
  cmake \
  git \
  wget \
  curl \
  python3 \
  python3-pip \
  python3-venv \
  pkg-config \
  libfftw3-dev \
  libopenmpi-dev \
  openmpi-bin

mkdir -p "${HOME}/warfarin_project"
echo "[OK] Base packages installed."
echo "Next: bash setup_cuda.sh (after Windows NVIDIA driver is installed)"
