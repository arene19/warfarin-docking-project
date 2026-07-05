# MD PC setup — fresh Windows + WSL2

Step-by-step guide for the RTX 4060 Ti workstation. Run scripts from this repo after cloning.

## Quick start (copy-paste order)

| Step | Where | Command |
|------|-------|---------|
| 1 | Windows PowerShell **(Admin)** | `wsl --install` or `wsl --install -d Ubuntu-22.04` |
| 2 | Reboot Windows | — |
| 3 | PowerShell | `wsl --list --verbose` — note the **exact** distro name |
| 4 | PowerShell | `wsl --set-default <exact-name-from-step-3>` |
| 5 | Windows | Install [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) for RTX 4060 Ti |
| 6 | Reboot Windows | — |
| 7 | Ubuntu (WSL) | `bash md_gromacs/scripts/setup_pc_all.sh` |

Or run the helper script in step 1: `.\md_gromacs\scripts\setup_windows_wsl.ps1`

## WSL troubleshooting

### `wsl --set-default Ubuntu-22.04` → "no distribution with the supplied name"

This means **Ubuntu is not installed yet**, or it was installed under a **different name**.

**Fix — run in PowerShell (Admin), in this order:**

```powershell
# 1. See what is actually installed (may be empty)
wsl --list --verbose

# 2. See what you CAN install
wsl --list --online

# 3. Install Ubuntu (pick one that appears in the online list)
wsl --install -d Ubuntu-22.04
# If that fails, try:
wsl --install -d Ubuntu
# Or:
wsl --install -d Ubuntu-24.04

# 4. Reboot Windows, then open "Ubuntu" from Start menu and create your Linux user

# 5. List again — use the EXACT name from the NAME column
wsl --list --verbose

# 6. Set default using that exact name (examples):
wsl --set-default Ubuntu-22.04
# or
wsl --set-default Ubuntu
# or
wsl --set-default Ubuntu-24.04
```

**Ubuntu 24.04 is fine** for this project if 22.04 is unavailable — the setup scripts work on both.

### Other common errors

| Error | Fix |
|-------|-----|
| `Wsl/WSL_E_DEFAULT_DISTRO_NOT_FOUND` | Install a distro first (`wsl --install`), reboot, create user |
| WSL installs but no GPU | Install NVIDIA driver **on Windows**, reboot, then `nvidia-smi` inside Ubuntu |
| `wsl --install` does nothing | Enable virtualization in BIOS; run Windows Update |

### `nvcc --version` shows CUDA 11.5 (too old for GROMACS 2024)

`sudo apt install nvidia-cuda-toolkit` installs **11.5**. Upgrade to **CUDA 12.x** inside **Ubuntu WSL**:

```bash
# Option A — project script (recommended)
bash ~/warfarin_project/md_gromacs/scripts/setup_cuda.sh

# Option B — manual (NVIDIA WSL repo)
sudo apt remove -y nvidia-cuda-toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

Install only **`cuda-toolkit-12-x`** — not `cuda` or `cuda-drivers` (those install Linux GPU drivers in WSL and break things).

Guide: [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## What you need vs. what you can skip

**Required for MD**

- Windows 11 + WSL2 + Ubuntu 22.04
- NVIDIA Windows driver (GPU passthrough to WSL)
- Git + cloned `warfarin-docking-project`
- GROMACS 2023.3+ or 2024.x **with CUDA**
- Python 3 (stdlib; for `analyze_md.py` after runs)
- CHARMM-GUI account (browser)

**Skip on MD PC** (laptop-only workflow)

- `pip install -r requirements.txt`, PyTorch, REINVENT4, AutoDock Vina
- `coagulation_admet_gnn.pth` (not used for MD)

## Scripts reference

| Script | Phase | Runs on |
|--------|-------|---------|
| [setup_windows_wsl.ps1](scripts/setup_windows_wsl.ps1) | WSL2 + Ubuntu install | Windows PowerShell (Admin) |
| [setup_ubuntu_base.sh](scripts/setup_ubuntu_base.sh) | apt build tools | WSL Ubuntu |
| [setup_cuda.sh](scripts/setup_cuda.sh) | CUDA toolkit + GPU check | WSL Ubuntu |
| [setup_clone_repo.sh](scripts/setup_clone_repo.sh) | git clone + verify MD inputs | WSL Ubuntu |
| [install_gromacs_cuda.sh](scripts/install_gromacs_cuda.sh) | Build GROMACS 2024.3 CUDA | WSL Ubuntu |
| [setup_pc_all.sh](scripts/setup_pc_all.sh) | Runs phases 1–5 + preflight | WSL Ubuntu |
| [pc_checklist.sh](scripts/pc_checklist.sh) | Pre-flight before CHARMM-GUI | WSL Ubuntu |
| [validate_charmm_export.sh](scripts/validate_charmm_export.sh) | Verify CHARMM-GUI download | WSL Ubuntu |

## Verify success

After `setup_pc_all.sh`:

```bash
cd ~/warfarin_project
bash md_gromacs/scripts/pc_checklist.sh
```

Expected:

- `nvidia-smi` → RTX 4060 Ti
- `gmx -version` → reports CUDA
- ≥8 complex PDBs in `md_poses/complexes/` (8 systems in manifest)

## Manuscript MD systems (4 RL-only)

| System | Manuscript ID | CGenFF risk |
|--------|---------------|-------------|
| RL_Gen_37 | MD-1 | high |
| RL_Gen_29_isoA | MD-2 | high |
| RL_Gen_22 | MD-3 | high |
| RL_Gen_45 | MD-4 | high |

See [manifest_manuscript_rl.csv](manifest_manuscript_rl.csv) for RL-only run order. Reference systems (S_Warfarin_ref, p_nitro_*, dimethoxy) remain in the full [manifest.csv](manifest.csv) for pilot/QC but are **not** in the manuscript case study.

## Disk and runtime

- Reserve **50–100 GB** free per system (CHARMM export + trajectories)
- Run **one complex at a time** on 4060 Ti (~8 GB VRAM)
- 20 ns pilot ≈ 12–24 h; 100 ns production ≈ 3–5 days per system

## After setup

1. [CHARMM_GUI_CHECKLIST.md](CHARMM_GUI_CHECKLIST.md) — membrane build in browser
2. [README_MD_GROMACS.md](README_MD_GROMACS.md) — full MD workflow
3. Pilot EM then `./run_one_complex.sh <SYSTEM> <export_dir> pilot`

## Local-only paths (do not git commit)

- `md_gromacs/runs/` — simulation outputs
- `md_gromacs/charmm_gui_export/` — CHARMM-GUI downloads
