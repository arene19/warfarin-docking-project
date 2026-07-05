# Run in PowerShell as Administrator on a fresh Windows 11 PC.
# Installs WSL2 + Ubuntu 22.04 for the warfarin MD workflow.

$ErrorActionPreference = "Stop"

Write-Host "=== Warfarin MD — Windows WSL2 setup ===" -ForegroundColor Cyan

Write-Host "`n[1/4] Enabling WSL and Virtual Machine Platform..."
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

Write-Host "`n[2/4] Setting WSL2 as default..."
wsl --set-default-version 2

Write-Host "`n[3/4] Listing available Ubuntu distributions..."
wsl --list --online | Select-String -Pattern "Ubuntu"

Write-Host "`nInstalling Ubuntu 22.04 LTS (if install fails, try 'Ubuntu' or 'Ubuntu-24.04' from the list above)..."
wsl --install -d Ubuntu-22.04
if ($LASTEXITCODE -ne 0) {
  Write-Host "[WARN] Ubuntu-22.04 install failed — trying generic Ubuntu..." -ForegroundColor Yellow
  wsl --install -d Ubuntu
}

Write-Host "`nInstalled distributions:"
wsl --list --verbose

Write-Host @"

IMPORTANT — do NOT run 'wsl --set-default Ubuntu-22.04' until Ubuntu is installed.
After reboot, check the exact name with:  wsl --list --verbose
Then set default using that EXACT name, e.g.:
  wsl --set-default Ubuntu-22.04
  wsl --set-default Ubuntu
  wsl --set-default Ubuntu-24.04

[4/4] NEXT STEPS (manual):

1. Reboot Windows when prompted.
2. After reboot, open "Ubuntu 22.04" from Start and create your Linux user.
3. Install the latest NVIDIA Game Ready / Studio driver for RTX 4060 Ti on Windows
   (https://www.nvidia.com/Download/index.aspx) — NOT inside WSL.
4. Reboot again, then open Ubuntu and run:

   cd ~/warfarin_project/md_gromacs/scripts
   bash setup_ubuntu_base.sh
   bash setup_cuda.sh
   bash setup_clone_repo.sh
   bash install_gromacs_cuda.sh
   bash setup_pc_all.sh

Or run the all-in-one helper after WSL user exists:

   bash setup_pc_all.sh

"@ -ForegroundColor Yellow
