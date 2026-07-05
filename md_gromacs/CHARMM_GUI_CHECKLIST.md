# CHARMM-GUI checklist — first membrane system

Complete this in a **browser** after WSL setup passes [pc_checklist.sh](scripts/pc_checklist.sh).

## Before you start

- [ ] Register / log in at [CHARMM-GUI](https://www.charmm-gui.org/)
- [ ] WSL preflight passed (`nvidia-smi`, `gmx` with CUDA)
- [ ] Choose first system:
  - **Pilot (lowest CGenFF risk):** `S_Warfarin_ref`
  - **Manuscript RL (expect ParamChem):** `RL_Gen_37`

## Per-system steps

Repeat for each row in [manifest.csv](manifest.csv) or [manifest_manuscript_rl.csv](manifest_manuscript_rl.csv).

### 1. Open Membrane Builder

1. **Input Generator** → **Membrane Builder** → **PDB Reader**

### 2. Upload structure

1. Upload `md_gromacs/systems/<SYSTEM>/charmm_gui/complex_clean.pdb`
2. Chain A = protein (VKORC1, 6WV3)
3. Chain B = hetero **LIG** — mark as **non-standard ligand**
4. Protonation: pH **7.4**
5. Terminal patching: accept CHARMM36m defaults (CT1/NT1)

### 3. Ligand parameterization (CGenFF)

1. Upload `md_gromacs/systems/<SYSTEM>/ligand_params/ligand.sdf`
2. Read `ligand_params/CGenFF_NOTES.md` for charge/stereo notes
3. If automated CGenFF fails or penalty > 50:
   - Submit SMILES to [ParamChem](https://cgenff.umaryland.edu/)
   - Re-upload returned mol2 to CHARMM-GUI Ligand Reader
4. Rename ligand residue to **`LIG`**

**High-risk (ParamChem likely):** `RL_Gen_37`, `RL_Gen_29_isoA`, `RL_Gen_22`, `RL_Gen_45`

### 4. Orientation and membrane

1. Orientation: PPM 2.0 default (TM helix normal to Z)
2. Membrane: **Homogeneous** → ER-like (POPC + POPE + POPS + cholesterol) or Mammalian ER preset
3. Box: default (~10 Å water slab; ~80–120k atoms)
4. Ions: **0.15 M NaCl**, neutralize
5. Temperature: 310 K (optional at build)

### 5. Download GROMACS export

1. Force field: **CHARMM36m**, water **TIP3P**
2. Output format: **GROMACS**
3. Download `.tgz` and unpack on PC:

```bash
mkdir -p ~/warfarin_project/md_gromacs/charmm_gui_export/<SYSTEM>
tar xzf charmm-gui-<SYSTEM>.tgz -C ~/warfarin_project/md_gromacs/charmm_gui_export/<SYSTEM> --strip-components=1
bash ~/warfarin_project/md_gromacs/scripts/validate_charmm_export.sh <SYSTEM>
```

### 6. EM pilot (before full equilibration)

```bash
CHARMM_DIR=~/warfarin_project/md_gromacs/charmm_gui_export/<SYSTEM>
mkdir -p ~/warfarin_project/md_gromacs/runs/<SYSTEM>
cd ~/warfarin_project/md_gromacs/runs/<SYSTEM>

gmx grompp -f ../../mdp/em.mdp \
  -c ${CHARMM_DIR}/step5_input.gro \
  -p ${CHARMM_DIR}/topol.top \
  -o em.tpr -maxwarn 2

gmx mdrun -v -deffnm em -nb gpu -pme gpu -bonded gpu -update gpu
```

If EM completes without LINCS crashes → run full pilot:

```bash
cd ~/warfarin_project/md_gromacs/scripts
export GMX=gmx GPU_ID=0 NTOMP=8
./run_one_complex.sh <SYSTEM> \
  ~/warfarin_project/md_gromacs/charmm_gui_export/<SYSTEM> \
  pilot
```

## Manuscript RL run order

| Order | System | Rationale |
|-------|--------|-----------|
| MD-1 | RL_Gen_37 | Top RL hit |
| MD-2 | RL_Gen_29_isoA | Richest H-bond network |
| MD-3 | RL_Gen_22 | Third-ranked; fluorinated scaffold |
| MD-4 | RL_Gen_45 | Fourth-ranked; spirocyclic diversity |

## Troubleshooting

| Issue | Action |
|-------|--------|
| CGenFF fails on RL_Gen_* | ParamChem → reupload mol2; see `CGenFF_NOTES.md` |
| `tc-grps` mismatch in mdp | `gmx make_ndx`; edit group names in mdp |
| CUDA not used | Re-run `install_gromacs_cuda.sh`; check `gmx -version` |
| Ligand high RMSD later | Extend NPT equil; verify enolate charge / stereo |

Full workflow: [README_MD_GROMACS.md](README_MD_GROMACS.md)
