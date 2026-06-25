# GROMACS membrane MD workflow — VKORC1_Human × 6 ligands

CHARMM36m + CGenFF | ER-like bilayer | GROMACS 2023/2024 CUDA | RTX 4060 Ti

---

## A) Laptop — what was generated

This folder (`md_gromacs/`) was built from `md_poses/` on the laptop. **No MD was run.**

```
md_gromacs/
├── README_MD_GROMACS.md          ← this file
├── manifest.csv                  ← run order + paths per system
├── prepare_workflow.py           ← regenerate inputs from md_poses/
├── mdp/                          ← shared GROMACS parameter templates
│   ├── em.mdp
│   ├── nvt_posres.mdp
│   ├── npt_posres.mdp
│   ├── npt_equil.mdp
│   ├── md_20ns_pilot.mdp
│   └── md_100ns_production.mdp
├── systems/
│   └── <SYSTEM_ID>/
│       ├── charmm_gui/
│       │   ├── complex.pdb       ← protein (A) + ligand (B, LIG)
│       │   ├── complex_clean.pdb ← TER-cleaned upload variant
│       │   ├── protein.pdb
│       │   └── ligand.pdb
│       └── ligand_params/
│           ├── ligand.sdf        ← from config SMILES (3D embed)
│           ├── ligand.mol2
│           └── CGenFF_NOTES.md
├── scripts/
│   ├── run_one_complex.sh        ← equilibration + production (PC only)
│   ├── analyze_md.py             ← RMSD + H-bond analysis
│   └── pc_checklist.sh           ← pre-flight on 4060 Ti
└── runs/                         ← created on PC during MD (gitignored)
```

### Systems (run order)

| Order | System | CGenFF risk | SMILES source |
|-------|--------|-------------|---------------|
| 1 | **S_Warfarin_ref** | low | `config.yaml` |
| 2 | p_nitro_R | medium | `config.yaml` |
| 3 | p_nitro_S | medium | `config.yaml` |
| 4 | dimethoxy_23_S | medium | `config.yaml` |
| 5 | RL_Gen_37 | **high** | `config_master.yaml` → `RL_Gen_37_isoA` (CIP R) |
| 6 | RL_Gen_29_isoA | **high** | `config_master.yaml` |

Coumarin references use **enolate** `[O-]` charge state at pH 7.4.

### Push to GitHub (laptop)

```bash
cd ~/warfarin_project
git add md_gromacs/
git commit -m "Add GROMACS membrane MD workflow for md_poses complexes"
git push origin main
```

---

## B) CHARMM-GUI Membrane Builder (manual, browser)

Do this **once per ligand** (6 times total). Estimated time: ~30–60 min submission + queue per system.

### B.1 Register and open

1. Go to [CHARMM-GUI](https://www.charmm-gui.org/) → **Log in / Register**
2. **Input Generator** → **Membrane Builder** → **PDB Reader**

### B.2 Upload structure

1. Upload `systems/<SYSTEM>/charmm_gui/complex_clean.pdb`
2. **PDB reading:**
   - Chain A = protein (VKORC1, 6WV3)
   - Chain B = hetero **LIG** — mark as **non-standard ligand**
3. **Protonation:** pH **7.4**
4. **Terminal patching:** protein N/C termini as suggested (CT1/NT1 for CHARMM36m)

### B.3 Ligand parameterization (CGenFF)

1. When prompted for ligand, upload `systems/<SYSTEM>/ligand_params/ligand.sdf`
2. Read `ligand_params/CGenFF_NOTES.md` for charge/stereo notes
3. If automated CGenFF fails or penalties > 50:
   - Submit SMILES to [ParamChem](https://cgenff.umaryland.edu/)
   - Re-upload returned mol2 to CHARMM-GUI Ligand Reader
4. **Rename ligand residue to `LIG`** to match complex PDB

**High-risk ligands (expect manual ParamChem):** `RL_Gen_37`, `RL_Gen_29_isoA`

### B.4 Orientation and membrane

1. **Orientation:** align transmembrane helix bundle normal to Z (default PPM 2.0 is fine)
2. **Membrane type:** **Homogeneous** → ER-like lipid mixture or closest available:
   - Prefer: **POPC + POPE + POPS + cholesterol** (approximate ER)
   - Alternative: CHARMM-GUI "Mammalian ER" if listed
3. **Box size:** default (~10 Å water slab); expect **~80–120k atoms** total
4. **Ions:** **0.15 M NaCl**, neutralize system
5. **Temperature:** 310 K (optional at build stage)

### B.5 Force field and output

1. Force field: **CHARMM36m**
2. Water: **TIP3P**
3. Output: **GROMACS**
4. Download the `.tgz` → unpack on PC as e.g.:
   ```
   md_gromacs/charmm_gui_export/S_Warfarin_ref/
   ├── topol.top
   ├── step5_input.gro
   ├── *.itp
   └── toppar/
   ```

### B.6 Merge custom mdp (optional)

CHARMM-GUI ships its own mdp files. You may either:

- Use CHARMM-GUI mdps for steps 1–6, then switch to `md_gromacs/mdp/md_20ns_pilot.mdp` for production, **or**
- Replace equilibration mdps with templates in `md_gromacs/mdp/` (GPU-tuned for 4060 Ti)

**Important:** Verify `tc-grps` and `pcoupltype` group names match your `index.ndx` after solvation (`PROTEIN`, `SOLU`, `MEMB`).

---

## C) PC checklist (RTX 4060 Ti + WSL2)

Path: `/home/amar/warfarin_project`

### C.1 Sync repo

```bash
cd ~/warfarin_project
git pull origin main
bash md_gromacs/scripts/pc_checklist.sh
```

Expect:

- 6 files in `md_poses/complexes/`
- `nvidia-smi` shows RTX 4060 Ti
- `gmx -version` reports **CUDA**

### C.2 Install GROMACS (if missing)

Use GROMACS **2023.3+** or **2024.x** built with CUDA. Example (adapt to your install):

```bash
# Verify GPU path
gmx mdrun -version 2>&1 | grep -i cuda
```

Set OpenMM/CHARMM-GUI paths only if you use their conversion tools — otherwise CHARMM-GUI web export is sufficient.

### C.3 Pilot test (S_Warfarin_ref only)

**Do not batch all 6 systems until pilot passes.**

```bash
# After CHARMM-GUI export unpacked to:
CHARMM_DIR=~/warfarin_project/md_gromacs/charmm_gui_export/S_Warfarin_ref

# EM only (~minutes)
mkdir -p ~/warfarin_project/md_gromacs/runs/S_Warfarin_ref
cd ~/warfarin_project/md_gromacs/runs/S_Warfarin_ref
gmx grompp -f ../../mdp/em.mdp -c ${CHARMM_DIR}/step5_input.gro -p ${CHARMM_DIR}/topol.top -o em.tpr -maxwarn 2
gmx mdrun -v -deffnm em -nb gpu -pme gpu -bonded gpu -update gpu
```

If EM completes without LINCS warnings or crashes → proceed to full script.

### C.4 Full run (one system at a time)

```bash
cd ~/warfarin_project/md_gromacs/scripts
export GMX=gmx
export GPU_ID=0
export NTOMP=8

./run_one_complex.sh S_Warfarin_ref \
  ~/warfarin_project/md_gromacs/charmm_gui_export/S_Warfarin_ref \
  pilot

# After reviewing 20 ns stability:
./run_one_complex.sh S_Warfarin_ref \
  ~/warfarin_project/md_gromacs/charmm_gui_export/S_Warfarin_ref \
  production
```

Repeat for systems 2–6 in `manifest.csv` order.

**Rough timing (4060 Ti, ~100k atoms):**

| Stage | Duration |
|-------|----------|
| EM | minutes |
| NVT + NPT equil | ~1 h |
| 20 ns pilot | ~12–24 h |
| 100 ns production | ~3–5 days |

Run **one complex at a time** to avoid VRAM contention.

### C.5 Analysis

```bash
python3 md_gromacs/scripts/analyze_md.py \
  --system S_Warfarin_ref \
  --run-dir md_gromacs/runs/S_Warfarin_ref
```

Outputs in `runs/<SYSTEM>/analysis/`:

- `rmsd_protein_ca.xvg` — protein Cα RMSD
- `rmsd_ligand.xvg` — ligand RMSD
- `hbond_*.xvg` — H-bonds to Asn80, Ser81, Tyr139, Phe55, Trp59, Thr138, Val134

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| CGenFF fails on RL_Gen_* | ParamChem manual → reupload mol2; see `CGenFF_NOTES.md` |
| `tc-grps` mismatch | `gmx make_ndx -f step5_input.gro -o index.ndx`; edit mdp group names |
| CUDA not detected | Reinstall GROMACS with `-DGMX_GPU=CUDA`; check `nvidia-smi` in WSL2 |
| Ligand drifts / high RMSD | Extend NPT equil; check enolate charge; verify stereo (RL_Gen_37 = isoA R) |
| Push failed (file size) | `runs/` and `charmm_gui_export/` stay local — do not git add |

---

## Regenerate inputs (laptop)

If `md_poses/` changes:

```bash
cd ~/warfarin_project
source venv/bin/activate
python md_gromacs/prepare_workflow.py
```

---

## References

- Receptor: **6WV3** chain A (VKORC1_Human protonated)
- Docking poses: `md_poses/README_MD_POSES.md`, `md_pose_manifest.csv`
- Interaction hotspots: deposition `interaction_profile.csv` (Asn80, Ser81, Tyr139, …)
