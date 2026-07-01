# MD pilot handover — warfarin project

**Date:** 2026-07-01 (updated)  
**Machine:** WSL2 Ubuntu, user `amar`, repo `/home/amar/warfarin_project`  
**GPU:** NVIDIA RTX 4060 Ti 8 GB | **Driver:** 610.62 | **GROMACS:** 2024.3 + CUDA at `~/gromacs/bin/gmx`  
**CPU:** Intel i9-13900KF | **RAM:** 16 GB DDR5 (WSL often sees ~7.6 GB — see §1.1)

---

## 1. Environment

### 1.1 WSL memory (recommended)

Create or edit `C:\Users\Urim\.wslconfig`:

```ini
[wsl2]
memory=12GB
swap=8GB
```

Then from PowerShell: `wsl --shutdown`, reopen Ubuntu.

### 1.2 GROMACS PATH

```bash
export PATH="$HOME/gromacs/bin:$PATH"
export GMX=gmx GPU_ID=0 NTOMP=8
```

Add to `~/.bashrc` for new shells.

### 1.3 Disk

Reserve **~50–100 GB** per system under `md_gromacs/runs/<SYSTEM_ID>/`.

---

## 2. Strategic decisions

| Decision | Status |
|----------|--------|
| S_Warfarin_ref CHARMM-GUI | **Complete** (job ID **8266243616**) |
| S_Warfarin_ref 20 ns pilot | **Complete** — pipeline validated |
| S_Warfarin_ref 100 ns production | **Skipped** — pilot sufficient for pipeline QA |
| RL_Gen_37 CHARMM-GUI | **Complete** (job ID **8273553994**, 133,746 atoms) |
| RL_Gen_37 20 ns pilot | **Complete** — **GO** for 100 ns (2026-06-29) |
| RL_Gen_37 100 ns production | **Complete** (2026-07-01 21:25) — **publication-ready** |
| Next system | RL_Gen_29_isoA or p_nitro_R / dimethoxy_23_S |

---

## 3. S_Warfarin_ref — CHARMM-GUI settings

| Step | Setting | Value |
|------|---------|-------|
| Upload PDB | `systems/S_Warfarin_ref/charmm_gui/complex_clean_charmm.pdb` | chain X = LIG |
| Ligand SDF | `systems/S_Warfarin_ref/ligand_params/ligand_docked.sdf` | CGenFF |
| Step 2 | PPM 2.0 | **Z translation ≈ −55 Å** (protein between membrane planes) |
| Step 3 lipids | POPC 6/6, POPE 2/2, POPS 1/1, Cholesterol 2/2 | water 22.5 Å |
| Box XY | **120 Å** (90 Å and 100 Å failed packing) | |
| Step 4 | NaCl **0.15 M** | neutralize |
| Step 5 | CHARMM36m, TIP3P, **310 K**, GROMACS output | |
| System size | **133,718 atoms** | box ~122 × 122 × 99 Å |

**Export path:** `md_gromacs/charmm_gui_export/S_Warfarin_ref/`

Unpack from Windows download (`.tgz`, not zip):

```bash
./md_gromacs/scripts/unpack_charmm_export.sh S_Warfarin_ref /mnt/c/Users/Urim/Downloads/charmm-gui.tgz
./md_gromacs/scripts/validate_charmm_export.sh md_gromacs/charmm_gui_export/S_Warfarin_ref
```

---

## 4. S_Warfarin_ref — GROMACS pilot results

### 4.1 Workflow used

| Stage | Method | Notes |
|-------|--------|-------|
| EM | `mdp/em.mdp` | **CPU-only** (GPU EM fails in GROMACS 2024) |
| Equilibration | CHARMM **step6.1–6.6** + `index.ndx` | GPU mdrun |
| Production | `mdp/md_20ns_pilot.mdp` | 20 ns, GPU |

**Do not use** repo `nvt_posres.mdp` / `npt_posres.mdp` — they trigger 3000+ `-DPOSRES` errors with CHARMM itps.

Updated automation: `scripts/run_one_complex.sh` now runs step6.1–6.6 by default.

### 4.2 Performance

| Metric | Value |
|--------|-------|
| Wall time | **5 h 54 min** |
| Performance | **~81 ns/day** |
| Mean temperature | **310.06 K** |

### 4.3 Analysis (20 ns)

| Metric | 2nd-half mean | Max |
|--------|---------------|-----|
| Protein backbone RMSD (fit backbone) | **~1.8 Å** | ~2.2 Å |
| Ligand RMSD (fit backbone) | **~3.2 Å** | ~4.8 Å |
| Visual (ChimeraX) | Pocket OK | user confirmed |

**Analysis files:**

```text
md_gromacs/runs/S_Warfarin_ref/
├── md_20ns.{xtc,gro,cpt,edr,tpr,log}
├── step6.*_equilibration.{gro,cpt}
├── em.gro
├── analysis/rmsd_protein.xvg
├── analysis/rmsd_ligand.xvg
├── analysis/view_prot_lig.pdb      # multi-model for ChimeraX
├── analysis/view_last_frame.pdb
└── temperature.xvg
```

**ChimeraX note:** Raw `md_20ns.xtc` from WSL paths is awkward (134k atoms). Use exported PDBs or `\\wsl.localhost\Ubuntu\home\amar\warfarin_project\...`.

Re-run analysis:

```bash
python3 md_gromacs/scripts/analyze_md.py \
  --system S_Warfarin_ref \
  --run-dir md_gromacs/runs/S_Warfarin_ref
```

---

## 5. Lessons learned

1. **Ligand chain X** — avoid chain B collision with CHARMM segment PROB.
2. **Docked SDF** — use `ligand_docked.sdf` (graph-matched coordinates), not bare `ligand.sdf`.
3. **PPM + Z translation** — default PPM may leave protein outside bilayer; adjust Z (~−55 Å for warfarin).
4. **Large fusion protein** — GFP–VKORC1 needs **XY ≥ 120 Å**.
5. **CHARMM equilibration** — use export `step6.*` mdps + `index.ndx`, not generic POSRES mdps.
6. **Unpack** — CHARMM download is `.tgz` with nested `gromacs/` folder; use `unpack_charmm_export.sh`.
7. **EM on CPU** — steep minimization in GROMACS 2024.3 is CPU-only.
8. **Production grompp** — pass `-n index.ndx` for `md_20ns` / `md_100ns` (SOLU/MEMB/SOLV groups).
9. **CHARMM_DIR path** — use absolute path or rely on fixed `run_one_complex.sh` (relative paths break after `cd` to run dir).
10. **Checkpoint restart** — only use `-cpi` when `.cpt` exists; script fixed 2026-06-29.
11. **Long runs** — use `tmux` (`tmux new -s rl100`); detach with `Ctrl+B` then `D`.
12. **Checkpoint resume without .gro** — if only `.cpt` + `.tpr` exist, skip grompp and `mdrun -cpi … -append` (fixed in `run_one_complex.sh` 2026-06-30).

---

## 6. RL_Gen_37 — manuscript lead compound

### 6.1 Manuscript context

- Primary RL manuscript lead compound.
- VKORC1 docking score: **−12.24 kcal/mol** (`publication/output/tables/table2_vkorc1_docking_leaderboard.csv`).
- CGenFF auto risk: **HIGH** — ParamChem fallback likely.

### 6.2 Input files

| File | Path |
|------|------|
| Complex PDB | `md_gromacs/systems/RL_Gen_37/charmm_gui/complex_clean_charmm.pdb` |
| Ligand SDF | `md_gromacs/systems/RL_Gen_37/ligand_params/ligand_docked.sdf` |
| CGenFF notes | `md_gromacs/systems/RL_Gen_37/ligand_params/CGenFF_NOTES.md` |

SMILES (`RL_Gen_37_isoA`, CIP R):

```
O=C1N(CC2CC2)CCC12CCN([C@H](c1ccccc1)c1ccc(F)cc1)CC2
```

### 6.3 CHARMM-GUI workflow

**Full click-by-click guide:** `md_gromacs/CHARMM_GUI_RL_Gen_37.md`

Summary:

1. New Membrane Builder job — upload PDB + SDF; chain **X** = **LIG**.
2. If auto CGenFF fails: ParamChem → re-upload mol2/str via Ligand Reader.
3. PPM 2.0; Z translation if protein not between membrane planes.
4. Same lipids as warfarin; **XY = 120 Å** initially.
5. NaCl 0.15 M; CHARMM36m; TIP3P; 310 K; GROMACS export.
6. Download: `charmm-gui (1).tgz` in Windows Downloads (not the warfarin `charmm-gui.tgz`).

**CHARMM job ID (RL_Gen_37):** **8273553994**

**Download file:** `/mnt/c/Users/Urim/Downloads/charmm-gui (1).tgz`

**Export path:** `md_gromacs/charmm_gui_export/RL_Gen_37/` — **133,746 atoms**, LIG present, grompp OK

### 6.4 GROMACS workflow (EM → equilibration → production)

```bash
export PATH="$HOME/gromacs/bin:$PATH"
export GMX=gmx GPU_ID=0 NTOMP=8
cd ~/warfarin_project/md_gromacs/scripts

# 20 ns pilot (after CHARMM export validated):
./run_one_complex.sh RL_Gen_37 ../charmm_gui_export/RL_Gen_37 pilot

# 100 ns production (after pilot go/no-go):
./run_one_complex.sh RL_Gen_37 ../charmm_gui_export/RL_Gen_37 production
```

Equilibration wall time (2026-06-29): **~42 min** (EM + step6.1–6.6).

### 6.5 RL_Gen_37 — 20 ns pilot results (2026-06-29)

| Metric | Value | vs S_Warfarin_ref |
|--------|-------|-------------------|
| Wall time | **5 h 41 min** | 5 h 54 min |
| Performance | **84.5 ns/day** | 81 ns/day |
| Mean temperature | **310.08 K** | 310.06 K |
| Mean pressure | **1.74 bar** | — |
| Protein RMSD 2nd-half mean (fit backbone) | **2.25 Å** | 1.76 Å |
| Protein RMSD 2nd-half max | **3.04 Å** | 2.16 Å |
| Ligand RMSD 2nd-half mean (fit backbone) | **2.51 Å** | 3.18 Å |
| Ligand RMSD 2nd-half max | **3.17 Å** | 3.77 Å |
| LINCS warnings | **None** | None |
| Go/no-go for 100 ns | **GO** | — |

Pilot go/no-go rationale: T and P stable; no simulation errors; ligand RMSD **better** than warfarin reference; protein RMSD slightly higher but within acceptable range (2nd-half max ~3 Å). Last-frame PDB exported for visual check.

**Analysis files:**

```text
md_gromacs/runs/RL_Gen_37/
├── md_20ns.{xtc,gro,cpt,edr,tpr,log}     # 20 ns pilot (complete)
├── md_100ns.{xtc,gro,cpt,edr,tpr,log}     # 100 ns production (complete)
├── step6.*_equilibration.{gro,cpt,xtc}
├── em.gro
├── analysis/rmsd_protein.xvg               # from md_100ns (longest traj)
├── analysis/rmsd_ligand.xvg
├── analysis/temperature_100ns.xvg
└── analysis/view_last_frame.pdb            # export at t=100 ns (see below)
```

Re-run analysis (uses longest available trajectory — `md_100ns` when complete):

```bash
python3 md_gromacs/scripts/analyze_md.py \
  --system RL_Gen_37 \
  --run-dir md_gromacs/runs/RL_Gen_37
```

Export last frame for ChimeraX (100 ns):

```bash
cd md_gromacs/runs/RL_Gen_37
echo "System" | gmx trjconv -s md_100ns.tpr -f md_100ns.xtc \
  -o analysis/view_last_frame_100ns.pdb -dump 100000
```

### 6.6 RL_Gen_37 — 100 ns production results (2026-07-01)

| Metric | Value | vs S_Warfarin_ref (20 ns) |
|--------|-------|---------------------------|
| Finished | **2026-07-01 21:25** | — |
| Trajectory length | **100 ns** (50 M steps @ 2 fs) | 20 ns |
| Checkpoint resumes | 2 (Ctrl+C stops; resumed without data loss) | — |
| Final segment wall time | **10 h 50 min** (~38 ns remaining) | — |
| Performance | **84.6 ns/day** | 81 ns/day |
| Mean temperature | **310.06 K** | 310.06 K |
| Mean pressure | **1.02 bar** | — |
| Protein RMSD full mean (fit backbone) | **2.53 Å** | 1.54 Å |
| Protein RMSD 2nd-half mean | **2.53 Å** | 1.76 Å |
| Protein RMSD last 25% mean | **2.32 Å** | 1.77 Å |
| Protein RMSD max | **3.70 Å** (early relaxation) | 2.16 Å |
| Ligand RMSD 2nd-half mean (fit backbone) | **1.92 Å** | 3.18 Å |
| Ligand RMSD last 25% mean | **1.90 Å** | 3.17 Å |
| Ligand RMSD max | **2.83 Å** | 4.83 Å |
| 2nd-half RMSD drift (protein / ligand) | **−0.019 / −0.002 Å/ns** | ~0 |
| LINCS warnings | **None** | None |

**Quarterly ligand RMSD (100 ns, means):** Q1 1.85 → Q2 1.80 → Q3 1.93 → Q4 1.90 Å — stable throughout.

**Interpretation (publication):**

- **Thermodynamics:** T and P stable over 100 ns; bilayer and barostat behaved normally.
- **Ligand:** RMSD **lower and more stable** than warfarin reference — consistent with retained VKORC1 pocket binding over 100 ns despite HIGH CGenFF risk.
- **Protein:** Higher RMSD than warfarin reflects GFP–VKORC1 fusion flexibility; early spike (~3.3 Å @ 25 ns) is post-equilibration relaxation, not sustained drift. **Last 25 ns shows lowest protein RMSD (2.32 Å mean).**
- **Verdict:** **Suitable for manuscript** — binding-mode stability claim supported; trajectory ready for MM-GBSA / interaction analysis / figures.

#### Stop / resume (used successfully)

Production was split across sessions via checkpoint:

1. **`Ctrl+C`** once → wait for prompt (writes `md_100ns.cpt`)
2. Resume: `./run_one_complex.sh RL_Gen_37 ../charmm_gui_export/RL_Gen_37 production`
3. If `.gro` missing but `.cpt` + `.tpr` exist, script uses existing `.tpr` (see lesson #12)

#### Start (with tmux)

```bash
tmux new -s rl100

export PATH="$HOME/gromacs/bin:$PATH"
export GMX=gmx GPU_ID=0 NTOMP=8
cd ~/warfarin_project/md_gromacs/scripts
./run_one_complex.sh RL_Gen_37 ../charmm_gui_export/RL_Gen_37 production
```

Detach (leave running): **`Ctrl+B`** then **`D`**

Reattach: `tmux attach -t rl100`

#### Stop mid-session (continue tomorrow)

1. Attach: `tmux attach -t rl100`
2. Press **`Ctrl+C` once** in the terminal running `mdrun`
3. Wait for GROMACS to exit cleanly (writes `md_100ns.cpt`)
4. Detach: `Ctrl+B` then `D`

Verify checkpoint:

```bash
ls -lh ~/warfarin_project/md_gromacs/runs/RL_Gen_37/md_100ns.cpt
```

**Do not** use `kill -9`.

#### Resume next day

```bash
tmux attach -t rl100   # or: tmux new -s rl100

export PATH="$HOME/gromacs/bin:$PATH"
export GMX=gmx GPU_ID=0 NTOMP=8
cd ~/warfarin_project/md_gromacs/scripts
./run_one_complex.sh RL_Gen_37 ../charmm_gui_export/RL_Gen_37 production
```

Expect: `Restarting md_100ns from checkpoint` / `Restarting md_100ns mdrun from checkpoint`

Manual resume alternative:

```bash
cd ~/warfarin_project/md_gromacs/runs/RL_Gen_37
gmx mdrun -v -ntmpi 1 -ntomp 8 \
  -nb gpu -pme gpu -bonded gpu -update gpu -gpu_id 0 \
  -deffnm md_100ns -cpi md_100ns.cpt -append
```

#### Monitor

```bash
tail -f ~/warfarin_project/md_gromacs/runs/RL_Gen_37/md_100ns.log
pgrep -af "gmx mdrun.*md_100ns"
grep "will finish" ~/warfarin_project/md_gromacs/runs/RL_Gen_37/md_100ns.log | tail -1
```

Progress: `step N / 50,000,000` → N/50M × 100 = % complete.

---

## 7. Systems manifest (run order)

| System | CGenFF risk | CHARMM status | MD status |
|--------|-------------|---------------|-----------|
| S_Warfarin_ref | low | Done (8266243616) | 20 ns pilot done |
| p_nitro_R / S | medium | Not started | — |
| dimethoxy_23_S | medium | Not started | — |
| **RL_Gen_37** | **high** | Done (8273553994) | **100 ns complete** (2026-07-01) |
| RL_Gen_29_isoA | high | Not started | — |

Manuscript RL targets (not all in repo): RL_Gen_37, RL_Gen_29_isoA, RL_Gen_22, RL_Gen_45.

---

## 8. Key scripts and mdps

| File | Purpose |
|------|---------|
| `scripts/unpack_charmm_export.sh` | Unpack `.tgz`/`.zip` CHARMM download |
| `scripts/validate_charmm_export.sh` | Check topol, gro, LIG, grompp dry-run |
| `scripts/run_one_complex.sh` | EM → step6.1–6.6 → pilot/production |
| `scripts/run_rl_gen_37_after_export.sh` | Unpack + validate + RL_Gen_37 pilot (one-shot) |
| `scripts/analyze_md.py` | RMSD + H-bond occupancy |
| `mdp/em.mdp` | Energy minimization |
| `mdp/md_20ns_pilot.mdp` | 20 ns pilot (CHARMM groups) |
| `mdp/md_100ns_production.mdp` | 100 ns production (aligned with pilot) |

---

## 9. Methods text (draft)

> VKORC1–ligand complexes from rigid-body docking (`md_poses/`) were embedded in an ER-like homogeneous bilayer (POPC/POPE/POPS/cholesterol) using CHARMM-GUI Membrane Builder (CHARMM36m, TIP3P, 0.15 M NaCl, 310 K). Ligand coordinates from docking were merged with CGenFF parameters by graph-matching docked heavy-atom coordinates to RDKit-generated SDF topology (`build_docked_ligand_mol2.py`), with ligand on chain X to avoid collision with CHARMM segment PROB. Protein orientation used PPM 2.0 with manual Z repositioning when required. Equilibration followed CHARMM-GUI step6.1–6.6 protocols. Production MD used GROMACS 2024.3 (CUDA) on an RTX 4060 Ti (WSL2).
>
> **S_Warfarin_ref** (20 ns pilot): mean temperature 310.06 K; protein backbone RMSD ~1.8 Å; ligand RMSD ~3.2 Å (both vs fit protein backbone).
>
> **RL_Gen_37** (manuscript lead; **100 ns production**): mean temperature 310.06 K; mean pressure 1.02 bar; protein backbone RMSD 2.53 Å (full trajectory mean, fit backbone), 2.32 Å (last 25 ns); ligand RMSD 1.92 Å (2nd-half mean), max 2.83 Å; GROMACS 2024.3 CUDA, RTX 4060 Ti, ~84.6 ns/day. Trajectory demonstrates stable ligand retention in the VKORC1 binding site over 100 ns.

---

## 10. New chat starter prompt

```
Continue warfarin MD project. Read md_gromacs/MD_PILOT_HANDOVER.md.

S_Warfarin_ref: 20 ns pilot done (T 310 K, protein RMSD ~1.8 Å, ligand ~3.2 Å). Skip warfarin 100 ns.

RL_Gen_37: 100 ns COMPLETE (2026-07-01). T 310.06 K, protein RMSD 2.53 Å (last 25 ns: 2.32 Å), ligand RMSD 1.92 Å (2nd half). Publication-ready.

Run dir: md_gromacs/runs/RL_Gen_37/  (md_100ns.xtc, 479 MB)
Export: md_gromacs/charmm_gui_export/RL_Gen_37/

Next: RL_Gen_29_isoA CHARMM-GUI or analysis/MM-GBSA on RL_Gen_37.
```
