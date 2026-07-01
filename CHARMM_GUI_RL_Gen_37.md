# CHARMM-GUI — RL_Gen_37 (manuscript lead)

**CHARMM job ID:** **8273553994**  
**Download file:** `charmm-gui (1).tgz` in Windows Downloads (`/mnt/c/Users/Urim/Downloads/charmm-gui (1).tgz`)  
**CGenFF risk:** **HIGH** (spiro/piperidinone, cyclopropyl, benzhydryl-like center)  
**Docking score (VKORC1):** −12.24 kcal/mol  
**Export:** unpacked to `md_gromacs/charmm_gui_export/RL_Gen_37/` — **133,746 atoms**, validation passed

Start a **new** CHARMM-GUI Membrane Builder job. Do not reuse the S_Warfarin_ref job.

---

## Files to upload

| Role | Path (from repo root) |
|------|------------------------|
| Protein–ligand PDB | `md_gromacs/systems/RL_Gen_37/charmm_gui/complex_clean_charmm.pdb` |
| Ligand for CGenFF | `md_gromacs/systems/RL_Gen_37/ligand_params/ligand_docked.sdf` |
| Backup ligand | `md_gromacs/systems/RL_Gen_37/ligand_params/ligand_docked.mol2` |
| CGenFF notes | `md_gromacs/systems/RL_Gen_37/ligand_params/CGenFF_NOTES.md` |

**SMILES** (`RL_Gen_37_isoA`, CIP R):

```
O=C1N(CC2CC2)CCC12CCN([C@H](c1ccccc1)c1ccc(F)cc1)CC2
```

---

## Step 1 — PDB Reader / Ligand Reader & Modeler

1. Go to [CHARMM-GUI](https://www.charmm-gui.org/) → **Input Generator** → **Ligand Reader & Modeler** (or Membrane Builder with ligand upload).
2. Upload **`complex_clean_charmm.pdb`**.
3. Check/correct PDB format: **Yes**.
4. Verify segments: **PROA, PROB, PROC, HETA (LIG)** on chain **X**.
5. pH **7.4**; protonate/deprotonate: **Yes**.
6. For hetero **LIG** → CGenFF → upload **`ligand_docked.sdf`** (not bare `ligand.sdf`).
7. Confirm ligand is **non-covalent** in the binding site (do not apply covalent CYS patches).

### If auto CGenFF fails or penalties are high

1. Upload `ligand_docked.sdf` to [ParamChem](https://cgenff.umaryland.edu/).
2. Inspect penalty scores — target **< 50** per torsion for production.
3. Download mol2 + str; re-upload via CHARMM-GUI Ligand Reader.
4. Ensure residue name remains **LIG** and atom count matches the complex PDB.

**Pass criterion:** RL_Gen_37 visible in VKORC1 pocket in the web viewer.

Record job ID: **8273553994**

---

## Step 2 — Protein Orientation (PPM 2.0)

| Setting | Value |
|---------|-------|
| Method | **Run PPM 2.0** |
| Do **not** use | “Use PDB Orientation” |

**Pass criteria:**

- Transmembrane helices roughly normal to membrane plane (Z).
- `LIG` faces lipid headgroup region / pocket accessible to bilayer interface.
- Protein between predicted membrane planes.

### Z translation (warfarin lesson)

If the protein sits **above or below** the membrane slab (not embedded):

- Use **Translate along Z** on the repositioning page.
- Warfarin required **Z ≈ −55 Å**; RL_Gen_37 (larger GFP–VKORC1 fusion) may need similar — adjust until cross-section looks correct.

If `LIG` ends up in the membrane hydrophobic core, stop and fix orientation before lipid build.

---

## Step 3 — Membrane Components

| Lipid | Upper | Lower |
|-------|-------|-------|
| POPC | 6 | 6 |
| POPE | 2 | 2 |
| POPS | 1 | 1 |
| Cholesterol | 2 | 2 |

| Setting | Value |
|---------|-------|
| Water thickness | **22.5 Å** (each side) |
| XY box guess | **120 Å** (start here; warfarin failed at 90/100 Å) |
| Homogeneous bilayer | Yes |

If packing fails (“protein larger than bilayer”), increase XY to 120 Å (or 130 Å if needed).

---

## Step 4 — Assembly / lipid penetration check

- Fix **minor tail penetrations** (POPS/POPE) if flagged.
- **No ring penetration** into protein interior.
- Proceed if only minor lipid tail issues remain.

---

## Step 5 — Input Generation

| Setting | Value |
|---------|-------|
| Ions | **NaCl 0.15 M** (use NaCl, not KCl) |
| Neutralize | Yes |
| Force field | **CHARMM36m** |
| Water | **TIP3P** |
| Temperature | **310 K** |
| Output | **GROMACS** |

Submit and wait for the queue (often 10–60 min).

---

## Step 6 — Download and unpack

Download the **GROMACS** `.tgz` to Windows Downloads.

```bash
export PATH="$HOME/gromacs/bin:$PATH"
cd ~/warfarin_project/md_gromacs/scripts

./run_rl_gen_37_after_export.sh
# or: ./run_rl_gen_37_after_export.sh "/mnt/c/Users/Urim/Downloads/charmm-gui (1).tgz"

# Step by step:
./unpack_charmm_export.sh RL_Gen_37 "/mnt/c/Users/Urim/Downloads/charmm-gui (1).tgz"
./validate_charmm_export.sh ../charmm_gui_export/RL_Gen_37
```

Expected layout:

```text
md_gromacs/charmm_gui_export/RL_Gen_37/
├── topol.top
├── step5_input.gro
├── index.ndx
├── step6.*_equilibration.mdp
└── toppar/
```

---

## Step 7 — GROMACS pilot (20 ns)

```bash
export GMX=gmx GPU_ID=0 NTOMP=8
./run_one_complex.sh RL_Gen_37 ../charmm_gui_export/RL_Gen_37 pilot
```

Pipeline: **EM (CPU)** → **step6.1–6.6 (GPU)** → **20 ns production (GPU)**.

Disk: reserve ~50–100 GB under `md_gromacs/runs/RL_Gen_37/`.

---

## Step 8 — Analysis and 100 ns decision

```bash
python3 analyze_md.py --system RL_Gen_37 --run-dir ../runs/RL_Gen_37
```

**Go for 100 ns** if T ~310 K, protein RMSD stable, ligand in pocket, bilayer intact:

```bash
./run_one_complex.sh RL_Gen_37 ../charmm_gui_export/RL_Gen_37 production
```

---

## Quick visual check (ChimeraX)

Open `step5_input.gro` or `analysis/view_last_frame.pdb` after pilot:

- `LIG` in VKORC1 pocket.
- Fusion protein spans bilayer sensibly.
- No obvious lipid–aromatic ring clashes.
