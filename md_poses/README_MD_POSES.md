# MD pose package — VKORC1_Human (corrected enantiomers)

Prepared for molecular dynamics starting structures.

## Contents

- `receptor/VKORC1_Human_chainA_protonated.pdb` — protonated receptor (6WV3, chain A)
- `complexes/*_VKORC1_Human_model1_complex.pdb` — receptor + ligand (MODEL 1 dock pose)
- `ligands/*_model1_ligand.pdb` — ligand only (chain B, resname `LIG`)
- `md_pose_manifest.csv` — affinities and file paths

## Ligands (8)

| Ligand | Role |
|--------|------|
| S_Warfarin_ref | Clinical S-warfarin reference |
| p_nitro_R | Top para-nitro (R) reference |
| p_nitro_S | Para-nitro (S) stereochemistry pair |
| dimethoxy_23_S | Dimethoxy reference (S) |
| RL_Gen_37 | Top GNN/REINVENT hit (manuscript MD-1) |
| RL_Gen_29_isoA | GNN hit; rich H-bond network (manuscript MD-2) |
| RL_Gen_22 | Third-ranked RL hit (manuscript MD-3) |
| RL_Gen_45 | Fourth-ranked RL hit; spirocyclic (manuscript MD-4) |

## Docking provenance

- **Target:** VKORC1_Human, flexible side-chains (A:217, A:276, A:269, A:272)
- **Pose:** AutoDock Vina MODEL 1 (best score) from corrected-enantiomer screening
- **HSA values:** from `results/docked_poses/HSA/` (corrected enantiomers, Jun 2026)

## Suggested MD workflow

1. Load complex PDB in GROMACS/AMBER/OpenMM
2. Assign ligand charges (AM1-BCC / RESP) — ligand PDB has no charges beyond 0.00 occupancy
3. Solvate, ionize, minimize, equilibrate, production
4. Optional: MM-PBSA on production frames

## RL_Gen_37 stereochemistry

Use the flat-dock MODEL 1 pose with **R** configuration (CIP R from 3D embed matches isoA).
