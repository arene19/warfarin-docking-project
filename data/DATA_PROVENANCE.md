# Data provenance

## Frozen training corpus

The publication uses the **frozen CSV** at `coagulation_admet_multi_task.csv` (not live ChEMBL API pulls).

| Field | Value |
|-------|--------|
| Source | ChEMBL bioactivity records + RL_Gen flexible-docking pseudo-labels |
| Compounds | 18,966 unique structures (after active-learning merges) |
| VKORC1 labels | 109 (ChEMBL + docking-derived) |
| Last curated | June 2026 |

To rebuild from ChEMBL (optional, not required for reproduction):

```bash
python build_admet_multitask_dataset.py   # live API — snapshot will differ
python active_learning_merger.py          # merge RL_Gen docking labels
```

## Scaffold split

Murcko scaffold assignment and train/val/test SMILES are saved after training:

```bash
python active_learning_merger.py          # merge RL_Gen labels (skips val/test SMILES)
python dynamic_gnn_pipeline.py --split-from publication/data/gnn_scaffold_split.json
python gnn_baseline_evaluation.py --split-from publication/data/gnn_scaffold_split.json
```

## Regenerating assets from deposition only

If `results/` or checkpoints are missing on a fresh clone, restore from the Zenodo bundle:

```bash
python scripts/bootstrap_results_from_deposition.py
# Flat Zenodo unzip (metrics/ and docking/ at repo root):
python scripts/bootstrap_results_from_deposition.py --package-dir .
python gnn_baseline_evaluation.py --split-from publication/data/gnn_scaffold_split.json
python publication/generate_manuscript_assets.py
```

Bootstrap restores master CSV, checkpoints, metrics JSON, scaffold split, docking CSVs, ligand SDFs, and core Python modules (`gnn_model.py`, `morgan_fp_baseline.py`, etc.).

## Publication rebuild guard

`publication/build_all.sh` exits if the screening CSV is newer than the master CSV or GNN checkpoint. After merging new RL labels, retrain before rebuilding:

```bash
python active_learning_merger.py
python dynamic_gnn_pipeline.py --split-from publication/data/gnn_scaffold_split.json
bash publication/build_all.sh
```

## Checkpoints

`coagulation_admet_gnn.pth` uses format v2 (state_dict + hyperparameters + metadata).
Legacy weights-only files are upgraded with:

```bash
python scripts/migrate_checkpoint.py
```

## Random seed

GNN training and baseline evaluation default to **seed = 42** (`gnn_model.DEFAULT_SEED`).
