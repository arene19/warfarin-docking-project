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
python dynamic_gnn_pipeline.py   # writes publication/data/gnn_scaffold_split.json
```

## Checkpoints

`coagulation_admet_gnn.pth` uses format v2 (state_dict + hyperparameters + metadata).
Legacy weights-only files are upgraded with:

```bash
python scripts/migrate_checkpoint.py
```

## Random seed

GNN training and baseline evaluation default to **seed = 42** (`gnn_model.DEFAULT_SEED`).
