# VKORC1-Targeted Coumarin Anticoagulant Discovery

Multi-task graph neural network (GAT), flexible AutoDock Vina docking, active-learning data merger, and REINVENT4 reinforcement-learning loop for coumarin anticoagulant lead discovery against human VKORC1 (PDB 6WV3).

## Repository layout

```
├── gnn_model.py                   # Shared GAT architecture, featurization, checkpoints
├── config_utils.py                # config_master.yaml ↔ config.yaml sync
├── dynamic_gnn_pipeline.py        # Train multi-task GAT (seed=42, v2 checkpoint)
├── gnn_baseline_evaluation.py     # GAT vs Morgan-RF vs VKORC1-only ablation
├── morgan_fp_baseline.py          # Morgan ECFP4 + RF baseline
├── gnn_predict.py                 # REINVENT4 ExternalProcess scorer
├── main_pipeline.py               # Flexible Vina screening (respects active: flags)
├── active_learning_merger.py      # Merge RL_Gen docking → master CSV
├── inject_ai_leads.py             # Top REINVENT hits → config_master.yaml
├── configs/coumarin_rl.toml       # REINVENT4 RL config (GNN weight 2.5)
├── data/coagulation_admet_multi_task.csv   # Frozen training corpus
├── coagulation_admet_gnn.pth      # Trained checkpoint (v2 format)
├── deposition/prepare_deposition.py # Bundle Zenodo upload package
├── publication/                   # Figures, tables, Word export
└── manuscript_draft.md
```

Large docking pose trees under `results/` are gitignored. Run `python deposition/prepare_deposition.py` to bundle CSVs and metrics for Zenodo.

## Quick start

### 1. Environment

```bash
conda create -n warfarin python=3.10
conda activate warfarin
pip install -r requirements.txt
pip install torch torch-geometric   # match your CUDA build
```

REINVENT4 runs use a separate env (`reinvent4`). See `publication/data/reinvent_provenance.json` for which generation CSV is canonical.

### 2. Train / evaluate GNN

```bash
python dynamic_gnn_pipeline.py --seed 42
python gnn_baseline_evaluation.py
```

### 3. Regenerate manuscript + deposition

```bash
bash publication/build_all.sh
```

### 4. Docking (active ligands only)

```bash
python main_pipeline.py --config config.yaml --plip
```

`config.yaml` is auto-synced from `config_master.yaml` (flat SMILES, active ligands only) via `config_utils.sync_active_config()`.

### 5. Active-learning loop

```bash
python active_learning_merger.py      # merges into master CSV, preserves multi-task rows
python dynamic_gnn_pipeline.py
```

### 6. REINVENT4

```bash
conda activate reinvent4
reinvent -l reinvent.log configs/coumarin_rl.toml
python inject_ai_leads.py
python scripts/validate_reinvent_provenance.py
```

## Data & reproducibility

- Frozen dataset provenance: `data/DATA_PROVENANCE.md`
- Scaffold split indices: `publication/data/gnn_scaffold_split.json` (after training)
- REINVENT run provenance: `publication/data/reinvent_provenance.json`
- Zenodo bundle: `deposition/package/` (via `prepare_deposition.py`)

## Manuscript

Draft: `manuscript_draft.md` / `manuscript_draft.docx`  
Section **3.5 (MD simulations)** is reserved for upcoming GROMACS/AMBER work.

## License

MIT — see [LICENSE](LICENSE).
