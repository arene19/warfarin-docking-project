# Publication data files

| File | Description |
|------|-------------|
| `gnn_evaluation_report.json` | Test metrics + baselines from `gnn_baseline_evaluation.py` |
| `gnn_vkorc1_label_audit.json` | Train/test VKORC1 label composition (ChEMBL vs RL-library overlaps) |
| `gnn_vkorc1_chembl_only_benchmark.json` | Table S5 ChEMBL-only VKORC1 metrics |
| `flexible_redock_spotcheck.csv` | Canonical flexible re-dock QC (8 RL_Gen + 6 reference rows; Figure 5) |
| `gnn_training_history.json` | Per-epoch train/val loss (Figure 1) — created by `dynamic_gnn_pipeline.py` |
| `LABEL_PROVENANCE.md` | Label-source documentation |

Regenerate after re-running baselines:

```bash
python gnn_baseline_evaluation.py --split-from publication/data/gnn_scaffold_split.json
python publication/generate_manuscript_assets.py
```
