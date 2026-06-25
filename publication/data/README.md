# Publication data files

| File | Description |
|------|-------------|
| `gnn_evaluation_report.json` | Test metrics + baselines from `gnn_baseline_evaluation.py` |
| `flexible_redock_spotcheck.csv` | 15-ligand flexible re-dock QC (Figure S3) |
| `gnn_training_history.json` | Per-epoch train/val loss (Figure 1) — created by `dynamic_gnn_pipeline.py` |

Copy fresh evaluation JSON after re-running baselines:

```bash
cp results/gnn_evaluation_report.json publication/data/
```
