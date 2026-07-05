# Publication assets

Scripts and outputs for the VKORC1 de novo ligand discovery manuscript.

## Regenerate everything

From a full checkout with `results/` and trained checkpoints:

```bash
python scripts/freeze_publication_master.py   # once, or after RUN_ACTIVE_LEARNING_MERGE=1
ALLOW_STALE_CHECKPOINT=1 bash publication/build_all.sh
```

The build uses the **frozen** master at `publication/data/coagulation_admet_multi_task_publication.csv` and does **not** run `active_learning_merger.py` unless `RUN_ACTIVE_LEARNING_MERGE=1`.

If screening CSV is newer than the frozen master or GNN checkpoint, either re-freeze after merging or set `ALLOW_STALE_CHECKPOINT=1`.

Or bypass the guard (not recommended for publication rebuilds):

```bash
ALLOW_STALE_CHECKPOINT=1 bash publication/build_all.sh
```

From a Zenodo deposition only:

```bash
python scripts/bootstrap_results_from_deposition.py
# or, if you unzipped the bundle at repo root:
python scripts/bootstrap_results_from_deposition.py --package-dir .
python gnn_baseline_evaluation.py --split-from publication/data/gnn_scaffold_split.json
python publication/generate_manuscript_assets.py
```

## Outputs

| Path | Content |
|------|---------|
| `output/figures/` | Manuscript figures (`figure1`–`figure9`) |
| `output/tables/` | Manuscript tables only (RL_Gen case study + GNN metrics) |
| `output/internal_qc/` | Reference-ligand QC tables (not cited in manuscript) |
| `output/data/` | Build-time synced spot-check copy (canonical QC in `data/flexible_redock_spotcheck.csv`) |
| `data/gnn_evaluation_report.json` | GNN baseline metrics |
| `data/flexible_redock_spotcheck.csv` | Canonical flexible re-dock QC (8 RL_Gen + 6 reference rows) |
| `data/gnn_training_history.json` | Created by `dynamic_gnn_pipeline.py` |

## Figure list (manuscript numbering)

| Figure | File | Content |
|--------|------|---------|
| 1 | `figure1_gnn_training_curves` | GNN train/val loss and R² curves |
| 2 | `figure2_baseline_comparison` | GAT vs Morgan RF vs VKORC1-only GAT R² |
| 3 | `figure3_scaffold_split` | Murcko scaffold split bar chart |
| 4 | `figure4_morgan_rf_analysis` | Morgan ECFP4 + RF split / R² / Spearman / scatter |
| 5 | `figure5_flexible_redock_spotcheck` | Flexible vs. re-dock affinity spot-check |
| 6 | `figure6_interaction_heatmap` | VKORC1 interaction fingerprint (top RL_Gen) |
| 7 | `figure7_reinvent_vs_training_distribution` | REINVENT predicted VKORC1 vs. train-split labels |
| 8 | `figure8_md_rmsd_rl_gen_37` | Comparative membrane MD RMSD time series (3 systems) |
| 9 | `figure9_md_hbond_occupancy` | Binding-site H-bond occupancy heatmap |

MD metrics source: `publication/data/md_results_summary.json` (`scripts/parse_md_results_summary.py`).

Word export embeds figures via `publication/md_to_docx.py` (`FIGURE_FILES` map).
