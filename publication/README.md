# Publication assets

Scripts and outputs for the VKORC1 coumarin discovery manuscript.

## Regenerate everything

From the repository root (requires `reinvent4` or equivalent env with PyTorch, RDKit, matplotlib):

```bash
conda activate reinvent4
python gnn_baseline_evaluation.py          # if evaluation JSON missing
python dynamic_gnn_pipeline.py             # optional: refreshes checkpoint + training history
python publication/generate_manuscript_assets.py
python publication/md_to_docx.py           # Word export
```

## Outputs

| Path | Content |
|------|---------|
| `output/figures/` | Main figures (1–3) and supplementary (S1–S3) |
| `output/tables/` | CSV + Markdown tables for manuscript |
| `data/gnn_evaluation_report.json` | GNN baseline metrics (copy of `results/`) |
| `data/flexible_redock_spotcheck.csv` | 15-ligand flexible re-dock QC |
| `data/gnn_training_history.json` | Created by `dynamic_gnn_pipeline.py` |

## Figure list

- **Figure 1** — GNN train/val loss and R² curves
- **Figure 2** — VKORC1 interaction heatmap (H-bonds, hydrophobic, π-stacking)
- **Figure 3** — REINVENT predicted VKORC1 vs. training label distribution
- **Figure S1** — Murcko scaffold split bar chart
- **Figure 4** — Morgan ECFP4 + RF under Murcko scaffold split (split context, R², Spearman, VKORC1 scatter)
- **figure_morgan_rf_standalone** — Compact Morgan-only R² / Spearman / scatter row
- **Figure S3** — Flexible vs. re-dock affinity spot-check

Figures S4–S6 are reserved for MD simulations (Section 3.5).
