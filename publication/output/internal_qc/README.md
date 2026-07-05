# Internal QC artifacts (not cited in main manuscript)

| File | Description |
|------|-------------|
| `table_s_stereoselectivity.csv` | Reference enantiomer ΔΔG (S − R) for six scaffolds including m_bromo |
| `table2_vkorc1_combined_leaderboard.csv` | RL_Gen + reference flexible-docking ranks |
| `table_docking_enrichment_summary.csv` | Retrospective VKORC1 enrichment (ROC-AUC, EF) |
| `docking_enrichment_summary.txt` | Full enrichment protocol summary |
| `figure_docking_enrichment_roc.png` | ROC curve from `docking_validation.py` |
| `docking_seed_reproducibility_summary.txt` | Multi-seed reference-ligand reproducibility |
| `run_docking_validation.sh` | Rerun `docking_validation.py` (requires full repo + Vina) |
| `receptor_validation_report.json` | VKORC1_Human setup checks + S-warfarin re-dock RMSD |

Regenerate: `python docking_validation.py` then `bash publication/build_all.sh`.
