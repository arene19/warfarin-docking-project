# Publication master CSV snapshot

## Files

| File | Role |
|------|------|
| `coagulation_admet_multi_task_publication.csv` | Frozen master table used by `publication/build_all.sh`, deposition, and GNN benchmark regeneration |
| `master_snapshot_meta.json` | Row counts, SHA-256, split semantics |
| `gnn_scaffold_split.json` | Frozen Murcko partition for benchmarks (18,966 compounds) |

## Semantics

- **18,976** compounds in the frozen publication master (includes post-split RL active-learning merges).
- **18,966** compounds assigned to the frozen Murcko scaffold split used for Table 1 / S5.
- **10** compounds sit outside the split (RL-derived VKORC1 pseudo-labels merged after the split was frozen); they are **not** used in benchmark metrics.

Regenerate the snapshot after intentional active-learning updates:

```bash
python active_learning_merger.py          # optional: mutates data/coagulation_admet_multi_task.csv
python scripts/freeze_publication_master.py
bash publication/build_all.sh
```

`build_all.sh` does **not** run the merger unless `RUN_ACTIVE_LEARNING_MERGE=1`.
