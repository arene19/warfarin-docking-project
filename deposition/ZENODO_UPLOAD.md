# Zenodo deposition instructions

Upload the contents of [`deposition/package/`](../package/) as a single Zenodo record before or at submission.

## Steps

1. Create account at [https://zenodo.org](https://zenodo.org) (or log in with GitHub/ORCID).
2. **New upload** → drag the entire `deposition/package/` folder (or create a `.zip` of it).
3. Metadata (suggested):
   - **Title:** VKORC1 de novo ligand discovery — supplementary data (GNN, docking, REINVENT4)
   - **Upload type:** Dataset
   - **Description:** Frozen master CSV, GNN checkpoint, Murcko scaffold split, RL_Gen docking results, comparative membrane MD summaries (RL_Gen_37_isoA, RL_Gen_29_isoA, S-warfarin reference), manuscript tables/figures, REINVENT4 provenance, and reproduction scripts for the warfarin-docking-project manuscript.
   - **License:** Match repository LICENSE
   - **Related identifier:** `https://github.com/arene19/warfarin-docking-project` (is supplement to)
4. Publish the record and copy the **DOI** (e.g. `10.5281/zenodo.XXXXXXX`).
5. Update [`vkorc1_integrated_workflow_manuscript.md`](../../vkorc1_integrated_workflow_manuscript.md) Data Availability:

   ```text
   Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX
   ```

6. Tag the matching GitHub release (e.g. `v1.0-submission`) with the same commit used to build `deposition/package/` (full checklist: [`RELEASE.md`](../RELEASE.md)).

## Verify before upload

```bash
python scripts/freeze_publication_master.py
ALLOW_STALE_CHECKPOINT=1 bash publication/build_all.sh
```

Check `deposition/package/manifest.json` reports `"missing": []`.

Confirm the deposition bundle includes:
- `manuscript/vkorc1_integrated_workflow_manuscript.md`
- `metrics/md_results_summary.json` and `md_analysis/` per-system summaries
- `publication_output/figures/` and `publication_output/tables/`

## After upload — assign DOI in manuscript

1. Publish the Zenodo record and copy the DOI (e.g. `10.5281/zenodo.XXXXXXX`).
2. In [`vkorc1_integrated_workflow_manuscript.md`](../vkorc1_integrated_workflow_manuscript.md) §Data Availability, replace `https://doi.org/10.5281/zenodo.TBD` with the published DOI.
3. Regenerate Word: `python publication/md_to_docx.py` → `vkorc1_integrated_workflow_manuscript.docx`.

## Optional metadata file

See [`zenodo-metadata.json`](zenodo-metadata.json) for a draft JSON metadata block you can paste or adapt in the Zenodo form.
