# Release checklist — GitHub ↔ Zenodo ↔ manuscript

Use this checklist before journal submission so the cited GitHub repo, Zenodo archive, and manuscript build date refer to the same snapshot.

## 1. Regenerate publication assets

```bash
python scripts/freeze_publication_master.py
ALLOW_STALE_CHECKPOINT=1 bash publication/build_all.sh
```

Verify:

- `deposition/package/manifest.json` → `"missing": []`
- `python scripts/validate_reinvent_provenance.py` → `VALIDATION: PASS`
- `publication/output/manifest.json` lists current figures/tables only

## 2. Commit and tag GitHub

```bash
git add -A
git status   # review; do not commit secrets (.env, local paths)
git commit -m "Publication snapshot: frozen master, manuscript assets, deposition bundle"
git tag -a v1.0-submission -m "Manuscript submission snapshot (YYYY-MM-DD)"
git push origin main
git push origin v1.0-submission
```

Record the tag commit SHA:

```bash
git rev-parse v1.0-submission
```

## 3. Upload Zenodo

Follow [`deposition/ZENODO_UPLOAD.md`](deposition/ZENODO_UPLOAD.md):

1. Zip or upload `deposition/package/`
2. Set related identifier → `https://github.com/arene19/warfarin-docking-project` (is supplement to)
3. Note GitHub tag `v1.0-submission` and commit SHA in the Zenodo description
4. Publish and copy the DOI

## 4. Update manuscript

In `vkorc1_integrated_workflow_manuscript.md` Data Availability, replace:

```text
https://doi.org/10.5281/zenodo.21208445
```

with the published DOI. Regenerate Word export:

```bash
python publication/md_to_docx.py
```

Output: `vkorc1_integrated_workflow_manuscript.docx`

## 5. Cross-reference table

| Artifact | Location | Version pin |
|----------|----------|-------------|
| Source code | GitHub tag `v1.0-submission` | commit SHA from step 2 |
| Data bundle | Zenodo DOI | built from same commit |
| Frozen master | `publication/data/coagulation_admet_multi_task_publication.csv` | SHA in `master_snapshot_meta.json` |
| REINVENT4 run | `REPRODUCE_REINVENT.md` | git `f9486d7` |
| Manuscript figures | `publication/output/figures/` | listed in `manifest.json` |
| MD summaries | `publication/data/md_results_summary.json`, `deposition/package/md_analysis/` | parsed from GROMACS `.xvg` |

## Bootstrap path for reviewers

Reviewers without the full `results/` tree:

```bash
git clone https://github.com/arene19/warfarin-docking-project
cd warfarin-docking-project
git checkout v1.0-submission
python scripts/bootstrap_results_from_deposition.py   # after downloading Zenodo package
ALLOW_STALE_CHECKPOINT=1 bash publication/build_all.sh
```

See also `deposition/package/README_DEPOSITION.txt`.
