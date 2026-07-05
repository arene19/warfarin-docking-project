# Zenodo deposition instructions

Upload the contents of [`deposition/package/`](package/) as a single Zenodo record (or upload `deposition/vkorc1_deposition_package.zip`).

## Published record

- **Version DOI (cite this):** https://doi.org/10.5281/zenodo.21209303 (version 1.0.1)
- **Concept DOI (all versions):** https://doi.org/10.5281/zenodo.21208444
- **Related GitHub:** https://github.com/arene19/warfarin-docking-project (tag `v1.0-submission`)

Version 1.0 (https://doi.org/10.5281/zenodo.21208445) is superseded; cite **21209303** for the post-audit bundle.

## Re-upload after refreshing the bundle

When manuscript or deposition assets change:

```bash
python scripts/freeze_publication_master.py
ALLOW_STALE_CHECKPOINT=1 bash publication/build_all.sh
rm -f deposition/vkorc1_deposition_package.zip
(cd deposition && zip -r vkorc1_deposition_package.zip package/)
```

Verify before upload:

```bash
cat deposition/package/manifest.json   # expect "missing": []
python3 -c "
import zipfile
z = zipfile.ZipFile('deposition/vkorc1_deposition_package.zip')
for path in ['package/manuscript/vkorc1_integrated_workflow_manuscript.md', 'package/RELEASE.md']:
    assert b'zenodo.TBD' not in z.read(path), path
    assert b'10.5281/zenodo.21209303' in z.read(path), path
print('ZIP OK')
"
```

Then on Zenodo:

1. Open [latest record](https://doi.org/10.5281/zenodo.21209303) → **New version** (recommended).
2. Remove the old zip in the draft and upload the freshly built `vkorc1_deposition_package.zip`.
3. Confirm metadata: MIT license, related identifier → GitHub repo, title includes MD.
4. Publish.

Confirm the deposition bundle includes:

- `manuscript/vkorc1_integrated_workflow_manuscript.md` (Data Availability must cite `10.5281/zenodo.21209303`)
- `metrics/md_results_summary.json` and `md_analysis/` per-system summaries
- `publication_output/figures/` and `publication_output/tables/`

## Optional metadata file

See [`zenodo-metadata.json`](zenodo-metadata.json) for a JSON metadata block matching the live record.
