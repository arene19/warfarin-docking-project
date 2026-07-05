# Zenodo deposition instructions

Upload the contents of [`deposition/package/`](package/) as a single Zenodo record (or upload `deposition/vkorc1_deposition_package.zip`).

## Published record

- **Version DOI (cite this):** https://doi.org/10.5281/zenodo.21208445 (version 1.0)
- **Concept DOI (all versions):** https://doi.org/10.5281/zenodo.21208444
- **Related GitHub:** https://github.com/arene19/warfarin-docking-project (tag `v1.0-submission`)

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
print('ZIP OK')
"
```

Then on Zenodo:

1. Open [record 21208445](https://doi.org/10.5281/zenodo.21208445) → **New version** (recommended).
2. Replace `vkorc1_deposition_package.zip` with the freshly built archive.
3. Confirm metadata: MIT license, related identifier → GitHub repo, title includes MD.
4. Publish (version 1.0.1 if creating a new version).

Confirm the deposition bundle includes:

- `manuscript/vkorc1_integrated_workflow_manuscript.md` (Data Availability must cite `10.5281/zenodo.21208445`)
- `metrics/md_results_summary.json` and `md_analysis/` per-system summaries
- `publication_output/figures/` and `publication_output/tables/`

## Optional metadata file

See [`zenodo-metadata.json`](zenodo-metadata.json) for a JSON metadata block matching the live record.
