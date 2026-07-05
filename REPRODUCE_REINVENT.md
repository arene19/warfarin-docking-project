# Reproducing the REINVENT4 generative run

This document complements the Zenodo deposition (`reinvent/` bundle) with environment pins and rerun commands. The archived generation CSV (`REINVENT4/coumarin_generation_1.csv`, 9,600 rows) is the canonical manuscript output; checksum validation: `python scripts/validate_reinvent_provenance.py`.

## Pinned software

| Component | Version / identifier |
|-----------|----------------------|
| REINVENT4 | **4.7.15** (git `f9486d7ba26d7b3eaa1eb783eb404f041ccea64d`, tag `v4.7-38-gf9486d7`) |
| Python | 3.10.20 (`reinvent4` conda env) |
| PyTorch | 2.9.1+cu128 |
| RDKit | 2026.03.3 |
| Main pipeline (GNN/docking) | separate venv — see `ENVIRONMENT.md` (torch 2.12.0) |
| Project GNN checkpoint | `coagulation_admet_gnn.pth` (deposited under `models/`) |

Conda env file: `environment/reinvent4-environment.yml`

Clone REINVENT4 at the pinned commit:

```bash
git clone https://github.com/MolecularAI/REINVENT4.git
cd REINVENT4
git checkout f9486d7ba26d7b3eaa1eb783eb404f041ccea64d
pip install -e .
```

## Prior model (not in Zenodo bundle)

The REINVENT prior/agent weights (~23 MB) are **not** included in `deposition/package/` due to size. They ship with a standard REINVENT4 install:

```text
REINVENT4/models/reinvent.prior
```

After cloning REINVENT4, copy or symlink this file to the path referenced in `configs/coumarin_rl.toml`:

```toml
prior_file = "models/reinvent.prior"
agent_file = "models/reinvent.prior"
```

On the original workstation the prior lived at `REINVENT4/models/reinvent.prior`. Verify against `publication/data/reinvent_prior_provenance.json` (hash `173568c36e1fc3d95cab289c7d31ce0b`).

## GNN external scorer

REINVENT calls the project checkpoint via ExternalProcess (`configs/coumarin_rl.toml`):

```toml
params.executable = "python"
params.args = "gnn_predict.py --checkpoint coagulation_admet_gnn.pth --target Pred_VKORC1_pXC50"
```

From the repository root (with `requirements.txt` venv active):

```bash
pip install -r requirements.txt
# ensure coagulation_admet_gnn.pth is present
```

## Exact rerun command

```bash
cd REINVENT4
# copy configs/coumarin_rl.toml from repo root; adjust prior_file paths if needed
reinvent -l coumarin_rl.log ../configs/coumarin_rl.toml
```

Expected outputs (matching deposition):

- `coumarin_generation_1.csv` — scored SMILES with `Pred_VKORC1_pXC50`
- `coumarin_agent.chkpt` — trained agent checkpoint
- `coumarin_rl.json` — run configuration snapshot

Scoring weights (geometric mean): SA 1.5×, QED 1.0×, MW 0.5×, GNN VKORC1 2.5×.

## Bootstrap from deposition only

```bash
python scripts/bootstrap_results_from_deposition.py
python scripts/validate_reinvent_provenance.py
```

This restores the generation CSV and TOML but **cannot** rerun generation without installing REINVENT4 and obtaining `reinvent.prior`.

## Audit note

Root `coumarin_rl.json` reflects an earlier SA/QED/MW-only pilot (no GNN ExternalProcess). Manuscript Figure 7 and Table S3 cite the GNN-scored run in `REINVENT4/coumarin_generation_1.csv`.
