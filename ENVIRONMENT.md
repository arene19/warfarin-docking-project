# Environment pinning

This project uses **two separate Python environments** by design. Do not merge them into one env.

## 1. Main pipeline (`warfarin` venv)

Used for GNN training, docking, publication build, and deposition.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

| Package | Pinned version (`requirements.txt`) |
|---------|-------------------------------------|
| Python | 3.10+ recommended |
| torch | **2.12.0** |
| torch-geometric | 2.8.0 |
| rdkit | 2026.3.1 |
| vina | 1.2.7 |

GNN checkpoint provenance is recorded in `coagulation_admet_gnn.pth` metadata (`benchmark_data_path`, `split_from`, `publication_master_sha256`).

## 2. REINVENT4 generative run (`reinvent4` conda env)

Used only for staged-learning generation. Recorded in `REINVENT4/coumarin_rl.log`:

| Package | Run-time version |
|---------|------------------|
| Python | 3.10.20 |
| REINVENT4 | 4.7.15 (git `f9486d7`) |
| torch | 2.9.1+cu128 |
| rdkit | 2026.03.3 |

```bash
conda env create -f environment/reinvent4-environment.yml
conda activate reinvent4
# clone REINVENT4 at f9486d7 and pip install -e .
```

See `REPRODUCE_REINVENT.md` for the full generative rerun. Prior model: `REINVENT4/models/reinvent.prior` (hash in `publication/data/reinvent_prior_provenance.json`).

## 3. Word export (optional)

```bash
pip install python-docx   # or use .docx_venv/
```

## Why two PyTorch versions?

The GNN was trained and benchmarked under torch 2.12 (main venv). REINVENT4 4.7.15 was executed under torch 2.9.1 in an isolated conda env. The archived generation CSV (`REINVENT4/coumarin_generation_1.csv`) is the canonical generative output; rerunning REINVENT requires the reinvent4 env, not the main venv.
