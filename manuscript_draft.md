# A Multi-Task Graph Neural Network and Active-Learning Framework for VKORC1-Targeted Coumarin Anticoagulant Discovery

**Authors:** [Author names]  
**Affiliations:** [Institutions]  
**Corresponding author:** [Email]

---

## Abstract

Vitamin K epoxide reductase complex subunit 1 (VKORC1) is the primary pharmacological target of coumarin anticoagulants such as warfarin. Structure-based virtual screening and generative molecular design can accelerate lead discovery, but both approaches suffer from sparse bioactivity labels, assay heterogeneity, and the cost of exhaustive experimental validation. Here we present an integrated computational pipeline that combines (i) a multi-task graph neural network (GNN) trained on coagulation and ADMET endpoints, (ii) flexible AutoDock Vina docking against a human VKORC1 crystal structure (PDB 6WV3), (iii) an active-learning loop that converts docking affinities into pseudo–pXC50 labels for model retraining, and (iv) reinforcement-learning–guided de novo generation via REINVENT4, with the trained GNN embedded as a custom scoring component. The GNN employs graph attention convolutions, masked multi-task regression across six endpoints, and Bemis–Murcko scaffold splitting to mitigate data leakage. On the held-out scaffold test set, pooled multi-task regression achieved RMSE = 1.05, MAE = 0.73, and R² = 0.56 across all labeled endpoints; VKORC1 absolute potency prediction remained weak (test R² = −0.14, *n* = 41), as expected given label sparsity (109 compounds total). Accordingly, the GNN is deployed in REINVENT4 as a **relative ranker and exploration bias**, not as a quantitative potency oracle—final prioritization relies on flexible docking and planned MD. Active-learning iterations expanded the training corpus with reinforcement-learning–generated candidates (RL_Gen series), whose top docked lead (RL_Gen_37) achieved a VKORC1 binding affinity of −12.24 kcal/mol under confirmed flexible docking. Rigorous quality control—including correction of enantiomer label swaps in reference ligands and empirical verification that historical training affinities are flexible-docking–equivalent (maximum |Δ| = 0.52 kcal/mol across 15 spot-checked ligands)—supports retention of the current model without retraining. **Molecular dynamics (MD) simulations of top-ranked complexes are planned to evaluate binding stability, induced-fit behavior, and interaction persistence beyond the rigid/flexible docking approximation; results will be reported in Section 3.5.** This work establishes a closed-loop, GNN-informed discovery platform for next-generation coumarin anticoagulants and provides a reproducible foundation for physics-based validation.

**Keywords:** VKORC1; coumarin; graph neural network; multi-task learning; active learning; AutoDock Vina; REINVENT4; drug discovery

---

## 1. Introduction

Oral anticoagulant therapy remains a cornerstone of thromboembolism prevention. Warfarin and related 4-hydroxycoumarins exert their effect by inhibiting VKORC1, blocking the recycling of vitamin K and thereby suppressing γ-carboxylation of clotting factors. Despite decades of clinical use, warfarin’s narrow therapeutic index, metabolic variability (notably CYP2C9 polymorphisms), and off-target human serum albumin (HSA) binding motivate the search for structurally related leads with improved selectivity and predictable pharmacokinetics.

High-throughput experimental profiling of coagulation targets and ADMET properties is expensive and incomplete. Machine learning on molecular graphs offers a complementary route: a single model can learn shared representations across related endpoints (VKORC1 potency, serine proteases, CYP2C9, HSA) while tolerating missing labels via multi-task masking. When experimental data are scarce, structure-based docking can supply additional weak supervision by translating binding free energies (ΔG) into approximate potency scales (pXC50), enabling an active-learning cycle in which each docking campaign enriches the next generation of predictive models and generative designs.

Generative molecular design adds another axis: reinforcement learning (RL) agents such as REINVENT4 can explore chemical space under composite reward functions that balance synthetic accessibility, drug-likeness, and model-predicted potency. Embedding a project-specific GNN as an external scoring process closes the loop between learned potency estimates and de novo structure generation.

In this study, we describe the design, training, validation, and deployment of such a pipeline applied to coumarin-derived VKORC1 ligands. We emphasize methodological rigor in data curation (scaffold splits, stereochemistry handling, flexible-receptor docking verification) and reserve a dedicated results section for all-atom MD simulations that will assess whether top docking poses remain stable under explicit solvent and thermal fluctuation.

---

## 2. Materials and Methods

### 2.1 Target and structural models

The primary target was human VKORC1 in complex with S-warfarin (PDB ID **6WV3**, chain A). The receptor was protonated and prepared for docking with flexible side-chain treatment at residues **A:217, A:269, A:272, and A:276**, selected based on proximity to the native ligand and prior flexible-docking benchmarks. A reduced VKORC1 model and additional off-target receptors (Factor Xa, Factor XIIa, thrombin, CYP2C9, HSA) were configured for multi-target screening but are reported here primarily in the context of the multi-task GNN labels derived from ChEMBL and related compilations.

Reference coumarin enantiomers included R/S warfarin, BENZ_R/S, and substituted analogues (p-nitro, m-nitro, m-bromo, dimethoxy-23), used for docking validation and RMSD assessment against the co-crystal pose.

### 2.2 Master dataset construction

The master training table (`coagulation_admet_multi_task.csv`) aggregates:

| Column | Description |
|--------|-------------|
| `canonical_smiles` | Unique molecular identifier |
| `is_coumarin` | Scaffold class flag |
| `VKORC1_pXC50` | VKORC1 potency (primary RL reward endpoint) |
| `Factor_XIIa_pXC50`, `Factor_Xa_pXC50`, `Thrombin_pXC50` | Coagulation cascade targets |
| `CYP2C9_pXC50` | Metabolic liability |
| `HSA_pXC50` | Plasma protein binding proxy |

The baseline corpus was compiled from ChEMBL and project-specific bioactivity records (~18,900 compounds after curation). Missing assay values are represented as NaN and excluded from the loss via per-task masking.

### 2.3 Graph featurization and multi-task GNN architecture

Each molecule was converted to a PyTorch Geometric graph:

- **Nodes:** RDKit atoms with **10** features: atomic number, degree, formal charge, hybridization, aromaticity, mass, implicit/explicit valence, total hydrogens, ring membership.
- **Edges:** Undirected bonds from RDKit topology.
- **Stereochemistry:** Explicit `@`/`@@` tags in SMILES are **not** used as input features; stereochemistry is stripped during Murcko scaffold generation to avoid spurious scaffold mismatches. The model is therefore chirality-blind at the graph level—a deliberate choice validated when enantiomer label corrections were shown not to alter learned graph→target mappings.

**Architecture (DynamicMultiTaskGNN):**

- **Encoder:** Four graph attention (GAT) convolution layers with hidden dimensions [64 → 128 → 128 → 64], ReLU activation, dropout (p = 0.1).
- **Pooling:** Global mean pooling over nodes.
- **Readout:** Two-layer MLP (128 hidden units → 6 task outputs).
- **Loss:** Masked mean squared error (MSE); only tasks with valid labels contribute.
- **Split:** Bemis–Murcko scaffold split (80% train / 10% validation / 10% test) to reduce analogue leakage.

**Training protocol:**

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr = 1×10⁻³, weight decay = 1×10⁻⁵) |
| Batch size | 64 |
| Max epochs | 150 |
| LR scheduler | ReduceLROnPlateau (patience = 15, factor = 0.5) |
| Early stopping | Patience = 25 (after 15-epoch warmup) |
| Checkpoint | Best validation masked MSE → `coagulation_admet_gnn.pth` |

Regularization was tuned iteratively (weight decay and dropout adjusted to balance under- and over-fitting on the masked multi-task objective). Final test-set metrics (pooled across all valid task labels): **RMSE = 1.05**, **MAE = 0.73**, **R² = 0.56** (report per-task breakdown in Table 1 and Supplementary Table S1).

**Baseline comparators** (same Murcko scaffold split, evaluated with `gnn_baseline_evaluation.py`):

1. **Morgan fingerprint + random forest (RF):** radius-2, 2048-bit ECFP4 fingerprints; per-task RF regressor (500 trees, scikit-learn defaults otherwise).
2. **VKORC1-only single-task GAT:** identical GAT encoder/readout architecture but one output head; trained only on the VKORC1 masked objective (same hyperparameters and early stopping as the multi-task model).

These baselines isolate whether multi-task graph learning adds value beyond classical fingerprints and whether VKORC1-specific training outperforms shared representation learning when VKORC1 labels are scarce (67 train / 41 test compounds with VKORC1 labels in the current split).

### 2.4 Flexible AutoDock Vina docking pipeline

Ligands were prepared from SMILES via conformer enumeration (20 conformers), Meeko protonation/torsion assignment, and conversion to PDBQT. Receptors used a cached flexible PDBQT map (`VKORC1_Human_chainA_protonated_flex.pdbqt`) with rigid receptor core and flexible side chains at the defined residues. AutoDock Vina was run with exhaustiveness = 20 (project default after pipeline hardening), energy range = 3.0 kcal/mol, and nine binding modes retained per ligand.

**Thermodynamic mapping for active learning:**

$$\mathrm{pXC}_{50} = -\frac{\Delta G}{1.36}$$

where ΔG is Vina’s best-scoring mode affinity (kcal/mol). This linear rescaling provides a consistent pseudo-label scale for merging docking results into the GNN training set.

### 2.5 Active-learning merger

The script `active_learning_merger.py` implements one iteration of the loop:

1. Read `VKORC1_Human_screening_results.csv`.
2. Filter ligands with names matching `RL_Gen_*`.
3. Resolve SMILES from `config.yaml` or, if absent, from `results/ligands/<name>.sdf` (RDKit fallback).
4. Convert ΔG → pXC50 and append to the master CSV.
5. **Deduplicate by canonical SMILES, retaining the strongest (maximum) VKORC1_pXC50** per structure.

This deduplication rule ensures that when a flat ligand and its enumerated stereoisomer represent the same graph (e.g., RL_Gen_37 and RL_Gen_37_isoA), the best docking pose survives.

After multiple merger cycles, the master dataset contained **18,966** unique structures, of which **109** carry a VKORC1_pXC50 label (ChEMBL + docking-derived). The trained checkpoint `coagulation_admet_gnn.pth` (v2 format with embedded hyperparameters; last saved June 2026) is consumed by both `dynamic_gnn_pipeline.py` and the REINVENT4 `gnn_predict.py` scorer.

### 2.6 Reinforcement-learning generative design (REINVENT4)

De novo generation used REINVENT4 staged learning (`configs/coumarin_rl.toml`) with:

| Component | Weight | Role |
|-----------|--------|------|
| Synthetic Accessibility (SA Score) | 1.5 | Penalize difficult syntheses |
| QED | 1.0 | Drug-likeness |
| Molecular Weight | 0.5 | Prefer 150–500 Da |
| **GNN VKORC1 potency (ExternalProcess)** | **2.5** | Custom `gnn_predict.py` endpoint |

**Run provenance.** The primary GNN-scored generation archive is `REINVENT4/coumarin_generation_1.csv` (includes `Pred_VKORC1_pXC50`; see `publication/data/reinvent_provenance.json`). An earlier pilot export at the repository root (`coumarin_rl.json`, SA/QED/MW only) predates GNN integration and is retained for audit only.

The external process invokes `gnn_predict.py`, which loads `coagulation_admet_gnn.pth`, featurizes stdin SMILES, predicts task index 0 (VKORC1), and returns JSON:

```json
{"version": 1, "payload": {"Pred_VKORC1_pXC50": [float, ...]}}
```

A sigmoid transform (low = 5.0, high = 9.0 pXC50) maps predictions into REINVENT’s [0, 1] reward scale. Diversity filtering used identical Murcko scaffolds (bucket size = 25, min score = 0.4). Top generations were injected into the ligand library via `inject_ai_leads.py` (auto-incrementing `RL_Gen_NN` names from the latest `coumarin_generation_*.csv`).

### 2.7 Stereochemistry enumeration

For ligands with undefined chiral centers, `enumerate_stereocenters.py` expands each entry in `config_master.yaml` into explicit `_isoA`, `_isoB`, … SMILES using RDKit’s `EnumerateStereoisomers`. Predefined enantiomers (reference warfarin, BENZ, nitro/bromo/dimethoxy pairs) were left unchanged.

### 2.8 Data-quality controls

**Enantiomer label correction:** Six reference pairs had R/S SMILES swapped relative to CIP convention. Re-docking with corrected labels and confirmed flexible side chains showed a pure label swap: each corrected R matched the previous S affinity within ±0.1 kcal/mol (Vina noise), with no systematic rigid→flexible shift.

**Flexible-docking equivalence of training labels:** Fifteen ligands spanning the training ΔG range (−12.2 to −5.6 kcal/mol)—six references plus nine RL_Gen spot-checks—were re-docked with confirmed flexible parameters. Maximum |Δ| between training and re-dock = **0.52 kcal/mol**; mean |Δ| = **0.087 kcal/mol** (see `publication/data/flexible_redock_spotcheck.csv`), consistent with flexible-vs-flexible reproducibility rather than a rigid-fallback artifact. Combined with the chirality-blind featurizer, these findings support **no GNN retraining** after label correction.

---

## 3. Results

### 3.1 Multi-task GNN performance and baseline comparison

On the held-out Murcko scaffold test set (*n* = 1,897 compounds), the final multi-task GAT achieved pooled masked regression performance of **RMSE = 1.05**, **MAE = 0.73**, and **R² = 0.56** (2,210 valid task–compound pairs across six endpoints). Performance was strongest on coagulation protease tasks with denser ChEMBL labels (Factor XIIa R² = 0.65; thrombin R² = 0.56; Factor Xa R² = 0.45) and modest on CYP2C9 (R² = 0.06). VKORC1 and HSA remained data-limited (*n* = 41 and 5 test labels, respectively) and are **not** interpreted as quantitative potency models.

**Table 1. Test-set regression metrics by endpoint (Murcko scaffold hold-out)**

| Task | *n* (test) | Multi-task GAT RMSE | Multi-task GAT R² | Morgan FP + RF R² | VKORC1-only GAT R² |
|------|------------|----------------------|-------------------|-------------------|---------------------|
| VKORC1 | 41 | 1.16 | −0.14 | 0.28 | −0.03 |
| Factor XIIa | 222 | 0.76 | 0.65 | 0.82 | — |
| Factor Xa | 623 | 1.21 | 0.45 | 0.67 | — |
| Thrombin | 599 | 1.01 | 0.56 | 0.69 | — |
| CYP2C9 | 720 | 0.87 | 0.06 | 0.17 | — |
| HSA | 5 | 6.19 | −0.20 | −0.01 | — |
| **Pooled (all tasks)** | **2,210** | **1.05** | **0.56** | — | — |

Full MAE values and Spearman rank correlations are given in Supplementary Table S1. **Figure S2** (`publication/output/figures/figure4_morgan_rf_analysis.png`) compares Morgan ECFP4 + random forest against the multi-task GAT under the same Murcko scaffold split (R², Spearman ρ, and VKORC1 scatter).

**Interpretation for deployment.** Absolute VKORC1 regression is weak under scaffold split (multi-task GAT R² = −0.14; VKORC1-only GAT R² = −0.03), confirming that the model must **not** be treated as a potency oracle. However, multi-task pretraining on ~6,500 thrombin and ~5,800 Factor Xa labels yields a shared graph encoder that transfers productively to related serine-protease endpoints and supports **multi-endpoint selectivity readouts** on partially characterized RL leads (Section 3.4). For VKORC1 specifically, Morgan FP + RF achieved higher test R² (0.28) and Spearman ρ (**0.55**) than either GAT variant (multi-task ρ = **0.21**; single-task ρ = **0.01**), indicating that simple fingerprint models remain competitive when labels are extremely sparse. We nevertheless retain the multi-task GAT in REINVENT4 because (i) it provides simultaneous ADMET/coagulation predictions in one forward pass, (ii) its REINVENT role is **relative ranking within generated libraries**, where even modest rank correlation can steer exploration toward coumarin-rich, drug-like regions before expensive docking, and (iii) the active-learning loop continuously augments VKORC1 pseudo-labels from Vina, improving future retraining cycles.

Learning curves (train/validation masked MSE, LR reductions, early-stopping epoch) are shown in **Figure 1** (`publication/output/figures/figure1_gnn_training_curves.png`).

### 3.2 Active-learning expansion of the VKORC1 label set

Each REINVENT4 campaign produced ranked CSVs (`REINVENT4/coumarin_generation_*.csv`) with model-predicted `Pred_VKORC1_pXC50`. After Vina validation, RL_Gen ligands were merged into the master dataset. The RL_Gen series spans **RL_Gen_01 through RL_Gen_49+** (with stereoisomer suffixes where enumerated), contributing **55** docked RL_Gen entries in the screening CSV (**49** unique base IDs after deduplication by canonical SMILES).

**Table 2. Top VKORC1 AutoDock Vina affinities (RL_Gen series, confirmed flexible docking)**

| Rank | Ligand | ΔG (kcal/mol) | pXC50 (derived) |
|------|--------|---------------|-----------------|
| 1 | RL_Gen_37 | −12.244 | 9.003 |
| 2 | RL_Gen_22 | −11.545 | 8.489 |
| 3 | RL_Gen_29 / isoA | −10.969 / −11.409 | 8.066 / 8.389 |
| 4 | RL_Gen_07 | −10.921 | 8.030 |
| 5 | RL_Gen_26 | −10.902 | 8.016 |
| … | … | … | … |

Reference **R-warfarin** (`R_Warfarin_ref`) docked at **−11.29 kcal/mol**; **S-warfarin** (`S_Warfarin_ref`) at **−10.92 kcal/mol** (co-crystal ligand in PDB 6WV3). RL_Gen_37 exceeds R-warfarin by ~0.95 kcal/mol in the docking score, nominating it as the primary lead for downstream biophysical validation.

Interaction fingerprints (hydrogen bonds, hydrophobic contacts, π-stacking) for top ligands against VKORC1 were profiled in `interaction_profile.csv` and summarized in **Figure 2** (`publication/output/figures/figure2_interaction_heatmap.png`).

### 3.3 REINVENT4–GNN closed-loop generation

Integrating the GNN as an ExternalProcess scoring component (weight 2.5× relative to QED) steered generation toward structures with **higher model-predicted VKORC1 scores** while SA/QED/MW filters maintained synthesizable, drug-like candidates. Importantly, the GNN score functions as a **composite reward channel and ranker**, not a calibrated pXC50 estimator: REINVENT optimizes relative scores within each generation batch, and downstream **flexible Vina docking** provides the primary structure-based filter (Section 3.2). Representative generations from `coumarin_generation_1.csv` were filtered (`gnn_filter_ai_leads.py`) and the top 50 exported for docking triage.

**Figure 3** (`publication/output/figures/figure3_reinvent_vs_training_distribution.png`): Distribution of `Pred_VKORC1_pXC50` across REINVENT generations vs. ChEMBL/docking training labels—demonstrates exploration beyond the training manifold while remaining within chemically accessible space.

### 3.4 ADMET and selectivity profiling

RDKit-derived descriptors (MW, LogP, TPSA, QED, Lipinski compliance) were computed for all docked ligands (`admet_profile.csv`). Coumarin RL leads generally satisfied Lipinski rules; larger RL_Gen scaffolds with extended aliphatic/linker regions showed elevated MW and LogP (flagged for medicinal chemistry optimization).

Multi-task GNN predictions for coagulation proteases and CYP2C9/HSA provide an *in silico* selectivity panel prior to synthesis.

**Table 3. Multi-task GNN predictions for top five RL_Gen leads (by VKORC1 docking score)**

| Ligand | ΔG (kcal/mol) | Pred VKORC1 | Pred Factor XIIa | Pred Factor Xa | Pred Thrombin | Pred CYP2C9 | Pred HSA |
|--------|---------------|-------------|------------------|----------------|---------------|-------------|----------|
| RL_Gen_37 | −12.24 | 7.52 | 4.63 | 7.24 | 5.61 | 4.66 | −6.19 |
| RL_Gen_22 | −11.55 | 7.38 | 5.29 | 6.20 | 4.87 | 5.28 | −3.39 |
| RL_Gen_45 | −11.40 | 6.77 | 4.35 | 5.58 | 5.57 | 4.98 | −1.62 |
| RL_Gen_49 | −11.24 | 7.58 | 5.09 | 6.37 | 4.84 | 5.37 | −1.74 |
| RL_Gen_29 | −10.97 | 7.28 | 4.22 | 7.37 | 4.23 | 5.05 | −1.65 |

*Source: `publication/output/tables/table3_top5_rl_gen_selectivity.csv` (regenerate with `python publication/generate_manuscript_assets.py`). Predictions are model outputs, not experimental potencies.*

### 3.5 Molecular dynamics simulations *(planned — results pending)*

> **This section is reserved for upcoming all-atom MD work. Insert results upon completion.**

Docking identifies low-energy binding modes but does not guarantee thermodynamic stability or kinetic residence times under explicit solvent. To address this limitation, MD simulations will be performed on selected complexes using **[GROMACS / AMBER — specify force field, e.g., AMBER ff14SB + GAFF2 for ligands]**:

**Planned systems:**

| System ID | Ligand | Receptor | Rationale |
|-----------|--------|----------|-----------|
| MD-1 | S-warfarin (reference) | VKORC1 (6WV3) | Positive control; RMSD vs. crystal |
| MD-2 | RL_Gen_37 | VKORC1 | Top RL lead (ΔG = −12.24 kcal/mol) |
| MD-3 | RL_Gen_22 | VKORC1 | Second-ranked RL lead |
| MD-4 | [SELECT] | VKORC1 | Stereoisomer pair (e.g., RL_Gen_37_isoA vs. isoB) |
| MD-5 | [SELECT] | VKORC1 | Weaker binder control (e.g., RL_Gen_06) |

**Simulation protocol (template — fill in actual parameters):**

1. **System preparation:** Docked poses merged into receptor structure; protonation state validated (PropKa / H++); orthorhombic box with **[TIP3P / TIP4P]** water, **[NaCl concentration]** mM; neutralizing ions as needed.
2. **Equilibration:** Energy minimization → NVT ( **[T]** K, **[duration]** ) → NPT ( **[P]** bar, **[duration]** ).
3. **Production:** **[100–500] ns** per replicate, **[N ≥ 3]** independent replicates per system.
4. **Analysis:**
   - Ligand RMSD and RMSF (global and per-residue)
   - Protein RMSD relative to crystal
   - Hydrogen-bond occupancy and key interaction persistence (Tyr139, Ser81, Asn80, etc.)
   - MM-PBSA / MM-GBSA binding free energy estimates
   - Comparison of flexible-residue conformational sampling (A:217, A:269, A:272, A:276) vs. docked pose

**Expected figures upon completion:**

- **Figure 4:** Time evolution of ligand RMSD (warfarin vs. RL_Gen_37 vs. controls)
- **Figure 5:** Representative MD snapshot with persistent H-bond network
- **Figure 6:** MM-PBSA ΔG_bind ranking vs. Vina ΔG (correlation plot)
- **Table 4:** MM-PBSA energies, H-bond occupancies, and residence times

**Placeholder results text (replace after simulations):**

> MD simulations of [N] VKORC1–ligand complexes were run for [X] ns. The reference warfarin complex remained stable (ligand RMSD < [ ] Å after [ ] ns), validating the simulation protocol. RL_Gen_37 [remained stable / underwent partial dissociation / exhibited alternative binding mode], with persistent contacts at [residues]. MM-PBSA estimated ΔG_bind was [ ] kcal/mol for RL_Gen_37 vs. [ ] kcal/mol for warfarin, [consistent / inconsistent] with the Vina ranking. These results [support / do not support] prioritization of RL_Gen_37 for synthesis.

---

## 4. Discussion

We have implemented a reproducible, closed-loop discovery platform that links multi-task molecular graph learning, flexible structure-based screening, and RL-driven de novo design for VKORC1-targeted coumarin anticoagulants. Several design choices warrant discussion.

**Chirality-blind graph featurization** simplified the pipeline when historical data mixed flat SMILES, enumerated isomers, and corrected enantiomer labels. Empirical verification showed that R/S pairs map to identical graphs and that label swaps do not alter the training signal—only human-readable annotations change. For future work, explicit tetrahedral stereochemistry features or separate enantiomer-resolved models may improve potency prediction when chiral discrimination becomes experimentally critical.

**Pseudo-labels from docking** extend supervision but inherit Vina’s approximations (fixed protonation, incomplete solvation, scoring-function error ~±1.5 kcal/mol). The active-learning merger’s “keep strongest affinity per SMILES” rule mitigates pose-selection noise. Spot-check re-docking confirmed that historical training affinities are flexible-docking–equivalent, avoiding a costly full retrain.

**Multi-task learning** shares representation across coagulation and ADMET endpoints, enabling simultaneous selectivity predictions on partially characterized RL leads even when VKORC1 labels are sparse. Test-set results (Table 1) show that transfer is meaningful for thrombin, Factor Xa, and Factor XIIa (R² = 0.45–0.65), while VKORC1 absolute regression remains unreliable (**R² = −0.14** on 41 test labels). This validates our deployment choice: the GNN is embedded in REINVENT4 as a **live ranker**—biasing generation toward graph regions associated with higher predicted VKORC1 scores—while **docking and planned MD** serve as the authoritative structure-based validators. A VKORC1-only GAT ablation (R² = −0.03, Spearman ρ = **0.01** on 41 test labels) underperformed the multi-task encoder (ρ = **0.21**), suggesting that coagulation-task pretraining provides useful representation even when the VKORC1 head itself is underdetermined.

**Limitations:** (1) GNN test R² on VKORC1 is negative (R² = −0.14; *n* = 41 test labels); Morgan FP + RF achieves better VKORC1 ranking (Spearman ρ = **0.55**) on the same split—reviewers should interpret REINVENT GNN scores as heuristic, not experimental potency; (2) docking scores are not experimental IC50 values; (3) RL_Gen_38 and higher combinatorial stereoisomer sets (e.g., 64 isomers for six undefined centers) remain incompletely docked; (4) **MD simulations (Section 3.5) are required to confirm that top-scoring poses are dynamically stable** and to rank leads by estimated binding free energy rather than docking score alone.

---

## 5. Conclusions

We developed and validated a multi-task GNN trained on a curated coagulation/ADMET corpus augmented by active-learning iterations from flexible VKORC1 docking of RL-generated coumarin analogues. The model achieves pooled test R² = 0.56 across six endpoints but weak VKORC1 regression (R² = −0.14); it is therefore deployed in REINVENT4 as a **multi-task ranker and selectivity panel**, not a potency oracle, with flexible docking as the primary validation gate. Data-quality audits (enantiomer relabeling, flexible-docking equivalence) support the current checkpoint without retraining. RL_Gen_37 emerges as the leading *in silico* candidate (ΔG = −12.24 kcal/mol). **Completion of all-atom MD simulations (Section 3.5) will provide the definitive biophysical assessment of binding stability and inform synthesis prioritization.** The pipeline code, configuration files, and trained weights (`coagulation_admet_gnn.pth`) constitute a reusable framework for structure-guided anticoagulant discovery.

---

## Author Contributions

[CRediT taxonomy — e.g., Conceptualization, Software, Validation, Writing — to be assigned]

## Data and Code Availability

- Master dataset: `data/coagulation_admet_multi_task.csv` (frozen snapshot; see `data/DATA_PROVENANCE.md`)
- Trained model: `coagulation_admet_gnn.pth` (v2 checkpoint with hyperparameters)
- Zenodo archive: run `python deposition/prepare_deposition.py` → upload `deposition/package/` (docking CSVs, interaction profiles, REINVENT generations, metrics)
- Training script: `dynamic_gnn_pipeline.py`
- Baseline evaluation: `gnn_baseline_evaluation.py` → `results/gnn_evaluation_report.json`
- Active-learning merger: `active_learning_merger.py`
- REINVENT scoring endpoint: `gnn_predict.py`
- REINVENT configuration: `configs/coumarin_rl.toml`
- Manuscript figures/tables: `publication/output/` (generate via `publication/generate_manuscript_assets.py`)
- Docking results (local): `results/docked_poses/VKORC1_Human/` *(not versioned; see README)*

## Acknowledgments

[Funding sources, compute resources, collaborators]

## References

1. Rost S, Fregin A, Ivaskevicius V, et al. Mutations in VKORC1 cause warfarin resistance and multiple coagulation factor deficiency type 2. *Nature* **2004**, *427*, 537–541.  
2. Bemis GW, Murcko MA. The properties of known drugs. 1. Molecular frameworks. *J Med Chem* **1996**, *39*, 2887–2893.  
3. Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. *Proc ICLR* **2018**.  
4. Trott O, Olson AJ. AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. *J Comput Chem* **2010**, *31*, 455–461.  
5. Blaschke T, Arús-Pous H, Chen H, et al. REINVENT 4: Modern AI-driven generative molecule design. *J Cheminform* **2024**, *16*, 20.  
6. Gaulton A, Hersey A, Bento AP, et al. The ChEMBL database in 2017. *Nucleic Acids Res* **2017**, *45*, D945–D954.  
7. Fey M, Lenssen JE. Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds* **2019**.  
8. Abraham MJ, Murtola T, Schulz R, et al. GROMACS: High performance molecular simulations through multi-level parallelism from laptops to supercomputers. *SoftwareX* **2015**, *1–2*, 19–25.  
9. Landrum G. RDKit: Open-source cheminformatics. https://www.rdkit.org (accessed 2025).  
10. Forli S, Huey R, Pique ME, Sanner MF, Goodsell DS, Olson AJ. Computational protein–ligand docking and virtual drug screening with the AutoDock suite. *Nat Protoc* **2016**, *11*, 905–919.  
11. Pedregosa F, et al. Scikit-learn: Machine learning in Python. *J Mach Learn Res* **2011**, *12*, 2825–2830.  
12. Rogers D, Hahn M. Extended-connectivity fingerprints. *J Chem Inf Model* **2010**, *50*, 742–754.  

---

## Supplementary Information

All supplementary tables and figures are auto-generated under `publication/output/`:

| Asset | File |
|-------|------|
| **Table S1** | `publication/output/tables/table_s1_full_gnn_metrics.csv` |
| **Table S2** | `publication/output/tables/table_s2_rl_gen_docking_full.csv` |
| **Table S3** | `publication/output/tables/table_s3_reinvent_scoring.csv` |
| **Figure S1** | `publication/output/figures/figure_s1_scaffold_split.png` |
| **Figure S2** | `publication/output/figures/figure4_morgan_rf_analysis.png` — Morgan RF + Murcko split (2×2) |
| **Figure S2b** | `publication/output/figures/figure_s2_baseline_comparison.png` — GAT vs Morgan RF vs VKORC1-only bars |
| **Morgan standalone** | `publication/output/figures/figure_morgan_rf_standalone.png` |
| **Figure S3** | `publication/output/figures/figure_s3_flexible_redock_spotcheck.png` |
| **Figure S4–S6** | *Reserved for MD trajectories, RMSD plots, MM-PBSA (Section 3.5)* |
| **REINVENT provenance** | `publication/data/reinvent_provenance.json` |

Regenerate: `python publication/generate_manuscript_assets.py`

---

*Manuscript draft. Remaining placeholders: author block, acknowledgments, Section 3.5 MD results.*
