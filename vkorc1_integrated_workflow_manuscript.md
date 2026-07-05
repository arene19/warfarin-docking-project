# An Integrated Multi-Task Graph Neural Network and Active-Learning Workflow for VKORC1-Targeted De Novo Ligand Design

**Author:** Amar Statovci  
**Affiliation:** Department of Chemistry, Chemical Engineering Program, University of Prishtina, Prishtina, Kosovo  
**Corresponding author:** amar.statovci@student.uni-pr.edu (amar.stato@gmail.com)

---

## Abstract

Computational discovery of VKORC1-targeted ligands is limited by sparse bioactivity labels, heterogeneous assay data, and fragmented workflows that treat machine learning, docking, and generative design as separate steps. Here we describe an integrated, reproducible **computational workflow** — not an experimentally validated lead-optimization campaign — that combines (i) a multi-task graph attention network (GAT) trained on coagulation and ADMET endpoints under Bemis–Murcko scaffold splitting, (ii) flexible AutoDock Vina docking against human VKORC1 (PDB 6WV3; S-warfarin co-crystal structure) with thermodynamic mapping of binding affinities to pseudo–pXC₅₀ labels, (iii) an active-learning merger that appends reinforcement-learning–generated (RL_Gen) candidates to the training corpus, and (iv) REINVENT4 staged learning with a custom ExternalProcess GNN scorer. On the held-out scaffold test set, pooled multi-task regression achieved RMSE = 1.00, MAE = 0.70, and R² = 0.60 across 2,210 valid task–compound pairs; per-endpoint performance was strongest for coagulation proteases (Factor XIIa R² = 0.64; thrombin R² = 0.60). VKORC1 absolute potency prediction remained weak (test R² = −0.38, *n* = 41), with Morgan fingerprint + random forest outperforming the GAT on VKORC1 rank correlation (Spearman ρ = 0.55 vs. 0.21). Accordingly, the GNN is deployed in REINVENT4 as a **relative ranker and multi-endpoint selectivity panel**, not as a quantitative potency oracle; flexible docking provides the primary structure-based filter. An illustrative VKORC1 case study ranked **RL_Gen_37** highest by **flat-SMILES** flexible docking among generated compounds (ΔG = −12.24 kcal/mol); after explicit stereochemistry assignment, **RL_Gen_37_isoA** (−10.70 kcal/mol) was used for membrane MD parametrization — the flat parent rank is **not** treated as the final potency estimate. **No compounds were synthesized or assayed in vitro.** **Comparative explicit-solvent membrane MD** (CHARMM-GUI, CHARMM36m, GROMACS 2024.3) was completed for **RL_Gen_37_isoA** and **RL_Gen_29_isoA** (100 ns each) and a **20 ns S-warfarin reference** pilot. RL_Gen_37_isoA showed stable temperature (310.06 K), second-half mean ligand RMSD of 1.92 Å, and persistent ASN80 H-bond occupancy (99.4%); RL_Gen_29_isoA exhibited elevated ligand RMSD (4.74 Å) and weaker ASN80 contact (10.3%), while the warfarin reference H-bonded primarily via SER81 (62.2%). Code, trained weights, and deposition bundle are openly available.

**Keywords:** VKORC1; de novo design; graph neural network; multi-task learning; active learning; AutoDock Vina; REINVENT4; cheminformatics workflow

---

## 1. Introduction

Oral anticoagulant therapy remains a cornerstone of thromboembolism prevention. Warfarin and related 4-hydroxycoumarins inhibit VKORC1, blocking vitamin K recycling and suppressing γ-carboxylation of clotting factors. Despite decades of clinical use, warfarin's narrow therapeutic index, metabolic variability (notably CYP2C9 polymorphisms), and off-target human serum albumin (HSA) binding motivate structure-based exploration of related scaffolds with improved selectivity profiles.

High-throughput experimental profiling of coagulation targets and ADMET endpoints is expensive and incomplete. Machine learning on molecular graphs can learn shared representations across related endpoints while tolerating missing labels via multi-task masking. When experimental VKORC1 data are scarce, structure-based docking can supply weak supervision by translating AutoDock Vina binding free energies (ΔG) into approximate potency scales (pXC₅₀), enabling an active-learning cycle in which each docking campaign enriches subsequent model retraining and generative design. Reinforcement-learning agents such as REINVENT4 explore chemical space under composite rewards; embedding a project-specific GNN as an external scoring process closes the loop between learned estimates and de novo generation.

Existing toolchains often treat predictive modeling, docking, and generative design as disconnected steps without unified benchmarks or reproducible deposition. This work addresses that gap with the following contributions:

- A **multi-task GAT** with masked MSE across six coagulation/ADMET endpoints, evaluated under **Murcko scaffold split** against Morgan ECFP4 + random forest and a VKORC1-only GAT ablation.
- An **active-learning merger** converting flexible Vina affinities to pseudo–pXC₅₀ labels with canonical-SMILES deduplication (retain strongest label per structure).
- A **REINVENT4 ExternalProcess** integration exposing checkpoint predictions to the RL reward function.
- Rigorous **data-quality controls**: enantiomer label correction on training records, flexible-docking reproducibility spot-checks, and stereochemistry enumeration for AI-generated leads.
- An open **deposition package** with RL_Gen docking results, GNN checkpoint, and REINVENT4 scoring configuration for reproducible reuse.

We emphasize honest reporting of VKORC1 model limitations and frame generative GNN scores as exploration biases rather than experimental potency predictions. **VKORC1 structure-based case studies in this paper are restricted to REINVENT4-generated RL_Gen ligands.** The multi-task training corpus combines ChEMBL bioactivity with RL-derived Vina pseudo-labels from the active-learning loop (Methods §2.5); benchmark label composition is summarized in Supplementary Table S5.

---

## 2. Materials and Methods

### 2.1 Target and structural models

The primary target was human VKORC1 in complex with S-warfarin (PDB ID **6WV3**, chain A); the co-crystal ligand defines the binding site geometry but is **not** included in the RL case-study ligand set reported here. The receptor was protonated and prepared for docking with flexible side-chain treatment at residues **A:217, A:269, A:272, and A:276**. Additional off-target receptors (Factor Xa, Factor XIIa, thrombin, CYP2C9, HSA) were configured for multi-target screening; their bioactivity labels enter the multi-task GNN but are secondary to the VKORC1 RL case study reported here.

### 2.2 Master dataset construction

The master training table aggregates ChEMBL bioactivity and project-specific RL_Gen pseudo-labels (**18,976** unique structures in the frozen publication master; **18,966** assigned to the Murcko scaffold split in Table S6; ten additional entries lack split assignment and are excluded from Table 1 benchmarks but retained for provenance). Experimental IC50 and Ki records from ChEMBL were pooled **without assay-type stratification** and converted to a common potency scale. For each measurement with concentration value_nM (nanomolar):

pXC₅₀ = −log₁₀(value_nM × 10⁻⁹)

This is the standard molar −log₁₀ transform used as the regression target for all six GNN tasks (Section 2.3) and as the Morgan RF baseline labels. When multiple ChEMBL records mapped to the same canonical SMILES and target, their pXC₅₀ values were averaged before pivoting to the wide multi-task table. Columns include canonical SMILES, scaffold class flag, and masked multi-task pXC₅₀ labels for VKORC1, Factor XIIa, Factor Xa, thrombin, CYP2C9, and HSA. VKORC1 labels: **119** in the frozen master (**109** within the benchmark split: 67 train / 1 validation / 41 test). Missing assay values are excluded from training and evaluation via per-task masking (Section 2.3).

### 2.3 Graph featurization and multi-task GNN architecture

Each molecule was converted to a PyTorch Geometric graph:

- **Nodes:** RDKit atoms with **10** features: atomic number, degree, formal charge, hybridization, aromaticity, mass, implicit/explicit valence, total hydrogens, ring membership.
- **Edges:** Undirected bonds from RDKit topology.
- **Stereochemistry:** Explicit `@`/`@@` tags in SMILES are **not** encoded as node features; Murcko scaffold generation strips stereochemistry tags. This **chirality-blind** featurization is a documented design choice (Section 4), not an oversight.

**Architecture (DynamicMultiTaskGNN):**

- **Encoder:** Four GAT layers [64 → 128 → 128 → 64], ReLU, dropout *p* = 0.1.
- **Pooling:** Global mean pooling over atom embeddings.
- **Readout:** Two-layer MLP → six task-specific pXC₅₀ outputs.
- **Partition:** Bemis–Murcko scaffold split (defined below; underlies Table 1, Figures 1–4).

**Loss and optimization.** Training minimizes masked mean squared error so that sparse, heterogeneous labels do not bias the objective toward data-rich tasks. For compound *n*, task *t*, model prediction ŷₙₜ, observed target *y*ₙₜ, and mask *m*ₙₜ ∈ {0, 1} (1 = label present, 0 = missing):

L_MSE = Σₙ,ₜ mₙₜ (ŷₙₜ − yₙₜ)² / Σₙ,ₜ mₙₜ

This loss is computed each epoch on the training set and validation set; the model checkpoint used for REINVENT4 scoring (Section 2.6) was saved at the epoch with lowest validation L_MSE (Figure 1).

**Benchmark metrics.** Model quality on the held-out test partition (Table 1, Supplementary Table S1) was assessed only on pairs with *m*ₙₜ = 1. Per-task metrics used the valid labels for that endpoint; the **Pooled** row concatenates all valid task–compound pairs across six endpoints (*N* = 2,210 in the reported benchmark):

RMSE = √( (1/N) Σᵢ (ŷᵢ − yᵢ)² )

MAE = (1/N) Σᵢ |ŷᵢ − yᵢ|

R² = 1 − Σᵢ(ŷᵢ − yᵢ)² / Σᵢ(yᵢ − ȳ)²

where ȳ is the mean observed label over the evaluated subset. The same definitions were applied to the Morgan FP + random forest baseline. **Spearman rank correlation** ρ was additionally computed as the Pearson correlation of rank-transformed (observed, predicted) pairs; ρ assesses ordinal agreement and is reported for VKORC1 in Table 1 and Supplementary Table S5 because absolute R² is unreliable for that sparse endpoint (Section 3.1).

**Scaffold split.** To prevent analogue leakage into benchmarks, molecules were grouped by Bemis–Murcko scaffold (stereochemistry stripped). Scaffold groups were sorted by descending size and assigned greedily to train, validation, and test until approximate fractions of 80%, 10%, and 10% of compounds were reached; all members of a scaffold group share the same partition (Figure 3, Table S6). The same partition was used for GAT training, Morgan RF fitting, and all reported test metrics.

**Training protocol:**

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr = 1×10⁻³, weight decay = 1×10⁻⁵) |
| Batch size | 64 |
| Max epochs | 150 |
| LR scheduler | ReduceLROnPlateau (patience = 15, factor = 0.5) |
| Early stopping | Patience = 25 (after 15-epoch warmup) |
| Checkpoint | Best validation masked MSE |

‡The validation split contains only one VKORC1 label; checkpoint selection is therefore dominated by data-rich coagulation protease tasks rather than VKORC1 performance.

**Baseline comparators** (same scaffold split): (1) Morgan ECFP4 (radius 2, 2048 bits) + per-task random forest (500 trees); (2) VKORC1-only GAT ablation with identical encoder/readout.

### 2.4 Flexible AutoDock Vina docking pipeline

Ligands were prepared from SMILES via conformer enumeration (20 conformers), Meeko protonation/torsion assignment, and PDBQT conversion. Receptors used a flexible PDBQT map with rigid core and flexible side chains at the defined residues. AutoDock Vina: exhaustiveness = 20, energy range = 3.0 kcal/mol, nine binding modes per ligand.

**Thermodynamic mapping for active learning.** Flexible Vina affinities from the RL screening campaign were converted to pseudo-labels on the same pXC₅₀ scale as ChEMBL data (Section 2.2), using the best-scoring docking mode ΔG (kcal/mol):

pXC₅₀ = −ΔG / 1.36

The divisor 1.36 kcal/mol corresponds to 2.303 × *R*T at ~298 K (one log-unit in binding free energy). Values produced by this mapping are reported in tables as **pXC₅₀ (Vina-derived)** to distinguish them from ChEMBL experimental pXC₅₀ and from GNN model predictions (Table 2, Table 4).

### 2.5 Active-learning merger

The active-learning merger closes the loop between generative screening and GNN retraining. It reads flexible Vina output, retains RL_Gen entries, resolves SMILES from project configuration or saved structures, and maps each docking score to a pseudo–pXC₅₀ label using Section 2.4. New rows are appended to the live master table with canonical-SMILES deduplication: when a structure already exists, the **strongest** (most potent) VKORC1 label is kept,

pXC₅₀^stored = max(pXC₅₀^old, pXC₅₀^new)

reflecting the conservative assumption that the best observed docking pose provides the upper bound on apparent affinity for that graph. **Val/test SMILES from the frozen scaffold split are never overwritten**, preserving benchmark integrity in Table 1 and Supplementary Table S5. Of the 67 train-split VKORC1 labels, 26 share canonical SMILES with the RL screening library (RL-library overlap by SMILES, not necessarily Vina-derived labels); test-split composition (23 ChEMBL-only / 18 RL-library SMILES overlaps under the audit filter) is documented in the deposited label-audit metadata. Supplementary Table S5 excludes RL-library SMILES from the held-out test set. Of 55 RL_Gen ligands docked in the screening campaign, **54** received VKORC1 pseudo-labels in the frozen master (**44** represented within the benchmark partition per label audit; **10** merged outside the split and excluded from Table 1/S5); one docked ligand was excluded by hold-out guards without a new merge.

### 2.6 Reinforcement-learning generative design (REINVENT4)

De novo generation used REINVENT4 staged learning in which each candidate SMILES receives a composite desirability score from four components: synthetic accessibility (SA), QED, molecular weight (MW), and GNN-predicted VKORC1 pXC₅₀ (ExternalProcess). Each component *c* is mapped to a bounded reward *s*ᶜ ∈ [0, 1] (GNN outputs pass through a sigmoid transform on the pXC₅₀ window 5.0–9.0). Component rewards are aggregated as a **weighted geometric mean** so that a single poor channel penalizes the overall score multiplicatively:

S_total = ( Π_c s_c^w_c )^(1 / Σ_c w_c)

with weights *w*ᶜ = 1.5 (SA), 1.0 (QED), 0.5 (MW), and 2.5 (GNN VKORC1). S_total steers the RL agent during generation (Figure 7); candidates with high Pred_VKORC1 scores were prioritized for flexible docking (Section 3.3). The external process loads the GAT checkpoint from Section 2.3 via a custom inference script. Top generations were injected into the screening library with sequential RL_Gen identifiers.

### 2.7 Stereochemistry enumeration

Undefined chiral centers in generated SMILES were expanded into explicit stereoisomer enumerations (_isoA, _isoB, …) using RDKit prior to flexible docking. **RL_Gen_37** was initially docked from a stereo-undefined parent SMILES; the parent flat dock (−12.24 kcal/mol) differs from enumerated isomers by ~1.5 kcal/mol (isoA −10.70; isoB −9.75 kcal/mol), while isoA and isoB differ by ~1.0 kcal/mol. Membrane MD parametrization used the **RL_Gen_37_isoA** (CIP R) 3D embed graph-matched to the parent docked pose.

### 2.8 Data-quality controls

**Enantiomer label correction:** Five reference enantiomer pairs in the training corpus had R/S SMILES swapped relative to CIP convention. Re-docking with corrected labels showed a pure label swap within Vina reproducibility noise, confirming that graph inputs (chirality-blind) were unaffected.

**Reference pose validation:** S-warfarin reference re-docking against the 6WV3 crystal pose yielded RMSD = **1.64 Å** (threshold < 2.0 Å), as documented in the deposited receptor-validation report.

**Retrospective docking enrichment:** Ten literature VKORC1/vitamin-K antagonists and 50 property-matched decoys were docked with the same flexible protocol. ROC-AUC = **0.964**; enrichment factor @ 1% = **6.0** (60/60 ligands docked successfully). Full enrichment tables are included in the deposition bundle.

**Flexible-docking reproducibility:** Eight RL_Gen ligands spanning ΔG −12.2 to −7.5 kcal/mol were re-docked with confirmed flexible parameters (RL_Gen_06 excluded: stored re-dock predates current screening affinity). Maximum |Δ| = **0.34 kcal/mol** (RL_Gen_31); mean |Δ| = **0.12 kcal/mol** among included RL_Gen rows. Six reference ligands (warfarin, BENZ, p-nitro enantiomers) show |Δ| = **0.018–0.055 kcal/mol** under the same protocol (Figure 5, right panel).

### 2.9 Membrane molecular dynamics

VKORC1–ligand complexes from flexible docking were embedded in an ER-like homogeneous bilayer using CHARMM-GUI Membrane Builder. Docked ligand coordinates were graph-matched to CGenFF-compatible topology generated via CHARMM-GUI Ligand Reader auto-CGenFF; manual ParamChem reparameterization was documented as a fallback if automated parameterization failed (not required for the completed RL systems). Lipids POPC/POPE/POPS/cholesterol (6:2:1:2 per leaflet); 0.15 M NaCl; TIP3P water; CHARMM36m protein/lipid parameters; 310 K; ~134,000 atoms in a 120 Å cubic box. Equilibration used CHARMM-GUI step6.1–6.6 protocols; production MD used **GROMACS 2024.3** with CUDA (2 fs timestep).

**Systems simulated.** Comparative validation comprised two prioritized RL isoA embeds (**RL_Gen_37_isoA** and **RL_Gen_29_isoA**) at 100 ns production each, plus a **20 ns S-warfarin reference** pilot docked to the same 6WV3-based receptor construct.

**Analysis.** Protein Cα and ligand heavy-atom RMSD after least-squares fit to protein backbone were computed from saved production trajectories. Binding-site hydrogen-bond occupancy was quantified on production frames using GROMACS analysis tools; production trajectories were generated with GROMACS 2024.3, and post-hoc RMSD and H-bond occupancy were recomputed from saved trajectories using GROMACS 2026 analysis utilities (**no re-simulation**). Binding-site residue labels follow PLIP/PDB numbering (e.g., ASN80, SER81); GROMACS topology residue indices differ by system and are reported alongside occupancy in Supplementary Table S7. Canonical merged metrics are included in the Zenodo deposition bundle.

---

## 3. Results

### 3.1 Multi-task GNN performance and baseline comparison

On the held-out Murcko scaffold test set (*n* = 1,897 compounds), the multi-task GAT achieved pooled masked regression **RMSE = 1.00**, **MAE = 0.70**, and **R² = 0.60** (2,210 valid task–compound pairs). Performance was strongest on coagulation protease tasks with denser ChEMBL labels (Factor XIIa R² = 0.64; thrombin R² = 0.60; Factor Xa R² = 0.56) and modest on CYP2C9 (R² = 0.03). VKORC1 and HSA remained data-limited (*n* = 41 and 5 test labels) and are **not** interpreted as quantitative potency models; HSA regression metrics in particular are unstable at *n* = 5 and too small for meaningful model comparison (Table 1, footnote). **For VKORC1 ranker assessment, primary claims rely on Supplementary Table S5** (*n* = 23 ChEMBL-only labels excluding RL-library SMILES overlaps), not the full *n* = 41 test row in Table 1.

**Table 1.** Test-set regression metrics by endpoint (Murcko scaffold hold-out)

| Task | *n* (test) | GAT RMSE | GAT R² | GAT Spearman ρ | RF R² | RF Spearman ρ | VKORC1-only GAT R² |
|------|------------|----------|--------|----------------|-------|---------------|---------------------|
| VKORC1 | 41 | 1.28 | −0.38 | 0.21 | 0.28 | 0.55 | 0.02 |
| Factor XIIa | 222 | 0.78 | 0.64 | — | 0.82 | — | — |
| Factor Xa | 623 | 1.09 | 0.56 | — | 0.67 | — | — |
| Thrombin | 599 | 0.96 | 0.60 | — | 0.69 | — | — |
| CYP2C9 | 720 | 0.88 | 0.03 | — | 0.17 | — | — |
| HSA† | 5 | 5.67 | −0.00 | — | −0.01 | — | — |
| **Pooled** | **2,210** | **1.00** | **0.60** | — | — | — | — |

†HSA test metrics (*n* = 5) are reported for completeness but are not statistically meaningful at this sample size; RMSE and R² are unstable and should not be used for model comparison.

Full MAE values and per-task details are in Supplementary Table S1.

**Figure 1.** Multi-task GAT training dynamics on the Murcko scaffold-split corpus (15,172 train / 1,897 validation compounds). Left panel: masked mean squared error (MSE) on training and validation sets across epochs, showing convergence without severe overfitting. Right panel: validation R² aggregated over all masked tasks per epoch; the checkpoint used for REINVENT4 scoring was selected at minimum validation MSE. Together, these curves document stable multi-task learning prior to generative deployment.

**Figure 2.** Baseline comparison of test-set R² by endpoint under Murcko scaffold hold-out for three models: multi-task GAT (blue), Morgan ECFP4 + random forest (orange), and VKORC1-only GAT ablation (green, VKORC1 task only). Bars highlight that Morgan RF outperforms the GAT on VKORC1 absolute regression while the multi-task GAT achieves stronger performance on data-rich coagulation protease tasks. This panel motivates deploying the GAT as a multi-endpoint ranker rather than a VKORC1 potency oracle.

**Figure 3.** Bemis–Murcko scaffold split statistics for the 18,966-compound frozen partition of the master dataset. Bar chart shows compound and unique-scaffold counts assigned to train (80%), validation (10%), and test (10%) partitions, ensuring that structurally related analogues do not leak between splits. This partition underlies all GNN benchmark results reported below.

**Figure 4.** Morgan ECFP4 + random forest baseline analysis on the Murcko scaffold test set. Three panels report (A) per-task test R², (B) per-task Spearman rank correlation ρ, and (C) VKORC1 predicted-vs-observed scatter (*n* = 41). Morgan RF achieves the best VKORC1 rank correlation (ρ = 0.55), supporting its use as a ranking comparator while the multi-task GAT provides simultaneous multi-endpoint readouts for REINVENT4 integration.

**Interpretation.** VKORC1 absolute regression fails under scaffold split (GAT R² = −0.38), confirming the model must **not** serve as a potency oracle. Morgan RF achieves better VKORC1 **ranking** on the full test set (Spearman ρ = 0.55 vs. 0.21). Of the 41 held-out VKORC1 test labels, 23 are ChEMBL-assigned and 18 share canonical SMILES with the RL screening library under the audit filter (Methods §2.5); on the ChEMBL-only subset excluding RL-library SMILES (*n* = 23), multi-task GAT R² = −0.54 (ρ = −0.07) and Morgan RF R² = 0.11 (ρ = 0.17) (Supplementary Table S5). Pooled test R² (0.60) exceeds best validation R² (0.51) because the test partition contains proportionally more data-rich protease labels than validation; this is not indicative of overfitting. We retain the multi-task GAT in REINVENT4 because (i) it outputs all six endpoints in one forward pass for selectivity triage, (ii) Morgan RF lacks the multi-endpoint single-pass inference required for REINVENT4 ExternalProcess integration, (iii) its role is **relative ranking within generated libraries**, and (iv) the active-learning loop continuously augments VKORC1 pseudo-labels from Vina (train partition only after hold-out guards). GNN predictions and Vina-derived pXC₅₀ operate on different scales (Section 3.4).

### 3.2 Data quality controls

Enantiomer relabeling produced symmetric affinity swaps without systematic rigid→flexible shifts, supporting retention of the current checkpoint without full retraining. Flexible re-docking spot-checks (Figure 5) confirmed RL_Gen stored affinities match flexible re-docks for eight ligands (max |Δ| = 0.34 kcal/mol for RL_Gen_31; RL_Gen_06 excluded as stale re-dock).

**Figure 5.** Flexible-docking reproducibility spot-check. Left: eight RL_Gen ligands re-docked with confirmed flexible side-chain parameters (A:217, A:269, A:272, A:276); dashed line marks maximum |Δ| (0.34 kcal/mol, RL_Gen_31). Right: six reference enantiomers (|Δ| = 0.018–0.055 kcal/mol), demonstrating tighter reproducibility for the curated reference set. RL_Gen_06 is omitted because its archived re-dock predates the current screening affinity in Table S2.

### 3.3 Case study — VKORC1 flexible docking of RL_Gen leads

Flexible AutoDock Vina screening of the REINVENT4-generated RL_Gen library against VKORC1 (6WV3) ranked **RL_Gen_37** highest by **flat-SMILES** flexible docking among generated ligands (ΔG = −12.24 kcal/mol; pXC₅₀ Vina-derived = 9.00). When stereochemistry is explicitly assigned, **RL_Gen_37_isoA** docks at −10.70 kcal/mol (~12th among RL_Gen entries; Supplementary Table S2). The **1.54 kcal/mol flat-vs-isoA gap reflects undefined-centre stereoisomer enumeration** (Section 2.7), not flexible re-dock protocol noise (max |Δ| = 0.34 kcal/mol among RL_Gen spot-check ligands; Figure 5). **Table 2 therefore reports the flat parent for transparency, but downstream MD and prioritization used the isoA embed** — the flat #1 rank is not interpreted as the definitive stereochemical potency estimate. Reference nitro-coumarin enantiomers remain competitive in the same flexible protocol (e.g., p_nitro_R at −12.12 kcal/mol) and are retained for internal QC only; they are excluded from the case-study tables below. The top ten RL candidates span ΔG −12.2 to −10.8 kcal/mol, with scaffold and stereoisomer variants (e.g., RL_Gen_29, RL_Gen_29_isoA/B) clustering within ~0.4 kcal/mol — consistent with the chirality-blind GNN featurization and subsequent explicit enumeration of stereocenters for docking.

**Table 2. VKORC1 flexible docking leaderboard (top RL_Gen leads)**

| Ligand | ΔG (kcal/mol) | pXC₅₀ (Vina-derived) |
|--------|---------------|----------------------|
| RL_Gen_37‡ | −12.244 | 9.003 |
| RL_Gen_22 | −11.545 | 8.489 |
| RL_Gen_29_isoA | −11.419 | 8.396 |
| RL_Gen_45 | −11.395 | 8.379 |
| RL_Gen_49 | −11.239 | 8.264 |
| RL_Gen_29_isoB | −11.069 | 8.139 |
| RL_Gen_29 | −10.969 | 8.065 |
| RL_Gen_07 | −10.921 | 8.030 |
| RL_Gen_26 | −10.902 | 8.016 |
| RL_Gen_39 | −10.823 | 7.958 |

‡RL_Gen_37 parent uses stereo-undefined SMILES; enumerated isoA (−10.702 kcal/mol) and isoB (−9.751 kcal/mol) affinities are in Supplementary Table S2 (Section 2.7).

Full RL_Gen ranking: Supplementary Table S2.

**Figure 6.** VKORC1 interaction fingerprint for the top twelve RL_Gen ligands ranked by flexible-docking score. Heatmap columns report hydrogen-bond count, hydrophobic contact count, π-stacking count, and AutoDock Vina ΔG (kcal/mol; color intensity uses |ΔG|). RL_Gen_29_isoA shows the richest H-bond network (seven contacts) despite ranking below RL_Gen_37 on raw ΔG, illustrating that interaction pattern — not affinity alone — informs downstream prioritization.

### 3.4 REINVENT4–GNN closed-loop generation

Integrating the GNN as an ExternalProcess scorer (weight 2.5×) steered generation toward higher model-predicted VKORC1 scores while SA/QED/MW filters maintained synthesizable, drug-like candidates. The GNN channel functions as a **composite reward and ranker**; flexible Vina docking provides the primary structure-based filter (Section 3.3). GNN predictions and Vina-derived pXC₅₀ values operate on different scales: for example, **RL_Gen_45** ranks fourth by docking (ΔG = −11.40 kcal/mol; pXC₅₀ Vina-derived ≈ 8.38) but receives the lowest Pred_VKORC1 score (6.18) among the top five RL hits (Table 4), illustrating that the GNN reward shapes generative exploration rather than reproducing docking-derived potency rankings.

**Figure 7.** Distribution of REINVENT4 GNN-predicted VKORC1 pXC₅₀ scores for generated candidates overlaid with the empirical distribution of **train-split** VKORC1 labels from ChEMBL and active-learning merges (hold-out val/test labels excluded). The generative library extends into regions of predicted potency space underrepresented in the training set. Vertical dashed lines mark the training-label median and 90th percentile; candidates above the 90th percentile were prioritized for flexible docking (Section 3.3).

### 3.5 ADMET-informed prioritization within the RL_Gen library

RDKit-derived descriptors (MW, LogP, TPSA, QED, Lipinski compliance) were computed for the top-ranked RL_Gen docked poses. Among the six leading RL hits, **QED ranged from 0.49 to 0.74** and **TPSA from 23.6 to 88.2 Å²**, indicating substantial developability spread within a narrow docking window (~1.2 kcal/mol). RL_Gen_37 combines the strongest VKOR affinity with high QED (0.74) but low TPSA (23.6 Å²); RL_Gen_22 fails Lipinski **LogP** rules (LogP ≈ 5.4; MW 487 Da) despite favorable docking. RL_Gen_29_isoA balances moderate QED (0.63) with the richest VKOR H-bond network (seven contacts). HSA off-target docking (where available) was comparable for RL_Gen_37 (ΔG = −9.33 kcal/mol) and RL_Gen_29_isoA (−8.67 kcal/mol).

**Table 3. ADMET and interaction summary for top RL_Gen leads (by VKORC1 docking rank)**

| Ligand | VKOR ΔG | QED | TPSA | Lipinski | VKOR H-bonds | HSA ΔG |
|--------|---------|-----|------|----------|--------------|--------|
| RL_Gen_37 | −12.244 | 0.740 | 23.6 | Yes | 1 | −9.328 |
| RL_Gen_22 | −11.545 | 0.539 | 67.4 | No | 2 | — |
| RL_Gen_29_isoA | −11.419 | 0.630 | 88.2 | Yes | 7 | −8.672 |
| RL_Gen_45 | −11.395 | 0.491 | 72.5 | Yes | 4 | — |
| RL_Gen_49 | −11.239 | 0.624 | 71.8 | Yes | 2 | — |
| RL_Gen_29_isoB | −11.069 | 0.630 | 88.2 | Yes | 5 | — |

HSA ΔG reported where HSA flexible docking was performed; em dash (—) indicates not evaluated for that ligand.

**Table 4. Multi-task GNN predictions for top five RL_Gen leads (by VKORC1 docking score)**

| Ligand | ΔG (kcal/mol) | Pred VKORC1 | Pred Factor XIIa | Pred Factor Xa | Pred Thrombin | Pred CYP2C9 | Pred HSA |
|--------|---------------|-------------|------------------|----------------|---------------|-------------|----------|
| RL_Gen_37 | −12.244 | 7.524 | 4.634 | 7.243 | 5.612 | 4.662 | −6.191 |
| RL_Gen_22 | −11.545 | 7.382 | 5.286 | 6.198 | 4.873 | 5.278 | −3.391 |
| RL_Gen_29_isoA | −11.419 | 7.846 | 4.059 | 4.967 | 5.296 | 4.951 | −1.318 |
| RL_Gen_45 | −11.395 | 6.178 | 5.299 | 6.308 | 6.057 | 4.907 | 0.789 |
| RL_Gen_49 | −11.239 | 7.038 | 4.879 | 6.337 | 5.325 | 5.130 | −1.036 |

*Predictions are model outputs, not experimental potencies.*

Multi-criteria prioritization selected **RL_Gen_37_isoA** (top docking score + high QED) and **RL_Gen_29_isoA** (interaction-rich scaffold) for 100 ns membrane MD validation, with a **20 ns S-warfarin reference** pilot for contact-pattern comparison (Section 3.6).

### 3.6 Comparative membrane MD validation

All-atom explicit-solvent MD was performed for **RL_Gen_37_isoA** (CIP R stereochemical embed graph-matched to the parent flat dock; parent **RL_Gen_37** flat-SMILES ΔG = −12.24 kcal/mol), **RL_Gen_29_isoA** (interaction-rich scaffold; isoA ΔG = −11.42 kcal/mol), and a **20 ns S-warfarin reference** pilot to test whether flexible-docking poses remain stable in a membrane environment (Section 2.9).

**Table 5. Membrane MD trajectory metrics (three completed systems)**

| Ligand | VKOR ΔG flat (kcal/mol) | VKOR ΔG isoA (kcal/mol) | Production (ns) | *T* (K) | *P* (bar) | Protein RMSD (Å) | Ligand RMSD 2nd half (Å) | ASN80 H-bond (%) |
|--------|-------------------------|-------------------------|-----------------|---------|-----------|------------------|--------------------------|------------------|
| RL_Gen_37_isoA | −12.244 | −10.702 | 100 | 310.06 | 1.02 | 2.53 (mean); 2.32 (last 25%) | 1.92 (max 2.83) | 99.4 |
| RL_Gen_29_isoA | — | −11.419 | 100 | 310.06 | 1.28 | 3.17 (mean); 3.78 (last 25%) | 4.74 (max 5.34) | 10.3 |
| S-warfarin (ref) | — | — | 20 | 310.06 | −0.11 | 1.54 (mean); 1.77 (last 25%) | 3.18 (max 4.83) | 1.0 |

RMSD values are backbone Cα (protein) or ligand heavy atoms after least-squares fit to protein backbone. The RL_Gen_37 simulated system used the **RL_Gen_37_isoA** 3D embed (Section 2.7). H-bond occupancy uses PLIP residue labels mapped to GROMACS topology indices per system (Supplementary Table S7).

**Figure 8.** Protein and ligand RMSD time series for RL_Gen_37_isoA (100 ns), RL_Gen_29_isoA (100 ns), and S-warfarin reference (20 ns) membrane MD at 310 K.

**Figure 9.** Binding-site H-bond occupancy (% production frames) for ASN80, SER81, TYR139, THR138, and hydrophobic pocket residues across the three completed systems.

**Interpretation.** Thermodynamic observables remained near target conditions (310 K; ~1 bar mean pressure for 100 ns RL runs). **RL_Gen_37_isoA** showed the lowest second-half ligand RMSD (1.92 Å) and near-complete ASN80 H-bond occupancy (99.4%), consistent with a persistent isoA-embed docked pose under bilayer conditions despite elevated CGenFF parameterization risk for this scaffold. **RL_Gen_29_isoA**—despite favorable isoA docking (−11.42 kcal/mol) and a rich static H-bond network in flexible docking—exhibited substantially higher ligand RMSD (4.74 Å second-half mean) and only intermittent ASN80/TYR139 contacts, indicating pose drift or partial dissociation over 100 ns. The **S-warfarin reference** pilot (20 ns) showed low protein RMSD but moderate ligand RMSD (3.18 Å) and dominant SER81 H-bond occupancy (62.2%) rather than ASN80; this short trajectory is included for **contact-pattern comparison only**, not as an equilibrated pose-stability benchmark (mean pressure −0.11 bar reflects incomplete barostat equilibration over 20 ns). These comparative results support—but do not replace—experimental validation; docking rank alone does not predict MD stability among prioritized RL scaffolds.

---

## 4. Discussion

We implemented a reproducible, closed-loop computational platform linking multi-task graph learning, flexible structure-based screening, and RL-driven de novo design for **VKORC1-targeted ligand discovery** (warfarin-pocket geometry from PDB 6WV3). The primary methodological contribution is the **integrated workflow with scaffold-split benchmarks**, not a claim of discovering a clinical candidate without experimental validation.

**VKORC1 model behavior.** Negative test R² on VKORC1 (41 labels) is expected under Murcko split with chirality-blind graphs. Morgan FP + RF achieves superior VKORC1 rank correlation (ρ = 0.55), yet the multi-task GAT provides simultaneous coagulation/ADMET readouts and shares representation with data-rich protease tasks (R² = 0.56–0.64). On the ChEMBL-only VKORC1 subset (Table S5), GAT Spearman ρ = −0.07 versus Morgan RF ρ = 0.17 — confirming that the GNN channel is a weak absolute ranker on sparse VKORC1 labels. We nevertheless retain GAT in REINVENT4 (weight 2.5×) because it returns all six endpoints in a **single forward pass**, whereas Morgan RF would require separate per-endpoint models and cannot supply the multi-task selectivity readouts used downstream without additional inference cost. Deployment is therefore justified for **exploration steering**, not potency estimation.

**Case study interpretation.** Flat-SMILES flexible docking ranks RL_Gen_37 highest among REINVENT4 outputs for reporting transparency, but the **isoA stereochemical embed** (−10.70 kcal/mol; ~12th in the RL library) was used for membrane MD and is the operative structural hypothesis. The 1.54 kcal/mol flat-vs-isoA shift arises from assigning an undefined chiral centre after generation, not from docking irreproducibility; it underscores that flat-SMILES leaderboard positions for generated chemotypes can be misleading when stereochemistry is resolved post hoc. ADMET profiling within the RL library reveals that affinity and developability are not monotonically coupled. **Comparative membrane MD** (100 ns RL_Gen_37_isoA and RL_Gen_29_isoA; 20 ns S-warfarin reference) showed a clear stability split: RL_Gen_37_isoA maintained low ligand RMSD (1.92 Å second-half mean) and persistent ASN80 H-bond occupancy (99.4%), whereas RL_Gen_29_isoA—despite strong isoA docking and a rich static interaction fingerprint—drifted to 4.74 Å ligand RMSD with weak ASN80 contact (10.3%) and partial TYR139 engagement (29.1%). This counterexample demonstrates that **docking rank and static interaction counts do not guarantee bilayer pose persistence** and motivates MD triage among prioritized scaffolds. The S-warfarin reference H-bonded primarily via SER81 (62.2%), not ASN80, providing a mechanistic contrast to the RL_Gen_37 contact pattern rather than a direct potency benchmark (20 ns pilot; longer equilibration would be needed for definitive reference behaviour). Manual ParamChem reoptimization was not performed; standard CHARMM-GUI auto-CGenFF parameters were used because RL_Gen_37 equilibrated and remained stable over 100 ns, although force-field uncertainty for spiro/piperidinone chemotypes remains a limitation.

**Reference enantiomer docking.** Supplementary stereoselectivity QC shows that flexible Vina prefers R-warfarin over S-warfarin (ΔΔG = +0.37 kcal/mol) despite the 6WV3 co-crystal containing S-warfarin (SWF). This stereoselectivity inversion is a known limitation of empirical scoring functions and underscores that docking ranks are approximate; reference ligands are retained for protocol QC only and are excluded from the RL case-study tables.

**Chirality-blind featurization** simplified handling of mixed flat/enumerated SMILES in the training corpus. Stereochemistry-aware extensions are warranted when experimental chiral discrimination becomes critical.

**Limitations:** (1) VKORC1 GNN regression is unreliable for absolute potency (validation split contains only one VKORC1 label, so early stopping is dominated by data-rich tasks); ChEMBL-only VKORC1 benchmark (Table S5) shows GAT Spearman ρ = −0.07, reinforcing ranker-not-oracle deployment; (2) ChEMBL IC50/Ki records were pooled without assay-type harmonization beyond pXC₅₀ conversion; (3) docking pseudo-labels inherit Vina approximations (~±1 kcal/mol); (4) **no experimental IC50, synthesis, or biological assay data** — this work reports an integrated computational workflow and an illustrative RL_Gen_37 case study only; potency and developability claims are computational and not validated in vitro or in vivo; (5) membrane MD covers only RL_Gen_37_isoA, RL_Gen_29_isoA, and a 20 ns S-warfarin reference pilot; MM-GBSA rescoring remains outstanding; the warfarin reference trajectory is too short for equilibrated benchmarking; full production trajectories are summarized in the deposition bundle rather than bundled at multi-gigabyte scale; (6) RL systems used auto-CGenFF despite high-penalty warnings for some scaffolds; manual ParamChem reparameterization remains future work; (7) reference enantiomer docking does not reproduce crystal S-warfarin stereochemical preference (see above).

**Future work:** Longer S-warfarin reference production; additional RL_Gen membrane MD on prioritized scaffolds; MM-GBSA on production frames; stereochemistry-aware GNN featurization; experimental validation of prioritized poses.

---

## 5. Conclusions

We developed and benchmarked a multi-task GAT-based active-learning **computational workflow** for VKORC1-directed de novo ligand design, integrating flexible docking, pseudo-label merging, and REINVENT4 generative design with an embedded GNN ranker. Pooled test R² = 0.60 across six endpoints demonstrates meaningful multi-task transfer; VKORC1-specific regression remains weak (R² = −0.38), validating deployment as ranker rather than potency oracle. An illustrative **RL_Gen_37** case study ranked highest by flat-SMILES flexible docking among generated ligands (ΔG = −12.24 kcal/mol); the stereochemistry-resolved **RL_Gen_37_isoA** embed (−10.70 kcal/mol) persisted in comparative membrane MD (1.92 Å second-half ligand RMSD; 99.4% ASN80 H-bond occupancy), while **RL_Gen_29_isoA**—a higher static-interaction scaffold—showed pose instability (4.74 Å ligand RMSD), underscoring the need for MD triage beyond docking rank. **No experimental synthesis or bioassay data were generated.** Code, checkpoint, and deposition bundle are openly available.

---

## Author Contributions

**Amar Statovci:** Conceptualization, Software, Validation, Formal analysis, Writing — original draft, Writing — review & editing, Visualization, Data curation, Project administration.

---

## Data and Code Availability

Source code is available at [https://github.com/arene19/warfarin-docking-project](https://github.com/arene19/warfarin-docking-project) (release tag v1.0-submission). A versioned Zenodo archive containing the frozen training master, trained GNN weights, Murcko scaffold split, RL_Gen docking and ADMET tables, manuscript figures and tables, membrane MD analysis summaries, REINVENT4 configuration and provenance metadata, and reproduction scripts is available at **https://doi.org/10.5281/zenodo.21208445** (version 1.0). Reference-ligand QC tables (stereoselectivity, docking enrichment, receptor validation) are included in the deposition bundle. The REINVENT4 prior model is documented in supplementary provenance metadata but is not redistributed.

---

## Acknowledgments

The author used personal computational resources (laptop and NVIDIA RTX 4060 Ti GPU workstation) for model training, docking, and membrane MD simulations. No external funding was received for this work. Generative AI coding assistants were used across software development, manuscript preparation, and analysis workflows (see Generative AI declaration below); the author independently verified all outputs and takes full responsibility for the scientific content.

## Conflict of Interest

The author declares no competing financial or non-financial interests.

## Ethics Statement

Not applicable. This study used publicly available chemical and structural data only; no human participants, animal subjects, or new experimental biological samples were involved.

## Generative AI declaration

Large language model tools (Cursor IDE with integrated coding assistants) were used to assist with Python pipeline implementation, debugging, manuscript drafting, literature organization, data-workflow documentation, and exploratory interpretation of computational results. The author independently verified all citations against primary sources, reproduced numerical results from archived data and scripts, and validated all scientific claims before submission. The author takes full responsibility for the content of this work.

---

## References

1. Rost S, Fregin A, Ivaskevicius V, et al. Mutations in VKORC1 cause warfarin resistance and multiple coagulation factor deficiency type 2. *Nature* **2004**, *427*, 537–541.
2. Bemis GW, Murcko MA. The properties of known drugs. 1. Molecular frameworks. *J Med Chem* **1996**, *39*, 2887–2893.
3. Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. *Proc ICLR* **2018**.
4. Trott O, Olson AJ. AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. *J Comput Chem* **2010**, *31*, 455–461.
5. Blaschke T, Arús-Pous H, Chen H, et al. REINVENT 4: Modern AI-driven generative molecule design. *J Cheminform* **2024**, *16*, 20.
6. Zdrazil B, Felix E, Hunter F, et al. The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. *Nucleic Acids Res* **2024**, *52*, D1180–D1192.
7. Fey M, Lenssen JE. Fast graph representation learning with PyTorch Geometric. *ICLR Workshop* **2019**.
8. Abraham MJ, Murtola T, Schulz R, et al. GROMACS: High performance molecular simulations. *SoftwareX* **2015**, *1–2*, 19–25.
9. Landrum G. RDKit: Open-source cheminformatics. https://www.rdkit.org (accessed 2025).
10. Forli S, Huey R, Pique ME, Sanner MF, Goodsell DS, Olson AJ. Computational protein–ligand docking and virtual drug screening with the AutoDock suite. *Nat Protoc* **2016**, *11*, 905–919.
11. Pedregosa F, et al. Scikit-learn: Machine learning in Python. *J Mach Learn Res* **2011**, *12*, 2825–2830.
12. Rogers D, Hahn M. Extended-connectivity fingerprints. *J Chem Inf Model* **2010**, *50*, 742–754.
13. Jo S, Kim T, Iyer VG, Im W. CHARMM-GUI: a web-based graphical user interface for CHARMM. *J Comput Chem* **2008**, *29*, 1859–1865.
14. Lee J, Cheng X, Swails JM, Yeom MS, Eastman PK, Lemkul JA, Wei S, Buckner J, Jeong JC, Qi Y, et al. CHARMM-GUI input generator for NAMD, GROMACS, AMBER, OpenMM, and CHARMM/OpenMM simulations using the CHARMM36 additive force field. *J Chem Theory Comput* **2016**, *12*, 405–413.
15. Huang J, Rauscher S, Nawrocki G, Ran T, Feig M, de Groot BL, Grubmüller H, MacKerell AD Jr. CHARMM36m: an improved force field for folded and intrinsically disordered proteins. *Nat Methods* **2017**, *14*, 71–73.
16. Vanommeslaeghe K, et al. CHARMM general force field: A force field for drug-like molecules compatible with the CHARMM all-atom additive biological force fields. *J Comput Chem* **2010**, *31*, 671–690.

---

## Supplementary Information

| Asset | Description |
|-------|-------------|
| **Table S1** | Full GNN metrics (MAE, Spearman ρ) |
| **Table S2** | Full RL_Gen VKORC1 docking table |
| **Table S3** | REINVENT4 scoring component weights |
| **Table S4** | Membrane MD metrics (mirrors main-text Table 5) |
| **Table S5** | VKORC1 test metrics excluding RL-library SMILES overlaps |
| **Table S6** | Murcko scaffold split statistics |
| **Table S7** | Binding-site H-bond occupancy across completed MD systems |

Reference-ligand QC tables (stereoselectivity, retrospective docking enrichment, combined docking leaderboard) are archived in the Zenodo deposition bundle but are not cited in the main-text supplementary table list.

Main-text Figures 1–9 cover GNN training, baseline comparison, scaffold split, Morgan FP + RF analysis, flexible re-dock spot-check, VKORC1 interaction fingerprint, REINVENT4 generation distribution, comparative membrane MD RMSD (three systems), and H-bond occupancy heatmap (Sections 3.1–3.6).

---
