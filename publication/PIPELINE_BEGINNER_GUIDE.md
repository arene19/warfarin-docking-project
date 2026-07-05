# Complete Beginner's Guide to the VKORC1 GNN + Active-Learning Pipeline

**Companion document for:** *A Multi-Task Graph Neural Network and Active-Learning Framework for VKORC1-Targeted De Novo Ligand Discovery*  
**Author:** Amar Statovci  
**Purpose:** Read this document to understand every major concept, formula, script, and design choice in the study — even if you are new to cheminformatics, machine learning, or molecular docking.

---

## How to use this guide

1. Start with **Section 1 (Biological context)** if you are new to anticoagulants or VKORC1.
2. Read **Section 2 (Big picture)** to see the full pipeline order before diving into details.
3. Use **Section 3 (Glossary)** as a dictionary while reading the rest.
4. **Sections 4–6** explain every formula and statistical term used in the paper.
5. **Sections 7–14** walk through each pipeline stage, script, and result table.
6. **Section 15** maps manuscript sections to this guide so you can read the paper section-by-section.

This guide is intentionally long. You do not need to memorize it in one sitting — treat it as a textbook chapter.

---

## 1. Biological and clinical context

### 1.1 What is VKORC1?

**VKORC1** (Vitamin K Epoxide Reductase Complex subunit 1) is a human protein found mainly in the liver and in the endoplasmic reticulum (ER) membrane. It recycles vitamin K, which is required to activate clotting factors II, VII, IX, and X. Without functional VKORC1, blood clotting is impaired.

**Warfarin** — the classic oral anticoagulant — works by inhibiting VKORC1. Patients on warfarin need careful dose monitoring because:

- Warfarin has a **narrow therapeutic index** (small dose changes cause large clinical effects).
- **CYP2C9** (a liver enzyme) metabolizes warfarin; genetic variants change exposure.
- Warfarin binds **HSA** (human serum albumin) in blood, affecting free drug concentration.

This project asks a computational question: *Can we design new coumarin-like molecules that might bind VKORC1, while using machine learning to estimate selectivity across related coagulation and ADMET endpoints?*

**Important honesty note:** Nothing in this study was synthesized or tested in a wet lab. All results are **computational predictions and simulations**.

### 1.2 What are coumarins and 4-hydroxycoumarins?

Coumarins are a family of organic scaffolds (ring systems) that include warfarin-like **4-hydroxycoumarins**. The REINVENT4 generative model in this project is biased toward coumarin-like chemistry via its prior model and scoring filters.

### 1.3 What is ADMET?

**ADMET** = Absorption, Distribution, Metabolism, Excretion, Toxicity. In this project, "ADMET" mainly refers to:

- **CYP2C9** — metabolic liability (how fast the liver might break the drug down).
- **HSA** — plasma protein binding (how much drug is "trapped" on albumin vs free to act).

The GNN predicts six endpoints simultaneously: VKORC1 + three coagulation proteases (Factor XIIa, Factor Xa, thrombin) + CYP2C9 + HSA.

### 1.4 Why structure-based docking if we have machine learning?

Experimental bioactivity data for VKORC1 is **sparse** in public databases (ChEMBL). Docking estimates how a 3D ligand might fit into a 3D protein binding site. It is imperfect but provides **weak supervision** when experiments are missing — especially for AI-generated molecules that have never been measured.

---

## 2. The big picture: pipeline order and closed loop

### 2.1 One-sentence summary

The project builds a **multi-task graph neural network** on ChEMBL + docking-derived labels, uses it to **score REINVENT4-generated molecules**, **docks** the best candidates against VKORC1, **merges** docking scores back into the training table, and optionally **retrains** — a computational active-learning loop.

### 2.2 Pipeline stages in chronological order

Think of the workflow as a repeating cycle. The **first-time setup** and **one full iteration** look like this:

| Step | What happens | Main script / file |
|------|----------------|-------------------|
| **A** | Build master training CSV from ChEMBL (+ any existing RL labels) | `data/coagulation_admet_multi_task.csv` |
| **B** | Assign Murcko scaffold train/val/test split (frozen for publication) | `gnn_model.py` → `publication/data/gnn_scaffold_split.json` |
| **C** | Train multi-task GAT on graphs | `dynamic_gnn_pipeline.py` → `coagulation_admet_gnn.pth` |
| **D** | Evaluate GAT vs Morgan RF vs VKORC1-only ablation | `gnn_baseline_evaluation.py` |
| **E** | Run REINVENT4 with GNN as external scorer | `configs/coumarin_rl.toml` + `gnn_predict.py` |
| **F** | Inject top generations into ligand library | `inject_ai_leads.py` → `config_master.yaml` |
| **G** | Flexible AutoDock Vina docking of active ligands | `main_pipeline.py` |
| **H** | Merge RL_Gen docking affinities into master CSV (train only) | `active_learning_merger.py` |
| **I** | Retrain GNN on enriched master | `dynamic_gnn_pipeline.py` (repeat C) |
| **J** | Regenerate manuscript tables/figures + deposition bundle | `publication/build_all.sh` |
| **K** | Membrane MD on prioritized hits (completed: RL_Gen_37_isoA, RL_Gen_29_isoA, S-warfarin ref) | CHARMM-GUI + GROMACS; `md_gromacs/` + `scripts/parse_md_results_summary.py` |

**Publication freeze:** For the manuscript, step H is **not** run automatically during `build_all.sh`. Instead, a frozen snapshot is used: `publication/data/coagulation_admet_multi_task_publication.csv`. This prevents accidental benchmark contamination while writing the paper.

### 2.3 The closed-loop diagram (conceptual)

```
ChEMBL bioactivity ──┐
                     ├──► Master CSV ──► GNN training ──► Checkpoint (.pth)
RL_Gen Vina labels ──┘         ▲                              │
                               │                              │
                               │         REINVENT4 ◄──────────┘
                               │              │
                               │              ▼
                               │      New SMILES (RL_Gen_NN)
                               │              │
                               │              ▼
                               └── Active-learning merger ◄── Flexible Vina docking
```

**Key idea:** Each docking campaign can add new **pseudo-labels** (estimated potencies) for generated molecules. Those labels enrich the next training round. Validation and test compounds are **protected** — their labels are never overwritten by the merger.

### 2.4 What "active learning" means here

In classical active learning, a model chooses which experiments to run next. Here, the loop is **human-steered but automated in software**:

1. The GNN guides REINVENT4 toward promising chemical space.
2. Top generations are docked (expensive 3D step).
3. Docking scores become training labels.
4. The model is retrained.

It is "active" because each cycle **actively improves the training set** rather than using a fixed frozen dataset forever.

---

## 3. Glossary of terms (A–Z)

| Term | Plain-English meaning | Why it matters in this study |
|------|----------------------|------------------------------|
| **Active learning** | Iteratively adding new labeled data based on model-guided exploration | RL_Gen docking labels feed back into GNN training |
| **Adam** | A popular neural-network optimizer (adaptive learning rate) | Used to train the GAT |
| **ADMET** | Drug-likeness and safety-related properties | CYP2C9 and HSA are predicted tasks |
| **Affinity (ΔG)** | Estimated binding free energy from docking (kcal/mol); more negative = stronger predicted binding | Converted to pseudo–pXC₅₀ for training |
| **AutoDock Vina** | Fast molecular docking program | Screens RL_Gen library against VKORC1 |
| **Bemis–Murcko scaffold** | Core ring framework of a molecule with side chains stripped | Used to split train/test without analogue leakage |
| **Canonical SMILES** | Standardized text representation of a molecule | Deduplication key in master CSV |
| **ChEMBL** | Public database of drug-like bioactivity measurements | Primary source of experimental labels |
| **Checkpoint** | Saved neural-network weights after training | `coagulation_admet_gnn.pth` |
| **Chirality / stereochemistry** | 3D handedness at tetrahedral centers (@/@@ in SMILES) | GNN is chirality-blind; docking uses enumerated isomers |
| **Coumarin** | Scaffold family including warfarin-like inhibitors | Generative prior bias |
| **CYP2C9** | Liver cytochrome P450 enzyme | Metabolism off-target task |
| **Dropout** | Randomly zeroing neurons during training to reduce overfitting | *p* = 0.1 in GAT |
| **Early stopping** | Stop training when validation loss stops improving | Patience = 25 epochs |
| **ECFP4 / Morgan fingerprint** | Binary vector encoding molecular substructures (radius 2, 2048 bits) | Baseline ML features |
| **Epoch** | One full pass through the training set | Up to 150 epochs |
| **ExternalProcess** | REINVENT4 hook that calls an external program to score molecules | Runs `gnn_predict.py` |
| **Factor Xa / XIIa / Thrombin** | Blood coagulation proteases | Multi-task coagulation endpoints |
| **Flexible docking** | Receptor side chains can move during docking | Residues A:217, 269, 272, 276 on VKORC1 |
| **GAT (Graph Attention Network)** | GNN layer that learns weighted attention over neighbor atoms | Encoder in DynamicMultiTaskGNN |
| **GNN** | Graph Neural Network — ML on molecular graphs | Core predictive model |
| **Global mean pooling** | Average all atom embeddings into one molecule vector | Graph → vector step |
| **H-bond occupancy** | Fraction of MD frames where ligand and protein residue form a hydrogen bond | Binding-site persistence (Table S7, Figure 9) |
| **GROMACS resid** | Residue index in the MD topology (may differ from PDB/PLIP numbering) | Mapped via `md_gromacs/scripts/hbond_mapping.py` |
| **Hold-out set** | Data never used during training; final honest test | Murcko test partition |
| **Hyperparameter** | User-chosen training setting (learning rate, batch size, etc.) | Table in Methods §2.3 |
| **IC50 / Ki** | Experimental concentration metrics (lower = more potent) | Converted to pXC₅₀ |
| **Masked loss** | Loss computed only where labels exist | Enables multi-task with missing data |
| **Meeko** | Tool converting RDKit molecules to PDBQT for Vina | Ligand preparation |
| **MLP** | Multi-layer perceptron (fully connected neural net) | Readout head after pooling |
| **Morgan RF** | Random forest on Morgan fingerprints | Strong VKORC1 ranking baseline |
| **Multi-task learning** | One model predicts several endpoints at once | Six pXC₅₀ tasks |
| **Murcko split** | Train/test split by scaffold clusters | Prevents memorizing close analogues |
| **PDB / PDBQT** | Protein/ligand structure file formats for docking | 6WV3 receptor |
| **PLIP** | Protein–ligand interaction profiler | Optional interaction analysis |
| **Pseudo-label** | Label derived from computation, not experiment | Vina-derived pXC₅₀ |
| **pXC₅₀** | −log₁₀ of molar potency; higher = more potent | Common ML target scale |
| **QED** | Quantitative Estimate of Drug-likeness (0–1) | REINVENT4 reward component |
| **Random forest (RF)** | Ensemble of decision trees | Morgan baseline |
| **REINVENT4** | Reinforcement-learning molecular generator | De novo design engine |
| **RDKit** | Cheminformatics toolkit | SMILES parsing, scaffolds, conformers |
| **RL_Gen_NN** | Project ID for REINVENT-generated ligand number NN | e.g. RL_Gen_37 |
| **RMSD** | Root-mean-square deviation — structural similarity measure | MD pose stability |
| **SA score** | Synthetic accessibility (lower = easier to make) | REINVENT4 reward |
| **Scaffold** | Core molecular framework | Split unit |
| **SMILES** | Line notation for molecular structure | Universal input format |
| **Spearman ρ** | Rank correlation (−1 to +1) | VKORC1 ranking metric |
| **Staged learning** | REINVENT4 training in phases with increasing difficulty | `coumarin_rl.toml` |
| **TPSA** | Topological polar surface area | ADMET descriptor in Table 3 |
| **VKORC1** | Target protein (vitamin K epoxide reductase) | Primary design target |
| **Weighted geometric mean** | Multiplicative score combination with weights | REINVENT4 total reward |

---

## 4. Representing molecules: SMILES, graphs, and fingerprints

### 4.1 SMILES

**SMILES** (Simplified Molecular Input Line Entry System) is a string that encodes a molecule's connectivity. Example: warfarin-like structures are represented as text that RDKit can parse into atoms and bonds.

**Canonical SMILES** is a unique standardized form produced by RDKit. The master CSV uses canonical SMILES so the same molecule isn't duplicated under different string variants.

**Flat SMILES** means stereochemistry at chiral centers is **not specified** (no `@`/`@@`). **Explicit isomers** (`_isoA`, `_isoB`) assign chirality before docking.

### 4.2 Molecular graphs (for the GNN)

The GNN does not read SMILES directly. RDKit converts each molecule into a **graph**:

- **Nodes** = atoms, each with 10 numeric features (atomic number, degree, charge, hybridization, aromaticity, mass, valence, hydrogens, ring membership).
- **Edges** = bonds (undirected; each bond becomes two directed edges).

**Why graphs?** They let the network learn from atom connectivity and local chemical environment, which is closer to physical structure than a flat string.

**Chirality-blind design:** The 10 atom features do **not** encode `@`/`@@` stereochemistry. Murcko scaffolds also strip stereo. This is intentional but limits enantiomer discrimination by the GNN.

### 4.3 Morgan fingerprints (for the baseline)

**Morgan fingerprints** (also called **ECFP4** when radius = 2) hash circular substructures into a fixed-length bit vector (2048 bits here). A **random forest** (500 trees) maps fingerprints → pXC₅₀ per task.

**Why include this baseline?** Fingerprints + RF are a strong, interpretable classical ML approach. The paper shows Morgan RF actually ranks VKORC1 better than the GAT (Spearman ρ 0.55 vs 0.21), which is why the GNN is used as a **multi-endpoint ranker** in REINVENT4, not as a sole potency oracle.

---

## 5. Every mathematical formula — what it means and where it is used

### 5.1 ChEMBL potency conversion (experimental pXC₅₀)

**Formula:**

pXC₅₀ = −log₁₀(value_nM × 10⁻⁹)

**Variables:**

- `value_nM` = measured IC50 or Ki in nanomolar units from ChEMBL.
- `× 10⁻⁹` converts nanomolar to molar (M).
- `−log₁₀` puts potency on a log scale where **bigger numbers = more potent**.

**Intuition:** If IC50 = 100 nM = 10⁻⁷ M, then pXC₅₀ = 7. If IC50 = 10 nM, pXC₅₀ = 8 (tenfold more potent).

**Where used:**

- Building the master CSV (§2.2).
- Regression targets for GNN training and Morgan RF.
- All tables comparing experimental vs predicted potency.

**Duplicate handling:** Multiple ChEMBL rows with the same canonical SMILES and target are **averaged** before training.

---

### 5.2 Masked mean squared error (training loss)

**Formula:**

L_MSE = Σₙ,ₜ mₙₜ (ŷₙₜ − yₙₜ)² / Σₙ,ₜ mₙₜ

**Variables:**

- *n* = compound index, *t* = task index (1 of 6 endpoints).
- ŷₙₜ = model prediction.
- yₙₜ = true label (pXC₅₀).
- mₙₜ = mask (1 if label exists, 0 if missing).

**Intuition:** Ordinary MSE would punish missing labels as if they were zero. Masking **skips** missing tasks so the model only learns from real measurements. The denominator normalizes by how many labels are actually present in the batch.

**Where used:**

- Every training and validation step in `dynamic_gnn_pipeline.py`.
- Checkpoint selection: save weights at **lowest validation L_MSE** (Figure 1).
- The trained checkpoint is loaded by `gnn_predict.py` for REINVENT4.

**Why MSE?** Standard choice for regression; penalizes large errors more than small ones (squaring).

---

### 5.3 RMSE (Root Mean Squared Error)

**Formula:**

RMSE = √( (1/N) Σᵢ (ŷᵢ − yᵢ)² )

**Intuition:** Average prediction error magnitude in **the same units as pXC₅₀** (log potency). RMSE = 1.0 means typical errors around 1 log unit (~10-fold potency).

**Where used:** Table 1 test metrics. Pooled RMSE = 1.00 across 2,210 valid task–compound pairs.

**RMSE vs MAE:** RMSE punishes occasional huge mistakes more harshly because of the square.

---

### 5.4 MAE (Mean Absolute Error)

**Formula:**

MAE = (1/N) Σᵢ |ŷᵢ − yᵢ|

**Intuition:** Average absolute deviation. Less sensitive to outliers than RMSE. MAE = 0.70 here means predictions are off by ~0.7 log units on average.

**Where used:** Table 1 alongside RMSE.

---

### 5.5 R² (coefficient of determination)

**Formula:**

R² = 1 − Σᵢ(ŷᵢ − yᵢ)² / Σᵢ(yᵢ − ȳ)²

**Intuition:**

- R² = 1 → perfect predictions.
- R² = 0 → model is no better than always predicting the mean ȳ.
- **R² < 0** → model is **worse** than the mean (happens for VKORC1: R² = −0.38).

**Where used:** Table 1 per-task. Strong on Factor XIIa (0.64), weak/negative on sparse VKORC1.

**Caution:** R² is unstable with tiny test sets (HSA has only 5 test labels).

---

### 5.6 Spearman rank correlation (ρ)

**Definition:** Pearson correlation applied to **ranked** data instead of raw values.

**Intuition:** Measures whether the model **orders** compounds correctly (best vs worst), even if absolute values are miscalibrated. ρ = +1 perfect ranking, 0 no relationship, −1 reversed.

**Where used:** VKORC1 in Table 1 and Supplementary Table S5. Morgan RF achieves ρ = 0.55 vs GAT ρ = 0.21.

**Why emphasize ρ for VKORC1?** With only 41 test labels and weak absolute R², ranking quality is the more honest success metric — especially for REINVENT4, which uses the GNN as a **relative** reward.

---

### 5.7 Bemis–Murcko scaffold split (algorithm, not a single equation)

**Procedure:**

1. Compute Murcko scaffold for each molecule (stereo stripped).
2. Group molecules sharing the same scaffold.
3. Sort scaffold groups largest-first.
4. Greedily assign whole groups to train (80%), validation (10%), or test (10%) until size cutoffs are met.

**Why?** Random splitting would put close analogues (same scaffold, different substituents) in both train and test. The model could **memorize scaffold patterns** and look artificially good. Scaffold splitting forces generalization to **new core structures**.

**Where used:** All benchmark metrics (Table 1, Figures 1–4). Frozen in `publication/data/gnn_scaffold_split.json`.

---

### 5.8 Vina affinity → pseudo–pXC₅₀

**Formula:**

pXC₅₀ = −ΔG / 1.36

**Variables:**

- ΔG = best AutoDock Vina score (kcal/mol) for the top docking pose.
- 1.36 kcal/mol ≈ 2.303 × R × T at room temperature — the free-energy change corresponding to **one log₁₀ unit** of binding constant.

**Intuition:** More negative ΔG (stronger predicted binding) → higher pXC₅₀. This puts docking on the same numeric scale as ChEMBL labels for training.

**Where used:**

- `active_learning_merger.py` when merging RL_Gen results.
- Table 2 (docking leaderboard), Table 4 (alongside GNN predictions).

**Caution:** This is a **rough thermodynamic mapping**, not experimental potency. Docking scores are useful for ranking and weak supervision, not as ground truth.

---

### 5.9 Active-learning merge rule (retain strongest label)

**Formula:**

pXC₅₀^stored = max(pXC₅₀^old, pXC₅₀^new)

**Intuition:** If the same canonical SMILES is docked again with a better (higher) pseudo-label, keep the optimistic bound. If a worse score arrives, keep the old stronger label.

**Where used:** `active_learning_merger.py` for train-split rows only.

**Protection rule:** If a SMILES is in the frozen validation or test split, **never overwrite** its labels — this prevents cheating on the benchmark.

---

### 5.10 REINVENT4 weighted geometric mean reward

**Formula:**

S_total = ( Π_c s_c^w_c )^(1 / Σ_c w_c)

**Variables:**

- *c* indexes scoring components: SA, QED, MW, GNN VKORC1.
- s_c ∈ [0, 1] = transformed component score.
- w_c = component weight.

**Weights in this project:** SA 1.5, QED 1.0, MW 0.5, GNN VKORC1 2.5.

**Intuition — geometric vs arithmetic mean:**

- **Arithmetic mean:** one bad score can be compensated by high others.
- **Geometric mean:** if any component is near zero, total score collapses — **all channels must be decent**.

The exponent 1/Σw_c normalizes weights so the result stays on a comparable scale.

**GNN transform:** Raw predicted pXC₅₀ is passed through a **sigmoid** squashing to [0, 1] with window 5.0–9.0 (see `coumarin_rl.toml`).

**Where used:** Every REINVENT4 generation step. High S_total molecules are more likely to be retained and elaborated. Figure 7 compares generated Pred_VKORC1 distribution to training labels.

---

## 6. Statistical and ML concepts explained for beginners

### 6.1 Train / validation / test — why three splits?

| Split | Fraction | Purpose |
|-------|----------|---------|
| **Train** | 80% | Learn model weights |
| **Validation** | 10% | Tune stopping, pick checkpoint, monitor overfitting |
| **Test** | 10% | **One-time** honest evaluation reported in the paper |

**Never** tune hyperparameters using test data — that would leak information and inflate metrics.

**Quirk in this dataset:** Validation has only **1** VKORC1 label, so checkpoint selection is dominated by coagulation protease tasks, not VKORC1.

### 6.2 Overfitting and dropout

**Overfitting** = model memorizes training noise instead of learning general patterns. Signs: train loss keeps dropping while validation loss rises.

**Dropout (p = 0.1)** randomly disables 10% of neurons during training, forcing redundant representations — a regularization trick.

**Early stopping (patience 25)** halts training if validation loss doesn't improve for 25 epochs (after 15-epoch warmup).

### 6.3 Multi-task learning — why six tasks at once?

**Shared encoder** learns chemical features useful across endpoints. Coagulation proteases have **many** ChEMBL labels; VKORC1 has few. Multi-task training can **transfer** representation learning from data-rich tasks to sparse ones.

**Trade-off:** VKORC1-specific signal may be diluted — consistent with Morgan RF beating GAT on VKORC1 ranking.

### 6.4 Ablation study

The **VKORC1-only GAT** uses the same architecture but trains only on VKORC1 labels. Comparing it to the full multi-task model tests whether auxiliary tasks help or hurt VKORC1 (Figure 2).

### 6.5 Baseline fairness

All models use the **same Murcko split** and the **same frozen master** for publication benchmarks. Without this, comparisons would be meaningless.

### 6.6 Pooled metrics

**Pooled** = concatenate all valid (compound, task) pairs across six endpoints and compute one RMSE/MAE/R². This weights tasks by how many test labels they have (proteases dominate).

---

## 7. The GNN architecture step by step

### 7.1 DynamicMultiTaskGNN structure

```
SMILES → RDKit graph (atoms + bonds)
       → 4× GAT layers [64 → 128 → 128 → 64] + ReLU + dropout
       → Global mean pool (graph-level vector)
       → MLP (128 hidden) → 6 outputs (one pXC₅₀ per task)
```

### 7.2 What is a GAT layer?

A **Graph Attention Network** layer updates each atom's embedding by looking at its bonded neighbors, with **learned attention weights** — the model decides which neighbors matter most. Compare to **GCN** (Graph Convolutional Network), which uses fixed aggregation; this project uses GAT.

### 7.3 Training hyperparameters (what each knob does)

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| Optimizer Adam | lr 1×10⁻³ | Step size for weight updates |
| Weight decay | 1×10⁻⁵ | L2 penalty; discourages huge weights |
| Batch size | 64 | Molecules per gradient step |
| Max epochs | 150 | Upper limit on training passes |
| ReduceLROnPlateau | patience 15, factor 0.5 | Halve learning rate if validation stalls |
| Seed | 42 | Reproducible randomness |

### 7.4 Inference (scoring new molecules)

`gnn_predict.py` loads the checkpoint, converts each input SMILES to a graph, runs a forward pass, and returns the first output head (VKORC1 pXC₅₀) as JSON for REINVENT4.

---

## 8. Docking pipeline explained

### 8.1 What is molecular docking?

Docking **poses** a ligand in a protein binding site and **scores** binding geometry using a physics-inspired energy function. It predicts **binding mode** (how the molecule sits) and **affinity estimate** (ΔG).

It is faster than experiments but imperfect: receptor flexibility, water, entropy, and force-field errors limit accuracy.

### 8.2 Receptor: PDB 6WV3

Human VKORC1 co-crystal with **S-warfarin** defines the binding site on chain A. Flexible side chains at residues **217, 269, 272, 276** move during docking; the rest of the receptor is effectively rigid.

### 8.3 Ligand preparation steps (`main_pipeline.py`)

1. Read active ligands from `config.yaml` (synced from `config_master.yaml`).
2. Generate **3D conformers** from SMILES (20 conformers per ligand).
3. Protonate and assign rotatable bonds (**Meeko**).
4. Convert to **PDBQT** format for Vina.
5. Run Vina: exhaustiveness 20, 9 modes, energy range 3 kcal/mol.
6. Write `VKORC1_Human_screening_results.csv` with `best_affinity` per ligand.

### 8.4 Flexible vs rigid docking

**Rigid docking** = protein frozen. **Flexible docking** = selected side chains rotate — often improves pose realism for induced-fit pockets at the cost of compute.

### 8.5 Quality controls (§2.8)

| Control | What it checks | Result |
|---------|----------------|--------|
| Reference re-dock | S-warfarin reproduces crystal pose | RMSD 1.64 Å (< 2.0 Å threshold) |
| Enrichment ROC-AUC | Actives vs decoys separate by score | AUC 0.964 |
| Flex re-dock spot-check | Stored vs re-run affinities | Mean absolute Δ 0.12 kcal/mol (RL_Gen) |
| Enantiomer label fix | Swapped R/S labels | Pure swap, no model retrain needed |

---

## 9. REINVENT4 generative design explained

### 9.1 What REINVENT4 does

REINVENT4 is a **reinforcement learning** agent that generates SMILES strings. It starts from a **prior** model (pretrained on drug-like molecules) and learns an **agent** policy that maximizes a **scoring function**.

### 9.2 Staged learning

Training proceeds in stages (`coumarin_rl.toml`): each stage runs up to 150 steps, checkpointing the agent. Stages can tighten objectives (e.g., max_score 0.8 termination).

### 9.3 Diversity filter

**IdenticalMurckoScaffold** bucket prevents the agent from filling memory with the same scaffold repeatedly (bucket size 25, min score 0.4).

### 9.4 Scoring components in plain language

| Component | Weight | What it rewards |
|-----------|--------|-----------------|
| **SA** (synthetic accessibility) | 1.5 | Molecules that look synthesizable (not overly complex) |
| **QED** | 1.0 | Drug-likeness (size, polarity, flexibility balance) |
| **MW** | 0.5 | Molecular weight near drug-like range (150–500 Da window) |
| **GNN VKORC1** | 2.5 | High model-predicted VKORC1 pXC₅₀ (strongest weight) |

### 9.5 From generation to docking

1. REINVENT4 writes generations to CSV (see `publication/data/reinvent_provenance.json`).
2. `inject_ai_leads.py` picks top hits, assigns `RL_Gen_NN` names, updates `config_master.yaml`.
3. `main_pipeline.py` docks active ligands.
4. Results feed `active_learning_merger.py`.

**Important:** GNN reward and Vina affinity are **different scales**. Table 4 shows RL_Gen_45: good docking but low GNN score — the GNN shapes exploration, not final ranking.

---

## 10. Stereochemistry and the RL_Gen_37 case study

### 10.1 Why stereochemistry matters

Many drug-like molecules are **chiral** — mirror-image isomers can have different biological activity. SMILES without `@`/`@@` specifies **topology only** (flat), not 3D handedness.

### 10.2 Enumeration (_isoA, _isoB)

RDKit expands undefined centers into explicit stereoisomers before docking. **RL_Gen_37** flat parent docked at −12.24 kcal/mol; **RL_Gen_37_isoA** at −10.70; **RL_Gen_37_isoB** at −9.75.

**Lesson:** The #1 flat-SMILES rank is **not** the final potency story. MD parametrization used **isoA** (CIP R).

### 10.3 GNN blindness to chirality

Because node features omit stereo, the GNN cannot distinguish enantiomers with identical connectivity. Docking with explicit isomers partially compensates at the structure-based stage.

### 10.4 RL_Gen_29_isoA — the MD counterexample

**RL_Gen_29_isoA** ranked highly by isoA flexible docking (−11.42 kcal/mol) and had the richest static H-bond network among top RL hits in flexible docking (seven contacts in Table 3). In **100 ns membrane MD**, however, second-half mean ligand RMSD reached **4.74 Å** with only **10.3%** ASN80 H-bond occupancy (vs **99.4%** for RL_Gen_37_isoA).

**Lesson:** Prioritize MD for top docking hits, but expect some interaction-rich scaffolds to fail bilayer stability even when Vina scores look favorable.

---

## 11. Membrane molecular dynamics (MD) — comparative validation

### 11.1 Why MD after docking?

Docking is a **static snapshot**. MD simulates **motion over time** in a solvated lipid bilayer approximating the ER membrane environment of VKORC1. The manuscript uses MD to ask: *Does a flexible-docking pose stay in the pocket, and which protein residues maintain contact?*

**Key lesson from this study:** A strong Vina score and a rich static H-bond network (from docking analysis) do **not** guarantee pose stability in MD. **RL_Gen_29_isoA** is the clearest example.

### 11.2 Systems simulated (as of publication)

| Run folder | Manuscript label | Production | Role |
|------------|------------------|------------|------|
| `RL_Gen_37` | **RL_Gen_37_isoA** | 100 ns | Top flat-dock hit; isoA embed used for MD |
| `RL_Gen_29_isoA` | **RL_Gen_29_isoA** | 100 ns | Interaction-rich scaffold; MD counterexample |
| `S_Warfarin_ref` | **S-warfarin (ref)** | 20 ns | Crystal-ligand reference pilot |

**Naming note:** The GROMACS run directory for the lead compound is `RL_Gen_37`, but the manuscript always calls it **RL_Gen_37_isoA** because the isoA stereochemical embed was used for parametrization.

### 11.3 Setup summary (all systems)

- **CHARMM-GUI** Membrane Builder embeds the docked complex in an ER-like bilayer.
- Lipids: POPC/POPE/POPS/cholesterol (6:2:1:2 per leaflet); 0.15 M NaCl; TIP3P water.
- **CHARMM36m** protein/lipid parameters; **CGenFF** (auto) for ligands.
- Equilibration: CHARMM-GUI step6.1–6.6; production: **GROMACS 2024.3**, 310 K, 2 fs timestep.
- Analysis outputs live under `md_gromacs/runs/<SYSTEM>/analysis/`.

### 11.4 Metrics reported (Table 5)

| Metric | Meaning |
|--------|---------|
| **Protein Cα RMSD** | Backbone movement vs starting structure (mean; last 25% of trajectory) |
| **Ligand RMSD** | Ligand heavy atoms after fit to protein backbone — **lower = more stable pose** |
| **Temperature / pressure** | Sanity checks (~310 K; ~1 bar for 100 ns RL runs) |
| **ASN80 H-bond (%)** | Occupancy at the key pocket asparagine (PLIP label; GROMACS resid ~217) |

Canonical numbers are merged in `publication/data/md_results_summary.json` (built by `scripts/parse_md_results_summary.py` from the package summary, `.xvg` files, and `hbond_summary.json`).

### 11.5 Comparative results (what the numbers mean)

| System | Ligand RMSD 2nd half (Å) | ASN80 H-bond (%) | SER81 H-bond (%) | Interpretation |
|--------|--------------------------|------------------|------------------|----------------|
| **RL_Gen_37_isoA** | **1.92** (max 2.83) | **99.4** | 0.0 | Stable isoA pose; persistent ASN contact |
| **RL_Gen_29_isoA** | **4.74** (max 5.34) | 10.3 | 0.4 | Pose drift despite good isoA docking (−11.42 kcal/mol) |
| **S-warfarin (ref)** | 3.18 (max 4.83) | 1.0 | **62.2** | Different contact pattern than RL_Gen_37; 20 ns pilot only |

**Stability ranking by ligand RMSD:** RL_Gen_37_isoA ≪ S-warfarin ref < RL_Gen_29_isoA (lower is better). Warfarin has moderate ligand RMSD but low protein RMSD — the reference ligand does not H-bond primarily through ASN80 in this construct.

**H-bond detail (Table S7, Figure 9):** Per-residue occupancy is tracked for ASN80, SER81, TYR139, THR138, PHE55, TRP59, and VAL134. PLIP residue numbers (from docking interaction analysis) are mapped to GROMACS `resid` per system because the GFP–VKORC1 fusion topology renumbers residues.

### 11.6 H-bond analysis pipeline (no MD rerun needed)

If production `.xtc` trajectories already exist, H-bond occupancy can be recomputed **without** rerunning 100 ns simulations:

1. `md_gromacs/scripts/hbond_mapping.py` — align PLIP labels to GROMACS `resid` via docked complex + `step5_input.pdb`.
2. `md_gromacs/scripts/analyze_md.py` — runs `gmx hbond`, writes `hbond_summary.json` and per-residue `.xvg`.
3. `scripts/parse_md_results_summary.py` — merges RMSD, T/P, and H-bond JSON into `publication/data/md_results_summary.json`.

GROMACS path used in this project: `/home/amar/miniforge3/envs/gromacs_env/bin/gmx`.

### 11.7 Figures 8 and 9

| Figure | File | Content |
|--------|------|---------|
| **8** | `figure8_md_rmsd_rl_gen_37` | Three-panel RMSD time series (protein + ligand) for all completed systems |
| **9** | `figure9_md_hbond_occupancy` | Heatmap of binding-site H-bond occupancy across systems |

Regenerate via `bash publication/build_all.sh` (runs the MD parser, then `generate_manuscript_assets.py`).

### 11.8 What MD does *not* prove

- No experimental binding affinity or synthesis.
- S-warfarin reference is only **20 ns** — not a fully equilibrated benchmark.
- RL_Gen_22 and RL_Gen_45 were **not** simulated for this submission.
- Full `.xtc` trajectories are large; the Zenodo bundle includes analysis summaries (`md_analysis/`) rather than multi-GB trajectory files.

---

## 12. Data files and scripts — what each one does

### 12.1 Core data files

| File | Role |
|------|------|
| `data/coagulation_admet_multi_task.csv` | Live master training table |
| `publication/data/coagulation_admet_multi_task_publication.csv` | Frozen publication snapshot (18,976 rows) |
| `publication/data/gnn_scaffold_split.json` | Frozen train/val/test SMILES lists |
| `publication/data/gnn_vkorc1_label_audit.json` | VKORC1 label provenance and overlap stats |
| `coagulation_admet_gnn.pth` | Trained GAT checkpoint |
| `results/docked_poses/.../VKORC1_Human_screening_results.csv` | Flexible docking output |
| `publication/data/md_results_summary.json` | Canonical multi-system MD metrics (RMSD + H-bond) |
| `publication/data/md/MD_RESULTS_SUMMARY.json` | Raw GROMACS analysis summary from MD publication package |
| `publication/data/md_rl_gen_37_summary.json` | Legacy single-system stub (auto-synced from md_results_summary) |
| `md_gromacs/runs/*/analysis/` | Per-system RMSD `.xvg`, `hbond_summary.json`, H-bond `.xvg` |

### 12.2 Core Python scripts

| Script | Role |
|--------|------|
| `dynamic_gnn_pipeline.py` | Train GAT; save checkpoint |
| `gnn_baseline_evaluation.py` | Metrics for GAT, Morgan RF, ablation |
| `gnn_model.py` | Shared model, featurization, split, loss |
| `morgan_fp_baseline.py` | Fingerprint + RF training/prediction |
| `gnn_predict.py` | REINVENT4 ExternalProcess endpoint |
| `main_pipeline.py` | Full flexible docking workflow |
| `active_learning_merger.py` | Merge RL_Gen Vina → master CSV |
| `inject_ai_leads.py` | Top REINVENT hits → config |
| `publication/generate_manuscript_assets.py` | Tables and figures (incl. Table 5, S7, Figures 8–9) |
| `scripts/parse_md_results_summary.py` | Merge MD package + hbond JSON → md_results_summary.json |
| `md_gromacs/scripts/analyze_md.py` | H-bond occupancy from existing trajectories |
| `md_gromacs/scripts/hbond_mapping.py` | PLIP label → GROMACS resid mapping |
| `publication/build_all.sh` | One-command rebuild |
| `deposition/prepare_deposition.py` | Zenodo bundle |

### 12.3 Configuration files

| File | Role |
|------|------|
| `config_master.yaml` | Master ligand library (all RL_Gen entries) |
| `config.yaml` | Active subset for docking (auto-synced) |
| `configs/coumarin_rl.toml` | REINVENT4 scoring weights and stages |

---

## 13. Reading the results — table by table

### Table 1 — GNN test metrics

Per-task RMSE, MAE, R², Spearman ρ on the **held-out Murcko test set**. This is the primary honest benchmark. VKORC1 negative R² means don't trust absolute VKORC1 numbers.

### Table 2 — VKORC1 docking leaderboard

Top RL_Gen ligands by flexible Vina ΔG. Reports flat-SMILES ranks; see §3.3 for isoA caveat.

### Table 3 — ADMET summary

QED, TPSA, MW, H-bond counts for top docked hits. Shows developability spread among priors.

### Table 4 — Multi-task GNN predictions

Model outputs for top five docking hits across all six tasks. Use for **selectivity triage** (e.g., potent VKORC1 prediction but also high thrombin risk).

### Table 5 — MD metrics (comparative)

Three **completed** membrane MD systems only (RL_Gen_37_isoA, RL_Gen_29_isoA, S-warfarin ref; matches manuscript Table 5). Includes production length, T/P, protein/ligand RMSD, and ASN80 H-bond occupancy. Use this table to compare pose stability — not just whether MD was run.

### Supplementary tables

| Table | Content |
|-------|---------|
| S1 | Full per-task metrics both models |
| S2 | Complete RL_Gen docking |
| S3 | REINVENT scoring component details |
| S4 | Membrane MD summary (mirrors Table 5) |
| S5 | VKORC1 benchmark excluding RL-library SMILES |
| S6 | Scaffold split statistics |
| **S7** | **Per-residue H-bond occupancy across MD systems** |

---

## 14. Figures — what to look for

| Figure | What it shows | What to learn |
|--------|---------------|---------------|
| **1** | Training/validation loss and R² vs epoch | Model converges; checkpoint at min val MSE |
| **2** | Bar chart R² by task for 3 models | Morgan RF wins VKORC1; GAT wins dense protease tasks |
| **3** | Scaffold split sizes | Train/val/test are scaffold-disjoint |
| **4** | Morgan RF R², ρ, VKORC1 scatter | Classical baseline strength on ranking |
| **5** | Flex re-dock spot-check | Docking protocol is reproducible |
| **6** | Interaction fingerprint heatmap | H-bonds vs affinity trade-offs |
| **7** | REINVENT Pred_VKORC1 vs training distribution | Generator explores high-score tail |
| **8** | MD RMSD time series (3 systems) | Compare pose stability over time; RL_Gen_37 stable, RL_Gen_29 drifts |
| **9** | H-bond occupancy heatmap | ASN80 dominates RL_Gen_37; SER81 dominates warfarin ref |

---

## 15. Mapping manuscript sections to this guide

| Manuscript section | Read in this guide |
|--------------------|-------------------|
| Abstract | §1, §2, §6.6, limitations in §16 |
| §1 Introduction | §1, §2.3 |
| §2.1 Target | §1.1, §8.2 |
| §2.2 Master dataset | §4.1, §5.1 |
| §2.3 GNN | §4.2, §5.2–5.7, §7 |
| §2.4 Docking | §8, §5.8 |
| §2.5 Active learning | §2.3, §5.9 |
| §2.6 REINVENT4 | §9, §5.10 |
| §2.7 Stereochemistry | §10 |
| §2.8 QC | §8.5 |
| §2.9 MD | §11 |
| §3 Results | §13, §14 |
| §4 Discussion | §6.3, §16 |

---

## 16. Limitations you must understand (honest science)

1. **No experimental validation** — all potency claims are computational.
2. **VKORC1 GNN is weak** for absolute prediction; use as ranker / multi-task panel only.
3. **Vina pseudo-labels ≠ experiment** — thermodynamic mapping is approximate.
4. **Chirality-blind GNN** — enantiomers not distinguished by the network.
5. **Sparse endpoints** — HSA (5 test labels) and VKORC1 (41) metrics are noisy.
6. **Flat vs isoA docking gap** — top flat rank can mislead; always enumerate stereo for finals.
7. **MD is comparative but not exhaustive** — three systems (2×100 ns RL + 20 ns warfarin ref) only; illustrative, not experimental proof.
8. **Docking ≠ MD stability** — RL_Gen_29_isoA shows strong static interactions but high ligand RMSD in bilayer MD.
9. **Publication master is frozen** — live active-learning may diverge from deposited snapshot.

---

## 17. Reproducing the workflow (commands)

```bash
# Environment
conda create -n warfarin python=3.10
conda activate warfarin
pip install -r requirements.txt

# Train GNN
python dynamic_gnn_pipeline.py --seed 42

# Evaluate baselines
python gnn_baseline_evaluation.py --split-from publication/data/gnn_scaffold_split.json \
  --data publication/data/coagulation_admet_multi_task_publication.csv

# Full manuscript rebuild (includes MD summary parse + figures)
bash publication/build_all.sh

# Refresh MD JSON only (no MD rerun)
python scripts/parse_md_results_summary.py
python publication/generate_manuscript_assets.py

# Docking (active ligands)
python main_pipeline.py --config config.yaml

# Active-learning merge (mutates live master — not run in frozen publication build)
python active_learning_merger.py
python dynamic_gnn_pipeline.py

# REINVENT4 (separate env)
conda activate reinvent4
reinvent -l reinvent.log configs/coumarin_rl.toml
python inject_ai_leads.py
```

Set `RUN_ACTIVE_LEARNING_MERGE=1` before `build_all.sh` only when you intentionally want to merge new docking into the live master and re-freeze.

---

## 18. Quick-reference formula sheet

| Name | Formula | Used for |
|------|---------|----------|
| ChEMBL pXC₅₀ | −log₁₀(value_nM × 10⁻⁹) | Experimental labels |
| Masked MSE | Σ m(ŷ−y)² / Σ m | GNN training |
| RMSE | √(mean squared error) | Test error magnitude |
| MAE | mean absolute error | Test error magnitude |
| R² | 1 − SS_res/SS_tot | Explained variance |
| Spearman ρ | rank correlation | VKORC1 ordering |
| Vina pXC₅₀ | −ΔG / 1.36 | Pseudo-labels from docking |
| Merge rule | max(old, new) | Active-learning dedup |
| REINVENT reward | (∏ s^w)^(1/Σw) | Generative scoring |

---

## 19. Conceptual checklist — "know it cold"

After reading this guide, you should be able to answer:

1. **What is the scientific goal?** Computational VKORC1-targeted coumarin discovery with multi-endpoint ML, docking, and generative design.
2. **What is the pipeline order?** Master CSV → split → train GNN → REINVENT → inject → dock → merge → (retrain) → evaluate → publish.
3. **Why Murcko split?** Prevent analogue leakage; honest generalization metric.
4. **Why masked loss?** Six tasks with missing labels in one table.
5. **Why geometric mean in REINVENT?** Penalize molecules weak on any objective channel.
6. **Why is VKORC1 R² negative but we still use the GNN?** Multi-task ranker / REINVENT reward, not quantitative potency model.
7. **What is the difference between pXC₅₀ (ChEMBL), pXC₅₀ (Vina), and Pred_VKORC1?** Experiment vs docking estimate vs neural prediction.
8. **What protects benchmark integrity?** Frozen split; merger skips val/test SMILES; publication master freeze.
9. **What happened with RL_Gen_37 vs RL_Gen_29?** Flat SMILES over-ranked vs isoA; MD used isoA. RL_Gen_37 stayed stable (1.92 Å ligand RMSD, 99% ASN80 H-bond); RL_Gen_29_isoA drifted (4.74 Å) despite good docking.
10. **What do Figures 8–9 show?** Comparative RMSD trajectories and binding-site H-bond occupancy for completed MD systems.
11. **What would be needed for real drug discovery?** Synthesis, biochemical assays, ADME experiments, safety studies.

---

*End of guide. For the formal methods wording, see `vkorc1_integrated_workflow_manuscript.md`. For file provenance, see `publication/data/MASTER_SNAPSHOT.md` and `RELEASE.md`.*
