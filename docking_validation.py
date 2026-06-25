#!/usr/bin/env python3
"""
Retrospective enrichment (ROC-AUC, EF) and multi-seed reproducibility for VKORC1_Human.

Outputs:
  validation/enrichment_scores.csv
  validation/enrichment_summary.txt
  validation/seed_reproducibility.csv
  validation/seed_reproducibility_summary.txt
  validation/enrichment_roc.png
"""
from __future__ import annotations

import random
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED

from grid_box import auto_box_from_native_ligand
from ligand_library import prepare_ligand_library, generate_3d_conformers
from sdf_to_pdbqt import batch_convert_ligands
from docking_engine import virtual_screening

OUT = Path("validation")
WORK = OUT / "_work"

# Literature VKORC1 / vitamin-K-antagonist actives (experimental inhibitors).
ACTIVES = {
    "active_S_warfarin": "CC(=O)C[C@@H](C1=CC=CC=C1)C2=C([O-])C3=CC=CC=C3OC2=O",
    "active_R_warfarin": "CC(=O)C[C@H](C1=CC=CC=C1)C2=C([O-])C3=CC=CC=C3OC2=O",
    "active_acenocoumarol": "CC(=O)OC1=CC=C(C=C1)C(C2=CC=CC=C2)C3=C(C4=CC=CC=C4OC3=O)C",
    "active_phenprocoumon": "CC(C)C1=CC=C(C=C1)C(C2=CC=CC=C2)C3=C(C4=CC=CC=C4OC3=O)O",
    "active_dicumarol": "OC1=CC=CC2=C1C(=O)OC2=C1OC(=O)C2=C(O)C=CC=C2C1=O",
    "active_brodifacoum": "CC1(C)CC(CC(C1)(C2=C(C=C(C=C2)Cl)Cl)OC3=CC=C(C=C3)C4=CC=CC=C4C5=CC(=O)OC5=O)C6=CC=CC=N6",
    "active_coumachlor": "CC(=O)OC1=CC=C(C=C1)C(C2=CC=CC=C2)C3=C(C4=CC=CC=C4OC3=O)Cl",
    "active_fluindione": "O=C1C(C2=CC=CC=C2)C(=O)C3=CC=CC=C13",
    "active_tioclomarol": "CC1=C(C=C(C=C1)Cl)C(C2=CC=CC=C2)C3=C(C4=CC=CC=C4OC3=O)O",
    "active_4hydroxycoumarin": "OC1=CC=CC2=C1C(=O)OC2=O",
}

# Drug-like decoy pool (non-coumarin / non-VKOR antagonist scaffolds).
DECOY_POOL = [
    "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "CC(=O)Nc1ccccc1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccc(-c2cccnc2)cc1", "COc1ccccc1OC",
    "c1ccc2ccccc2c1", "c1ccccc1C(=O)O", "NS(=O)(=O)c1ccccc1", "CCOC(=O)c1ccccc1",
    "c1ccc(-c2ccccc2)cc1", "Nc1ccccc1", "O=C(c1ccccc1)c1ccccc1", "c1ccc(Oc2ccccc2)cc1",
    "O=C(O)Cc1ccccc1", "NCc1ccccc1", "COC(=O)c1ccccc1", "N#Cc1ccccc1", "COc1ccccc1",
    "CCN(CC)CC", "C1CCCCC1", "Oc1ccccc1", "Fc1ccccc1", "Clc1ccccc1",
    "CC(C)NCC(O)COc1ccccc1", "NC(=O)c1ccccc1", "CN(C)C(=O)c1ccccc1", "CSc1ccccc1",
    "CN(C)c1ccccc1", "CC(C)c1ccccc1", "CCNC(=O)c1ccccc1", "O=C(NCCO)c1ccccc1",
    "O=C(NCCN)c1ccccc1", "O=C(NCCC)c1ccccc1", "CC(C)CCNC(=O)c1ccccc1",
    "CC(C)(C)CCNC(=O)c1ccccc1", "O=C(Nc1ccccc1)c1ccccc1", "COCCOC(=O)c1ccccc1",
    "Cc1ccccc1", "Cc1ccc(O)cc1", "Cc1ccc(N)cc1", "Cc1ccc(Cl)cc1", "Cc1ccc(F)cc1",
    "Cc1ccc(C(=O)O)cc1", "NCCc1ccccc1", "OCCc1ccccc1", "CC(=O)c1ccccc1",
    "O=C(OCc1ccccc1)c1ccccc1", "CC(C)OC(=O)c1ccccc1", "CN(C)CC(=O)c1ccccn1",
    "Cc1ccccn1", "Brc1ccccc1", "c1ccc2[nH]ccc2c1", "c1ccncc1", "c1ccoc1",
    "c1ccsc1", "c1cncnc1", "C1CCNCC1", "C1CNCCN1", "C1COCCN1", "C1CCOCC1",
    "CC(=O)NC(C)C(=O)O", "NC(CC(=O)O)C(=O)O", "NC(CS)C(=O)O", "NC(Cc1ccccc1)C(=O)O",
    "CC(C)(C)O", "CC(C)O", "CCO", "CC(C)C", "CC(=O)OC(C)(C)C",
    "O=C(O)C(O)C(O)C(O)CO", "CC(=O)NCCCS(=O)(=O)c1ccccc1", "CC(C)(C)OC(=O)N1CCC(CC1)n1cnnc1",
    "Ic1ccccc1", "CC(=O)OCCOC(=O)C", "CC(C)COC(=O)c1ccccc1", "NCCOC(=O)c1ccccc1",
    "Cc1ccc(C(=O)N)cc1", "CC(=O)Nc1ccc(O)cc1", "O=C(NCC(C)C)c1ccccc1",
    "O=C(NCC(C)(C)C)c1ccccc1", "O=C(NCC(C)O)c1ccccc1", "O=C(NCC(C)N)c1ccccc1",
    "c1ccc(C2CCCCC2)cc1", "c1ccc(C2CCNCC2)cc1", "c1ccc(C2CCOCC2)cc1",
    "CC(=O)Nc1ccc(C(C)C)cc1", "CC(=O)Nc1ccc(C(C)(C)C)cc1", "CC(=O)Nc1ccc(C(F)(F)F)cc1",
    "CC(=O)Nc1ccc(C#N)cc1", "CC(=O)Nc1ccc(C(=O)O)cc1", "CC(=O)Nc1ccc(C(=O)N)cc1",
    "CC(=O)Nc1ccc(C(=O)OC)cc1", "CC(=O)Nc1ccc(C(=O)OCC)cc1", "CC(=O)Nc1ccc(C(=O)NCC)cc1",
    "CC(=O)Nc1ccc(C(=O)NCCO)cc1", "CC(=O)Nc1ccc(C(=O)NCCN)cc1", "CC(=O)Nc1ccc(C(=O)NCCC)cc1",
]

SEED_REFS = {
    "R_Warfarin_ref": "CC(=O)C[C@H](C1=CC=CC=C1)C2=C([O-])C3=CC=CC=C3OC2=O",
    "S_Warfarin_ref": "CC(=O)C[C@@H](C1=CC=CC=C1)C2=C([O-])C3=CC=CC=C3OC2=O",
    "p_nitro_R": "CC(=O)N[C@H](C1=CC=C([N+](=O)[O-])C=C1)C2=C([O-])C3=CC=CC=C3OC2=O",
    "p_nitro_S": "CC(=O)N[C@@H](C1=CC=C([N+](=O)[O-])C=C1)C2=C([O-])C3=CC=CC=C3OC2=O",
}
SEEDS = [1, 42, 123, 456, 789, 999, 2024, 31415]


def _props(smiles: str) -> dict | None:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return {
        "mw": Descriptors.MolWt(m),
        "logp": Descriptors.MolLogP(m),
        "hbd": Lipinski.NumHDonors(m),
        "hba": Lipinski.NumHAcceptors(m),
        "rot": Lipinski.NumRotatableBonds(m),
    }


def property_match_decoys(actives: dict, pool: list[str], per_active: int = 5) -> dict[str, str]:
    pool_mols = [(s, p) for s in pool if (p := _props(s))]
    chosen: dict[str, str] = {}
    used: set[str] = set()

    for aname, asmiles in actives.items():
        ap = _props(asmiles)
        if not ap:
            continue
        scored = []
        for s, dp in pool_mols:
            if s in used:
                continue
            dist = (
                abs(dp["mw"] - ap["mw"]) / 50.0
                + abs(dp["logp"] - ap["logp"])
                + abs(dp["hbd"] - ap["hbd"])
                + abs(dp["hba"] - ap["hba"])
                + abs(dp["rot"] - ap["rot"]) * 0.25
            )
            scored.append((dist, s))
        scored.sort(key=lambda x: x[0])
        picks = 0
        for _, s in scored:
            if picks >= per_active:
                break
            if s in used:
                continue
            chosen[f"decoy_{len(chosen)+1:03d}"] = s
            used.add(s)
            picks += 1
    return chosen


def prepare_ligands(smiles_dict: dict, num_confs: int, sdf_dir: Path, pdbqt_dir: Path) -> None:
    sdf_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_ligand_library(smiles_dict)
    for _, row in df.iterrows():
        generate_3d_conformers(row["Mol"], row["Name"], output_dir=str(sdf_dir), num_confs=num_confs)
    batch_convert_ligands(sdf_dir=str(sdf_dir) + "/", pdbqt_dir=str(pdbqt_dir) + "/")


def dock_set(
    ligand_names: list[str],
    seed: int,
    cfg: dict,
    label: str,
    pdbqt_dir: Path,
) -> pd.DataFrame:
    target = "VKORC1_Human"
    tinfo = cfg["receptors"][target]
    chain = tinfo["chain"]
    dock = cfg["docking_params"]
    protonated_pdb = f"proteins/protonated/{target}_chain{chain}_protonated.pdb"
    receptor_pdbqt = f"pdbqt_receptors/{target}_chain{chain}_protonated.pdbqt"
    raw_pdb = f"proteins/raw/{tinfo['pdb_id']}.pdb"

    box = auto_box_from_native_ligand(raw_pdb, tinfo["native_ligand_resname"], padding=tinfo.get("padding", 6.0))
    box["exhaustiveness"] = dock.get("exhaustiveness", 20)
    box["num_modes"] = dock.get("num_modes", 9)

    out_dir = WORK / "docked" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    res = virtual_screening(
        ligand_pdbqt_dir=str(pdbqt_dir),
        receptor_pdbqt=receptor_pdbqt,
        target_name=target,
        box_params=box,
        output_dir=str(out_dir),
        exhaustiveness=box["exhaustiveness"],
        receptor_pdb=protonated_pdb,
        flex_res_list=tinfo.get("flexible_residues", []),
        needed_ligands=ligand_names,
        num_modes=box["num_modes"],
        min_rmsd=dock.get("min_rmsd", 1.0),
        n_cpu=dock.get("n_cpu"),
        dock_timeout=dock.get("dock_timeout_s", 900),
        seed=seed,
    )
    res["run_label"] = label
    res["seed"] = seed
    return res


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranked_labels = labels[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = 0
    auc = 0.0
    for lab in ranked_labels:
        if lab == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


def enrichment_factor(scores: np.ndarray, labels: np.ndarray, fraction: float) -> float:
    n = max(1, int(np.ceil(len(scores) * fraction)))
    order = np.argsort(scores)
    top = labels[order[:n]]
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    return (top.sum() / n) / (n_pos / len(labels))


def plot_roc(scores: np.ndarray, labels: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    order = np.argsort(scores)
    ranked = labels[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    tprs, fprs = [0.0], [0.0]
    tp = fp = 0
    for lab in ranked:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fprs, tprs, "b-", lw=2, label=f"ROC-AUC = {roc_auc(scores, labels):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("VKORC1_Human retrospective enrichment")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open("config.yaml"))
    num_confs = int(cfg["docking_params"].get("num_conformers", 20))

    actives = {k: v for k, v in ACTIVES.items() if Chem.MolFromSmiles(v)}
    decoys = property_match_decoys(actives, DECOY_POOL, per_active=5)
    print(f"Actives: {len(actives)}, property-matched decoys: {len(decoys)}")

    enrich_smiles = {**actives, **decoys}
    enrich_sdf = WORK / "enrichment" / "ligands"
    enrich_pdbqt = WORK / "enrichment" / "pdbqt"
    prepare_ligands(enrich_smiles, num_confs, enrich_sdf, enrich_pdbqt)

    print("\n=== (d) Retrospective enrichment docking (seed=42) ===")
    enrich_df = dock_set(list(enrich_smiles.keys()), seed=42, cfg=cfg,
                         label="enrichment_seed42", pdbqt_dir=enrich_pdbqt)
    enrich_df["is_active"] = enrich_df["ligand_name"].str.startswith("active_").astype(int)
    enrich_df.to_csv(OUT / "enrichment_scores.csv", index=False)

    ok = enrich_df[enrich_df["success"] == True].copy()
    scores = ok["best_affinity"].to_numpy()
    labels = ok["is_active"].to_numpy()
    auc = roc_auc(scores, labels)
    ef1 = enrichment_factor(scores, labels, 0.01)
    ef5 = enrichment_factor(scores, labels, 0.05)
    ef10 = enrichment_factor(scores, labels, 0.10)

    summary = textwrap.dedent(f"""
    RETROSPECTIVE ENRICHMENT — VKORC1_Human
    Protocol: same as main pipeline (flexible residues, exhaustiveness={cfg['docking_params'].get('exhaustiveness', 20)}, num_conformers={num_confs})
    Actives: {len(actives)} literature VKORC1 / vitamin-K antagonists
    Decoys: {len(decoys)} property-matched non-antagonist scaffolds (5 per active)
    Successfully docked: {len(ok)} / {len(enrich_df)}

    ROC-AUC: {auc:.3f}
    Enrichment factor @ 1%:  {ef1:.2f}
    Enrichment factor @ 5%:  {ef5:.2f}
    Enrichment factor @ 10%: {ef10:.2f}

    Interpretation guide:
      ROC-AUC > 0.7 = acceptable discrimination
      ROC-AUC > 0.8 = good
      EF@1% > 5 = strong early enrichment of actives in top-ranked poses
    """)
    (OUT / "enrichment_summary.txt").write_text(summary.strip() + "\n")
    print(summary)
    plot_roc(scores, labels, OUT / "enrichment_roc.png")

    print("\n=== (e) Multi-seed reproducibility for reference ligands ===")
    seed_rows = []
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        sdf_dir = WORK / "seeds" / f"sdf_{seed}"
        pdbqt_dir = WORK / "seeds" / f"pdbqt_{seed}"
        prepare_ligands(SEED_REFS, num_confs, sdf_dir, pdbqt_dir)
        res = dock_set(list(SEED_REFS.keys()), seed=seed, cfg=cfg,
                        label=f"seed_{seed}", pdbqt_dir=pdbqt_dir)
        seed_rows.append(res)

    seed_df = pd.concat(seed_rows, ignore_index=True)
    seed_df.to_csv(OUT / "seed_reproducibility.csv", index=False)

    lines = ["MULTI-SEED REPRODUCIBILITY — reference ligands vs VKORC1_Human", ""]
    for lig in SEED_REFS:
        sub = seed_df[(seed_df["ligand_name"] == lig) & (seed_df["success"] == True)]["best_affinity"]
        if len(sub) == 0:
            lines.append(f"{lig}: no successful docks")
            continue
        lines.append(
            f"{lig}: mean={sub.mean():.3f}, std={sub.std():.3f}, "
            f"min={sub.min():.3f}, max={sub.max():.3f} kcal/mol  (n={len(sub)})"
        )
    lines.append("")
    lines.append(f"Seeds tested: {SEEDS}")
    seed_summary = "\n".join(lines)
    (OUT / "seed_reproducibility_summary.txt").write_text(seed_summary + "\n")
    print(seed_summary)
    print(f"\nOutputs written to {OUT}/")


if __name__ == "__main__":
    main()
