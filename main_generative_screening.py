#main_generative_screening.py
import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem
from scipy.spatial import distance
from typing import List, Dict, Any

from gnn_model import load_checkpoint, smiles_to_inference_graph

# ==========================================
# 3. Traditional Physicochemical Filters
# ==========================================
def calculate_admet(smiles: str) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'QED': QED.qed(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol)
    }

def passes_lipinski(props: Dict[str, float]) -> bool:
    if not props:
        return False
    violations = 0
    if props['MW'] >= 500: violations += 1
    if props['LogP'] >= 5: violations += 1
    if props['HBD'] >= 5: violations += 1
    if props['HBA'] >= 10: violations += 1
    return violations <= 1

def prepare_3d_conformer(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    embed_status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if embed_status != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        return None
    return mol

def verify_nitrogen_contacts(mol: Chem.Mol) -> bool:
    if mol is None:
        return False
    conf = mol.GetConformer()
    core_oxygens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8]
    nitrogens = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
    
    if not nitrogens or not core_oxygens:
        return False
        
    o_coords = np.array([conf.GetAtomPosition(idx) for idx in core_oxygens])
    core_center = np.mean(o_coords, axis=0)
    
    for n_idx in nitrogens:
        n_coord = np.array(conf.GetAtomPosition(n_idx))
        dist = distance.euclidean(core_center, n_coord)
        if 3.0 <= dist <= 7.0:
            return True
    return False

# ==========================================
# 4. Main Integrated Screening Execution
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("   INTEGRATED GNN & PHYSICS SCREENING PIPELINE    ")
    print("==================================================")
    
    # 1. Setup Environment and Acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Acceleration: {device}")
    
    # 2. Instantiate and Load the Trained Multi-Task GNN
    model_path = "coagulation_admet_gnn.pth"
    if not os.path.exists(model_path):
        print(f"Error: Trained checkpoint '{model_path}' missing. Train your model first.")
        exit(1)
        
    print("Loading 6-Task GNN Model Weights...")
    model, _ = load_checkpoint(model_path, device)
    model.eval()
    
    # Ordered labels matching output layer indexes
    target_labels = [
        'Pred_VKORC1_pXC50', 'Pred_FXIIa_pXC50', 'Pred_FXa_pXC50', 
        'Pred_Thrombin_pXC50', 'Pred_CYP2C9_pXC50', 'Pred_HSA_pXC50'
    ]

    # 3. Input Pool (Simulated Generative Output Library)
    generated_smiles = [
        "O=C1C=Cc2ccccc2O1",                        # Base Coumarin
        "O=C1C=Cc2cc(NCC)ccc2O1",                   # Coumarin with Ethylamine
        "O=C1C=Cc2cc(NC(=O)C)ccc2O1",               # Coumarin with Acetamide
        "O=C1C=Cc2cc(N3CCOCC3)ccc2O1",              # Coumarin with Morpholine
        "O=C1C=Cc2cc(c3cccnc3)ccc2O1",              # Coumarin with Pyridine
        "O=C1C=Cc2cc(N(CC)CC)ccc2O1",               # Coumarin with Diethylamine
        "CCCCCCCCCCCCCCC(=O)Nc1ccc2c(c1)ccc(=O)o2"    # Too bulky (Fails Lipinski)
    ]
    
    results = []
    
    for idx, smiles in enumerate(generated_smiles):
        print(f"\nEvaluating Molecule {idx + 1}/{len(generated_smiles)}: {smiles}")
        
        # Phase A: Physicochemical Screening
        props = calculate_admet(smiles)
        if not passes_lipinski(props):
            print(" -> [REJECTED] Violates Lipinski Parameters.")
            continue
            
        mol_3d = prepare_3d_conformer(smiles)
        if mol_3d is None:
            print(" -> [REJECTED] Conformer optimization failed.")
            continue
            
        has_critical_contact = verify_nitrogen_contacts(mol_3d)
        
        # Phase B: Graph Neural Network Deep Inference
        graph_data = smiles_to_inference_graph(smiles)
        if graph_data is None:
            print(" -> [REJECTED] RDKit graph conversion failed.")
            continue
            
        # Manually route data vectors onto device and set single-graph batch index
        graph_data = graph_data.to(device)
        graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=device)
        
        with torch.no_grad():
            gnn_outputs = model(graph_data).cpu().numpy().flatten()
            
        # Build evaluation profile
        mol_record = {
            'SMILES': smiles,
            'MW': round(props['MW'], 2),
            'QED': round(props['QED'], 3),
            'Valid_N_Contact': has_critical_contact
        }
        
        # Dynamically inject model output predictions
        for label, score in zip(target_labels, gnn_outputs):
            mol_record[label] = round(float(score), 3)
            
        results.append(mol_record)
        print(f" -> [PASSED] GNN Profiling Complete. Predicted VKORC1 pXC50: {mol_record['Pred_VKORC1_pXC50']}")

    # ==========================================
    # 5. Multi-Property Optimization Report
    # ==========================================
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("VIRTUAL SCREENING SCOREBOARD (ORDERED BY PREDICTED VKORC1 POTENCY)")
    print("="*80)
    
    if not df_results.empty:
        # Sort candidate leads by structural viability and neural-predicted core target affinity
        final_reporting_cols = [
            'SMILES', 'MW', 'QED', 'Valid_N_Contact', 
            'Pred_VKORC1_pXC50', 'Pred_CYP2C9_pXC50', 'Pred_HSA_pXC50'
        ]
        reporting_df = df_results[final_reporting_cols].sort_values(by='Pred_VKORC1_pXC50', ascending=False)
        print(reporting_df.to_string(index=False))
    else:
        print("No generated candidate molecules survived the baseline filters.")
    print("="*80)