# branch2_scaffold_hopping.py
import os
import sys
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, FilterCatalog
from typing import List, Dict, Any

from gnn_model import load_checkpoint, smiles_to_inference_graph

# Import RDKit's contrib module for SA Score
from rdkit.Chem.RDConfig import RDContribDir
sys.path.append(RDContribDir)
from SA_Score import sascorer
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
def run_filters(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    # 1. Basic ADMET
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    qed = QED.qed(mol)
    
    # 2. Synthetic Accessibility (SA Score)
    sa_score = sascorer.calculateScore(mol)
    if sa_score > 4.5: return None # Reject hard-to-build molecules
    
    # 3. PAINS Filter (Crucial for publication)
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    if catalog.HasMatch(mol): return None # Reject toxic/false-positive structures
    
    return {'MW': round(mw, 2), 'LogP': round(logp, 2), 'QED': round(qed, 3), 'SA_Score': round(sa_score, 2)}

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print(" BRANCH 2: DE NOVO SCAFFOLD HOPPING SCREENING     ")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model_path = "coagulation_admet_gnn.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        exit(1)
        
    model, _ = load_checkpoint(model_path, device)
    model.eval()
    print("Trained GNN model loaded.")

    # 2. Load REINVENT Data
    csv_path = os.path.join("data", "ai_generated_drugs.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        exit(1)
        
    df_ai = pd.read_csv(csv_path)
    smiles_col = 'SMILES' if 'SMILES' in df_ai.columns else 'smiles'
    
    raw_smiles = df_ai[smiles_col].dropna().unique().tolist()
    clean_smiles = [s for s in raw_smiles if "*" not in s and "|" not in s]
    
    print(f"🧠 Screening {len(clean_smiles)} novel scaffolds...")
    results = []
    target_labels = ['VKORC1', 'FXIIa', 'FXa', 'Thrombin', 'CYP2C9', 'HSA']
    
    for smiles in clean_smiles:
        # Phase 1: Publication Filters (SA Score & PAINS)
        props = run_filters(smiles)
        if props is None: continue # Drop if it fails strict lab criteria
            
        # Phase 2: Neural Network Scoring
        graph_data = smiles_to_inference_graph(smiles)
        if graph_data is None: continue
            
        graph_data = graph_data.to(device)
        graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=device)
        
        with torch.no_grad():
            preds = model(graph_data).cpu().numpy().flatten()
            
        record = {'SMILES': smiles, **props}
        for label, score in zip(target_labels, preds):
            record[f'Pred_{label}_pXC50'] = round(float(score), 3)
        results.append(record)

    # 4. Sort and Filter
    if not results:
        print("❌ No molecules passed the SA Score and PAINS filters.")
        exit(0)
        
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Pred_VKORC1_pXC50', ascending=False)
    
    top_50 = df_results.head(50)
    out_path = os.path.join("data", "branch2_novel_scaffolds.csv")
    top_50.to_csv(out_path, index=False)
    print(f"\n✅ Success! {len(top_50)} publication-ready novel scaffolds saved to {out_path}")
    
    # 5. Print YAML format for config_master.yaml
    print("\n" + "="*60)
    print(" COPY AND PASTE THIS INTO YOUR config_master.yaml UNDER 'ligands:'")
    print("="*60)
    for i, row in enumerate(top_50.head(20).iterrows()):
        _, data = row
        print(f"  Novel_Scaffold_{i+1:02d}:")
        print(f"    active: true")
        print(f"    smiles: {data['SMILES']}")
    print("="*60)