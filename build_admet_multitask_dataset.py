#build_admet_multitask_dataset.py
"""Build multi-task CSV from ChEMBL (optional — publication uses frozen data/coagulation_admet_multi_task.csv)."""
import os
import math
import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings to keep the console output clean
RDLogger.DisableLog('rdApp.*')

def fetch_chembl_data(targets: dict) -> pd.DataFrame:
    """
    Scavenges ChEMBL for IC50 and Ki bioactivity data for specified targets.
    """
    activity_client = new_client.activity
    records = []

    print("Starting ChEMBL Data Scavenging...")
    for target_name, chembl_id in targets.items():
        print(f" -> Querying {target_name} ({chembl_id})...")
        target_count = 0
        
        # Loop through both standard types to maximize data retrieval
        for std_type in ['IC50', 'Ki']:
            try:
                query = activity_client.filter(
                    target_chembl_id=chembl_id,
                    standard_type=std_type,
                    standard_units="nM"
                )
                
                type_count = 0
                for act in query:
                    smiles = act.get('canonical_smiles')
                    val = act.get('standard_value')
                    
                    if smiles and val is not None:
                        try:
                            val_float = float(val)
                            if val_float > 0:  # Value must be strictly positive for log conversion
                                records.append({
                                    'Target': target_name,
                                    'Raw_SMILES': smiles,
                                    'Value_nM': val_float,
                                    'Type': std_type
                                })
                                type_count += 1
                                target_count += 1
                        except ValueError:
                            continue
                            
                print(f"    [+] Retrieved {type_count} valid {std_type} records.")
                
            except Exception as e:
                print(f"    [!] Error fetching {std_type} data for {target_name}: {str(e)}")
                
        print(f"    [=] Total records for {target_name}: {target_count}\n")

    return pd.DataFrame(records)

def process_and_audit_chemicals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes SMILES, calculates pXC50, and flags Coumarin substructures.
    """
    print("Starting Chemical Auditing and Math Scaling...")
    
    if df.empty:
        return df
        
    # 1. Math Scaling: Compute pXC50
    # pXC50 = -log10(Value * 10^-9)
    df['pXC50'] = df['Value_nM'].apply(lambda x: -math.log10(x * 1e-9))
    
    # Define the Coumarin core scaffold
    coumarin_smiles = "O=C1C=Cc2ccccc2O1"
    coumarin_pattern = Chem.MolFromSmiles(coumarin_smiles)
    
    processed_data = []
    
    # 2. Chemical Auditing: RDKit processing
    unique_smiles = df['Raw_SMILES'].unique()
    smiles_mapping = {}
    
    print(f" -> Auditing {len(unique_smiles)} unique chemical structures...")
    
    for raw_smiles in unique_smiles:
        try:
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
                canonical_smiles = Chem.MolToSmiles(mol)
                is_coumarin = mol.HasSubstructMatch(coumarin_pattern)
                smiles_mapping[raw_smiles] = {
                    'canonical_smiles': canonical_smiles,
                    'is_coumarin': is_coumarin
                }
            else:
                smiles_mapping[raw_smiles] = None
        except Exception:
            # Gracefully handle any RDKit crashes from corrupted database entries
            smiles_mapping[raw_smiles] = None

    # Map the audited data back to the main dataframe
    valid_rows = []
    for _, row in df.iterrows():
        audit_result = smiles_mapping.get(row['Raw_SMILES'])
        if audit_result is not None:
            valid_rows.append({
                'Target': row['Target'],
                'canonical_smiles': audit_result['canonical_smiles'],
                'is_coumarin': audit_result['is_coumarin'],
                'pXC50': row['pXC50']
            })
            
    cleaned_df = pd.DataFrame(valid_rows)
    print(f"    [+] Retained {len(cleaned_df)} records after chemical sanitization.")
    return cleaned_df

def aggregate_pivot_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates duplicate assays and pivots into a multi-task matrix.
    """
    print("\nStarting Pivot Matrix Aggregation...")
    
    if df.empty:
        return df
        
    # Aggregate duplicate SMILES-Target pairs by taking the mean pXC50
    grouped_df = df.groupby(['canonical_smiles', 'is_coumarin', 'Target'])['pXC50'].mean().reset_index()
    
    # Pivot the table: one row per SMILES, columns for each Target's pXC50
    pivot_df = grouped_df.pivot(
        index=['canonical_smiles', 'is_coumarin'], 
        columns='Target', 
        values='pXC50'
    ).reset_index()
    
    # Clean up column names
    pivot_df.columns.name = None
    
    # Rename target columns to explicitly state they are pXC50 values
    rename_dict = {col: f"{col}_pXC50" for col in pivot_df.columns if col not in ['canonical_smiles', 'is_coumarin']}
    pivot_df.rename(columns=rename_dict, inplace=True)
    
    print(f"    [+] Final multi-task matrix shape: {pivot_df.shape[0]} molecules x {pivot_df.shape[1]} features/targets.")
    return pivot_df

if __name__ == "__main__":
    print("========================================================")
    print(" ADMET MULTI-TASK DATASET COMPILER FOR COAGULATION GNN  ")
    print("========================================================")
    
    # 1. Target Compilation (Efficacy + ADMET)
    TARGETS = {
        'VKORC1': 'CHEMBL1930',
        'Factor_XIIa': 'CHEMBL2821',
        'Factor_Xa': 'CHEMBL244',
        'Thrombin': 'CHEMBL204',
        'CYP2C9': 'CHEMBL3397',
        'HSA': 'CHEMBL3253'
    }
    
    # 2. Data Scavenging
    raw_df = fetch_chembl_data(TARGETS)
    
    if raw_df.empty:
        print("Error: No data retrieved from ChEMBL. Exiting.")
        exit(1)
        
    # 3 & 4. Math Scaling and Structure Auditing
    cleaned_df = process_and_audit_chemicals(raw_df)
    
    # 5. Pivot Matrix Aggregation
    final_multitask_df = aggregate_pivot_matrix(cleaned_df)
    
    # 6. Output
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "coagulation_admet_multi_task.csv")
    
    final_multitask_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] ADMET Multi-task dataset saved to: {output_path}")
    print("========================================================")