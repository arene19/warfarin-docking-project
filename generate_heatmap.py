import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_best_columns(df):
    # Find the column that looks like the Ligand name
    lig_col = [c for c in df.columns if 'Ligand' in c or 'Name' in c][0]
    # Find the column that looks like the Affinity
    aff_col = [c for c in df.columns if 'Affinity' in c or 'dG' in c][0]
    return lig_col, aff_col

# Load data
vkorc1 = pd.read_csv('results/docked_poses/VKORC1/VKORC1_screening_results.csv')
hsa = pd.read_csv('results/docked_poses/HSA/HSA_screening_results.csv')

v_lig, v_aff = get_best_columns(vkorc1)
h_lig, h_aff = get_best_columns(hsa)

# Combine using the detected columns
df = pd.DataFrame({
    'VKORC1': vkorc1.set_index(v_lig)[v_aff],
    'HSA': hsa.set_index(h_lig)[h_aff]
})

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, cmap='YlGnBu', fmt=".2f", cbar_kws={'label': 'kcal/mol'})
plt.title('Warfarin Derivatives: Target vs. Transport Affinity')
plt.tight_layout()
plt.savefig('figures/multi_target_heatmap.png')
print("[OK] Heatmap successfully saved to figures/multi_target_heatmap.png")
