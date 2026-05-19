import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# ==========================================
# 1. NEW BALANCED DATA ENTRY (8 LIGANDS)
# ==========================================
docking_data = {
    'Ligand': ['1a', '1b', '2a', '2b', '3a', '3b', 'Ref_R', 'Ref_S'],
    'Compound': ['1 (Base)', '1 (Base)', '2 (m-NO2)', '2 (m-NO2)', '3 (m-Br)', '3 (m-Br)', 'Warfarin (Ref)', 'Warfarin (Ref)'],
    'Enantiomer': ['R (a)', 'S (b)', 'R (a)', 'S (b)', 'R (a)', 'S (b)', 'R (a)', 'S (b)'],
    'VKORC1 (Primary)': [-10.54, -11.43, -11.52, -11.12, -11.18, -11.54, -10.82, -11.63],
    'HSA (Transport)': [-7.86, -9.12, -7.96, -9.02, -8.02, -8.94, -8.49, -8.75]
}
df_docking = pd.DataFrame(docking_data)

admet_data = {
    'Compound': ['1 (Base)', '2 (m-NO2)', '3 (m-Br)', 'Warfarin (Ref)'],
    'MW (g/mol)': [308.0, 353.0, 387.0, 307.0],
    'LogP': [2.09, 2.00, 2.85, 2.98],
    'QED (Drug-likeness)': [0.751, 0.434, 0.701, 0.695],
    'Lipinski Violations': [0, 0, 0, 0]
}
df_admet = pd.DataFrame(admet_data)

# ==========================================
# 2. PLOT 1: DUAL-TARGET HEATMAP
# ==========================================
def plot_heatmap(df):
    plt.figure(figsize=(9, 6))
    
    # Map friendly labels for heatmap rows
    label_map = {
        '1a': '1a (Base R)', '1b': '1b (Base S)',
        '2a': '2a (m-NO2 R)', '2b': '2b (m-NO2 S)',
        '3a': '3a (m-Br R)', '3b': '3b (m-Br S)',
        'Ref_R': 'Warfarin Reference (R)', 'Ref_S': 'Warfarin Reference (S)'
    }
    df_hm = df.copy()
    df_hm['Ligand'] = df_hm['Ligand'].map(label_map)
    df_hm = df_hm.set_index('Ligand')[['VKORC1 (Primary)', 'HSA (Transport)']]
    
    ax = sns.heatmap(df_hm, annot=True, fmt=".2f", cmap="YlGnBu_r", 
                     cbar_kws={'label': 'Binding Affinity (kcal/mol)'},
                     linewidths=1, linecolor='white')
    
    plt.title('Multi-Target Binding Affinity Profile', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('Compound Structural Variant', fontweight='bold')
    plt.xlabel('Target Protein', fontweight='bold')
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    plt.tight_layout()
    plt.savefig('figures/Publication_Heatmap.png', dpi=300)
    plt.close()

# ==========================================
# 3. PLOT 2: STEREOSELECTIVITY BAR CHART
# ==========================================
def plot_stereoselectivity(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.suptitle('Stereoselective Binding Dynamics: S vs. R Configuration', fontsize=18, fontweight='bold', y=1.02)

    targets = ['VKORC1 (Primary)', 'HSA (Transport)']
    colors = {'R (a)': '#e74c3c', 'S (b)': '#2ecc71'}

    for i, target in enumerate(targets):
        sns.barplot(data=df, x='Compound', y=target, hue='Enantiomer', 
                    palette=colors, ax=axes[i], edgecolor='black')
        
        axes[i].set_title(target, fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Binding Affinity (kcal/mol)' if i == 0 else '')
        axes[i].set_xlabel('Structural Group', fontweight='bold')
        axes[i].invert_yaxis() 
        
        if axes[i].get_legend() is not None:
            axes[i].get_legend().remove()
        
        for p in axes[i].patches:
            val = p.get_height()
            if val == 0 or np.isnan(val): 
                continue
            axes[i].annotate(format(val, '.2f'), 
                             (p.get_x() + p.get_width() / 2., val), 
                             ha='center', va='bottom', 
                             xytext=(0, 5), textcoords='offset points',
                             fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Stereocenter Configuration', loc='lower center', 
               bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=13, title_fontsize=14)

    plt.tight_layout()
    plt.savefig('figures/Publication_Stereoselectivity.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 4. PLOT 3: ADMET SUMMARY GRAPHIC
# ==========================================
def plot_admet_table(df):
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0) 
    
    for i in range(len(df.columns)):
        cell = table[0, i]
        cell.set_facecolor('#34495e')
        cell.set_text_props(color='white', weight='bold')

    plt.title('ADMET & Physicochemical Property Comparison', fontweight='bold', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('figures/Publication_ADMET_Table.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    print("Generating comprehensive 8-compound publication-quality figures...")
    plot_heatmap(df_docking)
    plot_stereoselectivity(df_docking)
    plot_admet_table(df_admet)
    print("All figures successfully synchronized with new pipeline results!")
