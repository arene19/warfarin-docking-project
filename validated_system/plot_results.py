import pandas as pd
import matplotlib.pyplot as plt

# Load your results
df = pd.read_csv('screening_results.csv')

# 1. CLEANING: Remove '.pdbqt' from ligand names for a cleaner plot
df['Ligand'] = df['Ligand'].str.replace('.pdbqt', '', regex=False)

# Add the Warfarin benchmark manually if it's not in your CSV
if 'Warfarin_S' not in df['Ligand'].values:
    new_row = pd.DataFrame({'Ligand': ['Warfarin_S'], 'Affinity(kcal/mol)': [-11.07]})
    df = pd.concat([df, new_row], ignore_index=True)

# Sort by affinity (best binders at the top)
df = df.sort_values(by='Affinity(kcal/mol)', ascending=True)

# Create the plot
plt.figure(figsize=(10, 6))

# Define colors: Gold for the top lead, Gray for Warfarin, Blue for others
colors = []
for name in df['Ligand']:
    if 'Warfarin' in name:
        colors.append('#808080') # Gray
    elif name == df['Ligand'].iloc[0]: # The top performer
        colors.append('#d4af37') # Gold
    else:
        colors.append('#3498db') # Sky Blue

bars = plt.barh(df['Ligand'], df['Affinity(kcal/mol)'], color=colors)

# Add a vertical line for the Warfarin threshold
plt.axvline(x=-11.07, color='red', linestyle='--', alpha=0.6, label='Warfarin Threshold (-11.07)')

# 2. CLEANING: Simplified Axis Labels
plt.title('VKOR Inhibitor Screening: Binding Affinities', fontsize=14, fontweight='bold')
plt.xlabel('Affinity (kcal/mol)', fontsize=12)
plt.gca().invert_yaxis()  # Best results on top
plt.legend(loc='lower right')
plt.grid(axis='x', linestyle=':', alpha=0.7)

# Add the exact values at the end of each bar
for bar in bars:
    width = bar.get_width()
    plt.text(width - 0.1, bar.get_y() + bar.get_height()/2, 
             f'{width:.2f}', va='center', ha='right', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('binding_affinity_plot_clean.png', dpi=300)
print("Clean plot saved as binding_affinity_plot_clean.png")
