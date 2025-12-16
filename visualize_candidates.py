#!/usr/bin/env python3
"""
Visualize the discovered superconductor candidates.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load results
results_file = Path("results/new_materials_predictions.csv")
if not results_file.exists():
    print("❌ Results file not found! Please run screen_new_materials.py first.")
    exit(1)

df = pd.read_csv(results_file)

print(f"Loaded {len(df)} candidates")
print(f"Tc range: {df['predicted_tc'].min():.2f}K - {df['predicted_tc'].max():.2f}K")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('New Superconductor Candidates - Analysis', fontsize=16, fontweight='bold')

# 1. Top 15 candidates bar chart
ax1 = axes[0, 0]
top_15 = df.nlargest(15, 'predicted_tc')
colors = plt.cm.viridis(top_15['predicted_tc'].values / top_15['predicted_tc'].max())
bars = ax1.barh(range(len(top_15)), top_15['predicted_tc'].values, color=colors)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels([f"{row['formula']} ({row['material_id']})" 
                      for _, row in top_15.iterrows()], fontsize=9)
ax1.set_xlabel('Predicted Tc (K)', fontsize=11)
ax1.set_title('Top 15 Candidates by Predicted Tc', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_15.iterrows()):
    ax1.text(row['predicted_tc'] + 0.3, i, f"{row['predicted_tc']:.2f}K", 
             va='center', fontsize=8)

# 2. Tc distribution histogram
ax2 = axes[0, 1]
ax2.hist(df['predicted_tc'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(df['predicted_tc'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["predicted_tc"].mean():.2f}K', linewidth=2)
ax2.axvline(df['predicted_tc'].median(), color='green', linestyle='--', 
            label=f'Median: {df["predicted_tc"].median():.2f}K', linewidth=2)
ax2.set_xlabel('Predicted Tc (K)', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Tc Distribution for All Candidates', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# 3. Scatter: Formation Energy vs Tc
ax3 = axes[1, 0]
scatter = ax3.scatter(df['formation_energy_per_atom'], df['predicted_tc'], 
                     c=df['density'], cmap='plasma', s=100, alpha=0.6, 
                     edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Formation Energy per Atom (eV/atom)', fontsize=11)
ax3.set_ylabel('Predicted Tc (K)', fontsize=11)
ax3.set_title('Stability vs Superconductivity\n(Color = Density)', 
              fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Density (g/cm³)', fontsize=10)

# Highlight top candidates
top_5 = df.nlargest(5, 'predicted_tc')
for _, row in top_5.iterrows():
    ax3.annotate(row['formula'], 
                (row['formation_energy_per_atom'], row['predicted_tc']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='yellow', alpha=0.7))

# 4. Elements analysis
ax4 = axes[1, 1]
element_counts = {}
for formula in df['formula']:
    # Simple element extraction (counts all capital letters as elements)
    import re
    elements = re.findall(r'[A-Z][a-z]?', formula)
    for elem in set(elements):
        element_counts[elem] = element_counts.get(elem, 0) + 1

# Get top 10 elements
top_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:10]
elements, counts = zip(*top_elements)

bars = ax4.bar(range(len(elements)), counts, color='coral', edgecolor='black', alpha=0.7)
ax4.set_xticks(range(len(elements)))
ax4.set_xticklabels(elements, fontsize=11, fontweight='bold')
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('Top 10 Elements in Candidates', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, (elem, count) in enumerate(top_elements):
    ax4.text(i, count + 0.3, str(count), ha='center', fontsize=10, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save figure
output_file = Path("results/candidates_analysis.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved to {output_file}")

# Show plot
plt.show()

print("\n📊 Analysis Complete!")
print(f"   Total candidates: {len(df)}")
print(f"   Tc > 10K: {len(df[df['predicted_tc'] > 10])}")
print(f"   Top candidate: {top_5.iloc[0]['formula']} ({top_5.iloc[0]['predicted_tc']:.2f}K)")


