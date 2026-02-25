import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ============================================================================
# Path setup
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PREDICTIONS_DIR = PROJECT_ROOT / 'notebooks' / 'label_propagation' / 'predictions'
FIGURES_DIR = SCRIPT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================================================
# Load data
# ============================================================================
with open(PREDICTIONS_DIR / 'within_service_predictions.json', 'r') as f:
    data = json.load(f)

# Extract metadata
total_manual_labels = data['metadata']['total_labeled_methods']

# Extract all predictions
all_predictions = []
for service, methods in data['predictions'].items():
    for method_name, pred_info in methods.items():
        all_predictions.append({
            'service': service,
            'method': method_name,
            'label': pred_info['label'],
            'confidence': pred_info['confidence'],
            'iteration': pred_info['iteration'],
            'similarity': pred_info.get('similarity', None),
            'neighbors': pred_info.get('neighbors', [])
        })

df = pd.DataFrame(all_predictions)

# The predictions file contains ONLY propagated predictions
# Manual labels are NOT in this file - they're the source, not predictions
# So all rows in df are propagated predictions
propagated = df
manual_labels_count = total_manual_labels

# ============================================================================
# Console statistics
# ============================================================================
print("=" * 70)
print("PREDICTION STATISTICS")
print("=" * 70)
print(f"\nManual labels (not in predictions file): {manual_labels_count}")
print(f"Total propagated predictions in file: {len(propagated)}")
print(f"Expected from metadata: {data['metadata']['total_predictions']}")

# Confidence analysis - PROPAGATED ONLY
high_conf_prop = propagated[propagated['confidence'] >= 0.90]
print(f"\n--- Propagated Predictions Only (Total: {len(propagated)}) ---")
print(f"Propagated with confidence >= 0.90: {len(high_conf_prop)} ({len(high_conf_prop)/len(propagated)*100:.1f}%)")
print(f"Mean confidence (propagated): {propagated['confidence'].mean():.4f}")
print(f"Median confidence (propagated): {propagated['confidence'].median():.4f}")

# Total including manual labels
total_with_manual = len(propagated) + manual_labels_count
# Assume manual labels have 100% confidence
high_conf_total = len(high_conf_prop) + manual_labels_count
print(f"\n--- Including Manual Labels (Total: {total_with_manual}) ---")
print(f"Total with confidence >= 0.90: {high_conf_total} ({high_conf_total/total_with_manual*100:.1f}%)")

# Breakdown by service
print("\n" + "=" * 70)
print("BY SERVICE ANALYSIS")
print("=" * 70)

for service in df['service'].unique():
    service_prop = propagated[propagated['service'] == service]
    
    print(f"\n{service.upper()}:")
    print(f"  Propagated predictions: {len(service_prop)}")
    if len(service_prop) > 0:
        high_conf = service_prop[service_prop['confidence'] >= 0.90]
        print(f"  Propagated with conf >= 0.90: {len(high_conf)}/{len(service_prop)} ({len(high_conf)/len(service_prop)*100:.1f}%)")
        print(f"  Mean confidence: {service_prop['confidence'].mean():.4f}")

# ============================================================================
# Plot style
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# ============================================================================
# FIGURE 1: Label Distribution by Service (Stacked Bar Chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Calculate label counts by service
label_counts = df.groupby(['service', 'label']).size().unstack(fill_value=0)
label_counts = label_counts.reindex(columns=['none', 'sink', 'source'], fill_value=0)

# Create stacked bar chart
label_counts.plot(kind='bar', stacked=True, ax=ax, 
                  color=['#6495ED', '#FF8C00', '#32CD32'],  # blue, orange, green
                  edgecolor='black', linewidth=0.5)

ax.set_xlabel('Services', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Methods', fontsize=13, fontweight='bold')
ax.set_title('Within-Service - Label Distribution', fontsize=15, fontweight='bold')
ax.legend(title='Label', labels=['none', 'sink', 'source'], loc='upper right', frameon=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Add percentage labels on dominant segments
for i, service in enumerate(label_counts.index):
    total = label_counts.loc[service].sum()
    cumsum = 0
    for label in ['none', 'sink', 'source']:
        count = label_counts.loc[service, label]
        if count > 0:
            percentage = (count / total) * 100
            if percentage > 15:  # Only show percentages > 15%
                y_pos = cumsum + count / 2
                ax.text(i, y_pos, f'{percentage:.1f}%', 
                       ha='center', va='center', fontweight='bold', fontsize=10)
            cumsum += count

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig1_label_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/fig1_label_distribution.png")
plt.close()

# ============================================================================
# FIGURE 2: High Confidence Percentage by Service
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

service_stats = []
for service in df['service'].unique():
    service_prop = propagated[propagated['service'] == service]
    
    if len(service_prop) > 0:
        high_conf_pct = (service_prop['confidence'] >= 0.90).sum() / len(service_prop) * 100
        similarities = service_prop['similarity'].dropna()
        mean_similarity = similarities.mean() if len(similarities) > 0 else 0
        
        service_stats.append({
            'service': service.upper(),
            'high_conf_pct': high_conf_pct,
            'total_propagated': len(service_prop),
            'high_conf_count': (service_prop['confidence'] >= 0.90).sum(),
            'mean_similarity': mean_similarity
        })

stats_df = pd.DataFrame(service_stats).sort_values('high_conf_pct', ascending=True)

# Create horizontal bar chart
bars = ax.barh(stats_df['service'], stats_df['high_conf_pct'], 
               color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)

# Color code bars by performance tier
for i, (idx, row) in enumerate(stats_df.iterrows()):
    if row['high_conf_pct'] >= 80:
        bars[i].set_color('#2E8B57')  # Green for high performance
    elif row['high_conf_pct'] >= 60:
        bars[i].set_color('#4682B4')  # Blue for medium
    else:
        bars[i].set_color('#CD853F')  # Brown for low performance

ax.axvline(90, color='red', linestyle='--', linewidth=2, label='90% threshold', alpha=0.7)
ax.set_xlabel('% of Propagated Predictions with Confidence ≥ 0.90', fontsize=13, fontweight='bold')
ax.set_ylabel('Service', fontsize=13, fontweight='bold')
ax.set_title('High Confidence Percentage by Service (Propagated Only)', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', frameon=True)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, 105)

# Add percentage and similarity labels on bars
for i, (idx, row) in enumerate(stats_df.iterrows()):
    label_text = f"{row['high_conf_pct']:.1f}%\n({row['high_conf_count']}/{row['total_propagated']})\nSim: {row['mean_similarity']:.3f}"
    ax.text(row['high_conf_pct'] + 2, i, label_text,
            ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig2_high_confidence_percentage.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/fig2_high_confidence_percentage.png")
plt.close()

# ============================================================================
# FIGURE 3: Confidence Distribution by Service (Box Plots)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

service_order = df.groupby('service')['confidence'].median().sort_values(ascending=False).index
service_order = [s.upper() for s in service_order]

# Prepare data with uppercase service names
df_plot = df.copy()
df_plot['service'] = df_plot['service'].str.upper()

box_plot = sns.boxplot(data=df_plot, y='service', x='confidence', 
                       order=service_order, ax=ax, 
                       palette='Set2', linewidth=1.5)

ax.axvline(0.90, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='0.90 threshold')
ax.set_xlabel('Confidence Score', fontsize=13, fontweight='bold')
ax.set_ylabel('Service', fontsize=13, fontweight='bold')
ax.set_title('Confidence Distribution by Service (Propagated Only)', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(loc='lower left', frameon=True)
ax.set_xlim(0.45, 1.02)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig3_confidence_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/fig3_confidence_boxplots.png")
plt.close()

# ============================================================================
# FIGURE 4: Predictions and Confidence by Iteration
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

iteration_conf = df.groupby('iteration').agg({
    'confidence': ['mean', 'count']
}).reset_index()
iteration_conf.columns = ['iteration', 'mean_confidence', 'count']

ax_twin = ax.twinx()

# Bar chart for count
bars = ax.bar(iteration_conf['iteration'], iteration_conf['count'], 
              alpha=0.7, color='lightblue', edgecolor='black', 
              linewidth=1.5, label='Number of Predictions', width=0.8)

# Line chart for mean confidence
line = ax_twin.plot(iteration_conf['iteration'], iteration_conf['mean_confidence'], 
                    color='red', marker='o', linewidth=3, markersize=10, 
                    label='Mean Confidence', markeredgecolor='darkred', markeredgewidth=1.5)

ax_twin.axhline(0.90, color='green', linestyle='--', linewidth=2.5, 
                alpha=0.7, label='0.90 threshold')

ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Predictions', fontsize=13, fontweight='bold', color='steelblue')
ax_twin.set_ylabel('Mean Confidence', fontsize=13, fontweight='bold', color='red')
ax.set_title('Predictions and Confidence by Iteration', fontsize=15, fontweight='bold')
ax.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
ax_twin.tick_params(axis='y', labelcolor='red', labelsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(iteration_conf['count']) * 1.15)
ax_twin.set_ylim(0.80, 1.02)

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig4_iteration_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/fig4_iteration_analysis.png")
plt.close()

# ============================================================================
# FIGURE 5: Overall Confidence Distribution (Histogram)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Histogram for propagated predictions
ax.hist(propagated['confidence'], bins=50, alpha=0.7, color='orange',
        label=f'Propagated predictions ({len(propagated)})', edgecolor='black', linewidth=0.5)

# Add reference lines
ax.axvline(0.90, color='red', linestyle='--', linewidth=2.5, 
           label='0.90 threshold', alpha=0.8)
ax.axvline(propagated['confidence'].mean(), color='green', linestyle='--', 
           linewidth=2.5, label=f'Mean: {propagated["confidence"].mean():.3f}', alpha=0.8)
ax.axvline(propagated['confidence'].median(), color='purple', linestyle='--', 
           linewidth=2.5, label=f'Median: {propagated["confidence"].median():.3f}', alpha=0.8)

ax.set_xlabel('Confidence Score', fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax.set_title('Confidence Distribution (Propagated Predictions)', fontsize=15, fontweight='bold')
ax.legend(loc='upper left', frameon=True)
ax.grid(True, alpha=0.3, axis='y')

# Add text box with statistics
textstr = f'Manual labels: {manual_labels_count}\n'
textstr += f'Propagated: {len(propagated)}\n'
textstr += f'Total labeled: {total_with_manual}\n\n'
textstr += f'Propagated high conf (≥0.90): {len(high_conf_prop)} ({len(high_conf_prop)/len(propagated)*100:.1f}%)\n'
textstr += f'Total high conf: {high_conf_total} ({high_conf_total/total_with_manual*100:.1f}%)'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.60, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig5_confidence_histogram.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/fig5_confidence_histogram.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("CORRECTED STATEMENT")
print("="*70)
print(f"\nOf the {len(propagated)} propagated predictions (from {manual_labels_count}")
print(f"manual labels), {len(high_conf_prop)} ({len(high_conf_prop)/len(propagated)*100:.1f}%) exceeded the 0.90 confidence")
print(f"threshold. Including manual labels, {high_conf_total} of {total_with_manual} total")
print(f"predictions ({high_conf_total/total_with_manual*100:.1f}%) achieved high confidence.")

print("\n" + "="*70)
print("FIGURE GENERATION COMPLETE")
print("="*70)
print("\nGenerated figures:")
print("  1. figures/fig1_label_distribution.png - Stacked bar chart of label types")
print("  2. figures/fig2_high_confidence_percentage.png - High confidence % by service (with similarity)")
print("  3. figures/fig3_confidence_boxplots.png - Confidence distribution box plots")
print("  4. figures/fig4_iteration_analysis.png - Predictions and confidence by iteration")
print("  5. figures/fig5_confidence_histogram.png - Overall confidence histogram")
