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
with open(PREDICTIONS_DIR / 'group_cross_service_prelabeled_only.json', 'r') as f:
    prelabeled_data = json.load(f)

with open(PREDICTIONS_DIR / 'group_cross_service_enhanced.json', 'r') as f:
    enhanced_data = json.load(f)

try:
    with open(PREDICTIONS_DIR / 'all_to_all_cross_service_predictions.json', 'r') as f:
        all_to_all_data = json.load(f)
    has_all_to_all = True
except FileNotFoundError:
    print("⚠️ All-to-all file not found, skipping that analysis")
    has_all_to_all = False

# ============================================================================
# Helper: extract predictions from grouped or flat data
# ============================================================================
def extract_grouped_predictions(data):
    predictions = []
    for group_name, group_data in data['predictions'].items():
        for service, methods in group_data.items():
            for method_name, pred_info in methods.items():
                predictions.append({
                    'group': group_name,
                    'service': service,
                    'method': method_name,
                    'label': pred_info['label'],
                    'confidence': pred_info['confidence'],
                    'iteration': pred_info['iteration']
                })
    return pd.DataFrame(predictions)


def extract_flat_predictions(data):
    predictions = []
    for service, methods in data['predictions'].items():
        for method_name, pred_info in methods.items():
            predictions.append({
                'service': service,
                'method': method_name,
                'label': pred_info['label'],
                'confidence': pred_info['confidence'],
                'method_type': pred_info.get('method', 'all_to_all')
            })
    return pd.DataFrame(predictions)


def calc_high_conf_pct(data, is_grouped=True):
    confidences = []
    if is_grouped:
        for group_name, group_data in data['predictions'].items():
            for service, methods in group_data.items():
                for method_name, pred_info in methods.items():
                    confidences.append(pred_info['confidence'])
    else:
        for service, methods in data['predictions'].items():
            for method_name, pred_info in methods.items():
                confidences.append(pred_info['confidence'])

    high_conf = sum(1 for c in confidences if c >= 0.90)
    return (high_conf / len(confidences)) * 100 if confidences else 0


# ============================================================================
# Build DataFrames
# ============================================================================
df_prelabeled = extract_grouped_predictions(prelabeled_data)
df_enhanced = extract_grouped_predictions(enhanced_data)

high_conf_prelabeled = df_prelabeled[df_prelabeled['confidence'] >= 0.90]
high_conf_enhanced = df_enhanced[df_enhanced['confidence'] >= 0.90]

if has_all_to_all:
    df_all = extract_flat_predictions(all_to_all_data)
    high_conf_all = df_all[df_all['confidence'] >= 0.90]

# ============================================================================
# CONSOLE STATISTICS
# ============================================================================
print("=" * 80)
print("CROSS-SERVICE LABEL PROPAGATION STATISTICS")
print("=" * 80)

# --- Group-based: Prelabeled Only ---
print("\n" + "=" * 80)
print("1. GROUP-BASED APPROACH (Prelabeled Data Only)")
print("=" * 80)

print(f"\nTotal predictions: {len(df_prelabeled)}")
print(f"Groups processed: {len(prelabeled_data['metadata']['groups_processed'])}")
print(f"\nGroup configuration:")
for group_name, config in prelabeled_data['metadata']['group_configuration'].items():
    print(f"  • {group_name}:")
    print(f"    - Core services: {', '.join(config['core_services'])}")
    print(f"    - Target services: {', '.join(config['target_services'])}")
    print(f"    - {config['description']}")

print(f"\nConfidence statistics:")
print(f"  High confidence (≥0.90): {len(high_conf_prelabeled)}/{len(df_prelabeled)} ({len(high_conf_prelabeled)/len(df_prelabeled)*100:.1f}%)")
print(f"  Mean confidence: {df_prelabeled['confidence'].mean():.4f}")
print(f"  Median confidence: {df_prelabeled['confidence'].median():.4f}")

print(f"\nPredictions by group:")
for group in df_prelabeled['group'].unique():
    group_df = df_prelabeled[df_prelabeled['group'] == group]
    high_conf = (group_df['confidence'] >= 0.90).sum()
    print(f"  • {group}: {len(group_df)} predictions ({high_conf}/{len(group_df)} = {high_conf/len(group_df)*100:.1f}% high conf)")

# --- Group-based: Enhanced ---
print("\n" + "=" * 80)
print("2. GROUP-BASED APPROACH (Enhanced with Within-Service Predictions)")
print("=" * 80)

print(f"\nTotal predictions: {len(df_enhanced)}")
print(f"Uses within-service predictions: {enhanced_data['metadata']['uses_within_service_predictions']}")

print(f"\nConfidence statistics:")
print(f"  High confidence (≥0.90): {len(high_conf_enhanced)}/{len(df_enhanced)} ({len(high_conf_enhanced)/len(df_enhanced)*100:.1f}%)")
print(f"  Mean confidence: {df_enhanced['confidence'].mean():.4f}")
print(f"  Median confidence: {df_enhanced['confidence'].median():.4f}")

print(f"\nPredictions by group:")
for group in df_enhanced['group'].unique():
    group_df = df_enhanced[df_enhanced['group'] == group]
    high_conf = (group_df['confidence'] >= 0.90).sum()
    print(f"  • {group}: {len(group_df)} predictions ({high_conf}/{len(group_df)} = {high_conf/len(group_df)*100:.1f}% high conf)")

# --- Comparison ---
print(f"\n" + "=" * 80)
print("COMPARISON: Prelabeled vs Enhanced")
print("=" * 80)
additional_predictions = len(df_enhanced) - len(df_prelabeled)
print(f"Additional predictions from using within-service data: {additional_predictions}")
print(f"Improvement in high-confidence percentage: {(len(high_conf_enhanced)/len(df_enhanced) - len(high_conf_prelabeled)/len(df_prelabeled))*100:.1f} percentage points")

# --- All-to-All ---
if has_all_to_all:
    print("\n" + "=" * 80)
    print("3. ALL-TO-ALL CROSS-SERVICE APPROACH")
    print("=" * 80)
    
    print(f"\nTotal predictions: {len(df_all)}")
    print(f"Services covered: {df_all['service'].nunique()}")
    
    print(f"\nConfidence statistics:")
    print(f"  High confidence (≥0.90): {len(high_conf_all)}/{len(df_all)} ({len(high_conf_all)/len(df_all)*100:.1f}%)")
    print(f"  Mean confidence: {df_all['confidence'].mean():.4f}")
    print(f"  Median confidence: {df_all['confidence'].median():.4f}")

# ============================================================================
# SUMMARY FOR THESIS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY FOR THESIS WRITING")
print("=" * 80)

print("\nKey Statistics to Include:")
print(f"  1. Group-based (prelabeled only): {len(df_prelabeled)} predictions, {len(high_conf_prelabeled)/len(df_prelabeled)*100:.1f}% high confidence")
print(f"  2. Group-based (enhanced): {len(df_enhanced)} predictions, {len(high_conf_enhanced)/len(df_enhanced)*100:.1f}% high confidence")
if has_all_to_all:
    print(f"  3. All-to-all: {len(df_all)} predictions, {len(high_conf_all)/len(df_all)*100:.1f}% high confidence")
print(f"\n  Number of service groups: {len(prelabeled_data['metadata']['groups_processed'])}")
print(f"  Target services covered across groups: {df_enhanced['service'].nunique()}")
print(f"\n  Adaptive thresholding parameters:")
print(f"    - Initial threshold: {prelabeled_data['metadata']['initial_threshold']}")
print(f"    - Minimum threshold: {prelabeled_data['metadata']['minimum_threshold']}")
print(f"    - k-neighbors: {prelabeled_data['metadata']['k_neighbors']}")

# ============================================================================
# TEXT REPLACEMENT VALUES
# ============================================================================
print("\n" + "=" * 80)
print("TEXT REPLACEMENT VALUES (Copy these into your subsection)")
print("=" * 80)

print("\n### GROUP-BASED APPROACH (Prelabeled Only) ###")
print(f"[PRELABELED_HIGH_CONF_PCT] = {len(high_conf_prelabeled)/len(df_prelabeled)*100:.1f}")
print(f"[PRELABELED_MEAN_CONF] = {df_prelabeled['confidence'].mean():.3f}")

# Best and worst performing groups
group_performance_prelabeled = []
for group in df_prelabeled['group'].unique():
    group_df = df_prelabeled[df_prelabeled['group'] == group]
    high_conf_pct = (group_df['confidence'] >= 0.90).sum() / len(group_df) * 100
    group_performance_prelabeled.append((group, high_conf_pct, len(group_df)))

group_performance_prelabeled.sort(key=lambda x: x[1], reverse=True)
best_group = group_performance_prelabeled[0]
worst_group = group_performance_prelabeled[-1]

print(f"\nExample group performance text:")
print(f"\"Storage services achieved {best_group[1]:.1f}% high-confidence predictions ({best_group[2]} total),")
print(f"while messaging services demonstrated {worst_group[1]:.1f}% high confidence ({worst_group[2]} total).\"")

print("\n### GROUP-BASED APPROACH (Enhanced) ###")
print(f"[ENHANCED_HIGH_CONF_PCT] = {len(high_conf_enhanced)/len(df_enhanced)*100:.1f}")
print(f"[ENHANCED_MEAN_CONF] = {df_enhanced['confidence'].mean():.3f}")

improvement = (len(high_conf_enhanced)/len(df_enhanced) - len(high_conf_prelabeled)/len(df_prelabeled)) * 100
print(f"[IMPROVEMENT] = a {improvement:.1f} percentage point improvement")

if has_all_to_all:
    print("\n### ALL-TO-ALL APPROACH ###")
    print(f"[ALL_TO_ALL_HIGH_CONF_PCT] = {len(high_conf_all)/len(df_all)*100:.1f}")
    print(f"[ALL_TO_ALL_MEAN_CONF] = {df_all['confidence'].mean():.3f}")
    
    comparison_to_enhanced = (len(high_conf_all)/len(df_all) - len(high_conf_enhanced)/len(df_enhanced)) * 100
    if comparison_to_enhanced > 0:
        print(f"\nComparison text: \"This represents a {comparison_to_enhanced:.1f} percentage point increase compared to the enhanced group-based approach.\"")
    else:
        print(f"\nComparison text: \"This represents a {abs(comparison_to_enhanced):.1f} percentage point decrease compared to the enhanced group-based approach.\"")

print("\n" + "=" * 80)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# ============================================================================
# FIGURE 1: Comparison of Three Approaches (Bar Chart)
# ============================================================================
if has_all_to_all:
    fig, ax = plt.subplots(figsize=(12, 7))

    approaches = ['Group-Based\n(Prelabeled)', 'Group-Based\n(Enhanced)', 'All-to-All']
    total_predictions = [
        prelabeled_data['metadata']['total_predictions'],
        enhanced_data['metadata']['total_predictions'],
        all_to_all_data['metadata']['total_predictions']
    ]

    high_conf_pcts = [
        calc_high_conf_pct(prelabeled_data, True),
        calc_high_conf_pct(enhanced_data, True),
        calc_high_conf_pct(all_to_all_data, False)
    ]

    x = np.arange(len(approaches))
    width = 0.35

    bars1 = ax.bar(x - width/2, total_predictions, width, label='Total Predictions',
                   color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, [tp * hcp / 100 for tp, hcp in zip(total_predictions, high_conf_pcts)], 
                   width, label='High Confidence (≥0.90)',
                   color='#2E8B57', edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Propagation Approach', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Predictions', fontsize=13, fontweight='bold')
    ax.set_title('Cross-Service Label Propagation: Approach Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add percentage labels
    for i, (x_pos, pct) in enumerate(zip(x, high_conf_pcts)):
        ax.text(x_pos, max(total_predictions) * 0.95, f'{pct:.1f}%',
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_cross_service_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/fig_cross_service_comparison.png")
    plt.close()

# ============================================================================
# FIGURE 2: Group-Based Predictions by Service Group (Prelabeled Only)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

group_stats_prelabeled = []
for group_name, group_data in prelabeled_data['predictions'].items():
    total_in_group = 0
    high_conf_in_group = 0
    similarities = []
    
    for service, methods in group_data.items():
        for method_name, pred_info in methods.items():
            total_in_group += 1
            if pred_info['confidence'] >= 0.90:
                high_conf_in_group += 1
            if 'similarity' in pred_info and pred_info['similarity'] is not None:
                similarities.append(pred_info['similarity'])
    
    group_stats_prelabeled.append({
        'group': group_name.replace('_', ' ').title(),
        'total': total_in_group,
        'high_conf': high_conf_in_group,
        'high_conf_pct': (high_conf_in_group / total_in_group * 100) if total_in_group > 0 else 0,
        'mean_similarity': np.mean(similarities) if similarities else 0
    })

stats_df_prelabeled = pd.DataFrame(group_stats_prelabeled).sort_values('high_conf_pct', ascending=True)

bars = ax.barh(stats_df_prelabeled['group'], stats_df_prelabeled['high_conf_pct'], 
               color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)

for i, (idx, row) in enumerate(stats_df_prelabeled.iterrows()):
    if row['high_conf_pct'] >= 80:
        bars[i].set_color('#2E8B57')
    elif row['high_conf_pct'] >= 60:
        bars[i].set_color('#4682B4')
    else:
        bars[i].set_color('#CD853F')

ax.axvline(90, color='red', linestyle='--', linewidth=2, label='90% threshold', alpha=0.7)
ax.set_xlabel('% of Predictions with Confidence ≥ 0.90', fontsize=13, fontweight='bold')
ax.set_ylabel('Service Group', fontsize=13, fontweight='bold')
ax.set_title('Cross-Service High Confidence by Service Group (Prelabeled Only)', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', frameon=True)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, 105)

for i, (idx, row) in enumerate(stats_df_prelabeled.iterrows()):
    label_text = f"{row['high_conf_pct']:.1f}%\n({row['high_conf']}/{row['total']})\nSim: {row['mean_similarity']:.3f}"
    ax.text(row['high_conf_pct'] + 2, i, label_text,
            ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_cross_service_by_group_prelabeled.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/fig_cross_service_by_group_prelabeled.png")
plt.close()

# ============================================================================
# FIGURE 3: Group-Based Predictions by Service Group (Enhanced)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

group_stats = []
for group_name, group_data in enhanced_data['predictions'].items():
    total_in_group = 0
    high_conf_in_group = 0
    similarities = []
    
    for service, methods in group_data.items():
        for method_name, pred_info in methods.items():
            total_in_group += 1
            if pred_info['confidence'] >= 0.90:
                high_conf_in_group += 1
            if 'similarity' in pred_info and pred_info['similarity'] is not None:
                similarities.append(pred_info['similarity'])
    
    group_stats.append({
        'group': group_name.replace('_', ' ').title(),
        'total': total_in_group,
        'high_conf': high_conf_in_group,
        'high_conf_pct': (high_conf_in_group / total_in_group * 100) if total_in_group > 0 else 0,
        'mean_similarity': np.mean(similarities) if similarities else 0
    })

stats_df = pd.DataFrame(group_stats).sort_values('high_conf_pct', ascending=True)

bars = ax.barh(stats_df['group'], stats_df['high_conf_pct'], 
               color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)

for i, (idx, row) in enumerate(stats_df.iterrows()):
    if row['high_conf_pct'] >= 80:
        bars[i].set_color('#2E8B57')
    elif row['high_conf_pct'] >= 60:
        bars[i].set_color('#4682B4')
    else:
        bars[i].set_color('#CD853F')

ax.axvline(90, color='red', linestyle='--', linewidth=2, label='90% threshold', alpha=0.7)
ax.set_xlabel('% of Predictions with Confidence ≥ 0.90', fontsize=13, fontweight='bold')
ax.set_ylabel('Service Group', fontsize=13, fontweight='bold')
ax.set_title('Cross-Service High Confidence by Service Group (Enhanced)', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', frameon=True)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, 105)

for i, (idx, row) in enumerate(stats_df.iterrows()):
    label_text = f"{row['high_conf_pct']:.1f}%\n({row['high_conf']}/{row['total']})\nSim: {row['mean_similarity']:.3f}"
    ax.text(row['high_conf_pct'] + 2, i, label_text,
            ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_cross_service_by_group_enhanced.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/fig_cross_service_by_group_enhanced.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated figures:")
print("  1. figures/fig_cross_service_comparison.png - Comparison of three approaches")
print("  2. figures/fig_cross_service_by_group_prelabeled.png - Performance by service group (Prelabeled Only)")
print("  3. figures/fig_cross_service_by_group_enhanced.png - Performance by service group (Enhanced)")
print("\nRecommendation: Use Figure 1 (comparison) and either Figure 2 or 3 (or both if comparing prelabeled vs enhanced).")
