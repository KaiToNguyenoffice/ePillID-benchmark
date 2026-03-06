import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

metrics_csv = os.path.join(os.path.dirname(__file__), 'outputs', 'metrics.csv')
agg_csv = os.path.join(os.path.dirname(__file__), 'outputs', 'agg_metrics.csv')
output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'plots')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(metrics_csv)
agg = pd.read_csv(agg_csv)

df['fold'] = df['val_imgs_csv'].str.extract(r'_(\d+)\.csv$').astype(int)

# ── Figure 1: Val vs Holdout – Key Accuracy Metrics (per fold) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Validation vs Holdout — Key Metrics per Fold', fontsize=16, fontweight='bold')

key_metrics = ['top1-acc', 'top5-acc', 'map', 'micro-ap']
titles = ['Top-1 Accuracy', 'Top-5 Accuracy', 'Mean Average Precision (MAP)', 'Micro Average Precision']

for ax, metric, title in zip(axes.flat, key_metrics, titles):
    val_data = df[(df['dataset'] == 'val') & (df['name'] == metric)].sort_values('fold')
    holdout_data = df[(df['dataset'] == 'holdout') & (df['name'] == metric)].sort_values('fold')

    x = np.arange(len(val_data))
    width = 0.35
    ax.bar(x - width / 2, val_data['value'].values, width, label='Validation', color='#2196F3', alpha=0.85)
    ax.bar(x + width / 2, holdout_data['value'].values, width, label='Holdout', color='#FF9800', alpha=0.85)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in val_data['fold'].values])
    ax.legend()
    ax.set_ylim(0, min(1.0, max(val_data['value'].max(), holdout_data['value'].max()) * 1.25))

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(output_dir, '1_val_vs_holdout_per_fold.png'), dpi=150, bbox_inches='tight')
print(f"Saved: 1_val_vs_holdout_per_fold.png")

# ── Figure 2: Aggregated Metrics (mean ± std) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Aggregated Metrics (Mean ± Std across Folds)', fontsize=16, fontweight='bold')

for ax, dataset, color, title in zip(
    axes, ['val', 'holdout'], ['#2196F3', '#FF9800'],
    ['Validation Set', 'Holdout Set']
):
    subset = agg[agg['dataset'] == dataset]
    show_metrics = ['top1-acc', 'top5-acc', 'map', 'map_at_1', 'gap_at_1', 'micro-ap']
    labels =       ['Top-1\nAcc', 'Top-5\nAcc', 'MAP', 'MAP@1', 'GAP@1', 'Micro-AP']
    means, stds = [], []
    for m in show_metrics:
        row = subset[subset['name'] == m]
        means.append(row['mean'].values[0] if len(row) > 0 else 0)
        stds.append(row['std'].values[0] if len(row) > 0 else 0)

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=color, alpha=0.85, edgecolor='white')
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(output_dir, '2_aggregated_metrics.png'), dpi=150, bbox_inches='tight')
print(f"Saved: 2_aggregated_metrics.png")

# ── Figure 3: Loss Components (per fold) ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Loss Components per Fold (Validation)', fontsize=16, fontweight='bold')

loss_metrics = ['loss', 'ce', 'metric_loss', 'contrastive', 'triplet']
loss_labels = ['Total Loss', 'Cross-Entropy', 'Metric Loss', 'Contrastive', 'Triplet']
colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

x = np.arange(4)
width = 0.15
for i, (metric, label, color) in enumerate(zip(loss_metrics, loss_labels, colors)):
    vals = df[(df['dataset'] == 'val') & (df['name'] == metric)].sort_values('fold')['value'].values
    ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

ax.set_xlabel('Fold')
ax.set_ylabel('Loss Value')
ax.set_xticks(x + width * 2)
ax.set_xticklabels([f'Fold {i}' for i in range(4)])
ax.legend(loc='upper right')
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(output_dir, '3_loss_components.png'), dpi=150, bbox_inches='tight')
print(f"Saved: 3_loss_components.png")

# ── Figure 4: Metric Learning Distances ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Metric Learning — Positive vs Negative Distances (Validation)', fontsize=16, fontweight='bold')

for ax, prefix, title in zip(axes, ['triplet', 'contrastive'], ['Triplet Loss', 'Contrastive Loss']):
    pos = df[(df['dataset'] == 'val') & (df['name'] == f'{prefix}_pos_distances')].sort_values('fold')['value'].values
    neg = df[(df['dataset'] == 'val') & (df['name'] == f'{prefix}_neg_distances')].sort_values('fold')['value'].values
    x = np.arange(4)
    width = 0.35
    ax.bar(x - width / 2, pos, width, label='Positive Dist', color='#43A047', alpha=0.85)
    ax.bar(x + width / 2, neg, width, label='Negative Dist', color='#E53935', alpha=0.85)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Distance')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in range(4)])
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(output_dir, '4_metric_distances.png'), dpi=150, bbox_inches='tight')
print(f"Saved: 4_metric_distances.png")

# ── Figure 5: Holdout — Front vs Back vs Simulated Side-pairs ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Holdout — Front vs Back vs Simulated Side-pairs (MAP)', fontsize=16, fontweight='bold')

side_metrics = ['f_map', 'b_map', 's_map', 'map']
side_labels = ['Front', 'Back', 'Simulated\nSide-pairs', 'Combined\n(Both sides)']
colors_side = ['#1E88E5', '#E53935', '#FB8C00', '#43A047']

x = np.arange(4)
width = 0.18
for i, (metric, label, color) in enumerate(zip(side_metrics, side_labels, colors_side)):
    vals = df[(df['dataset'] == 'holdout') & (df['name'] == metric)].sort_values('fold')['value'].values
    ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

ax.set_xlabel('Fold')
ax.set_ylabel('MAP Score')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f'Fold {i}' for i in range(4)])
ax.legend()
ax.set_ylim(0, 0.7)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(output_dir, '5_holdout_sides_map.png'), dpi=150, bbox_inches='tight')
print(f"Saved: 5_holdout_sides_map.png")

print(f"\nAll plots saved to: {output_dir}")
