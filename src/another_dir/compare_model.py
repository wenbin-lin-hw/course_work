"""
Script to compare test metrics across different models
Reads metrics from summary files and creates comparison visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def extract_metrics_from_file(filepath):
    """Extract test metrics from a summary file"""
    metrics = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract metrics using regex
    metrics['AUROC'] = float(re.search(r'AUROC:\s+([\d.]+)', content).group(1))
    metrics['AUPR'] = float(re.search(r'AUPR:\s+([\d.]+)', content).group(1))

    # Handle duplicate Accuracy lines (take the first one)
    accuracy_matches = re.findall(r'Accuracy:\s+([\d.]+)', content)
    metrics['Accuracy'] = float(accuracy_matches[0])

    metrics['Precision'] = float(re.search(r'Test Precision:\s+([\d.]+)', content).group(1))
    metrics['Recall'] = float(re.search(r'Test Recall:\s+([\d.]+)', content).group(1))
    metrics['F1-Score'] = float(re.search(r'Test F1-Score:\s+([\d.]+)', content).group(1))

    return metrics


def main():
    # Define paths
    results_dir = Path('../results')

    # Model names and their corresponding files
    models = {
        'Decision Tree': 'decision_tree_summary.txt',
        'KNN': 'knn_summary.txt',
        'Logistic Regression': 'logistic_regression_summary.txt',
        'Random Forest': 'random_forest_summary.txt'
    }

    # Extract metrics for all models
    all_metrics = {}
    for model_name, filename in models.items():
        filepath = results_dir / filename
        all_metrics[model_name] = extract_metrics_from_file(filepath)

    # Prepare data for plotting
    metric_names = ['AUROC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    model_names = list(models.keys())

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison on Test Set', fontsize=20, fontweight='bold', y=0.995)

    axes = axes.flatten()

    # Colors for each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    # Plot each metric
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = [all_metrics[model][metric] for model in model_names]

        # Create bar plot
        bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Customize subplot
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'Test {metric}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Highlight the best model
        best_idx = values.index(max(values))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {results_dir / 'model_comparison.png'}")

    # Create a grouped bar chart for all metrics
    fig, ax = plt.subplots(figsize=(16, 10))

    x = np.arange(len(metric_names))
    width = 0.2

    for i, model_name in enumerate(model_names):
        values = [all_metrics[model_name][metric] for metric in metric_names]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Comprehensive Model Performance Comparison on Test Set',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison_grouped.png', dpi=300, bbox_inches='tight')
    print(f"Saved grouped comparison plot to {results_dir / 'model_comparison_grouped.png'}")

    # Create a summary table
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON - TEST SET")
    print("=" * 80)
    print(
        f"{'Model':<25} {'AUROC':<10} {'AUPR':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)

    for model_name in model_names:
        metrics = all_metrics[model_name]
        print(f"{model_name:<25} {metrics['AUROC']:<10.4f} {metrics['AUPR']:<10.4f} "
              f"{metrics['Accuracy']:<10.4f} {metrics['Precision']:<10.4f} "
              f"{metrics['Recall']:<10.4f} {metrics['F1-Score']:<10.4f}")

    print("=" * 80)

    # Find best model for each metric
    print("\nBEST MODEL FOR EACH METRIC:")
    print("-" * 80)
    for metric in metric_names:
        values = [(model, all_metrics[model][metric]) for model in model_names]
        best_model, best_value = max(values, key=lambda x: x[1])
        print(f"{metric:<15}: {best_model:<25} ({best_value:.4f})")
    print("=" * 80)

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for heatmap
    data = []
    for model_name in model_names:
        data.append([all_metrics[model_name][metric] for metric in metric_names])

    # Create heatmap
    im = ax.imshow(data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_yticklabels(model_names, fontsize=12)

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{data[i][j]:.3f}',
                           ha="center", va="center", color="black", fontsize=11, fontweight='bold')

    ax.set_title('Model Performance Heatmap - Test Set', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved heatmap to {results_dir / 'model_comparison_heatmap.png'}")

    plt.show()


if __name__ == '__main__':
    main()
