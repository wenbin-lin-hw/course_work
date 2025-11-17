"""
Script to compare test metrics across different models
Reads metrics from summary files and creates comparison visualizations
Highlights the best performance for each metric
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


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
    
    # Colors for each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Create a figure with subplots (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle('Model Performance Comparison on Test Set\n(Best Results Highlighted with Gold Border)', 
                 fontsize=22, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    # Plot each metric
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = [all_metrics[model][metric] for model in model_names]
        
        # Create bar plot
        bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # Customize subplot
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric, fontsize=13, fontweight='bold')
        ax.set_title(f'Test {metric}', fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Highlight the best model with gold border and star
        best_idx = values.index(max(values))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(4)
        
        # Add a star marker above the best bar
        best_value = values[best_idx]
        ax.plot(best_idx, best_value + 0.08, marker='*', markersize=25, 
               color='gold', markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        
        # Add "BEST" label
        ax.text(best_idx, best_value + 0.12, 'BEST', ha='center', va='bottom',
               fontsize=10, fontweight='bold', color='darkgoldenrod',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.7, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison_highlighted.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {results_dir / 'model_comparison_highlighted.png'}")
    plt.close()
    
    # Create a grouped bar chart for all metrics
    fig, ax = plt.subplots(figsize=(18, 11))
    
    x = np.arange(len(metric_names))
    width = 0.2
    
    # Store bars for each model
    all_bars = []
    for i, model_name in enumerate(model_names):
        values = [all_metrics[model_name][metric] for metric in metric_names]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)
        all_bars.append(bars)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, rotation=0)
    
    # Highlight the best model for each metric
    for metric_idx, metric in enumerate(metric_names):
        values = [all_metrics[model_name][metric] for model_name in model_names]
        best_model_idx = values.index(max(values))
        
        # Highlight the best bar with gold border
        bar_to_highlight = all_bars[best_model_idx][metric_idx]
        bar_to_highlight.set_edgecolor('gold')
        bar_to_highlight.set_linewidth(4)
        
        # Add star above the best bar
        x_pos = bar_to_highlight.get_x() + bar_to_highlight.get_width()/2.
        y_pos = bar_to_highlight.get_height()
        ax.plot(x_pos, y_pos + 0.05, marker='*', markersize=20, 
               color='gold', markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    
    ax.set_xlabel('Metrics', fontsize=15, fontweight='bold')
    ax.set_ylabel('Score', fontsize=15, fontweight='bold')
    ax.set_title('Comprehensive Model Performance Comparison on Test Set\n(Best Results Marked with Gold Border and Star)', 
                 fontsize=17, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=13)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison_grouped_highlighted.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved grouped comparison plot to {results_dir / 'model_comparison_grouped_highlighted.png'}")
    plt.close()
    
    # Create a summary table
    print("\n" + "="*100)
    print("MODEL PERFORMANCE COMPARISON - TEST SET")
    print("="*100)
    print(f"{'Model':<25} {'AUROC':<12} {'AUPR':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*100)
    
    for model_name in model_names:
        metrics = all_metrics[model_name]
        print(f"{model_name:<25} {metrics['AUROC']:<12.4f} {metrics['AUPR']:<12.4f} "
              f"{metrics['Accuracy']:<12.4f} {metrics['Precision']:<12.4f} "
              f"{metrics['Recall']:<12.4f} {metrics['F1-Score']:<12.4f}")
    
    print("="*100)
    
    # Find best model for each metric
    print("\n" + "="*100)
    print("BEST MODEL FOR EACH METRIC:")
    print("="*100)
    for metric in metric_names:
        values = [(model, all_metrics[model][metric]) for model in model_names]
        best_model, best_value = max(values, key=lambda x: x[1])
        print(f"  {metric:<15}: {best_model:<25} ({best_value:.4f}) ⭐")
    print("="*100)
    
    # Create a heatmap with best values highlighted
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare data for heatmap
    data = []
    for model_name in model_names:
        data.append([all_metrics[model_name][metric] for metric in metric_names])
    
    data_array = np.array(data)
    
    # Create heatmap
    im = ax.imshow(data_array, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(metric_names, fontsize=13)
    ax.set_yticklabels(model_names, fontsize=13)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=25, fontsize=13, fontweight='bold')
    
    # Add text annotations and highlight best values
    for i in range(len(model_names)):
        for j in range(len(metric_names)):
            # Check if this is the best value for this metric
            is_best = data_array[i, j] == np.max(data_array[:, j])
            
            if is_best:
                # Highlight best value with bold text and star
                text = ax.text(j, i, f'{data_array[i][j]:.3f} ⭐',
                              ha="center", va="center", color="black", 
                              fontsize=12, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', 
                                       alpha=0.6, edgecolor='darkgoldenrod', linewidth=2))
            else:
                text = ax.text(j, i, f'{data_array[i][j]:.3f}',
                              ha="center", va="center", color="black", 
                              fontsize=11, fontweight='bold')
    
    ax.set_title('Model Performance Heatmap - Test Set\n(Best Values Highlighted with Gold Background and Star)', 
                 fontsize=17, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison_heatmap_highlighted.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved heatmap to {results_dir / 'model_comparison_heatmap_highlighted.png'}")
    plt.close()
    
    # Create a radar chart for comprehensive comparison
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    num_vars = len(metric_names)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot data for each model
    for i, model_name in enumerate(model_names):
        values = [all_metrics[model_name][metric] for metric in metric_names]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, 
               color=colors[i], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=12, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, framealpha=0.9)
    
    # Add title
    plt.title('Radar Chart: Model Performance Comparison\nAcross All Metrics', 
             fontsize=16, fontweight='bold', pad=30)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'model_comparison_radar.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved radar chart to {results_dir / 'model_comparison_radar.png'}")
    plt.close()
    
    # Save summary to text file
    with open(results_dir / 'model_comparison_summary.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("MODEL PERFORMANCE COMPARISON - TEST SET\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Model':<25} {'AUROC':<12} {'AUPR':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-"*100 + "\n")
        
        for model_name in model_names:
            metrics = all_metrics[model_name]
            f.write(f"{model_name:<25} {metrics['AUROC']:<12.4f} {metrics['AUPR']:<12.4f} "
                  f"{metrics['Accuracy']:<12.4f} {metrics['Precision']:<12.4f} "
                  f"{metrics['Recall']:<12.4f} {metrics['F1-Score']:<12.4f}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("BEST MODEL FOR EACH METRIC:\n")
        f.write("="*100 + "\n")
        
        for metric in metric_names:
            values = [(model, all_metrics[model][metric]) for model in model_names]
            best_model, best_value = max(values, key=lambda x: x[1])
            f.write(f"  {metric:<15}: {best_model:<25} ({best_value:.4f})\n")
        
        f.write("="*100 + "\n")
    
    print(f"✓ Saved summary to {results_dir / 'model_comparison_summary.txt'}")
    
    print("\n" + "="*100)
    print("ALL COMPARISON PLOTS AND SUMMARY HAVE BEEN SAVED SUCCESSFULLY!")
    print("="*100)
    print("\nGenerated files:")
    print("  1. model_comparison_highlighted.png - Individual metric comparisons with best highlighted")
    print("  2. model_comparison_grouped_highlighted.png - Grouped bar chart with best highlighted")
    print("  3. model_comparison_heatmap_highlighted.png - Heatmap with best values highlighted")
    print("  4. model_comparison_radar.png - Radar chart for comprehensive view")
    print("  5. model_comparison_summary.txt - Text summary of all results")
    print("="*100)


if __name__ == '__main__':
    main()
