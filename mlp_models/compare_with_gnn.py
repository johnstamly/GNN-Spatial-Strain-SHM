"""
Utility script to compare MLP model results with the best GNN model.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


def load_results(mlp_results_path, gnn_results_path):
    """
    Load results from both models.
    
    Args:
        mlp_results_path: Path to the MLP results JSON file
        gnn_results_path: Path to the GNN results JSON file
        
    Returns:
        Tuple of (mlp_results, gnn_results)
    """
    # Load MLP results
    with open(mlp_results_path, 'r') as f:
        mlp_results = json.load(f)
    
    # Load GNN results
    with open(gnn_results_path, 'r') as f:
        gnn_results = json.load(f)
    
    return mlp_results, gnn_results


def compare_metrics(mlp_results, gnn_results, output_dir):
    """
    Compare metrics between MLP and GNN models.
    
    Args:
        mlp_results: MLP results dictionary
        gnn_results: GNN results dictionary
        output_dir: Directory to save plots
    """
    # Extract fold keys (excluding summary)
    fold_keys = [key for key in mlp_results.keys() if key != 'summary']
    
    # Create data for the comparison table
    data = []
    
    for key in fold_keys:
        mlp_metrics = {
            'model': 'MLP',
            'fold': key,
            'MSE': mlp_results[key]['mse'],
            'RMSE': mlp_results[key]['rmse'],
            'MAPE': mlp_results[key]['mape']
        }
        data.append(mlp_metrics)
        
        gnn_metrics = {
            'model': 'GNN',
            'fold': key,
            'MSE': gnn_results[key]['mse'],
            'RMSE': gnn_results[key]['rmse'],
            'MAPE': gnn_results[key]['mape']
        }
        data.append(gnn_metrics)
    
    # Add summary metrics
    mlp_summary = {
        'model': 'MLP',
        'fold': 'Average',
        'MSE': mlp_results['summary']['mean_mse'],
        'RMSE': mlp_results['summary']['mean_rmse'],
        'MAPE': mlp_results['summary']['mean_mape']
    }
    data.append(mlp_summary)
    
    gnn_summary = {
        'model': 'GNN',
        'fold': 'Average',
        'MSE': gnn_results['summary']['mean_mse'],
        'RMSE': gnn_results['summary']['mean_rmse'],
        'MAPE': gnn_results['summary']['mean_mape']
    }
    data.append(gnn_summary)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print comparison table
    print("\nComparison of MLP vs. GNN metrics:")
    print("==================================")
    
    # Print by fold
    for key in fold_keys + ['Average']:
        print(f"\nFold: {key}")
        fold_df = df[df['fold'] == key]
        print(fold_df.set_index('model')[['MSE', 'RMSE', 'MAPE']].to_string(float_format=lambda x: f"{x:.4f}"))
    
    # Create bar chart for visual comparison
    plt.figure(figsize=(15, 10))
    
    # Set up the figure with 3 subplots (one for each metric)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plotting for each metric
    metrics = ['MSE', 'RMSE', 'MAPE']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Filter out Average for individual fold comparison
        plot_df = df[df['fold'] != 'Average'].copy()
        
        # Create grouped bar chart
        sns.barplot(x='fold', y=metric, hue='model', data=plot_df, ax=ax)
        
        ax.set_title(f'Comparison of {metric} across folds')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom', 
                        xytext = (0, 5), 
                        textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # Create table figure
    plt.figure(figsize=(12, 4))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Filter for averages only
    avg_df = df[df['fold'] == 'Average']
    avg_df = avg_df[['model', 'MSE', 'RMSE', 'MAPE']]
    
    # Calculate improvement percentages
    mlp_row = avg_df[avg_df['model'] == 'MLP'].iloc[0]
    gnn_row = avg_df[avg_df['model'] == 'GNN'].iloc[0]
    
    improvement = {
        'model': 'Improvement (%)',
        'MSE': ((mlp_row['MSE'] - gnn_row['MSE']) / mlp_row['MSE']) * 100 if mlp_row['MSE'] != 0 else float('inf'),
        'RMSE': ((mlp_row['RMSE'] - gnn_row['RMSE']) / mlp_row['RMSE']) * 100 if mlp_row['RMSE'] != 0 else float('inf'),
        'MAPE': ((mlp_row['MAPE'] - gnn_row['MAPE']) / mlp_row['MAPE']) * 100 if mlp_row['MAPE'] != 0 else float('inf')
    }
    
    # Add improvement row
    avg_df = avg_df.append(improvement, ignore_index=True)
    
    # Format table data
    cell_text = []
    for row in avg_df.itertuples():
        if row.model == 'Improvement (%)':
            cell_text.append([row.model, f"{row.MSE:.2f}%", f"{row.RMSE:.2f}%", f"{row.MAPE:.2f}%"])
        else:
            cell_text.append([row.model, f"{row.MSE:.4f}", f"{row.RMSE:.4f}", f"{row.MAPE:.2f}"])
    
    # Create table
    table = ax.table(cellText=cell_text, 
                    colLabels=['Model', 'MSE', 'RMSE', 'MAPE (%)'],
                    loc='center',
                    cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    # Color positive improvements green (GNN better than MLP)
    if cell_text[2][1].startswith('-'):
        table[(3, 1)]._text.set_color('red')
    else:
        table[(3, 1)]._text.set_color('green')
        
    if cell_text[2][2].startswith('-'):
        table[(3, 2)]._text.set_color('red')
    else:
        table[(3, 2)]._text.set_color('green')
        
    if cell_text[2][3].startswith('-'):
        table[(3, 3)]._text.set_color('red')
    else:
        table[(3, 3)]._text.set_color('green')
    
    plt.title('MLP vs GNN Performance Comparison (Average across all folds)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_table.png'), bbox_inches='tight')
    plt.close()
    
    return df


def plot_predictions_comparison(mlp_results, gnn_results, output_dir):
    """
    Plot comparison of predictions between MLP and GNN models.
    
    Args:
        mlp_results: MLP results dictionary
        gnn_results: GNN results dictionary
        output_dir: Directory to save plots
    """
    # Extract fold keys (excluding summary)
    fold_keys = [key for key in mlp_results.keys() if key != 'summary']
    
    for key in fold_keys:
        # Extract prediction data
        mlp_true = np.array(mlp_results[key]['true_values'])
        mlp_pred = np.array(mlp_results[key]['predicted_values'])
        
        gnn_true = np.array(gnn_results[key]['true_values'])
        gnn_pred = np.array(gnn_results[key]['predicted_values'])
        
        # Ensure same length by truncating to the shorter one if needed
        min_len = min(len(mlp_true), len(gnn_true))
        mlp_true = mlp_true[:min_len]
        mlp_pred = mlp_pred[:min_len]
        gnn_true = gnn_true[:min_len]
        gnn_pred = gnn_pred[:min_len]
        
        # Create prediction comparison plot
        plt.figure(figsize=(12, 6))
        
        # Plot data
        x_values = np.arange(min_len)
        plt.plot(x_values, mlp_true, label="True Values", color="blue", marker='.', linestyle='-', markersize=3, alpha=0.7)
        plt.plot(x_values, mlp_pred, label="MLP Predictions", color="red", marker='.', linestyle='--', markersize=3, alpha=0.6)
        plt.plot(x_values, gnn_pred, label="GNN Predictions", color="green", marker='.', linestyle='--', markersize=3, alpha=0.6)
        
        # Metrics text box
        mlp_metrics_text = (
            f"MLP Metrics:\n"
            f"MSE:  {mlp_results[key]['mse']:.2f}\n"
            f"RMSE: {mlp_results[key]['rmse']:.2f}\n"
            f"MAPE: {mlp_results[key]['mape']:.2f}%"
        )
        gnn_metrics_text = (
            f"GNN Metrics:\n"
            f"MSE:  {gnn_results[key]['mse']:.2f}\n"
            f"RMSE: {gnn_results[key]['rmse']:.2f}\n"
            f"MAPE: {gnn_results[key]['mape']:.2f}%"
        )
        
        plt.annotate(mlp_metrics_text, xy=(0.03, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", lw=0.5, alpha=0.8),
                    ha='left', va='top', fontsize=8, family='monospace')
        
        plt.annotate(gnn_metrics_text, xy=(0.03, 0.70), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="green", lw=0.5, alpha=0.8),
                    ha='left', va='top', fontsize=8, family='monospace')
        
        plt.xlabel("Time Steps")
        plt.ylabel("Stiffness (%)")
        plt.title(f"Fold {key}: MLP vs GNN Predictions")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"{key}_comparison.png"), bbox_inches='tight')
        plt.close()


def main():
    # Default paths
    mlp_results_path = 'mlp_models/results/loocv_results.json'
    gnn_results_path = 'results/loocv_results.json'  # Assuming this is where GNN results are stored
    output_dir = 'mlp_models/comparison'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading results...")
    mlp_results, gnn_results = load_results(mlp_results_path, gnn_results_path)
    
    print("\nComparing metrics...")
    comparison_df = compare_metrics(mlp_results, gnn_results, output_dir)
    
    print("\nPlotting prediction comparisons...")
    plot_predictions_comparison(mlp_results, gnn_results, output_dir)
    
    print(f"\nComparison completed. Results saved to {output_dir}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()