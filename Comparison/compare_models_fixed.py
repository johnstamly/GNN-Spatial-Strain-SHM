#!/usr/bin/env python3
"""
Script to compare different GNN model architectures for stiffness prediction.

This script runs Leave-One-Out Cross-Validation (LOOCV) for multiple GNN model architectures
and compares their performance.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set matplotlib backend to non-interactive 'Agg' to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

# Import utility functions
from gnn_utils import (
    setup_matplotlib_style,
    load_data,
    preprocess_data,
    identify_target_indexes,
    truncate_data,
    prepare_gnn_data,
    run_loocv_utility,
    summarize_loocv_results
)

# Import model variants
from gnn_utils.model_variants import (
    GENConvModel,
    SAGPoolModel,
    GATv2Model,
    GCNModel,
    EdgeConvModel,
    count_parameters
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare different GNN model architectures')
    
    # Data paths
    parser.add_argument('--stiffness-path', type=str, default='Data/Stiffness_Reduction',
                        help='Path to stiffness data directory')
    parser.add_argument('--strain-path', type=str, default='Data/Strain',
                        help='Path to strain data directory')
    
    # Model parameters (using best parameters from hyperparameter tuning)
    parser.add_argument('--num-nodes', type=int, default=16,
                        help='Number of nodes in each graph')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension for the GNN model')
    parser.add_argument('--num-gnn-layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.307,
                        help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping')
    
    # Other parameters
    parser.add_argument('--drop-level', type=int, default=85,
                        help='Stiffness reduction level for truncation')
    parser.add_argument('--output-dir', type=str, default='model_comparison',
                        help='Directory to save results and plots')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['GENConv', 'SAGPool', 'GATv2', 'GCN', 'EdgeConv'],
                        help='List of models to compare')
    
    return parser.parse_args()


def create_model(model_name, num_node_features, edge_feature_dim, hidden_dim, 
                output_dim, num_gnn_layers, dropout_p):
    """Create a model instance based on the model name."""
    if model_name == 'GENConv':
        return GENConvModel(
            num_node_features=num_node_features,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_gnn_layers=num_gnn_layers,
            dropout_p=dropout_p
        )
    elif model_name == 'SAGPool':
        return SAGPoolModel(
            num_node_features=num_node_features,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_gnn_layers=num_gnn_layers,
            dropout_p=dropout_p
        )
    elif model_name == 'GATv2':
        return GATv2Model(
            num_node_features=num_node_features,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_gnn_layers=num_gnn_layers,
            dropout_p=dropout_p
        )
    elif model_name == 'GCN':
        return GCNModel(
            num_node_features=num_node_features,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_gnn_layers=num_gnn_layers,
            dropout_p=dropout_p
        )
    elif model_name == 'EdgeConv':
        return EdgeConvModel(
            num_node_features=num_node_features,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_gnn_layers=num_gnn_layers,
            dropout_p=dropout_p
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_loocv_for_model(model_name, strain_data_list, stiffness_data_list, specimen_keys, args):
    """Run LOOCV for a specific model."""
    print(f"\n{'='*80}")
    print(f"Running LOOCV for {model_name} model")
    print(f"{'='*80}")
    
    # Create model directory
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Custom model creation function for run_loocv_utility
    def model_creator(num_node_features, edge_feature_dim, hidden_dim, output_dim, 
                     num_gnn_layers, dropout_p):
        return create_model(
            model_name, 
            num_node_features, 
            edge_feature_dim, 
            hidden_dim, 
            output_dim, 
            num_gnn_layers, 
            dropout_p
        )
    
    # Run LOOCV
    loocv_results = run_loocv_utility(
        strain_data_list,
        stiffness_data_list,
        specimen_keys,
        num_nodes=args.num_nodes,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout_p=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        visualize=False,
        save_plots=True,
        output_dir=model_dir,
        model_class=model_creator  # Pass custom model creator
    )
    
    # Summarize results
    mean_mse, mean_rmse, mean_mape = summarize_loocv_results(loocv_results)
    
    # Save results to file
    serializable_results = {}
    for key, result in loocv_results.items():
        serializable_results[key] = {
            'mse': float(result['mse']),
            'rmse': float(result['rmse']),
            'mape': float(result['mape']),
            'model_path': result['model_path']
        }
    
    # Add summary metrics
    serializable_results['summary'] = {
        'mean_mse': float(mean_mse),  # Convert numpy.float32 to Python float
        'mean_rmse': float(mean_rmse),  # Convert numpy.float32 to Python float
        'mean_mape': float(mean_mape)  # Convert numpy.float32 to Python float
    }
    
    # Save to JSON file
    with open(os.path.join(model_dir, 'loocv_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    return {
        'model_name': model_name,
        'mean_mse': float(mean_mse),  # Convert numpy.float32 to Python float
        'mean_rmse': float(mean_rmse),  # Convert numpy.float32 to Python float
        'mean_mape': float(mean_mape)  # Convert numpy.float32 to Python float
    }


def plot_model_comparison(comparison_results, output_dir):
    """Plot comparison of model performance."""
    # Create DataFrame for easier plotting
    df = pd.DataFrame(comparison_results)
    df.set_index('model_name', inplace=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot MSE
    df['mean_mse'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Mean MSE (lower is better)')
    axes[0].set_ylabel('MSE')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot RMSE
    df['mean_rmse'].plot(kind='bar', ax=axes[1], color='lightgreen')
    axes[1].set_title('Mean RMSE (lower is better)')
    axes[1].set_ylabel('RMSE')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot MAPE
    df['mean_mape'].plot(kind='bar', ax=axes[2], color='salmon')
    axes[2].set_title('Mean MAPE % (lower is better)')
    axes[2].set_ylabel('MAPE %')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Create a table for the report
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')
    
    # Create table data
    table_data = df.reset_index()
    table_data.columns = ['Model', 'MSE', 'RMSE', 'MAPE (%)']
    
    # Format numbers
    table_data['MSE'] = table_data['MSE'].map('{:.4f}'.format)
    table_data['RMSE'] = table_data['RMSE'].map('{:.4f}'.format)
    table_data['MAPE (%)'] = table_data['MAPE (%)'].map('{:.2f}'.format)
    
    # Create table
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title('Model Performance Comparison', fontsize=16, pad=20)
    
    # Save table
    plt.savefig(os.path.join(output_dir, 'model_comparison_table.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def main():
    """Main function to compare different GNN model architectures."""
    # Parse arguments
    args = parse_args()
    
    # Configure matplotlib
    setup_matplotlib_style()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    stiffness_dfs, strain_dfs = load_data(args.stiffness_path, args.strain_path)
    
    # Preprocess data
    print("\nPreprocessing data...")
    stiffness_post, strain_post, last_cycle = preprocess_data(stiffness_dfs, strain_dfs)
    
    # Identify target indexes
    print("\nIdentifying target indexes...")
    target_indexes = identify_target_indexes(stiffness_post)
    
    # Truncate data
    print("\nTruncating data...")
    stiffness_post_trunc, strain_post_trunc = truncate_data(
        stiffness_post, strain_post, target_indexes, args.drop_level
    )
    
    # Prepare data for GNN
    print("\nPreparing data for GNN...")
    strain_data_list, stiffness_data_list, specimen_keys = prepare_gnn_data(
        stiffness_post_trunc, strain_post_trunc
    )
    
    # Compare model parameter counts
    print("\nComparing model parameter counts:")
    model_params = {}
    for model_name in args.models:
        model = create_model(
            model_name,
            num_node_features=1,
            edge_feature_dim=1,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            num_gnn_layers=args.num_gnn_layers,
            dropout_p=args.dropout
        )
        param_count = count_parameters(model)
        model_params[model_name] = param_count
        print(f"  {model_name}: {param_count:,} parameters")
    
    # Save parameter counts
    with open(os.path.join(args.output_dir, 'model_parameters.json'), 'w') as f:
        json.dump(model_params, f, indent=4)
    
    # Run LOOCV for each model
    comparison_results = []
    for model_name in args.models:
        result = run_loocv_for_model(
            model_name,
            strain_data_list,
            stiffness_data_list,
            specimen_keys,
            args
        )
        comparison_results.append(result)
    
    # Plot comparison
    plot_model_comparison(comparison_results, args.output_dir)
    
    # Ensure all values are JSON serializable
    serializable_comparison = []
    for result in comparison_results:
        serializable_comparison.append({
            'model_name': result['model_name'],
            'mean_mse': float(result['mean_mse']),  # Convert numpy.float32 to Python float
            'mean_rmse': float(result['mean_rmse']),  # Convert numpy.float32 to Python float
            'mean_mape': float(result['mean_mape'])  # Convert numpy.float32 to Python float
        })
    
    # Save comparison results
    with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(serializable_comparison, f, indent=4)
    
    print(f"\nModel comparison completed successfully!")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")
    
    # Print best model
    best_model = min(comparison_results, key=lambda x: x['mean_mse'])
    print(f"\nBest model based on MSE: {best_model['model_name']}")
    print(f"  MSE: {best_model['mean_mse']:.4f}")
    print(f"  RMSE: {best_model['mean_rmse']:.4f}")
    print(f"  MAPE: {best_model['mean_mape']:.2f}%")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()