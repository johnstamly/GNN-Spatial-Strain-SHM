#!/usr/bin/env python3
"""
Main script to run LOOCV with the best model from the model comparison study.

This script loads the best model from the comparison results and runs it with more epochs
to get the best possible performance and detailed visualizations.
"""

import os
import argparse
import sys
import json
import numpy as np

# Set matplotlib backend to non-interactive 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import utility functions
from gnn_utils.utils import calculate_cycles_from_timesteps
from gnn_utils import (
    setup_matplotlib_style,
    load_data,
    preprocess_data,
    identify_target_indexes,
    truncate_data,
    prepare_gnn_data,
    run_loocv_utility,
    summarize_loocv_results,
    plot_loocv_predictions
)

# Import model variants
from gnn_utils.model_variants import (
    GENConvModel,
    SAGPoolModel,
    GATv2Model,
    GCNModel,
    EdgeConvModel
)

from gnn_utils.model_variants_no_edges import (
    GCNModel_NoEdges,
    GINModel,
    SGConvModel,
    GraphSAGEModel,
    ChebConvModel
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run LOOCV with the best model from comparison')
    
    # Data paths and output config
    parser.add_argument('--stiffness-path', type=str, default='Data/Stiffness_Reduction',
                        help='Path to stiffness data directory')
    parser.add_argument('--strain-path', type=str, default='Data/Strain',
                        help='Path to strain data directory')
    parser.add_argument('--drop-level', type=int, default=85,
                        help='Stiffness reduction level for truncation')
    
    # Model selection
    parser.add_argument('--model-type', type=str, default='with_edges',
                        choices=['with_edges', 'no_edges', 'both'],
                        help='Type of model to run (with_edges, no_edges, or both)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    
    # Visualization options
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files instead of displaying them')
    parser.add_argument('--output-dir', type=str, default='best_comparison_model',
                        help='Directory to save results and plots')
    
    return parser.parse_args()


def get_best_model_info(model_type):
    """Get information about the best model from comparison results."""
    if model_type == 'with_edges' or model_type == 'both':
        try:
            with open('model_comparison/comparison_results.json') as f:
                with_edges_results = json.load(f)
            best_with_edges = min(with_edges_results, key=lambda x: x['mean_mse'])
            print(f"\nBest model with edge attributes: {best_with_edges['model_name']}")
            print(f"  MSE: {best_with_edges['mean_mse']:.4f}")
            print(f"  RMSE: {best_with_edges['mean_rmse']:.4f}")
            print(f"  MAPE: {best_with_edges['mean_mape']:.2f}%")
        except FileNotFoundError:
            print("Error: model_comparison/comparison_results.json not found")
            sys.exit(1)
    else:
        best_with_edges = None
        
    if model_type == 'no_edges' or model_type == 'both':
        try:
            with open('model_comparison_no_edges/comparison_results.json') as f:
                no_edges_results = json.load(f)
            best_no_edges = min(no_edges_results, key=lambda x: x['mean_mse'])
            print(f"\nBest model without edge attributes: {best_no_edges['model_name']}")
            print(f"  MSE: {best_no_edges['mean_mse']:.4f}")
            print(f"  RMSE: {best_no_edges['mean_rmse']:.4f}")
            print(f"  MAPE: {best_no_edges['mean_mape']:.2f}%")
        except FileNotFoundError:
            print("Error: model_comparison_no_edges/comparison_results.json not found")
            sys.exit(1)
    else:
        best_no_edges = None
    
    return best_with_edges, best_no_edges


def create_model(model_name, num_node_features, edge_feature_dim, hidden_dim, 
                output_dim, num_gnn_layers, dropout_p, use_edge_attr=True):
    """Create a model instance based on the model name."""
    if use_edge_attr:
        # Models with edge attributes
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
        # Models without edge attributes
        if model_name == 'GCN_NoEdges':
            return GCNModel_NoEdges(
                num_node_features=num_node_features,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_gnn_layers=num_gnn_layers,
                dropout_p=dropout_p
            )
        elif model_name == 'GIN':
            return GINModel(
                num_node_features=num_node_features,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_gnn_layers=num_gnn_layers,
                dropout_p=dropout_p
            )
        elif model_name == 'SGConv':
            return SGConvModel(
                num_node_features=num_node_features,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_gnn_layers=num_gnn_layers,
                dropout_p=dropout_p
            )
        elif model_name == 'GraphSAGE':
            return GraphSAGEModel(
                num_node_features=num_node_features,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_gnn_layers=num_gnn_layers,
                dropout_p=dropout_p
            )
        elif model_name == 'ChebConv':
            return ChebConvModel(
                num_node_features=num_node_features,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_gnn_layers=num_gnn_layers,
                dropout_p=dropout_p
            )
    
    raise ValueError(f"Unknown model name: {model_name}")


def run_best_model(model_name, use_edge_attr, strain_data_list, stiffness_data_list, 
                  specimen_keys, args, output_subdir, preprocessed_data=None):
    """Run LOOCV with the best model."""
    # Create model creator function
    def model_creator(num_node_features, edge_feature_dim, hidden_dim, output_dim, 
                     num_gnn_layers, dropout_p):
        return create_model(
            model_name, 
            num_node_features, 
            edge_feature_dim, 
            hidden_dim, 
            output_dim, 
            num_gnn_layers, 
            dropout_p,
            use_edge_attr
        )
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load hyperparameters from best_params.json
    try:
        with open('results/best_params.json') as f:
            best_params = json.load(f).get('params', {})
    except FileNotFoundError:
        print("Warning: best_params.json not found, using default hyperparameters")
        best_params = {}
    
    # Run LOOCV with the best model
    print(f"\nRunning LOOCV with {model_name} model...")
    loocv_results = run_loocv_utility(
        strain_data_list,
        stiffness_data_list,
        specimen_keys,
        num_nodes=16,  # Default number of nodes
        batch_size=args.batch_size,
        hidden_dim=best_params.get('hidden_dim', 64),
        num_gnn_layers=best_params.get('num_gnn_layers', 3),
        dropout_p=best_params.get('dropout', 0.307),
        epochs=args.epochs,
        patience=args.patience,
        visualize=not args.no_visualize and not args.save_plots,
        save_plots=args.save_plots,
        output_dir=output_dir,
        model_class=model_creator
    )
    
    # Summarize and save results
    mean_mse, mean_rmse, mean_mape = summarize_loocv_results(loocv_results)
    
    # Calculate cycles for x-axis if preprocessed data is available
    cycles_dict = None
    if preprocessed_data is not None:
        strain_post = preprocessed_data.get('strain_post')
        stiffness_post = preprocessed_data.get('stiffness_post')
        last_cycle = preprocessed_data.get('last_cycle')
        
        if strain_post is not None and stiffness_post is not None and last_cycle is not None:
            cycles_dict = calculate_cycles_from_timesteps(strain_post, stiffness_post, last_cycle)
    
    if not args.no_visualize:
        plot_loocv_predictions(
            loocv_results,
            save_plots=args.save_plots,
            output_dir=output_dir,
            cycles_dict=cycles_dict
        )
    
    # Save detailed results
    serializable_results = {
        key: {
            'mse': float(result['mse']),  # Convert numpy.float32 to Python float
            'rmse': float(result['rmse']),  # Convert numpy.float32 to Python float
            'mape': float(result['mape']),  # Convert numpy.float32 to Python float
            'model_path': result['model_path']
        }
        for key, result in loocv_results.items()
    }
    serializable_results['summary'] = {
        'mean_mse': float(mean_mse),
        'mean_rmse': float(mean_rmse),
        'mean_mape': float(mean_mape)
    }
    
    with open(os.path.join(output_dir, 'loocv_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    # Create detailed plots
    create_detailed_plots(loocv_results, output_dir, 
                         strain_post=preprocessed_data.get('strain_post'),
                         stiffness_post=preprocessed_data.get('stiffness_post'),
                         last_cycle=preprocessed_data.get('last_cycle'))
    
    return float(mean_mse), float(mean_rmse), float(mean_mape)  # Convert numpy.float32 to Python float


def create_detailed_plots(loocv_results, output_dir, strain_post=None, stiffness_post=None, last_cycle=None):
    """Create detailed plots for the best model."""
    # Calculate cycles for x-axis if preprocessed data is available
    cycles_dict = None
    if strain_post is not None and stiffness_post is not None and last_cycle is not None:
        cycles_dict = calculate_cycles_from_timesteps(strain_post, stiffness_post, last_cycle)
    
    # Plot true vs predicted values for each fold
    for key, result in loocv_results.items():
        if key == 'summary':
            continue
            
        true_values = result['true_values']
        pred_values = result['predicted_values']
        
        # 1. Scatter plot with regression line
        plt.figure(figsize=(10, 8))
        plt.scatter(true_values, pred_values, alpha=0.7)
        
        # Add regression line
        z = np.polyfit(true_values, pred_values, 1)
        p = np.poly1d(z)
        plt.plot(true_values, p(true_values), "r--", alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(min(true_values), min(pred_values))
        max_val = max(max(true_values), max(pred_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Fold {key}: True vs Predicted Values')
        plt.grid(True, alpha=0.3)
        
        # Add metrics as text
        metrics_text = (
            f"MSE:  {result['mse']:.4f}\n"
            f"RMSE: {result['rmse']:.4f}\n"
            f"MAPE: {result['mape']:.2f}%\n"
            f"R²: {np.corrcoef(true_values, pred_values)[0, 1]**2:.4f}"
        )
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=0.5, alpha=0.8),
                    ha='left', va='top', fontsize=10, family='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{key}_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Residual plot
        residuals = np.array(pred_values) - np.array(true_values)
        plt.figure(figsize=(10, 6))
        plt.scatter(true_values, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        plt.xlabel('True Values')
        plt.ylabel('Residuals (Predicted - True)')
        plt.title(f'Fold {key}: Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # Add residual statistics
        residual_stats = (
            f"Mean: {np.mean(residuals):.4f}\n"
            f"Std Dev: {np.std(residuals):.4f}\n"
            f"Min: {np.min(residuals):.4f}\n"
            f"Max: {np.max(residuals):.4f}"
        )
        plt.annotate(residual_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=0.5, alpha=0.8),
                    ha='left', va='top', fontsize=10, family='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{key}_residuals.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Time series plot with error bands
        # Use cycles for x-axis if available, otherwise use timesteps
        if cycles_dict is not None and key in cycles_dict:
            x_values = cycles_dict[key][:len(true_values)]
            x_label = "Cycles"
        else:
            x_values = np.arange(len(true_values))
            x_label = "Timestep"
            
        plt.figure(figsize=(12, 6))
        
        # Plot true values
        plt.plot(x_values, true_values, 'b-', label='True Values', linewidth=2)
        
        # Plot predicted values with error band
        error = np.abs(np.array(true_values) - np.array(pred_values))
        plt.plot(x_values, pred_values, 'r--', label='Predicted Values', linewidth=2)
        plt.fill_between(x_values, 
                        np.array(pred_values) - error, 
                        np.array(pred_values) + error, 
                        color='r', alpha=0.2, label='Error Band')
        
        plt.xlabel(x_label)
        plt.ylabel('Stiffness (%)')
        plt.title(f'Fold {key}: True vs Predicted Values Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{key}_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the best model from comparison."""
    # Parse arguments
    args = parse_args()
    
    # Configure matplotlib
    setup_matplotlib_style()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('log_best_model', exist_ok=True)
    
    # Dictionary to store preprocessed data for cycle calculation
    preprocessed_data = {}
    
    # Get best model information
    best_with_edges, best_no_edges = get_best_model_info(args.model_type)
    
    # Load and process data
    print("Loading data...")
    stiffness_dfs, strain_dfs = load_data(args.stiffness_path, args.strain_path)
    
    print("\nPreprocessing data...")
    stiffness_post, strain_post, last_cycle = preprocess_data(stiffness_dfs, strain_dfs)
    
    # Store preprocessed data for cycle calculation
    preprocessed_data = {'strain_post': strain_post, 'stiffness_post': stiffness_post, 'last_cycle': last_cycle}
    
    print("\nIdentifying target indexes...")
    target_indexes = identify_target_indexes(stiffness_post)
    
    print("\nTruncating data...")
    stiffness_post_trunc, strain_post_trunc = truncate_data(
        stiffness_post, strain_post, target_indexes, args.drop_level
    )
    
    print("\nPreparing data for GNN...")
    strain_data_list, stiffness_data_list, specimen_keys = prepare_gnn_data(
        stiffness_post_trunc, strain_post_trunc
    )
    
    # Run best model(s)
    results = {}
    
    if args.model_type == 'with_edges' or args.model_type == 'both':
        model_name = best_with_edges['model_name']
        mean_mse, mean_rmse, mean_mape = run_best_model(
            model_name, 
            True, 
            strain_data_list, 
            stiffness_data_list, 
            specimen_keys, 
            args,
            f"{model_name}_with_edges",
            preprocessed_data
        )
        results['with_edges'] = {
            'model_name': model_name,
            'mean_mse': mean_mse,
            'mean_rmse': mean_rmse,
            'mean_mape': mean_mape
        }
        
    if args.model_type == 'no_edges' or args.model_type == 'both':
        model_name = best_no_edges['model_name']
        mean_mse, mean_rmse, mean_mape = run_best_model(
            model_name, 
            False, 
            strain_data_list, 
            stiffness_data_list, 
            specimen_keys, 
            args,
            f"{model_name}_no_edges",
            preprocessed_data
        )
        results['no_edges'] = {
            'model_name': model_name,
            'mean_mse': mean_mse,
            'mean_rmse': mean_rmse,
            'mean_mape': mean_mape
        }
    
    # Save overall results
    # Ensure all values are JSON serializable
    serializable_results = {}
    for key, result in results.items():
        serializable_results[key] = {
            'model_name': result['model_name'],
            'mean_mse': float(result['mean_mse']),  # Convert numpy.float32 to Python float
            'mean_rmse': float(result['mean_rmse']),  # Convert numpy.float32 to Python float
            'mean_mape': float(result['mean_mape'])  # Convert numpy.float32 to Python float
        }
    with open(os.path.join(args.output_dir, 'best_model_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    # Print summary
    print("\nBest Model Results Summary:")
    print("==========================")
    
    if 'with_edges' in results:
        print(f"\nBest model with edge attributes: {results['with_edges']['model_name']}")
        print(f"  MSE: {results['with_edges']['mean_mse']:.4f}")
        print(f"  RMSE: {results['with_edges']['mean_rmse']:.4f}")
        print(f"  MAPE: {results['with_edges']['mean_mape']:.2f}%")
        
    if 'no_edges' in results:
        print(f"\nBest model without edge attributes: {results['no_edges']['model_name']}")
        print(f"  MSE: {results['no_edges']['mean_mse']:.4f}")
        print(f"  RMSE: {results['no_edges']['mean_rmse']:.4f}")
        print(f"  MAPE: {results['no_edges']['mean_mape']:.2f}%")
    
    if 'with_edges' in results and 'no_edges' in results:
        # Calculate improvement
        improvement_mse = ((results['no_edges']['mean_mse'] - results['with_edges']['mean_mse']) / 
                          results['no_edges']['mean_mse']) * 100
        improvement_rmse = ((results['no_edges']['mean_rmse'] - results['with_edges']['mean_rmse']) / 
                           results['no_edges']['mean_rmse']) * 100
        improvement_mape = ((results['no_edges']['mean_mape'] - results['with_edges']['mean_mape']) / 
                           results['no_edges']['mean_mape']) * 100
        
        print(f"\nImprovement from using edge attributes:")
        print(f"  MSE improvement: {improvement_mse:.2f}%")
        print(f"  RMSE improvement: {improvement_rmse:.2f}%")
        print(f"  MAPE improvement: {improvement_mape:.2f}%")
    
    print(f"\nResults saved to: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)