#!/usr/bin/env python3
"""
Main script to run LOOCV with optimized hyperparameters from best_params.json
"""

import os
import argparse
import sys
import json

# Set matplotlib backend to non-interactive 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def parse_args():
    """Parse remaining command line arguments."""
    parser = argparse.ArgumentParser(description='Run LOOCV with optimized hyperparameters')
    
    # Data paths and output config
    parser.add_argument('--stiffness-path', type=str, default='Data/Stiffness_Reduction',
                        help='Path to stiffness data directory')
    parser.add_argument('--strain-path', type=str, default='Data/Strain',
                        help='Path to strain data directory')
    parser.add_argument('--drop-level', type=int, default=85,
                        help='Stiffness reduction level for truncation')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files instead of displaying them')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results and plots')
    
    return parser.parse_args()

def main():
    """Main function to run LOOCV with optimized params."""
    args = parse_args()
    
    try:
        # Load optimized hyperparameters with error handling
        with open('results/best_params.json') as f:
            best_params = json.load(f).get('params', {})
    except FileNotFoundError:
        print("Error: best_params.json not found in results directory")
        sys.exit(1)
    
    # Set parameters with fallback defaults
    optimized_params = {
        'num_nodes': best_params.get('num_nodes', 16),
        'batch_size': best_params.get('batch_size', 64),
        'hidden_dim': best_params.get('hidden_dim', 64),
        'num_gnn_layers': best_params.get('num_gnn_layers', 3),
        'dropout': best_params.get('dropout', 0.3),
        'epochs': best_params.get('epochs', 1000),
        'patience': best_params.get('patience', 20),
    }
    
    # Configure matplotlib
    setup_matplotlib_style()
    
    # Create directories
    os.makedirs('best_model', exist_ok=True)
    os.makedirs('log', exist_ok=True)
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process data
    print("Loading data...")
    stiffness_dfs, strain_dfs = load_data(args.stiffness_path, args.strain_path)
    
    print("\nPreprocessing data...")
    stiffness_post, strain_post, last_cycle = preprocess_data(stiffness_dfs, strain_dfs)
    
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
    
    # Run LOOCV with optimized parameters
    print("\nRunning LOOCV with optimized parameters...")
    loocv_results = run_loocv_utility(
        strain_data_list,
        stiffness_data_list,
        specimen_keys,
        num_nodes=optimized_params['num_nodes'],
        batch_size=optimized_params['batch_size'],
        hidden_dim=optimized_params['hidden_dim'],
        num_gnn_layers=optimized_params['num_gnn_layers'],
        dropout_p=optimized_params['dropout'],
        epochs=optimized_params['epochs'],
        patience=optimized_params['patience'],
        visualize=not args.no_visualize and not args.save_plots,
        save_plots=args.save_plots,
        output_dir=args.output_dir
    )
    
    # Summarize and save results
    mean_mse, mean_rmse, mean_mape = summarize_loocv_results(loocv_results)
    
    if not args.no_visualize:
        plot_loocv_predictions(
            loocv_results,
            save_plots=args.save_plots,
            output_dir=args.output_dir
        )
    
    if args.save_plots:
        serializable_results = {
            key: {
                'mse': float(result['mse']),
                'rmse': float(result['rmse']),
                'mape': float(result['mape']),
                'model_path': result['model_path']
            }
            for key, result in loocv_results.items()
        }
        serializable_results['summary'] = {
            'mean_mse': float(mean_mse),
            'mean_rmse': float(mean_rmse),
            'mean_mape': float(mean_mape)
        }
        
        with open(os.path.join(args.output_dir, 'loocv_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    print("\nLOOCV completed with optimized parameters!")
    print(f"Average MSE: {mean_mse:.4f}")
    print(f"Average RMSE: {mean_rmse:.4f}")
    print(f"Average MAPE: {mean_mape:.2f}%")
    
    if args.save_plots:
        print(f"\nResults saved to: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)