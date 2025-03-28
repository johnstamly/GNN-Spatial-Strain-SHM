#!/usr/bin/env python3
"""
Main script to run Leave-One-Out Cross-Validation (LOOCV) for stiffness prediction using GNN.

This script loads data, preprocesses it, and runs LOOCV to evaluate the model's performance.
"""

import os
import argparse
import sys

# Set matplotlib backend to non-interactive 'Agg' to avoid GUI issues
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

# Import utility functions
from gnn_utils import (
    setup_matplotlib_style,
    load_data,
    preprocess_data,
    identify_target_indexes,
    truncate_data,
    prepare_gnn_data,
    run_loocv,
    summarize_loocv_results,
    plot_loocv_predictions
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run LOOCV for stiffness prediction using GNN')
    
    # Data paths
    parser.add_argument('--stiffness-path', type=str, default='Data/Stiffness_Reduction',
                        help='Path to stiffness data directory')
    parser.add_argument('--strain-path', type=str, default='Data/Strain',
                        help='Path to strain data directory')
    
    # Model parameters
    parser.add_argument('--num-nodes', type=int, default=16,
                        help='Number of nodes in each graph')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension for the GNN model')
    parser.add_argument('--num-gnn-layers', type=int, default=4,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    
    # Other parameters
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
    """Main function to run LOOCV."""
    # Parse arguments
    args = parse_args()
    
    # Configure matplotlib
    setup_matplotlib_style()
    
    # Create directories if they don't exist
    os.makedirs('best_model', exist_ok=True)
    os.makedirs('log', exist_ok=True)
    if args.save_plots:
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
    
    # Run LOOCV
    print("\nRunning LOOCV...")
    loocv_results = run_loocv(
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
        visualize=not args.no_visualize and not args.save_plots,
        save_plots=args.save_plots,
        output_dir=args.output_dir
    )
    
    # Summarize results
    mean_mse, mean_rmse, mean_mape = summarize_loocv_results(loocv_results)
    
    # Plot predictions if not disabled
    if not args.no_visualize:
        plot_loocv_predictions(
            loocv_results,
            save_plots=args.save_plots,
            output_dir=args.output_dir
        )
    
    # Save results to file
    if args.save_plots:
        import json
        
        # Convert numpy arrays to lists for JSON serialization
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
            'mean_mse': float(mean_mse),
            'mean_rmse': float(mean_rmse),
            'mean_mape': float(mean_mape)
        }
        
        # Save to JSON file
        with open(os.path.join(args.output_dir, 'loocv_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    print("\nLOOCV completed successfully!")
    print(f"Average MSE: {mean_mse:.4f}")
    print(f"Average RMSE: {mean_rmse:.4f}")
    print(f"Average MAPE: {mean_mape:.2f}%")
    
    if args.save_plots:
        print(f"\nResults and plots saved to: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)