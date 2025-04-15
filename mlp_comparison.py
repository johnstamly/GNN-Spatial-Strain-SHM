#!/usr/bin/env python3
"""
Script to run MLP comparison study against the best GNN model (GENConv with edges)
"""

import os
import argparse
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gnn_utils import (
    setup_matplotlib_style,
    load_data,
    preprocess_data,
    identify_target_indexes,
    truncate_data,
    run_loocv_utility,
    summarize_loocv_results,
    plot_loocv_predictions
)

class MLPModel(nn.Module):
    """
    MLP model with comparable parameter count to GENConv GNN
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 4, dropout_p: float = 0.3):
        """
        Args:
            input_dim: Dimension of input features (16 HI values)
            hidden_dim: Dimension of hidden layers (matches GNN)
            num_layers: Number of hidden layers (adjusted for parameter count)
            dropout_p: Dropout probability (matches GNN)
        """
        super().__init__()
        
        layers = []
        in_features = input_dim
        out_features = hidden_dim
        
        # Input layer
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(p=dropout_p))
        
        # Hidden layers (adjusted to match GNN parameter count)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(out_features, out_features))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=dropout_p))
        
        # Output layer
        layers.append(nn.Linear(out_features, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def loss(self, pred, true):
        """Mean Squared Error loss (matches GNN)"""
        return F.mse_loss(pred, true)

def prepare_mlp_data(stiffness_data, strain_data):
    """
    Prepare data for MLP by flattening node features
    Returns:
        List of flattened strain features (num_samples x num_nodes*num_features)
        List of corresponding stiffness values
        List of specimen keys
    """
    strain_features = []
    stiffness_values = []
    specimen_keys = []
    
    for key in strain_data.keys():
        # Flatten node features (assuming shape [num_nodes, num_features])
        flattened = strain_data[key].view(-1).unsqueeze(0)  # [1, num_nodes*num_features]
        strain_features.append(flattened)
        stiffness_values.append(stiffness_data[key])
        specimen_keys.append(key)
    
    return strain_features, stiffness_values, specimen_keys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run MLP comparison study')
    
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
    parser.add_argument('--output-dir', type=str, default='mlp_comparison_results',
                        help='Directory to save results and plots')
    
    return parser.parse_args()

def main():
    """Main function to run MLP comparison study."""
    args = parse_args()
    
    # Use same hyperparameters as GNN for fair comparison
    model_params = {
        'input_dim': 16,  # Flattened 16 HI values
        'hidden_dim': 64,  # Matches GNN
        'num_layers': 4,   # Adjusted to match parameter count
        'dropout_p': 0.3   # Matches GNN
    }
    
    # Configure matplotlib
    setup_matplotlib_style()
    
    # Create directories
    os.makedirs('mlp_models', exist_ok=True)
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
    
    print("\nPreparing data for MLP...")
    strain_data_list, stiffness_data_list, specimen_keys = prepare_mlp_data(
        stiffness_post_trunc, strain_post_trunc
    )
    
    # Run LOOCV with MLP model
    print("\nRunning LOOCV with MLP model...")
    loocv_results = run_loocv_utility(
        strain_data_list,
        stiffness_data_list,
        specimen_keys,
        model_class=MLPModel,
        model_params=model_params,
        batch_size=64,      # Matches GNN
        epochs=1000,        # Matches GNN
        patience=20,        # Matches GNN
        visualize=not args.no_visualize and not args.save_plots,
        save_plots=args.save_plots,
        output_dir=args.output_dir,
        model_save_dir='mlp_models'
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
        
        with open(os.path.join(args.output_dir, 'mlp_loocv_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    print("\nMLP LOOCV completed!")
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