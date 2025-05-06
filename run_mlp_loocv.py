#!/usr/bin/env python3
"""
Main script to run Leave-One-Out Cross-Validation (LOOCV) for stiffness prediction using MLP.

This script loads data, preprocesses it, and runs LOOCV to evaluate the MLP model's performance
as a baseline for comparison with GNN models.
"""

import os
import argparse
import sys
import json
import numpy as np
import torch
import torch.optim as optim

# Set matplotlib backend to non-interactive 'Agg' to avoid GUI issues
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

# Import GNN utility functions for data loading and preprocessing
from gnn_utils import (
    setup_matplotlib_style,
    load_data,
    preprocess_data,
    identify_target_indexes,
    truncate_data,
    prepare_gnn_data
)

# Import MLP-specific utilities
from mlp_models.mlp_model import MLPModel
from mlp_models.data_processing import (
    prepare_mlp_data,
    compute_normalization_params,
    normalize_data,
    unnormalize_target,
    create_dataloaders
)
from mlp_models.training import (
    run_training,
    run_inference,
    plot_predictions,
    plot_scatter,
    plot_residuals
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run LOOCV for stiffness prediction using MLP')
    
    # Data paths
    parser.add_argument('--stiffness-path', type=str, default='Data/Stiffness_Reduction',
                        help='Path to stiffness data directory')
    parser.add_argument('--strain-path', type=str, default='Data/Strain',
                        help='Path to strain data directory')
    
    # Model parameters
    parser.add_argument('--hidden-dims', type=str, default='64,32',
                        help='Comma-separated hidden layer dimensions')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    
    # Other parameters
    parser.add_argument('--drop-level', type=int, default=85,
                        help='Stiffness reduction level for truncation')
    parser.add_argument('--save-plots', action='store_true', default=True,
                        help='Save plots to files instead of displaying them')
    parser.add_argument('--output-dir', type=str, default='mlp_models/results',
                        help='Directory to save results and plots')
    
    return parser.parse_args()


def run_mlp_loocv(strain_data, stiffness_data, specimen_keys, args):
    """
    Run Leave-One-Out Cross-Validation (LOOCV) for the MLP model.
    
    Args:
        strain_data: List of strain data arrays for each specimen
        stiffness_data: List of stiffness data arrays for each specimen
        specimen_keys: List of specimen keys (e.g., 'df1', 'df2', etc.)
        args: Command line arguments
        
    Returns:
        Dictionary of results for each fold
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if saving plots
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    results = {}
    
    # Run LOOCV
    for val_idx in range(len(specimen_keys)):
        val_key = specimen_keys[val_idx]
        print(f"\n{'='*50}\nFold {val_idx+1}/{len(specimen_keys)}: Validation on {val_key}\n{'='*50}")
        
        # Define training indices (all except validation)
        train_indices = [i for i in range(len(specimen_keys)) if i != val_idx]
        
        # Get training data
        train_inputs = [strain_data[i] for i in train_indices]
        train_targets = [stiffness_data[i] for i in train_indices]
        
        # Compute normalization parameters from training data only
        norm_params = compute_normalization_params(train_inputs, train_targets)
        
        print("\nNormalization Parameters (from Training Data):")
        print(f"  Input Mean: {norm_params['input_mean']:.4f}")
        print(f"  Input Std:  {norm_params['input_std']:.4f}")
        print(f"  Target Min: {norm_params['target_min']:.4f}")
        print(f"  Target Max: {norm_params['target_max']:.4f}")
        
        # Normalize all data
        norm_strain_data, norm_stiffness_data = normalize_data(
            strain_data, stiffness_data, norm_params
        )
        
        # Split Data into Train and Validation
        train_features = np.vstack([norm_strain_data[i] for i in train_indices])
        train_targets = np.vstack([norm_stiffness_data[i] for i in train_indices])
        
        val_features = norm_strain_data[val_idx]
        val_targets = norm_stiffness_data[val_idx]
        
        print(f"\nData Split:")
        print(f"  Training samples: {len(train_features)} (from {', '.join([specimen_keys[i] for i in train_indices])})")
        print(f"  Validation samples: {len(val_features)} (from {val_key})")
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_features, train_targets, 
            val_features, val_targets,
            batch_size=args.batch_size
        )
        
        # Create MLP model
        input_dim = train_features.shape[1]  # Should be 16 for the 16 sensors
        model = MLPModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout_p=args.dropout
        )
        model = model.to(device)
        
        # Optimizer and Scheduler
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=10
        )
        
        # TensorBoard Writer
        from tensorboardX import SummaryWriter
        from datetime import datetime
        log_dir = os.path.join("log", "mlp", f"fold_{val_idx+1}_{val_key}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        writer = SummaryWriter(log_dir)
        
        # Train model
        print(f"\nTraining model for fold {val_idx+1}/{len(specimen_keys)}...")
        fold_output_dir = os.path.join(args.output_dir, f"fold_{val_idx+1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        model_save_path = os.path.join(fold_output_dir, f"{val_key}_model_state.pth")
        model, train_losses, val_losses = run_training(
            model, train_loader, val_loader, device, writer, optimizer, scheduler,
            epochs=args.epochs, patience=args.patience, model_save_path=model_save_path
        )
        
        # Plot Training History
        plt.figure(figsize=(10, 6))
        plt.semilogy(train_losses, label='Training Loss')
        plt.semilogy(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'Fold {val_idx+1}: Training and Validation Loss History')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        
        loss_plot_path = os.path.join(fold_output_dir, f"{val_key}_loss.png")
        plt.savefig(loss_plot_path, bbox_inches='tight')
        plt.close()
        
        # Evaluate model
        print(f"\nEvaluating model on validation set ({val_key})...")
        unnormalize_fn = lambda x: unnormalize_target(x, norm_params['target_min'], norm_params['target_range'])
        true_val, pred_val, val_mse, val_rmse, val_mape = run_inference(
            model, val_loader, device, unnormalize_fn
        )
        
        # Generate plots
        time_series_title = f"Fold {val_idx+1}: True vs. Predicted Stiffness ({val_key})"
        time_series_path = os.path.join(fold_output_dir, f"{val_key}_timeseries.png")
        plot_predictions(true_val, pred_val, val_mse, val_rmse, val_mape, 
                        time_series_title, save_path=time_series_path)
        
        scatter_title = f"Fold {val_idx+1}: Scatter Plot ({val_key})"
        scatter_path = os.path.join(fold_output_dir, f"{val_key}_scatter.png")
        plot_scatter(true_val, pred_val, val_mse, val_rmse, val_mape, 
                    scatter_title, save_path=scatter_path)
        
        residuals_title = f"Fold {val_idx+1}: Residuals Plot ({val_key})"
        residuals_path = os.path.join(fold_output_dir, f"{val_key}_residuals.png")
        plot_residuals(true_val, pred_val, val_mse, val_rmse, val_mape, 
                      residuals_title, save_path=residuals_path)
        
        # Store results
        results[val_key] = {
            'true_values': true_val.tolist(),
            'predicted_values': pred_val.tolist(),
            'mse': float(val_mse),
            'rmse': float(val_rmse),
            'mape': float(val_mape),
            'train_losses': [float(loss) for loss in train_losses],
            'val_losses': [float(loss) for loss in val_losses],
            'model_path': model_save_path
        }
        
        # Close TensorBoard writer
        writer.close()
    
    return results


def summarize_loocv_results(loocv_results):
    """
    Summarize LOOCV results.
    
    Args:
        loocv_results: Dictionary of LOOCV results
        
    Returns:
        Tuple of (mean_mse, mean_rmse, mean_mape)
    """
    print("\nLOOCV Results Summary:")
    print("=======================")
    print(f"{'Fold':<10} {'MSE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 40)
    
    mse_values = []
    rmse_values = []
    mape_values = []
    
    for key, result in loocv_results.items():
        mse = result['mse']
        rmse = result['rmse']
        mape = result['mape']
        
        mse_values.append(mse)
        rmse_values.append(rmse)
        mape_values.append(mape)
        
        print(f"{key:<10} {mse:<10.4f} {rmse:<10.4f} {mape:<10.2f}%")
    
    print("-" * 40)
    mean_mse = np.mean(mse_values)
    mean_rmse = np.mean(rmse_values)
    mean_mape = np.mean(mape_values)
    std_mse = np.std(mse_values)
    std_rmse = np.std(rmse_values)
    std_mape = np.std(mape_values)
    
    print(f"{'Average':<10} {mean_mse:<10.4f} {mean_rmse:<10.4f} {mean_mape:<10.2f}%")
    print(f"{'Std Dev':<10} {std_mse:<10.4f} {std_rmse:<10.4f} {std_mape:<10.2f}%")
    
    return mean_mse, mean_rmse, mean_mape


def plot_loocv_predictions(loocv_results, output_dir):
    """
    Plot predictions for each fold.
    
    Args:
        loocv_results: Dictionary of LOOCV results
        output_dir: Directory to save plots
    """
    fig = plt.figure(figsize=(15, 10))
    
    for i, (key, result) in enumerate(loocv_results.items()):
        plt.subplot(2, 2, i+1)
        
        true_values = np.array(result['true_values'])
        pred_values = np.array(result['predicted_values'])
        
        x_values = np.arange(len(true_values))
        plt.plot(x_values, true_values, label="True Values", color="blue", marker='o', linestyle='-', markersize=1, alpha=0.7)
        plt.plot(x_values, pred_values, label="Predicted Values", color="red", marker='x', linestyle='--', markersize=1, alpha=0.7)
        
        # Metrics text box
        metrics_text = (
            f"MSE:  {result['mse']:.2f}\n"
            f"RMSE: {result['rmse']:.2f}\n"
            f"MAPE: {result['mape']:.2f}%"
        )
        plt.annotate(metrics_text, xy=(0.97, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=0.5, alpha=0.8),
                     ha='right', va='top', fontsize=10, family='monospace')
        
        plt.xlabel("Time Steps")
        plt.ylabel("Stiffness (%)")
        plt.title(f"Fold: {key}")
        plt.legend()
        plt.grid(True)
     
    plt.tight_layout()
    plt.suptitle("LOOCV: True vs. Predicted Stiffness for Each Fold (MLP Model)", fontsize=16, y=1.02)
    
    plt.savefig(os.path.join(output_dir, "loocv_predictions.png"), bbox_inches='tight')
    plt.close(fig)


def main():
    """Main function to run MLP LOOCV."""
    # Parse arguments
    args = parse_args()
    
    # Configure matplotlib
    setup_matplotlib_style()
    
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("log/mlp", exist_ok=True)
    
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
    print("\nPreparing data for GNN (intermediate step)...")
    strain_data_list, stiffness_data_list, specimen_keys = prepare_gnn_data(
        stiffness_post_trunc, strain_post_trunc
    )
    
    # Convert GNN data to MLP format
    print("\nConverting data format for MLP...")
    strain_data_mlp, stiffness_data_mlp, specimen_keys_mlp = prepare_mlp_data(
        strain_data_list, stiffness_data_list, specimen_keys
    )
    
    # Run LOOCV with MLP model
    print("\nRunning LOOCV with MLP model...")
    loocv_results = run_mlp_loocv(
        strain_data_mlp,
        stiffness_data_mlp,
        specimen_keys_mlp,
        args
    )
    
    # Summarize results
    mean_mse, mean_rmse, mean_mape = summarize_loocv_results(loocv_results)
    
    # Plot predictions
    plot_loocv_predictions(
        loocv_results,
        args.output_dir
    )
    
    # Save results to file
    serializable_results = loocv_results.copy()
    
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
    print(f"\nResults and plots saved to: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)