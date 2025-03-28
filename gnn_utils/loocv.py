"""
Leave-One-Out Cross-Validation (LOOCV) utilities for stiffness prediction GNN.
"""

import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Callable, Optional
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter

from gnn_utils import EdgeAttrGNN, create_graph_data_objects, run_training, run_inference, unnormalize_target, compute_normalization_params


def run_loocv(strain_data: List[np.ndarray], 
              stiffness_data: List[np.ndarray], 
              specimen_keys: List[str], 
              num_nodes: int = 16, 
              batch_size: int = 128, 
              hidden_dim: int = 64, 
              num_gnn_layers: int = 4, 
              dropout_p: float = 0.5, 
              epochs: int = 1000, 
              patience: int = 50,
              visualize: bool = True,
              save_plots: bool = False,
              output_dir: str = "results") -> Dict[str, Dict[str, Any]]:
    """
    Run Leave-One-Out Cross-Validation (LOOCV) for the GNN model.
    
    Args:
        strain_data: List of strain data arrays for each specimen
        stiffness_data: List of stiffness data arrays for each specimen
        specimen_keys: List of specimen keys (e.g., 'df1', 'df2', etc.)
        num_nodes: Number of nodes in each graph
        batch_size: Batch size for training
        hidden_dim: Hidden dimension for the GNN model
        num_gnn_layers: Number of GNN layers
        dropout_p: Dropout probability
        epochs: Maximum number of epochs
        patience: Patience for early stopping
        visualize: Whether to visualize training progress
        save_plots: Whether to save plots to files instead of displaying them
        output_dir: Directory to save plots if save_plots is True
        
    Returns:
        Dictionary of results for each fold
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if saving plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
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
        
        # Create Graph Data Objects
        specimen_graph_data = create_graph_data_objects(
            strain_data, 
            stiffness_data, 
            norm_params, 
            num_nodes
        )
        
        # Split Data into Train and Validation
        train_data = []
        for i in train_indices:
            train_data.extend(specimen_graph_data[i])
            
        val_data = specimen_graph_data[val_idx]
        
        print(f"\nData Split:")
        print(f"  Training samples: {len(train_data)} (from {', '.join([specimen_keys[i] for i in train_indices])})")
        print(f"  Validation samples: {len(val_data)} (from {val_key})")
        
        # Create model
        model = EdgeAttrGNN(
            num_node_features=1,  # Assuming HI is a single feature per node
            edge_feature_dim=1,   # Assuming HI_i - HI_j is a single feature per edge
            hidden_dim=hidden_dim,
            output_dim=1,         # Predicting a single stiffness value
            num_gnn_layers=num_gnn_layers,
            dropout_p=dropout_p
        ).to(device)
        
        # Optimizer and Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-8)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)
        
        # Data Loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # TensorBoard Writer
        log_dir = os.path.join("log", f"fold_{val_idx+1}_{val_key}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        writer = SummaryWriter(log_dir)
        
        # Train model
        print(f"\nTraining model for fold {val_idx+1}/{len(specimen_keys)}...")
        model_save_path = f"best_model/fold_{val_idx+1}_{val_key}_model_state.pth"
        model, train_losses, val_losses = run_training(
            model, train_loader, val_loader, optimizer, scheduler, device, writer,
            epochs=epochs, patience=patience, model_save_path=model_save_path
        )
        
        # Plot Training History if visualize is True
        if visualize or save_plots:
            fig = plt.figure(figsize=(10, 6))
            plt.semilogy(train_losses, label='Training Loss')
            plt.semilogy(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)')
            plt.title(f'Fold {val_idx+1}: Training and Validation Loss History')
            plt.legend()
            plt.grid(True, which="both", ls="--")
            
            if save_plots:
                plt.savefig(os.path.join(output_dir, f"fold_{val_idx+1}_{val_key}_loss.png"), bbox_inches='tight')
                plt.close(fig)
            elif visualize:
                plt.show()
            else:
                plt.close(fig)
        
        # Evaluate model
        print(f"\nEvaluating model on validation set ({val_key})...")
        unnormalize_fn = lambda x: unnormalize_target(x, norm_params['target_min'], norm_params['target_range'])
        true_val, pred_val, val_mse, val_rmse, val_mape = run_inference(
            model, val_loader, device, unnormalize_fn
        )
        
        # Store results
        results[val_key] = {
            'true_values': true_val,
            'predicted_values': pred_val,
            'mse': val_mse,
            'rmse': val_rmse,
            'mape': val_mape,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_path': model_save_path,
            'norm_params': norm_params
        }
        
        # Close TensorBoard writer
        writer.close()
    
    return results


def summarize_loocv_results(loocv_results: Dict[str, Dict[str, Any]]) -> Tuple[float, float, float]:
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


def plot_loocv_predictions(loocv_results: Dict[str, Dict[str, Any]], 
                          save_plots: bool = False, 
                          output_dir: str = "results") -> None:
    """
    Plot predictions for each fold.
    
    Args:
        loocv_results: Dictionary of LOOCV results
        save_plots: Whether to save plots to files instead of displaying them
        output_dir: Directory to save plots if save_plots is True
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(15, 10))
    
    for i, (key, result) in enumerate(loocv_results.items()):
        plt.subplot(2, 2, i+1)
        
        true_values = result['true_values']
        pred_values = result['predicted_values']
        x_values = np.arange(len(true_values))
        
        plt.plot(x_values, true_values, label="True Values", color="blue", marker='o', linestyle='-', markersize=3, alpha=0.7)
        plt.plot(x_values, pred_values, label="Predicted Values", color="red", marker='x', linestyle='--', markersize=4, alpha=0.7)
        
        # Metrics text box
        metrics_text = (
            f"MSE:  {result['mse']:.2f}\n"
            f"RMSE: {result['rmse']:.2f}\n"
            f"MAPE: {result['mape']:.2f}%"
        )
        plt.annotate(metrics_text, xy=(0.97, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=0.5, alpha=0.8),
                     ha='right', va='top', fontsize=10, family='monospace')
        
        plt.xlabel("Timestep")
        plt.ylabel("Stiffness (%)")
        plt.title(f"Fold: {key}")
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=min(0, plt.ylim()[0]))  # Ensure y-axis starts at or below 0
        
    plt.tight_layout()
    plt.suptitle("LOOCV: True vs. Predicted Stiffness for Each Fold", fontsize=16, y=1.02)
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, "loocv_predictions.png"), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()