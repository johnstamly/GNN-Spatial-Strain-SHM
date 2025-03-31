#!/usr/bin/env python3
"""
Script to compare different GNN model architectures for stiffness prediction without using edge attributes.

This script runs Leave-One-Out Cross-Validation (LOOCV) for multiple GNN model architectures
that only use node features (strain health indicators) without edge attributes.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torch_geometric.data import Data

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
    summarize_loocv_results,
    compute_normalization_params,
    create_fully_connected_edge_index
)

# Import model variants without edge attributes
from gnn_utils.model_variants_no_edges import (
    GCNModel_NoEdges,
    GINModel,
    SGConvModel,
    GraphSAGEModel,
    ChebConvModel,
    count_parameters
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare different GNN model architectures without edge attributes')
    
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
    parser.add_argument('--output-dir', type=str, default='model_comparison_no_edges',
                        help='Directory to save results and plots')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['GCN_NoEdges', 'GIN', 'SGConv', 'GraphSAGE', 'ChebConv'],
                        help='List of models to compare')
    
    return parser.parse_args()


def create_model(model_name, num_node_features, edge_feature_dim, hidden_dim, 
                output_dim, num_gnn_layers, dropout_p):
    """Create a model instance based on the model name."""
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_graph_data_objects_no_edges(strain_data, stiffness_data, norm_params, num_nodes=16):
    """
    Create PyTorch Geometric Data objects from strain and stiffness data without edge attributes.
    
    Args:
        strain_data: List of strain data arrays for each specimen
        stiffness_data: List of stiffness data arrays for each specimen
        norm_params: Dictionary of normalization parameters
        num_nodes: Number of nodes in each graph (default: 16 for 16 sensors)
        
    Returns:
        List of lists of Data objects, where each inner list contains the graph data
        for all timesteps of a specimen
    """
    specimen_graph_data = []
    
    # Precompute edge_index for a fully connected graph
    edge_index_static = create_fully_connected_edge_index(num_nodes)
    
    for i in range(len(strain_data)):
        # Normalize input (strain/HI) and target (stiffness)
        hi_tensor = (torch.tensor(strain_data[i], dtype=torch.float) - norm_params['input_mean']) / norm_params['input_std']
        
        stiffness_tensor = (torch.tensor(stiffness_data[i], dtype=torch.float) - norm_params['target_min']) / norm_params['target_range']
        
        data_list_for_specimen = []
        for t in range(hi_tensor.shape[0]):  # Iterate over timesteps
            # Node features: HI values for each sensor at timestep t
            x = hi_tensor[t].reshape(num_nodes, -1)  # Shape: [NUM_NODES, num_node_features=1]
            if x.shape[0] != num_nodes:
                print(f"Warning: Specimen {i}, Timestep {t}: Node feature shape mismatch ({x.shape[0]} vs {num_nodes}). Skipping timestep.")
                continue
                
            # Target: Graph-level stiffness value at timestep t
            # Assuming stiffness_tensor is [num_timesteps, 1]
            y = stiffness_tensor[t].reshape(1, 1)  # Shape: [1, 1]
            
            # Create Data object WITHOUT edge attributes
            data = Data(x=x, edge_index=edge_index_static, y=y)
            data_list_for_specimen.append(data)
            
        specimen_graph_data.append(data_list_for_specimen)
        
    return specimen_graph_data


def run_loocv_for_model(model_name, strain_data_list, stiffness_data_list, specimen_keys, args):
    """Run LOOCV for a specific model."""
    print(f"\n{'='*80}")
    print(f"Running LOOCV for {model_name} model (without edge attributes)")
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
    
    # Define training indices (all except validation)
    results = {}
    
    # Run LOOCV
    for val_idx in range(len(specimen_keys)):
        val_key = specimen_keys[val_idx]
        print(f"\n{'='*50}\nFold {val_idx+1}/{len(specimen_keys)}: Validation on {val_key}\n{'='*50}")
        
        # Define training indices (all except validation)
        train_indices = [i for i in range(len(specimen_keys)) if i != val_idx]
        
        # Get training data
        train_inputs = [strain_data_list[i] for i in train_indices]
        train_targets = [stiffness_data_list[i] for i in train_indices]
        
        # Compute normalization parameters from training data only
        norm_params = compute_normalization_params(train_inputs, train_targets)
        
        print("\nNormalization Parameters (from Training Data):")
        print(f"  Input Mean: {norm_params['input_mean']:.4f}")
        print(f"  Input Std:  {norm_params['input_std']:.4f}")
        print(f"  Target Min: {norm_params['target_min']:.4f}")
        print(f"  Target Max: {norm_params['target_max']:.4f}")
        
        # Create Graph Data Objects WITHOUT edge attributes
        specimen_graph_data = create_graph_data_objects_no_edges(
            strain_data_list, 
            stiffness_data_list, 
            norm_params, 
            args.num_nodes
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_creator(
            num_node_features=1,  # Assuming HI is a single feature per node
            edge_feature_dim=0,   # No edge features
            hidden_dim=args.hidden_dim,
            output_dim=1,         # Predicting a single stiffness value
            num_gnn_layers=args.num_gnn_layers,
            dropout_p=args.dropout
        ).to(device)
        
        # Optimizer and Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
        
        # Data Loaders
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        
        # TensorBoard Writer
        from tensorboardX import SummaryWriter
        log_dir = os.path.join("log_no_edges", f"fold_{val_idx+1}_{val_key}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        writer = SummaryWriter(log_dir)
        
        # Train model
        print(f"\nTraining model for fold {val_idx+1}/{len(specimen_keys)}...")
        model_save_path = f"{model_dir}/fold_{val_idx+1}_{val_key}_model_state.pth"
        
        from gnn_utils import run_training, run_inference, unnormalize_target
        model, train_losses, val_losses = run_training(
            model, train_loader, val_loader, device, writer, optimizer, scheduler,
            epochs=args.epochs, patience=args.patience, model_save_path=model_save_path
        )
        
        # Plot Training History
        fig = plt.figure(figsize=(10, 6))
        plt.semilogy(train_losses, label='Training Loss')
        plt.semilogy(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'Fold {val_idx+1}: Training and Validation Loss History')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(model_dir, f"fold_{val_idx+1}_{val_key}_loss.png"), bbox_inches='tight')
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
            'mse': float(val_mse),
            'rmse': float(val_rmse),
            'mape': float(val_mape),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_path': model_save_path,
            'norm_params': norm_params
        }
        
        # Close TensorBoard writer
        writer.close()
    
    # Summarize results
    mean_mse, mean_rmse, mean_mape = summarize_loocv_results(results)
    
    # Save results to file
    serializable_results = {}
    for key, result in results.items():
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
    plt.title('Model Performance Comparison (No Edge Attributes)', fontsize=16, pad=20)
    
    # Save table
    plt.savefig(os.path.join(output_dir, 'model_comparison_table.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def main():
    """Main function to compare different GNN model architectures without edge attributes."""
    # Parse arguments
    args = parse_args()
    
    # Configure matplotlib
    setup_matplotlib_style()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("log_no_edges", exist_ok=True)
    
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
            edge_feature_dim=0,  # No edge features
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