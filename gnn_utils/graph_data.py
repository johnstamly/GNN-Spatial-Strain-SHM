"""
Utilities for preparing graph data for the GNN model.
"""

import torch
import numpy as np
from torch_geometric.data import Data, DataLoader


def normalize_input(x, input_mean, input_std):
    """Normalize input features using mean and standard deviation."""
    return (x - input_mean) / input_std


def normalize_target(y, target_min, target_range):
    """Normalize target values to [0, 1] range."""
    return (y - target_min) / target_range


def unnormalize_target(y_norm, target_min, target_range):
    """Convert normalized target values back to original scale."""
    return y_norm * target_range + target_min


def compute_normalization_params(train_inputs, train_targets):
    """Compute normalization parameters from training data."""
    # Flatten inputs and compute mean/std
    train_inputs_flat = torch.cat([torch.tensor(x, dtype=torch.float).flatten() for x in train_inputs])
    input_mean = train_inputs_flat.mean().item()
    input_std = train_inputs_flat.std().item()
    # Add small epsilon to std to prevent division by zero if std is very small
    input_std = input_std if input_std > 1e-6 else 1.0
    
    # Flatten targets and compute min/max
    train_targets_flat = torch.cat([torch.tensor(y, dtype=torch.float).flatten() for y in train_targets])
    target_min = train_targets_flat.min().item()
    target_max = train_targets_flat.max().item()
    target_range = target_max - target_min
    # Add small epsilon to range if min and max are the same
    target_range = target_range if target_range > 1e-6 else 1.0
    
    return {
        'input_mean': input_mean,
        'input_std': input_std,
        'target_min': target_min,
        'target_max': target_max,
        'target_range': target_range
    }


def create_fully_connected_edge_index(num_nodes):
    """Create edge indices for a fully connected graph."""
    edge_list = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def create_graph_data_objects(strain_data, stiffness_data, norm_params, num_nodes=16):
    """
    Create PyTorch Geometric Data objects from strain and stiffness data.
    
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
        hi_tensor = normalize_input(
            torch.tensor(strain_data[i], dtype=torch.float),
            norm_params['input_mean'],
            norm_params['input_std']
        )
        
        stiffness_tensor = normalize_target(
            torch.tensor(stiffness_data[i], dtype=torch.float),
            norm_params['target_min'],
            norm_params['target_range']
        )
        
        data_list_for_specimen = []
        for t in range(hi_tensor.shape[0]):  # Iterate over timesteps
            # Node features: HI values for each sensor at timestep t
            x = hi_tensor[t].reshape(num_nodes, -1)  # Shape: [NUM_NODES, num_node_features=1]
            if x.shape[0] != num_nodes:
                print(f"Warning: Specimen {i}, Timestep {t}: Node feature shape mismatch ({x.shape[0]} vs {num_nodes}). Skipping timestep.")
                continue
                
            # Edge attributes: Difference in HI between connected nodes
            row, col = edge_index_static
            edge_attr = x[row] - x[col]  # Shape: [num_edges, num_node_features=1]
            
            # Target: Graph-level stiffness value at timestep t
            # Assuming stiffness_tensor is [num_timesteps, 1]
            y = stiffness_tensor[t].reshape(1, 1)  # Shape: [1, 1]
            
            # Create Data object
            data = Data(x=x, edge_index=edge_index_static, edge_attr=edge_attr, y=y)
            data_list_for_specimen.append(data)
            
        specimen_graph_data.append(data_list_for_specimen)
        
    return specimen_graph_data


def prepare_data_loaders(train_data, val_data, test_data, batch_size=128):
    """Create DataLoader objects for training, validation, and testing."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader