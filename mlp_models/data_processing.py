"""
Data processing utilities for MLP stiffness prediction.
Adapts strain data to be used directly with MLP models without graph structure.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any


class StiffnessDataset(Dataset):
    """
    Dataset for MLP stiffness prediction.
    """
    def __init__(self, features, targets, normalize=True):
        """
        Initialize the dataset.
        
        Args:
            features: Strain data features (shape: n_samples, n_sensors)
            targets: Stiffness data targets (shape: n_samples, 1)
            normalize: Whether to normalize the features
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def prepare_mlp_data(strain_data_list, stiffness_data_list, specimen_keys):
    """
    Prepare data for MLP input by reshaping strain data.
    
    Args:
        strain_data_list: List of strain data arrays from the GNN processing
        stiffness_data_list: List of stiffness data arrays from the GNN processing
        specimen_keys: List of specimen keys
        
    Returns:
        Tuple of (processed_strain_data, processed_stiffness_data, specimen_keys)
    """
    processed_strain_data = []
    processed_stiffness_data = []
    processed_keys = []
    
    for idx, (strain, stiffness, key) in enumerate(zip(strain_data_list, stiffness_data_list, specimen_keys)):
        # Ensure the data is not empty
        if strain.size == 0 or stiffness.size == 0:
            print(f"Skipping {key} due to empty data")
            continue
            
        # For MLP, we need each sample to have shape (16,) for the 16 sensors
        # The strain data from GNN processing has shape (n_samples, 16)
        # We can use it directly
        processed_strain_data.append(strain)
        processed_stiffness_data.append(stiffness)
        processed_keys.append(key)
        
        print(f"Prepared {key}: Strain shape {strain.shape}, Stiffness shape {stiffness.shape}")
    
    return processed_strain_data, processed_stiffness_data, processed_keys


def compute_normalization_params(train_inputs, train_targets):
    """
    Compute normalization parameters from training data.
    
    Args:
        train_inputs: List of training input arrays
        train_targets: List of training target arrays
        
    Returns:
        Dictionary of normalization parameters
    """
    # Concatenate all training data
    all_inputs = np.vstack([x for x in train_inputs if x.size > 0])
    all_targets = np.vstack([y for y in train_targets if y.size > 0])
    
    # Compute normalization parameters
    input_mean = np.mean(all_inputs)
    input_std = np.std(all_inputs)
    target_min = np.min(all_targets)
    target_max = np.max(all_targets)
    target_range = target_max - target_min
    
    # Avoid division by zero
    if input_std == 0:
        input_std = 1.0
    if target_range == 0:
        target_range = 1.0
    
    return {
        'input_mean': input_mean,
        'input_std': input_std,
        'target_min': target_min,
        'target_max': target_max,
        'target_range': target_range
    }


def normalize_data(strain_data, stiffness_data, norm_params):
    """
    Normalize data using the provided parameters.
    
    Args:
        strain_data: List of strain data arrays
        stiffness_data: List of stiffness data arrays
        norm_params: Dictionary of normalization parameters
        
    Returns:
        Tuple of (normalized_strain_data, normalized_stiffness_data)
    """
    normalized_strain_data = []
    normalized_stiffness_data = []
    
    for strain, stiffness in zip(strain_data, stiffness_data):
        # Normalize strain data
        norm_strain = (strain - norm_params['input_mean']) / norm_params['input_std']
        
        # Normalize stiffness data
        norm_stiffness = (stiffness - norm_params['target_min']) / norm_params['target_range']
        
        normalized_strain_data.append(norm_strain)
        normalized_stiffness_data.append(norm_stiffness)
    
    return normalized_strain_data, normalized_stiffness_data


def unnormalize_target(normalized_values, target_min, target_range):
    """
    Unnormalize the target values.
    
    Args:
        normalized_values: Normalized target values
        target_min: Minimum value of unnormalized targets
        target_range: Range of unnormalized targets
        
    Returns:
        Unnormalized target values
    """
    return normalized_values * target_range + target_min


def create_dataloaders(train_features, train_targets, val_features, val_targets, batch_size=128):
    """
    Create PyTorch DataLoaders for training and validation data.
    
    Args:
        train_features: Training features
        train_targets: Training targets
        val_features: Validation features
        val_targets: Validation targets
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = StiffnessDataset(train_features, train_targets)
    val_dataset = StiffnessDataset(val_features, val_targets)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader