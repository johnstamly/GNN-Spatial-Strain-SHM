"""
Data processing utilities for stiffness prediction GNN.
Contains functions for data loading, preprocessing, target index identification, and data preparation.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from gnn_utils import percentage_change_from_max, resample_stiffness_to_match_strain, find_closest_index


def load_data(stiffness_path: str, strain_path: str) -> Tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame]]:
    """
    Load stiffness and strain data from HDF5 files.
    
    Args:
        stiffness_path: Path to the stiffness data directory
        strain_path: Path to the strain data directory
        
    Returns:
        Tuple of (stiffness_dfs, strain_dfs)
    """
    # Load Stiffness Data
    stiff_file_paths = sorted([f.path for f in os.scandir(stiffness_path) if f.path.endswith('.h5')])
    stiffness_dfs = {f'df{i}': pd.read_hdf(file_path)['Stiffness'] for i, file_path in enumerate(stiff_file_paths)}
    
    # Load Strain Data
    strain_file_paths = sorted([f.path for f in os.scandir(strain_path) if f.path.endswith('.h5')])
    strain_dfs = {f'df{i}': pd.read_hdf(file_path) for i, file_path in enumerate(strain_file_paths)}
    
    print(f"Loaded {len(stiffness_dfs)} stiffness datasets.")
    print(f"Loaded {len(strain_dfs)} strain datasets.")
    
    return stiffness_dfs, strain_dfs


def preprocess_data(stiffness_dfs: Dict[str, pd.Series], strain_dfs: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, int]]:
    """
    Preprocess stiffness and strain data.
    
    Args:
        stiffness_dfs: Dictionary of stiffness dataframes
        strain_dfs: Dictionary of strain dataframes
        
    Returns:
        Tuple of (stiffness_post, strain_post, last_cycle)
    """
    stiffness_post = {}
    strain_post = {}
    last_cycle = {}
    
    for key, strain_df in strain_dfs.items():
        print(f"Processing {key}...")
        
        # Store original length for cycle calculation later
        last_cycle[key] = len(stiffness_dfs[key])
        
        # --- Strain Processing ---
        # Handle specific case for df2 (remove last 8 columns)
        if key == 'df2':
            strain_df = strain_df.iloc[:, :-8]
            print(f"  Adjusted {key} strain shape: {strain_df.shape}")
            
        # Resample strain to 200s intervals and smooth
        strain_resampled = strain_df.resample("200s").mean().rolling(10, min_periods=1).mean()
        strain_resampled = strain_resampled.dropna() # Drop NaNs resulting from resampling/rolling
        
        # --- Custom Feature Engineering (Cumulative Absolute Difference) ---
        if len(strain_resampled) > 1:
            strain_temp = np.cumsum(np.abs(np.diff(strain_resampled.values, axis=0)), axis=0)
            strain_temp_df = pd.DataFrame(strain_temp, columns=strain_resampled.columns, index=strain_resampled.index[1:])
            strain_processed = strain_temp_df
        else:
            # Handle cases with 0 or 1 data point after resampling
            strain_processed = pd.DataFrame(columns=strain_resampled.columns, index=strain_resampled.index)
            
        # --- Stiffness Processing ---
        stiffness_df = stiffness_dfs[key].rolling(50, min_periods=1).mean()
        stiffness_df = stiffness_df.dropna()
        
        # Normalize stiffness using percentage change from max
        stiffness_normalized = percentage_change_from_max(stiffness_df)
        
        # --- Alignment ---
        # Resample stiffness to match the processed strain length
        if not strain_processed.empty and not stiffness_normalized.empty:
            stiffness_aligned = resample_stiffness_to_match_strain(strain_processed, stiffness_normalized)
            stiffness_aligned.index = strain_processed.index # Align index
        else:
            # Handle empty dataframes after processing
            stiffness_aligned = pd.DataFrame(columns=[0] if isinstance(stiffness_normalized, pd.Series) else stiffness_normalized.columns, 
                                          index=strain_processed.index)
            
        # Store processed data
        strain_post[key] = strain_processed
        stiffness_post[key] = pd.DataFrame(stiffness_aligned) # Ensure DataFrame
    
        print(f"  Finished {key}. Strain shape: {strain_post[key].shape}, Stiffness shape: {stiffness_post[key].shape}")
    
    print("\nPreprocessing complete.")
    return stiffness_post, strain_post, last_cycle


def identify_target_indexes(stiffness_post: Dict[str, pd.DataFrame]) -> Dict[str, Dict[int, int]]:
    """
    Identify target indexes for stiffness reduction levels.
    
    Args:
        stiffness_post: Dictionary of preprocessed stiffness dataframes
        
    Returns:
        Dictionary mapping keys to target indexes
    """
    target_indexes = {}
    
    for key, stiffness_df in stiffness_post.items():
        if stiffness_df.empty or len(stiffness_df.iloc[:, 0]) < 2:
            print(f"Skipping {key} due to insufficient data points ({len(stiffness_df)}).")
            target_indexes[key] = {99: 0, 95: 0, 90: 0, 85: 0} # Default or placeholder
            continue
            
        stiffness_values = stiffness_df.iloc[:, 0].values # Assuming stiffness is in the first column
        
        # Find the index closest to 99% stiffness
        closest_index_99 = find_closest_index(stiffness_values, 99)
        index_99 = closest_index_99
        
        # Filter values *after* the 99% index to find subsequent drops
        if index_99 + 1 < len(stiffness_values):
            filtered_values = stiffness_values[index_99 + 1:]
            offset = index_99 + 1
            
            index_95 = find_closest_index(filtered_values, 95) + offset if len(filtered_values) > 0 else index_99
            index_90 = find_closest_index(filtered_values, 90) + offset if len(filtered_values) > 0 else index_99
            index_85 = find_closest_index(filtered_values, 70) + offset if len(filtered_values) > 0 else index_99 # Original code used 70 for key 85
        else:
            # If 99% is the last point or beyond, subsequent indices are the same
            index_95 = index_90 = index_85 = index_99
            
        target_indexes[key] = {
            99: index_99,
            95: index_95,
            90: index_90,
            85: index_85 # Key '85' corresponds to finding ~70% stiffness
        }
    
    print("Target Indexes:")
    for key, indices in target_indexes.items():
        print(f"  {key}: {indices}")
        
    return target_indexes


def truncate_data(stiffness_post: Dict[str, pd.DataFrame], strain_post: Dict[str, pd.DataFrame], 
                 target_indexes: Dict[str, Dict[int, int]], drop_level: int = 85) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Truncate data at specific stiffness reduction level.
    
    Args:
        stiffness_post: Dictionary of preprocessed stiffness dataframes
        strain_post: Dictionary of preprocessed strain dataframes
        target_indexes: Dictionary of target indexes
        drop_level: Stiffness reduction level for truncation (default: 85)
        
    Returns:
        Tuple of (stiffness_post_trunc, strain_post_trunc)
    """
    stiffness_post_trunc = {}
    strain_post_trunc = {}
    
    print(f"Truncating data at index corresponding to level '{drop_level}' (~70% stiffness drop)...")
    
    for key, stiffness_df in stiffness_post.items():
        if key not in target_indexes or stiffness_df.empty:
            print(f"Skipping truncation for {key} (no target index or empty data).")
            stiffness_post_trunc[key] = stiffness_df
            strain_post_trunc[key] = strain_post.get(key, pd.DataFrame())
            continue
            
        cut_index = target_indexes[key][drop_level]
        cut_index = min(cut_index, len(stiffness_df) - 1)  # Ensure cut_index is within bounds
        
        if cut_index < 0:
            print(f"Warning: Negative cut_index ({cut_index}) for {key}. Skipping truncation.")
            stiffness_post_trunc[key] = stiffness_df
            strain_post_trunc[key] = strain_post.get(key, pd.DataFrame())
            continue
              
        print(f"  Truncating {key} at index {cut_index} (up to {cut_index + 1} data points).")
        
        # Truncate dataframes
        stiffness_post_trunc[key] = stiffness_df.iloc[:cut_index + 1]
        strain_post_trunc[key] = strain_post.get(key, pd.DataFrame()).iloc[:cut_index + 1] if key in strain_post else pd.DataFrame()
    
    return stiffness_post_trunc, strain_post_trunc


def prepare_gnn_data(stiffness_post_trunc: Dict[str, pd.DataFrame], strain_post_trunc: Dict[str, pd.DataFrame]) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Prepare data for GNN input.
    
    Args:
        stiffness_post_trunc: Dictionary of truncated stiffness dataframes
        strain_post_trunc: Dictionary of truncated strain dataframes
        
    Returns:
        Tuple of (strain_data_list, stiffness_data_list, specimen_keys)
    """
    strain_data_list = []
    stiffness_data_list = []
    specimen_keys = []
    
    # Exclude FOD3 (df0) as per original code
    valid_keys = sorted([k for k in stiffness_post_trunc.keys() if k != 'df0']) # Exclude df0 and sort
    
    for key in valid_keys:
        if key in strain_post_trunc and not strain_post_trunc[key].empty:
            strain_data_list.append(strain_post_trunc[key].values)
            
            # Reshape stiffness data to ensure it is (N, 1)
            stiffness_values = stiffness_post_trunc[key].values
            if stiffness_values.ndim == 1:
                stiffness_values = stiffness_values.reshape(-1, 1)
            stiffness_data_list.append(stiffness_values)
            specimen_keys.append(key)
            
            print(f"Prepared {key}: Strain shape {strain_data_list[-1].shape}, Stiffness shape {stiffness_data_list[-1].shape}")
        else:
            print(f"Skipping {key} for GNN data preparation due to missing or empty data.")
    
    num_specimens = len(specimen_keys)
    print(f"\nPrepared data for {num_specimens} specimens for GNN input.")
    
    return strain_data_list, stiffness_data_list, specimen_keys