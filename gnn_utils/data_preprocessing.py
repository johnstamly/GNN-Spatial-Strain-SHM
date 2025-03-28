"""
Data preprocessing utilities for stiffness prediction GNN.
Contains functions for resampling, normalization, and other data transformations.
"""

import numpy as np
import pandas as pd

def resample_stiffness_to_match_strain(strain_df, stiffness_df):
    """Resamples stiffness data (upsampling or downsampling) to match the length of strain data.
    
    Args:
        strain_df (pd.DataFrame or pd.Series): Strain data with the target length.
        stiffness_df (pd.DataFrame or pd.Series): Stiffness data to be resampled.
        
    Returns:
        pd.DataFrame or pd.Series: Resampled stiffness data.
    """
    strain_length = len(strain_df)
    stiffness_length = len(stiffness_df)

    if strain_length == stiffness_length:
        return stiffness_df.reset_index(drop=True)

    x_old = np.linspace(0, 1, stiffness_length)  # Normalized index for stiffness
    x_new = np.linspace(0, 1, strain_length)    # Normalized index for strain

    if strain_length > stiffness_length:
        # Upsample using linear interpolation
        if isinstance(stiffness_df, pd.Series):
            stiffness_resampled = pd.Series(np.interp(x_new, x_old, stiffness_df.values))
        elif isinstance(stiffness_df, pd.DataFrame):
            interpolated_data = {col: np.interp(x_new, x_old, stiffness_df[col].values) for col in stiffness_df.columns}
            stiffness_resampled = pd.DataFrame(interpolated_data)
        else:
            raise TypeError("stiffness_df must be a pandas Series or DataFrame")
            
    else: # strain_length < stiffness_length
        # Downsample by selecting the closest indices
        idx_new = np.searchsorted(x_old, x_new, side='left')
        # Ensure indices are within bounds, handle potential edge cases
        idx_new = np.clip(idx_new, 0, stiffness_length - 1)
        # Correct for cases where searchsorted might pick index+1 due to floating point
        # If the point is closer to the previous index, choose that one
        prev_idx_dist = np.abs(x_new - x_old[np.clip(idx_new - 1, 0, stiffness_length - 1)])
        curr_idx_dist = np.abs(x_new - x_old[idx_new])
        idx_new[prev_idx_dist < curr_idx_dist] -= 1
        idx_new = np.clip(idx_new, 0, stiffness_length - 1) # Re-clip after adjustment
        
        stiffness_resampled = stiffness_df.iloc[idx_new].reset_index(drop=True)

    return stiffness_resampled

def percentage_change_from_max(stiffness_df):
    """Calculates the percentage change from the maximum value for each column.
    Sets values before the maximum index to 100%.
    
    Args:
        stiffness_df (pd.DataFrame or pd.Series): Stiffness data.
        
    Returns:
        pd.DataFrame or pd.Series: Stiffness data normalized to percentage of max,
                                   with pre-max values set to 100.
    """
    if isinstance(stiffness_df, pd.Series):
        if stiffness_df.empty:
            return stiffness_df
        max_index = stiffness_df.idxmax()
        max_value = stiffness_df[max_index]
        if max_value == 0:
            # Avoid division by zero if max value is 0
            percentage_change_df = pd.Series(np.zeros_like(stiffness_df.values), index=stiffness_df.index)
        else:
            percentage_change_df = (stiffness_df / max_value) * 100
        # Ensure correct assignment using .loc
        percentage_change_df.loc[:max_index] = 100.0
        return percentage_change_df

    elif isinstance(stiffness_df, pd.DataFrame):
        percentage_change_df = stiffness_df.copy()
        for col in stiffness_df.columns:
            if stiffness_df[col].empty:
                continue
            max_idx_col = stiffness_df[col].idxmax()
            max_val_col = stiffness_df[col][max_idx_col]
            if max_val_col == 0:
                percentage_change_df[col] = 0.0
            else:
                percentage_change_df[col] = (stiffness_df[col] / max_val_col) * 100
            percentage_change_df.loc[:max_idx_col, col] = 100.0
        return percentage_change_df
    else:
        raise TypeError("Input must be a pandas DataFrame or Series")

def find_closest_index(array, target):
    """Finds the index of the value closest to the target in a NumPy array."""
    return np.abs(array - target).argmin()