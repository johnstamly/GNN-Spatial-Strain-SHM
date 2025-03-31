"""
Utility functions for stiffness prediction GNN.
"""

import numpy as np


def calculate_cycles_from_timesteps(strain_post, stiffness_post, last_cycle):
    """
    Calculate rescaled x-values (cycles) from timesteps.
    
    Args:
        strain_post: Dictionary of preprocessed strain dataframes
        stiffness_post: Dictionary of preprocessed stiffness dataframes
        last_cycle: Dictionary mapping keys to last cycle values
        
    Returns:
        Dictionary mapping keys to rescaled x-values (cycles)
    """
    strain_x_rescaled = {}
    
    for key in strain_post.keys():
        if key in last_cycle:
            # Retrieve original x values (timestamps in seconds)
            strain_x = strain_post[key].index.total_seconds()
            # Scale the x-axis so the last point corresponds to last_cycle[key]
            max_time = strain_x.max()
            if max_time > 0:  # Avoid division by zero
                strain_x_rescaled[key] = strain_x * (last_cycle[key] / max_time)
            else:
                # Fallback to using indices if timestamps are not available
                strain_x_rescaled[key] = np.arange(len(strain_post[key])) * last_cycle[key] / len(strain_post[key])
    
    return strain_x_rescaled