"""
GNN Utilities Package for Stiffness Prediction.

This package contains modules for data preprocessing, model definition,
training, evaluation, graph data preparation, and visualization.
"""

from gnn_utils.data_preprocessing import (
    resample_stiffness_to_match_strain,
    percentage_change_from_max,
    find_closest_index
)

from gnn_utils.model import (
    EdgeAttrGNN,
    count_parameters,
    weighted_mse_loss
)

from gnn_utils.training import (
    train_epoch,
    validate_epoch,
    run_training,
    run_inference,
    plot_predictions,
    setup_tensorboard
)

from gnn_utils.graph_data import (
    normalize_input,
    normalize_target,
    unnormalize_target,
    compute_normalization_params,
    create_fully_connected_edge_index,
    create_graph_data_objects,
    prepare_data_loaders
)

from gnn_utils.visualization import (
    setup_matplotlib_style,
    plot_stiffness_vs_cycles,
    plot_training_history,
    visualize_graph
)

# Import new modules
from gnn_utils.data_processing import (
    load_data,
    preprocess_data,
    identify_target_indexes,
    truncate_data,
    prepare_gnn_data
)

from gnn_utils.loocv import (
    run_loocv,
    summarize_loocv_results,
    plot_loocv_predictions
)

__all__ = [
    # Data preprocessing
    'resample_stiffness_to_match_strain',
    'percentage_change_from_max',
    'find_closest_index',
    
    # Model
    'EdgeAttrGNN',
    'count_parameters',
    'weighted_mse_loss',
    
    # Training
    'train_epoch',
    'validate_epoch',
    'run_training',
    'run_inference',
    'plot_predictions',
    'setup_tensorboard',
    
    # Graph data
    'normalize_input',
    'normalize_target',
    'unnormalize_target',
    'compute_normalization_params',
    'create_fully_connected_edge_index',
    'create_graph_data_objects',
    'prepare_data_loaders',
    
    # Visualization
    'setup_matplotlib_style',
    'plot_stiffness_vs_cycles',
    'plot_training_history',
    'visualize_graph',
    
    # Data processing
    'load_data',
    'preprocess_data',
    'identify_target_indexes',
    'truncate_data',
    'prepare_gnn_data',
    
    # LOOCV
    'run_loocv',
    'summarize_loocv_results',
    'plot_loocv_predictions'
]