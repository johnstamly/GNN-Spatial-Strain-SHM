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

from gnn_utils.model_variants import (
    GENConvModel,
    SAGPoolModel,
    GATv2Model,
    GCNModel,
    EdgeConvModel
)

from gnn_utils.model_variants_no_edges import (
    GCNModel_NoEdges,
    GINModel,
    SGConvModel,
    GraphSAGEModel,
    ChebConvModel
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
    run_loocv_utility,
    summarize_loocv_results,
    plot_loocv_predictions
)

from gnn_utils.utils import calculate_cycles_from_timesteps

__all__ = [
    # Data preprocessing
    'resample_stiffness_to_match_strain',
    'percentage_change_from_max',
    'find_closest_index',
    
    # Model
    'EdgeAttrGNN',
    'count_parameters',
    'weighted_mse_loss',

    # Model variants
    'GENConvModel',
    'SAGPoolModel',
    'GATv2Model',
    'GCNModel',
    'EdgeConvModel',
    
    # Model variants without edge attributes
    'GCNModel_NoEdges',
    'GINModel',
    'SGConvModel',
    'GraphSAGEModel',
    'ChebConvModel',
    
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
    'calculate_cycles_from_timesteps',
    
    # LOOCV
    'run_loocv_utility',
    'summarize_loocv_results',
    'plot_loocv_predictions'
]