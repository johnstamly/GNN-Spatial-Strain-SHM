# GNN Utilities for Stiffness Prediction

This package contains utility functions and classes for stiffness prediction using Graph Neural Networks (GNNs). It was created to reduce code duplication and improve maintainability of the Jupyter notebook.

## Package Structure

The package is organized into the following modules:

- `data_preprocessing.py`: Functions for resampling, normalization, and other data transformations
- `model.py`: GNN model definition and related functions
- `training.py`: Training and evaluation utilities
- `graph_data.py`: Graph data preparation utilities
- `visualization.py`: Plotting and visualization functions

## Usage

To use this package, import it in your notebook:

```python
import gnn_utils
```

You can also import specific functions or classes:

```python
from gnn_utils import EdgeAttrGNN, percentage_change_from_max
```

## Main Components

### Data Preprocessing

- `resample_stiffness_to_match_strain`: Resamples stiffness data to match strain data length
- `percentage_change_from_max`: Calculates percentage change from maximum value
- `find_closest_index`: Finds the index of the closest value in an array

### Model

- `EdgeAttrGNN`: Graph Neural Network using GENConv layers
- `count_parameters`: Counts trainable parameters in a model
- `weighted_mse_loss`: Weighted MSE loss function

### Training

- `train_epoch`: Trains the model for one epoch
- `validate_epoch`: Validates the model on validation data
- `run_training`: Runs the training loop with early stopping
- `run_inference`: Runs inference and calculates metrics
- `plot_predictions`: Plots true vs. predicted values
- `setup_tensorboard`: Sets up TensorBoard writer

### Graph Data

- `normalize_input`: Normalizes input features
- `normalize_target`: Normalizes target values
- `unnormalize_target`: Unnormalizes target values
- `compute_normalization_params`: Computes normalization parameters
- `create_fully_connected_edge_index`: Creates edge indices for a fully connected graph
- `create_graph_data_objects`: Creates PyTorch Geometric Data objects
- `prepare_data_loaders`: Creates DataLoader objects

### Visualization

- `setup_matplotlib_style`: Configures Matplotlib for professional plots
- `plot_stiffness_vs_cycles`: Plots stiffness data against cycles
- `plot_training_history`: Plots training and validation loss history
- `visualize_graph`: Visualizes a PyTorch Geometric graph

## Example Workflow

1. Load and preprocess data
2. Compute normalization parameters
3. Create graph data objects
4. Define and train the model
5. Evaluate the model
6. Visualize results

See `Cline_test_simplified.ipynb` for a complete example.