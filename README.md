# Graph Neural Network for Stiffness Prediction

This project demonstrates the use of Graph Neural Networks (GNNs) for predicting stiffness reduction in materials based on strain sensor data.

## Project Structure

- `run_loocv.py`: Main Python script to run Leave-One-Out Cross-Validation
- `compare_models.py`: Script to compare different GNN architectures with edge attributes
- `compare_models_no_edges.py`: Script to compare GNN architectures without edge attributes
- `run_best_comparison_model.py`: Script to run the best performing model from comparisons
- `gnn_utils/`: Python package containing utility functions and classes
  - `data_preprocessing.py`: Basic data preprocessing functions
  - `data_processing.py`: Advanced data processing functions
  - `model.py`: GNN model definition
  - `training.py`: Training and evaluation utilities
  - `graph_data.py`: Graph data preparation utilities
  - `visualization.py`: Plotting and visualization functions
  - `loocv.py`: Leave-One-Out Cross-Validation utilities
  - `__init__.py`: Package initialization
- `Data/`: Directory containing the input data
  - `Stiffness_Reduction/`: Stiffness reduction data
  - `Strain/`: Strain sensor data

## Model Architectures

### With Edge Attributes
1. **GENConvModel**: Uses GENConv layers with edge attributes
2. **SAGPoolModel**: Implements SAGPooling for hierarchical pooling
3. **GATv2Model**: Uses Graph Attention Network v2 layers
4. **GCNModel**: Implements Graph Convolutional Network layers
5. **EdgeConvModel**: Uses EdgeConv layers with dynamic edge features

### Without Edge Attributes
1. **GCNModel_NoEdges**: Graph Convolutional Network without edge attributes
2. **GINModel**: Graph Isomorphism Network
3. **SGConvModel**: Simplified Graph Convolution
4. **GraphSAGEModel**: Graph SAmple and aggreGatE
5. **ChebConvModel**: Chebyshev Spectral Graph Convolution

## Running Model Comparisons

### With Edge Attributes
```bash
python compare_models.py [options]
```

### Without Edge Attributes
```bash
python compare_models_no_edges.py [options]
```

Common options for both scripts:
- `--stiffness-path`: Path to stiffness data directory
- `--strain-path`: Path to strain data directory
- `--batch-size`: Batch size for training
- `--hidden-dim`: Hidden dimension for GNN models
- `--num-gnn-layers`: Number of GNN layers
- `--dropout`: Dropout probability
- `--epochs`: Maximum number of epochs
- `--patience`: Patience for early stopping
- `--output-dir`: Directory to save results

## Running Best Models

After running comparisons, run the best model(s) with:
```bash
python run_best_comparison_model.py [options]
```

Options:
- `--model-type`: 'with_edges', 'no_edges', or 'both'
- `--epochs`: Maximum number of epochs
- `--patience`: Patience for early stopping
- `--save-plots`: Save plots to files
- `--output-dir`: Directory to save results

## Package Documentation (gnn_utils)

Key modules:
- `data_preprocessing.py`: Data resampling and normalization
- `model.py`: GNN model definitions
- `training.py`: Training utilities
- `graph_data.py`: Graph data preparation
- `visualization.py`: Plotting functions

## Interpreting Results

Key metrics:
- **MSE (Mean Squared Error)**: Lower is better
- **RMSE (Root Mean Squared Error)**: Lower is better
- **MAPE (Mean Absolute Percentage Error)**: Lower is better
- **R²**: Higher is better

Visualizations:
- Scatter plots of true vs predicted values
- Residual plots showing prediction errors
- Time series plots with error bands

## Dependencies

- Python 3.6+
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Matplotlib
- TensorBoardX
- scienceplots (optional)

Install with:
```bash
pip install torch torch-geometric numpy pandas matplotlib tensorboardx
pip install scienceplots  # Optional