# Graph Neural Network for Stiffness Prediction - Code for Scientific Paper

This repository contains the code accompanying the scientific paper "[Paper Title Here]" (link to paper once available), which explores the use of Graph Neural Networks (GNNs) for predicting stiffness reduction in materials based on strain sensor data.

## Project Structure

- `run_loocv.py`: Main Python script to run Leave-One-Out Cross-Validation for the best performing model.
- `run_best_comparison_model.py`: Script to run the best performing model from comparisons (used for generating results in the paper).
- `gnn_utils/`: Python package containing utility functions and classes for data processing, model definition, training, and visualization.
  - `data_preprocessing.py`: Basic data preprocessing functions.
  - `data_processing.py`: Advanced data processing functions.
  - `model.py`: GNN model definition.
  - `training.py`: Training and evaluation utilities.
  - `graph_data.py`: Graph data preparation utilities.
  - `visualization.py`: Plotting and visualization functions.
  - `loocv.py`: Leave-One-Out Cross-Validation utilities.
  - `__init__.py`: Package initialization.
- `Data/`: Directory containing the input data used in the paper.
  - `Stiffness_Reduction/`: Stiffness reduction data.
  - `Strain/`: Strain sensor data.
- `Comparison/`: This directory contains scripts, code, and results from model comparisons, edge attribute analysis, and MLP comparisons that were performed during the research phase. These are included for completeness but are not necessary to reproduce the main results presented in the paper.
  - `best_comparison_model/`: Results from running the best comparison models.
  - `edge_comparison/`: Results and analysis comparing models with and without edge attributes.
  - `model_comparison/`: Results from comparing different GNN architectures with edge attributes.
  - `model_comparison_no_edges/`: Results from comparing different GNN architectures without edge attributes.
  - `mlp_models/`: Code and results for MLP comparisons.
  - `compare_edge_vs_no_edge.py`: Script to compare models with and without edge attributes.
  - `compare_models_fixed.py`: Script for fixed model comparisons.
  - `compare_models_no_edges.py`: Script to compare GNN architectures without edge attributes.
  - `compare_models.py`: Script to compare different GNN architectures with edge attributes.
  - `mlp_comparison.py`: Script for MLP model comparison.
  - `run_mlp_loocv.py`: Script to run LOOCV for MLP models.
  - `model_comparison_no_edges_README.md`: README for no-edge model comparisons.
  - `model_comparison_README.md`: README for model comparisons with edges.
  - `model_comparison_report.md`: Report on model comparisons.
  - `mlp_model_report.md`: Report on MLP model comparison.
- `results/`: Directory containing the results from running the best model (e.g., LOOCV predictions, loss plots).
- `visualizations/`: Directory containing visualizations generated during hyperparameter tuning.
- `hpo_study.db`: Optuna study database for hyperparameter optimization.
- `hyperparameter_tuning.py`: Script for hyperparameter tuning.
- `mlp_comparison.py`: Script for MLP model comparison.
- `best_model_README.md`: README for the best model results.
- `log_best_model/`: Log files for the best model runs.
- `best_model/`: Saved state dictionaries for the best model from each LOOCV fold.
- `visualize_hpo_results.py`: Script to visualize hyperparameter optimization results.

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

## Reproducing Results

To reproduce the main results presented in the paper, you can use the `run_loocv.py` script. Ensure you have the necessary data in the `Data/` directory.

```bash
python run_loocv.py [options]
```

Options:
- `--stiffness-path`: Path to stiffness data directory (default: `Data/Stiffness_Reduction/`)
- `--strain-path`: Path to strain data directory (default: `Data/Strain/`)
- `--batch-size`: Batch size for training
- `--hidden-dim`: Hidden dimension for GNN models
- `--num-gnn-layers`: Number of GNN layers
- `--dropout`: Dropout probability
- `--epochs`: Maximum number of epochs
- `--patience`: Patience for early stopping
- `--output-dir`: Directory to save results (default: `results/`)

The script will perform Leave-One-Out Cross-Validation using the best performing model identified in the comparison phase and save the predictions and evaluation metrics to the specified output directory.

## Dependencies

- Python 3.6+
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Matplotlib
- TensorBoardX
- scienceplots (optional)
- Optuna (for hyperparameter tuning, not required for reproducing main results)

Install core dependencies with:
```bash
pip install torch torch-geometric numpy pandas matplotlib tensorboardx
```
Install optional dependencies:
```bash
pip install scienceplots optuna
```

## Interpreting Results

The `results/` directory will contain `loocv_results.json` with key metrics (MSE, RMSE, MAPE, R²) and `loocv_predictions.png` visualizing the true vs predicted stiffness reduction. Loss plots for each fold will also be saved.

## License

The code in this repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The data in the `Data/` directory is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Citation

If you use this code in your research, please cite the accompanying paper:

[Add Citation Information Here]