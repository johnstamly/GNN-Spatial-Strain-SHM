# GNN Model Comparison for Stiffness Prediction

This document describes the different Graph Neural Network (GNN) model architectures implemented for stiffness prediction and how to compare their performance.

## Model Architectures

Five different GNN architectures have been implemented for comparison:

1. **GENConvModel (Original)**: Uses GENConv layers with edge attributes. This is the original model that was used for hyperparameter tuning.

2. **SAGPoolModel**: Implements SAGPooling for hierarchical pooling, which selects the most important nodes in the graph based on their features. This model uses GCNConv layers for message passing.

3. **GATv2Model**: Uses Graph Attention Network v2 (GATv2Conv) layers, which learn to assign different weights to different neighbors during message passing, allowing the model to focus on the most relevant connections.

4. **GCNModel**: Implements Graph Convolutional Network (GCNConv) layers, which are a simpler but often effective approach for graph learning tasks.

5. **EdgeConvModel**: Uses EdgeConv layers (Dynamic Edge-Conditioned Convolution), which dynamically compute edge features based on the features of connected nodes.

All models have been designed to have similar parameter counts for fair comparison, and they all use the same hyperparameters that were found to be optimal for the original GENConv model:
- Number of GNN layers: 3
- Hidden dimension: 64
- Dropout: ~0.307
- Batch size: 64

## Running the Model Comparison

To compare the performance of all models, run the `compare_models.py` script:

```bash
python compare_models.py
```

This script will:
1. Load and preprocess the data
2. Run Leave-One-Out Cross-Validation (LOOCV) for each model
3. Compare the performance metrics (MSE, RMSE, MAPE)
4. Generate visualizations of the results
5. Save all results to the `model_comparison` directory

### Command-line Arguments

The script accepts several command-line arguments to customize the comparison:

- `--stiffness-path`: Path to stiffness data directory (default: 'Data/Stiffness_Reduction')
- `--strain-path`: Path to strain data directory (default: 'Data/Strain')
- `--num-nodes`: Number of nodes in each graph (default: 16)
- `--batch-size`: Batch size for training (default: 64)
- `--hidden-dim`: Hidden dimension for the GNN models (default: 64)
- `--num-gnn-layers`: Number of GNN layers (default: 3)
- `--dropout`: Dropout probability (default: 0.307)
- `--epochs`: Maximum number of epochs (default: 1000)
- `--patience`: Patience for early stopping (default: 20)
- `--drop-level`: Stiffness reduction level for truncation (default: 85)
- `--output-dir`: Directory to save results and plots (default: 'model_comparison')
- `--models`: List of models to compare (default: all five models)

Example with custom arguments:

```bash
python compare_models.py --batch-size=128 --epochs=500 --output-dir=custom_comparison --models GENConv GATv2 GCN
```

## Output

The script will generate the following outputs in the specified output directory:

1. **Model Parameter Counts**: A JSON file with the number of trainable parameters for each model.
2. **Individual Model Results**: For each model, a directory containing:
   - LOOCV results (MSE, RMSE, MAPE) for each fold
   - Training and validation loss plots
   - Prediction plots
3. **Comparison Visualizations**:
   - Bar charts comparing MSE, RMSE, and MAPE across models
   - A table summarizing the performance metrics
4. **Comparison Results**: A JSON file with the performance metrics for all models.

## Interpreting the Results

When comparing the models, consider the following:

- **MSE (Mean Squared Error)**: Lower is better. This is the primary metric for regression tasks.
- **RMSE (Root Mean Squared Error)**: Lower is better. This is in the same units as the target variable.
- **MAPE (Mean Absolute Percentage Error)**: Lower is better. This gives the percentage error.

The best model for this specific task will likely depend on how well it can capture the relationships between node features (strain health indicators) and edge features (differences between connected nodes) to predict the global stiffness value.

Different GNN architectures have different strengths:
- GENConv is good at capturing complex relationships with edge features
- SAGPooling can identify the most important nodes
- GATv2 can learn to focus on the most relevant connections
- GCN is simpler but often effective
- EdgeConv dynamically computes edge features

The comparison will reveal which approach works best for this specific graph regression task.