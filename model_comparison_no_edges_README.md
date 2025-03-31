# GNN Model Comparison Without Edge Attributes

This document describes the different Graph Neural Network (GNN) model architectures implemented for stiffness prediction without using edge attributes, and how to compare their performance.

## Model Architectures Without Edge Attributes

Five different GNN architectures have been implemented for comparison that only use node features (strain health indicators) without edge attributes:

1. **GCNModel_NoEdges**: A Graph Convolutional Network that only uses node features and the graph structure, without considering edge attributes. This is a simpler version of the GCN model that was used in the previous comparison.

2. **GINModel**: Graph Isomorphism Network, which is particularly good at capturing structural information without edge features. GIN uses MLPs to transform node features and is designed to be as powerful as the Weisfeiler-Lehman graph isomorphism test.

3. **SGConvModel**: Simplified Graph Convolution, which simplifies GCN by using a fixed power of the adjacency matrix. This reduces computational complexity while maintaining good performance for many tasks.

4. **GraphSAGEModel**: Graph SAmple and aggreGatE, which generates node embeddings by sampling and aggregating features from neighbors. GraphSAGE is designed to work well with inductive learning tasks.

5. **ChebConvModel**: Chebyshev Spectral Graph Convolution, which uses Chebyshev polynomials to approximate spectral graph convolutions. This approach can capture multi-scale information in the graph.

All models have been designed to have similar parameter counts for fair comparison, and they all use the same hyperparameters that were found to be optimal for the original GENConv model:
- Number of GNN layers: 3
- Hidden dimension: 64
- Dropout: ~0.307
- Batch size: 64

## Running the Model Comparison Without Edge Attributes

To compare the performance of all models without edge attributes, run the `compare_models_no_edges.py` script:

```bash
python compare_models_no_edges.py
```

This script will:
1. Load and preprocess the data
2. Create graph data objects without edge attributes
3. Run Leave-One-Out Cross-Validation (LOOCV) for each model
4. Compare the performance metrics (MSE, RMSE, MAPE)
5. Generate visualizations of the results
6. Save all results to the `model_comparison_no_edges` directory

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
- `--output-dir`: Directory to save results and plots (default: 'model_comparison_no_edges')
- `--models`: List of models to compare (default: all five models)

Example with custom arguments:

```bash
python compare_models_no_edges.py --batch-size=128 --epochs=500 --output-dir=custom_comparison_no_edges --models GCN_NoEdges GIN GraphSAGE
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

## Comparing Models With and Without Edge Attributes

After running both comparison scripts (`compare_models_fixed.py` and `compare_models_no_edges.py`), you can compare the results to understand the impact of edge attributes on model performance.

To do this, you can:

1. Compare the best model from each approach:
   - Best model with edge attributes (from `model_comparison` directory)
   - Best model without edge attributes (from `model_comparison_no_edges` directory)

2. Compare the overall performance metrics:
   - Are models with edge attributes consistently better?
   - How much improvement do edge attributes provide?
   - Are there specific models that benefit more from edge attributes?

3. Visualize the differences:
   - Create a combined bar chart showing the best models from both approaches
   - Plot the percentage improvement that edge attributes provide

This comparison will help demonstrate the value of including edge attributes (differences in strain health indicators between connected nodes) in your GNN models for stiffness prediction.