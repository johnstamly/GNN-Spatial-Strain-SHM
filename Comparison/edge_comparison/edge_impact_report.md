# Edge Attributes Impact Analysis

## Summary

This report compares the performance of Graph Neural Network (GNN) models with and without edge attributes for stiffness prediction.

### Best Model With Edge Attributes: GENConv
- MSE: 3.0478
- RMSE: 1.7141
- MAPE: 1.25%

### Best Model Without Edge Attributes: GraphSAGE
- MSE: 3.9237
- RMSE: 1.8839
- MAPE: 1.36%

### Performance Improvement from Edge Attributes
- MSE Improvement: 22.32%
- RMSE Improvement: 9.01%
- MAPE Improvement: 8.15%

## Analysis

The results demonstrate that including edge attributes (differences in strain health indicators between connected nodes) in the GNN models leads to significant performance improvements. The edge attributes provide valuable information about the relationships between nodes, which helps the models better capture the structural properties of the graph and make more accurate predictions.

The best-performing model with edge attributes (GENConv) outperforms the best model without edge attributes (GraphSAGE) by 22.32% in terms of MSE, which is the primary metric for this regression task.

This improvement highlights the importance of incorporating edge attributes in GNN models for stiffness prediction tasks, as they capture important information about the relationships between strain health indicators at different sensor locations.

## Visualizations

Please refer to the following visualizations for a detailed comparison:
- `best_models_comparison.png`: Bar chart comparing the best models from each approach
- `best_models_table.png`: Table with detailed metrics and improvement percentages
- `all_models_comparison.png`: Bar chart comparing all models from both approaches

