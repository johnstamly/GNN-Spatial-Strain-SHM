# MLP Model Technical Report

## Overview
This report details the Multi-Layer Perceptron (MLP) model implemented for stiffness prediction in structural health monitoring. The MLP model serves as a baseline comparison for Graph Neural Network (GNN) models in the project, providing a standard against which to measure the performance benefits of incorporating graph structure.

## Model Architecture

The MLP architecture consists of a simple feedforward neural network with the following characteristics:

- **Input Layer**: 16 neurons (corresponding to 16 sensors in the structural system)
- **Hidden Layers**: Two hidden layers with dimensions [64, 32] by default
- **Output Layer**: 1 neuron (for stiffness prediction)
- **Activation Function**: ReLU (Rectified Linear Unit) for all hidden layers
- **Regularization**: Dropout with probability 0.5 between layers
- **Loss Function**: Mean Squared Error (MSE)

The model implementation utilizes PyTorch and follows a modular design pattern that allows for flexible configuration of hyperparameters.

## Hyperparameters

The following hyperparameters were used for training the MLP model:

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Hidden Dimensions | [64, 32] | Neurons in each hidden layer |
| Dropout Rate | 0.5 | Probability of neuron deactivation during training |
| Batch Size | 128 | Number of samples per gradient update |
| Maximum Epochs | 1000 | Maximum number of training iterations |
| Early Stopping Patience | 20 | Epochs to wait before stopping if no improvement |
| Learning Rate | 0.01 | Initial learning rate for optimization |
| Weight Decay | 1e-5 | L2 regularization parameter |
| Optimizer | AdamW | Optimizer algorithm with weight decay |
| LR Scheduler | ReduceLROnPlateau | Reduces learning rate when validation loss plateaus |
| LR Reduction Factor | 0.8 | Factor by which to reduce learning rate |
| LR Scheduler Patience | 10 | Epochs to wait before reducing learning rate |

## Data Processing

The model processes strain data from structural sensors using the following steps:

1. **Data Preparation**: The strain data from 16 sensors is organized for direct input to the MLP model
2. **Normalization**:
   - Input strain data is standardized (z-score normalization) using the mean and standard deviation from the training data
   - Target stiffness values are normalized using min-max scaling to the range [0, 1]
3. **Data Loading**: The processed data is loaded into PyTorch DataLoaders with a specified batch size and shuffling for training data

## Training Methodology

The model was trained using a rigorous Leave-One-Out Cross-Validation (LOOCV) approach:

1. For each specimen/dataset in the collection, one is held out for validation while the rest are used for training
2. Normalization parameters are computed from the training data only
3. The model is trained with early stopping based on validation loss
4. Learning rate is reduced when the validation loss plateaus
5. The best model (with lowest validation loss) is saved during training
6. Evaluation is performed on the validation set using MSE, RMSE, and MAPE metrics

## Visualization and Analysis

Training and validation outcomes were monitored and visualized using:

- Loss curves plotting training and validation loss over epochs
- Time series plots of true vs. predicted stiffness values
- Scatter plots comparing true and predicted values
- Residual plots to analyze prediction errors

## Comparison with GNN Models

The MLP model serves as a baseline for comparison with more complex Graph Neural Network (GNN) architectures. The comparison enables quantification of the performance benefits gained by incorporating structural relationships between sensors through graph representations.

The comparative analysis includes:
- Performance metrics (MSE, RMSE, MAPE) for both model types
- Visualization of predictions from both models
- Calculation of improvement percentages when using GNN vs. MLP

## Summary

The MLP model provides a strong baseline that captures the direct relationship between sensor readings and structural stiffness. Its relatively simple architecture makes it computationally efficient and interpretable. The performance of this model demonstrates the basic predictive power achievable without explicitly modeling the spatial relationships between sensors, which serves as an important reference point when evaluating the added value of graph-based approaches.