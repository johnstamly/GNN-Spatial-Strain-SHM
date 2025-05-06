"""
MLP (Multi-Layer Perceptron) model for stiffness prediction.
This serves as a baseline comparison for GNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    """
    Simple MLP model for stiffness prediction.
    Takes sensor data directly as input without graph structure.
    """
    def __init__(self, input_dim=16, hidden_dims=[64, 32], output_dim=1, dropout_p=0.5):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Number of input features (16 sensors)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for stiffness prediction)
            dropout_p: Dropout probability
        """
        super(MLPModel, self).__init__()
        
        # Create the layers
        layer_sizes = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        
        # Create hidden layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
        # Output layer
        self.output_layer = nn.Linear(layer_sizes[-1], output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Apply hidden layers with ReLU activation and dropout
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
            
        # Apply output layer
        x = self.output_layer(x)
        
        return x
    
    def loss(self, pred, target):
        """
        Compute the mean squared error loss.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            Mean squared error loss
        """
        return F.mse_loss(pred, target)