"""
Graph Neural Network model definition for stiffness prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class EdgeAttrGNN(nn.Module):
    """
    Graph Neural Network using GENConv layers to process node and edge features.
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 2, dropout_p: float = 0.3,
                 genconv_aggr: str = 'add'):
        """
        Args:
            num_node_features: Dimensionality of input node features.
            edge_feature_dim: Dimensionality of input edge features.
            hidden_dim: Dimensionality of hidden layers.
            output_dim: Number of output values to predict.
            num_gnn_layers: Number of GENConv blocks.
            dropout_p: Dropout probability.
            genconv_aggr: Aggregation method for GENConv.
        """
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_gnn_layers = num_gnn_layers

        # Initial Node Feature Embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)

        # GNN Layers (GENConv + Normalization)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv = pyg_nn.GENConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggr=genconv_aggr,
                msg_norm=True,
                learn_msg_scale=True,
                num_layers=2, # Internal MLP depth in GENConv
                norm=None,    # Using external BatchNorm
                edge_dim=edge_feature_dim
            )
            self.convs.append(conv)
            self.norms.append(pyg_nn.BatchNorm(hidden_dim))

        # Global Pooling
        self.pool = pyg_nn.global_add_pool # Or global_mean_pool, global_max_pool

        # Readout Network (MLP)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GNN."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Ensure edge_attr has the correct shape [num_edges, edge_feature_dim]
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        # 1. Initial Node Embedding
        x = self.node_emb(x)

        # 2. GNN Blocks
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x)
            x = F.leaky_relu(x)
            # Apply dropout after activation, except potentially for the last GNN layer
            if i < self.num_gnn_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 3. Global Pooling
        x_graph = self.pool(x, batch) # Shape: [batch_size, hidden_dim]

        # Optional: Dropout after pooling
        x_graph = F.dropout(x_graph, p=self.dropout_p, training=self.training)

        # 4. Readout MLP
        out = self.readout(x_graph)

        return out

    def loss(self, pred, true):
        """Calculates the Mean Squared Error loss."""
        return F.mse_loss(pred, true)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weighted_mse_loss(pred, target, weight_range=(0.2, 0.5), weight_value=2.0):
    """Applies higher weight to MSE loss for target values within a specific range."""
    mse_loss = F.mse_loss(pred, target, reduction='none')
    weight_mask = (target >= weight_range[0]) & (target <= weight_range[1])
    weights = torch.ones_like(target)
    weights[weight_mask] = weight_value
    weighted_loss = mse_loss * weights
    return weighted_loss.mean()