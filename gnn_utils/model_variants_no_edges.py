"""
Graph Neural Network model variants for stiffness prediction without edge attributes.

This module contains different GNN architectures that only use node features (no edge attributes):
1. GCNModel_NoEdges (Graph Convolutional Network)
2. GINModel (Graph Isomorphism Network)
3. SGConvModel (Simplified Graph Convolution)
4. GraphSAGEModel (Graph SAmple and aggreGatE)
5. ChebConvModel (Chebyshev Spectral Graph Convolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data


class GCNModel_NoEdges(nn.Module):
    """
    Graph Neural Network using GCNConv layers without edge attributes.
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 3, dropout_p: float = 0.3):
        """
        Args:
            num_node_features: Dimensionality of input node features.
            edge_feature_dim: Not used in this model.
            hidden_dim: Dimensionality of hidden layers.
            output_dim: Number of output values to predict.
            num_gnn_layers: Number of GCNConv blocks.
            dropout_p: Dropout probability.
        """
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_gnn_layers = num_gnn_layers

        # Initial Node Feature Embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)

        # GNN Layers (GCNConv + Normalization)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv = pyg_nn.GCNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim
            )
            self.convs.append(conv)
            self.norms.append(pyg_nn.BatchNorm(hidden_dim))

        # Global Pooling
        self.pool = pyg_nn.global_add_pool

        # Readout Network (MLP)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GNN."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Initial Node Embedding
        x = self.node_emb(x)

        # 2. GNN Blocks
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.leaky_relu(x)
            if i < self.num_gnn_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 3. Global Pooling
        x_graph = self.pool(x, batch)  # Shape: [batch_size, hidden_dim]

        # Optional: Dropout after pooling
        x_graph = F.dropout(x_graph, p=self.dropout_p, training=self.training)

        # 4. Readout MLP
        out = self.readout(x_graph)

        return out

    def loss(self, pred, true):
        """Calculates the Mean Squared Error loss."""
        return F.mse_loss(pred, true)


class GINModel(nn.Module):
    """
    Graph Neural Network using GIN (Graph Isomorphism Network) layers.
    GIN is particularly good at capturing structural information without edge features.
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 3, dropout_p: float = 0.3):
        """
        Args:
            num_node_features: Dimensionality of input node features.
            edge_feature_dim: Not used in this model.
            hidden_dim: Dimensionality of hidden layers.
            output_dim: Number of output values to predict.
            num_gnn_layers: Number of GIN blocks.
            dropout_p: Dropout probability.
        """
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_gnn_layers = num_gnn_layers

        # Initial Node Feature Embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)

        # GNN Layers (GIN + Normalization)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            # MLP for GIN
            nn_module = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = pyg_nn.GINConv(
                nn=nn_module,
                train_eps=True  # Learn epsilon parameter
            )
            self.convs.append(conv)
            self.norms.append(pyg_nn.BatchNorm(hidden_dim))

        # Global Pooling
        self.pool = pyg_nn.global_add_pool

        # Readout Network (MLP)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GNN."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Initial Node Embedding
        x = self.node_emb(x)

        # 2. GNN Blocks
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.leaky_relu(x)
            if i < self.num_gnn_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 3. Global Pooling
        x_graph = self.pool(x, batch)  # Shape: [batch_size, hidden_dim]

        # Optional: Dropout after pooling
        x_graph = F.dropout(x_graph, p=self.dropout_p, training=self.training)

        # 4. Readout MLP
        out = self.readout(x_graph)

        return out

    def loss(self, pred, true):
        """Calculates the Mean Squared Error loss."""
        return F.mse_loss(pred, true)


class SGConvModel(nn.Module):
    """
    Graph Neural Network using SGConv (Simplified Graph Convolution) layers.
    SGConv simplifies GCN by using a fixed power of the adjacency matrix.
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 3, dropout_p: float = 0.3,
                 K: int = 2):
        """
        Args:
            num_node_features: Dimensionality of input node features.
            edge_feature_dim: Not used in this model.
            hidden_dim: Dimensionality of hidden layers.
            output_dim: Number of output values to predict.
            num_gnn_layers: Number of SGConv blocks.
            dropout_p: Dropout probability.
            K: Number of hops to consider (power of adjacency matrix).
        """
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_gnn_layers = num_gnn_layers
        self.K = K

        # Initial Node Feature Embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)

        # GNN Layers (SGConv + Normalization)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv = pyg_nn.SGConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                K=K  # Number of hops
            )
            self.convs.append(conv)
            self.norms.append(pyg_nn.BatchNorm(hidden_dim))

        # Global Pooling
        self.pool = pyg_nn.global_add_pool

        # Readout Network (MLP)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GNN."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Initial Node Embedding
        x = self.node_emb(x)

        # 2. GNN Blocks
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.leaky_relu(x)
            if i < self.num_gnn_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 3. Global Pooling
        x_graph = self.pool(x, batch)  # Shape: [batch_size, hidden_dim]

        # Optional: Dropout after pooling
        x_graph = F.dropout(x_graph, p=self.dropout_p, training=self.training)

        # 4. Readout MLP
        out = self.readout(x_graph)

        return out

    def loss(self, pred, true):
        """Calculates the Mean Squared Error loss."""
        return F.mse_loss(pred, true)


class GraphSAGEModel(nn.Module):
    """
    Graph Neural Network using GraphSAGE (Graph SAmple and aggreGatE) layers.
    GraphSAGE is designed to generate node embeddings by sampling and aggregating features from neighbors.
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 3, dropout_p: float = 0.3,
                 aggr: str = 'mean'):
        """
        Args:
            num_node_features: Dimensionality of input node features.
            edge_feature_dim: Not used in this model.
            hidden_dim: Dimensionality of hidden layers.
            output_dim: Number of output values to predict.
            num_gnn_layers: Number of GraphSAGE blocks.
            dropout_p: Dropout probability.
            aggr: Aggregation method ('mean', 'max', or 'add').
        """
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_gnn_layers = num_gnn_layers
        self.aggr = aggr

        # Initial Node Feature Embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)

        # GNN Layers (GraphSAGE + Normalization)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv = pyg_nn.SAGEConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggr=aggr  # Aggregation method
            )
            self.convs.append(conv)
            self.norms.append(pyg_nn.BatchNorm(hidden_dim))

        # Global Pooling
        self.pool = pyg_nn.global_add_pool

        # Readout Network (MLP)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GNN."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Initial Node Embedding
        x = self.node_emb(x)

        # 2. GNN Blocks
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.leaky_relu(x)
            if i < self.num_gnn_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 3. Global Pooling
        x_graph = self.pool(x, batch)  # Shape: [batch_size, hidden_dim]

        # Optional: Dropout after pooling
        x_graph = F.dropout(x_graph, p=self.dropout_p, training=self.training)

        # 4. Readout MLP
        out = self.readout(x_graph)

        return out

    def loss(self, pred, true):
        """Calculates the Mean Squared Error loss."""
        return F.mse_loss(pred, true)


class ChebConvModel(nn.Module):
    """
    Graph Neural Network using ChebConv (Chebyshev Spectral Graph Convolution) layers.
    ChebConv uses Chebyshev polynomials to approximate spectral graph convolutions.
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 3, dropout_p: float = 0.3,
                 K: int = 3):
        """
        Args:
            num_node_features: Dimensionality of input node features.
            edge_feature_dim: Not used in this model.
            hidden_dim: Dimensionality of hidden layers.
            output_dim: Number of output values to predict.
            num_gnn_layers: Number of ChebConv blocks.
            dropout_p: Dropout probability.
            K: Chebyshev filter size (order of Chebyshev polynomial).
        """
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_gnn_layers = num_gnn_layers
        self.K = K

        # Initial Node Feature Embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)

        # GNN Layers (ChebConv + Normalization)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv = pyg_nn.ChebConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                K=K  # Chebyshev filter size
            )
            self.convs.append(conv)
            self.norms.append(pyg_nn.BatchNorm(hidden_dim))

        # Global Pooling
        self.pool = pyg_nn.global_add_pool

        # Readout Network (MLP)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GNN."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Initial Node Embedding
        x = self.node_emb(x)

        # 2. GNN Blocks
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.leaky_relu(x)
            if i < self.num_gnn_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 3. Global Pooling
        x_graph = self.pool(x, batch)  # Shape: [batch_size, hidden_dim]

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