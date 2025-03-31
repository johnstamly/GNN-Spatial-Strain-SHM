"""
Graph Neural Network model variants for stiffness prediction.

This module contains different GNN architectures for comparison:
1. GENConvModel (original model)
2. SAGPoolModel (with SAGPooling)
3. GATv2Model (Graph Attention Network v2)
4. GCNModel (Graph Convolutional Network)
5. EdgeConvModel (Dynamic Edge-Conditioned Convolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data


class GENConvModel(nn.Module):
    """
    Original Graph Neural Network using GENConv layers to process node and edge features.
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 3, dropout_p: float = 0.3,
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
                num_layers=2,  # Internal MLP depth in GENConv
                norm=None,     # Using external BatchNorm
                edge_dim=edge_feature_dim
            )
            self.convs.append(conv)
            self.norms.append(pyg_nn.BatchNorm(hidden_dim))

        # Global Pooling
        self.pool = pyg_nn.global_add_pool  # Or global_mean_pool, global_max_pool

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
        x_graph = self.pool(x, batch)  # Shape: [batch_size, hidden_dim]

        # Optional: Dropout after pooling
        x_graph = F.dropout(x_graph, p=self.dropout_p, training=self.training)

        # 4. Readout MLP
        out = self.readout(x_graph)

        return out

    def loss(self, pred, true):
        """Calculates the Mean Squared Error loss."""
        return F.mse_loss(pred, true)


class SAGPoolModel(nn.Module):
    """
    Graph Neural Network using GCNConv layers with SAGPooling for hierarchical pooling.
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 3, dropout_p: float = 0.3,
                 pool_ratio: float = 0.5):
        """
        Args:
            num_node_features: Dimensionality of input node features.
            edge_feature_dim: Dimensionality of input edge features.
            hidden_dim: Dimensionality of hidden layers.
            output_dim: Number of output values to predict.
            num_gnn_layers: Number of GNN blocks.
            dropout_p: Dropout probability.
            pool_ratio: Ratio of nodes to keep in SAGPooling.
        """
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_gnn_layers = num_gnn_layers

        # Initial Node Feature Embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)
        
        # Edge Feature Embedding
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LeakyReLU()
        )

        # GNN Layers (GCNConv + Normalization)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            # Using GCNConv since it works well with SAGPooling
            conv = pyg_nn.GCNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim
            )
            self.convs.append(conv)
            self.norms.append(pyg_nn.BatchNorm(hidden_dim))

        # SAGPooling layer
        self.pool = pyg_nn.SAGPooling(hidden_dim, ratio=pool_ratio)
        
        # Global Pooling after SAGPooling
        self.global_pool = pyg_nn.global_add_pool

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
            
        # Process edge features if needed
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)

        # 1. Initial Node Embedding
        x = self.node_emb(x)

        # 2. GNN Blocks
        for i in range(self.num_gnn_layers):
            # For GCNConv, we don't directly use edge_attr
            # Instead, we can use it to modify edge_index or weights if needed
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.leaky_relu(x)
            if i < self.num_gnn_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 3. SAGPooling
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        
        # 4. Global Pooling
        x_graph = self.global_pool(x, batch)  # Shape: [batch_size, hidden_dim]

        # Optional: Dropout after pooling
        x_graph = F.dropout(x_graph, p=self.dropout_p, training=self.training)

        # 5. Readout MLP
        out = self.readout(x_graph)

        return out

    def loss(self, pred, true):
        """Calculates the Mean Squared Error loss."""
        return F.mse_loss(pred, true)


class GATv2Model(nn.Module):
    """
    Graph Neural Network using GATv2Conv layers (Graph Attention Network v2).
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 3, dropout_p: float = 0.3,
                 heads: int = 4):
        """
        Args:
            num_node_features: Dimensionality of input node features.
            edge_feature_dim: Dimensionality of input edge features.
            hidden_dim: Dimensionality of hidden layers.
            output_dim: Number of output values to predict.
            num_gnn_layers: Number of GATv2Conv blocks.
            dropout_p: Dropout probability.
            heads: Number of attention heads.
        """
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_gnn_layers = num_gnn_layers
        self.heads = heads

        # Initial Node Feature Embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)
        
        # Edge Feature Embedding
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LeakyReLU()
        )

        # GNN Layers (GATv2Conv + Normalization)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer with multiple heads
        self.convs.append(pyg_nn.GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,  # Divide by heads to keep param count similar
            heads=heads,
            edge_dim=hidden_dim,
            dropout=dropout_p
        ))
        self.norms.append(pyg_nn.BatchNorm(hidden_dim))
        
        # Middle layers
        for _ in range(num_gnn_layers - 2):
            self.convs.append(pyg_nn.GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                edge_dim=hidden_dim,
                dropout=dropout_p
            ))
            self.norms.append(pyg_nn.BatchNorm(hidden_dim))
        
        # Last layer with a single head for final representation
        if num_gnn_layers > 1:
            self.convs.append(pyg_nn.GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=1,
                edge_dim=hidden_dim,
                dropout=dropout_p
            ))
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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Ensure edge_attr has the correct shape [num_edges, edge_feature_dim]
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
            
        # Process edge features
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)

        # 1. Initial Node Embedding
        x = self.node_emb(x)

        # 2. GNN Blocks
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
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


class GCNModel(nn.Module):
    """
    Graph Neural Network using GCNConv layers (Graph Convolutional Network).
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 3, dropout_p: float = 0.3):
        """
        Args:
            num_node_features: Dimensionality of input node features.
            edge_feature_dim: Dimensionality of input edge features.
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
        
        # Edge Feature Network (for edge weighting)
        self.edge_network = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Normalize edge weights to [0, 1]
        )

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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Ensure edge_attr has the correct shape [num_edges, edge_feature_dim]
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
            
        # Process edge features to get edge weights
        edge_weight = None
        if edge_attr is not None:
            edge_weight = self.edge_network(edge_attr).view(-1)

        # 1. Initial Node Embedding
        x = self.node_emb(x)

        # 2. GNN Blocks
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
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


class EdgeConvModel(nn.Module):
    """
    Graph Neural Network using EdgeConv layers (Dynamic Edge-Conditioned Convolution).
    """
    def __init__(self, num_node_features: int, edge_feature_dim: int, hidden_dim: int,
                 output_dim: int, num_gnn_layers: int = 3, dropout_p: float = 0.3,
                 k: int = 8):
        """
        Args:
            num_node_features: Dimensionality of input node features.
            edge_feature_dim: Dimensionality of input edge features.
            hidden_dim: Dimensionality of hidden layers.
            output_dim: Number of output values to predict.
            num_gnn_layers: Number of EdgeConv blocks.
            dropout_p: Dropout probability.
            k: Number of nearest neighbors for dynamic graph construction.
        """
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_gnn_layers = num_gnn_layers
        self.k = k

        # Initial Node Feature Embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)

        # GNN Layers (EdgeConv + Normalization)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            # EdgeConv with MLP for edge features
            nn_module = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = pyg_nn.EdgeConv(nn=nn_module, aggr='add')
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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 1. Initial Node Embedding
        x = self.node_emb(x)

        # 2. GNN Blocks
        for i in range(self.num_gnn_layers):
            # EdgeConv dynamically computes edges based on feature similarity
            # We can use the provided edge_index or compute a new one
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