"""
Visualization utilities for stiffness prediction GNN.
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx


def setup_matplotlib_style():
    """Configure Matplotlib for professional plots."""
    try:
        # Try to use scienceplots style if available
        plt.style.use(['science', 'no-latex'])
        print("Using 'science' style from scienceplots package.")
    except (IOError, OSError):
        # Fall back to a standard style if scienceplots is not available
        print("Warning: 'science' style not found. Using default style.")
        plt.style.use('default')
    
    # Update specific parameters to override or customize the base style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False
    })


def plot_stiffness_vs_cycles(stiffness_data, x_rescaled, specimen_keys, title=None):
    """
    Plot stiffness data against cycles for multiple specimens.
    
    Args:
        stiffness_data: Dictionary mapping specimen keys to stiffness DataFrames
        x_rescaled: Dictionary mapping specimen keys to rescaled x-values (cycles)
        specimen_keys: List of specimen keys to plot
        title: Optional plot title
    """
    plt.figure(figsize=(12, 6))
    
    for key in specimen_keys:
        if key not in stiffness_data or stiffness_data[key].empty:
            print(f"Skipping plot for empty {key}")
            continue
            
        if key not in x_rescaled:
            print(f"No x-values for {key}, using default range")
            x_values = np.arange(len(stiffness_data[key]))
        else:
            x_values = x_rescaled[key]
            
        # Calculate FOD number (assuming df0=FOD3, df1=FOD4 etc.)
        fod_label = f"FOD{int(key.split('f')[-1]) + 3}"
        
        # Plot stiffness data
        plt.scatter(x_values, stiffness_data[key].iloc[:, 0], 
                   label=fod_label, s=5)  # Use scatter for potentially sparse data
    
    # Customize plot
    plt.legend(loc='best', markerscale=3)
    plt.title(title or "Normalized Stiffness Reduction vs. Cycles")
    plt.xlabel("Cycles")
    plt.ylabel("Normalized Stiffness (%)")
    plt.ylim(bottom=min(0, plt.ylim()[0]))  # Ensure y-axis starts at or below 0
    plt.grid(True)
    plt.show()


def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss history."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_losses, label='Training Loss')
    plt.semilogy(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


def visualize_graph(data, node_color_attr=None, title="Graph Visualization"):
    """
    Visualize a PyTorch Geometric graph using NetworkX.
    
    Args:
        data: PyTorch Geometric Data object
        node_color_attr: Node attribute to use for coloring (default: None)
        title: Plot title
    """
    G = to_networkx(data, to_undirected=True)
    
    plt.figure(figsize=(10, 10))
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Determine node colors if attribute is provided
    if node_color_attr is not None and hasattr(data, node_color_attr):
        node_attr = getattr(data, node_color_attr)
        if node_attr.dim() > 1:
            # If multi-dimensional, use first dimension
            node_colors = node_attr[:, 0].cpu().numpy()
        else:
            node_colors = node_attr.cpu().numpy()
        
        # Draw nodes with color mapping
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                      cmap=plt.cm.viridis, node_size=300)
        plt.colorbar(nodes)
    else:
        # Draw nodes with default color
        nx.draw_networkx_nodes(G, pos, node_size=300)
    
    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()