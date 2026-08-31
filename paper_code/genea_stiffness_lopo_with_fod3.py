# %% [markdown]
# # IMPORTS

# %%
import os
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots

# %% [markdown]
# # CONFIGURATION

# %%
CONFIG = {
    # Data and Preprocessing
    "stiffness_drop_threshold": 70,

    # Leave-One-Out Cross-Validation Setup
    "test_fold_key": 'df0', # Options: 'df1', 'df2', 'df3', 'df4'

    # Model Architecture
    "hidden_dim": 16,
    "output_dim": 1,
    "num_gnn_layers": 3,
    "dropout_p": 0.3,

    # Training
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "epochs": 2000,
    "patience": 100,
    "batch_size": 128,
    "clip_norm": 1.0,

    # Weighted Loss
    "use_weighted_loss": True,
    "weight_range": (0.2, 0.95),
    "weight_value": 2.0,
    
    # Inference
    "mc_dropout_samples": 100, # <-- This line was missing
    "use_augmentation": False,
    "noise_level": 0.01,
    "use_subsampling": False,
    "use_shuffling": False,
    "min_nodes_subsampling": 16
}
# Map dataframe names to indices for fold selection
# We exclude 'df0' as in the original script
DF_INDICES = {'df0': 0, 'df1': 1, 'df2': 2, 'df3': 3, 'df4': 4}

# %% [markdown]
# # PLOTTING STYLE

# %%
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif", "font.size": 24, "axes.labelsize": 26,
    "axes.titlesize": 24, "legend.fontsize": 24, "xtick.labelsize": 24,
    "ytick.labelsize": 24, "lines.linewidth": 1.5, "lines.markersize": 6,
    "grid.linestyle": "--", "grid.alpha": 0.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.grid": True,
    "axes.spines.top": True, "axes.spines.right": True,
})

# %% [markdown]
# # DATA LOADING & UTILITY FUNCTIONS & PREPROCESSING

# %%
# ### Loading strain and stiffness reduction data

# %%
stiffness_data_path = 'Data/Stiffness_Reduction'
strain_data_path = 'Data/Strain'

# Stiffness Data
stiff_file_paths = [f.path for f in os.scandir(stiffness_data_path) if f.path.endswith('.h5')]
stiff_file_paths.sort()
stiffness_dfs = {}
for i, file_path in enumerate(stiff_file_paths):
    stiffness_dfs[f'df{i}'] = pd.read_hdf(file_path)['Stiffness']

# Strain Data
strain_file_paths = [f.path for f in os.scandir(strain_data_path) if f.path.endswith('.h5')]
strain_file_paths.sort()
strain_dfs = {}
for i, file_path in enumerate(strain_file_paths):
    strain_dfs[f'df{i}'] = pd.read_hdf(file_path)

# %%
def resample_stiffness_to_match_strain(strain_df, stiffness_df):
    strain_length = len(strain_df)
    stiffness_length = len(stiffness_df)
    
    if stiffness_length == 0:
        return pd.DataFrame(np.nan, index=strain_df.index, columns=[0])


    if strain_length > stiffness_length:
        # Interpolation: Upsample stiffness_df to match strain_df length
        x_old = np.linspace(0, 1, stiffness_length)  # Normalized index for stiffness
        x_new = np.linspace(0, 1, strain_length)  # Normalized index for strain
        stiffness_df_resampled = pd.DataFrame(np.interp(x_new, x_old, stiffness_df))
    
    elif strain_length < stiffness_length:
        # Downsampling: Downsample stiffness_df to match strain_df length
        x_old = np.linspace(0, 1, stiffness_length)  # Normalized index for stiffness
        x_new = np.linspace(0, 1, strain_length)  # Normalized index for strain
        idx_new = np.searchsorted(x_old, x_new)
        idx_new = np.clip(idx_new, 0, stiffness_length - 1)  # Ensure indices are valid
        stiffness_df_resampled = stiffness_df.iloc[idx_new].reset_index(drop=True)
    
    else:
        # If already the same length, no action required
        stiffness_df_resampled = stiffness_df.reset_index(drop=True)
    
    return stiffness_df_resampled


########## Correcting starting stiffness values ##########
def percentage_change_from_max(stiffness_df):
    if not isinstance(stiffness_df, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if stiffness_df.empty or stiffness_df.isnull().all():
        return stiffness_df

    max_index = stiffness_df.idxmax()
    max_value = stiffness_df[max_index]

    if pd.isna(max_value) or max_value == 0:
        return pd.Series(np.nan, index=stiffness_df.index)

    percentage_change_df = (stiffness_df / max_value) * 100
    percentage_change_df.loc[:max_index] = 100  # Ensure correct assignment
    return percentage_change_df


# %%
last_cycle = {}
for key in stiffness_dfs.keys():
    last_cycle[key] = len(stiffness_dfs[key])

# %%
#### Resample Strain, Smooth Strain and Stiffness Daata, and Match the Time Stamps ####

stiffness_post = {}
strain_post = {}
# Use the key from strain_dfs
for key, strain_df in strain_dfs.items():
    
    # This check for 'df0' might be important if its index is problematic

    if key == 'df2':
        strain_df = strain_df.iloc[:,:-8]
        
    # Resample the strain data and smooth it with rolling mean
    strain_resampled = strain_df.resample("200s").mean().rolling(10).mean()
    strain_resampled = strain_resampled.dropna()

    ##### Custom Feature Engineering #####
    strain_temp = np.cumsum(abs(np.diff(strain_resampled, axis=0)), axis=0)
    strain_temp = pd.DataFrame(strain_temp, columns=strain_resampled.columns)
    strain_temp.index = strain_resampled.index[1:]
    strain_resampled = strain_temp
    
    strain_post[key] = strain_resampled
    
    # Get the corresponding stiffness_df using the same key from stiffness_dfs
    stiffness_df = stiffness_dfs[key].rolling(50).mean()
    stiffness_df = stiffness_df.dropna()
    
    # Calculate the percentage change from the maximum value
    stiffness_df = percentage_change_from_max(stiffness_df)
    
    # Resample the stiffness data to match the strain
    stiffness_resampled = resample_stiffness_to_match_strain(strain_resampled, stiffness_df)

    # Store the resampled stiffness in stiffness_post
    stiffness_post[key] = pd.DataFrame(stiffness_resampled)
    stiffness_post[key].index = strain_post[key].index

# %%
target_indexes = {}

def find_closest_index(array, target):
    # Find index of the closest value to the target in the array
    idx = np.abs(array - target).argmin()
    return idx

for key, values in stiffness_post.items():
    stiffness_values = np.array(values).flatten()

    if len(stiffness_values) < 2:
        print(f"Warning: Not enough data for '{key}' to find target indexes. Skipping.")
        target_indexes[key] = {99: 0, 95: 0, 90: 0, CONFIG["stiffness_drop_threshold"]: 0} # Default to 0
        continue
    
    closest_index_99 = find_closest_index(stiffness_values, 99)
    
    # Search for subsequent points in the remainder of the array
    filtered_values = stiffness_values[closest_index_99 + 1:]
    offset = closest_index_99 + 1
    
    if len(filtered_values) == 0:
        print(f"Warning: Not enough data for '{key}' after index 99. Using last available index.")
        last_idx = len(stiffness_values) - 1
        target_indexes[key] = {
            99: closest_index_99, 95: last_idx, 90: last_idx, CONFIG["stiffness_drop_threshold"]: last_idx
        }
        continue

    target_indexes[key] = {
        99: closest_index_99,
        95: find_closest_index(filtered_values, 95) + offset,
        90: find_closest_index(filtered_values, 90) + offset,
        # The original code had 70 here, which may be a typo. I'll use the config threshold.
        CONFIG["stiffness_drop_threshold"]: find_closest_index(filtered_values, CONFIG["stiffness_drop_threshold"]) + offset
    }

# %%
strain_x_rescaled = {}
# Truncate data
drop = CONFIG["stiffness_drop_threshold"]
keys_to_process = list(stiffness_post.keys()) # Create a copy to iterate over

for key in keys_to_process:
    if key not in target_indexes or len(stiffness_post[key]) <= target_indexes[key][drop]:
        print(f"Skipping truncation for {key} due to insufficient data.")
        # Remove problematic key if it exists
        if key in stiffness_post: del stiffness_post[key]
        if key in strain_post: del strain_post[key]
        continue

    cut_index = target_indexes[key][drop]
    stiffness_post[key] = stiffness_post[key][:cut_index+1]
    strain_post[key] = strain_post[key].iloc[:cut_index+1]

    # Create rescaled x-axis for plotting
    time_x = strain_post[key].index.total_seconds()
    max_time = time_x.max()
    strain_x_rescaled[key] = time_x * (last_cycle[key] / max_time) if max_time > 0 else time_x

# %%
strain_data = []
stiffness_data = []

# Iterate over a sorted list of keys to ensure consistent order
for key in sorted(strain_post.keys()):
    if key not in DF_INDICES: # Ensure we only use df1, df2, df3, df4
        continue

    strain_data.append(strain_post[key].values)
    
    # Reshape stiffness data to ensure it is (N, 1)
    stiffness_values = stiffness_post[key].values
    if stiffness_values.ndim == 1:
        stiffness_values = stiffness_values.reshape(-1, 1)
    
    stiffness_data.append(stiffness_values)

# %% [markdown]
# # VISUALIZE PREPROCESSED STRAIN DATA

# %%
plt.figure(figsize=(15, 10))
for key in sorted(strain_post.keys()):
    if key in strain_x_rescaled:
        # Plotting the mean of all strain sensors for each FOD
        plt.plot(strain_x_rescaled[key], strain_post[key].mean(axis=1), label=f"FOD {key}")
plt.title("Mean Strain vs. Cycles for each FOD")
plt.xlabel("Cycles")
plt.ylabel("Mean Strain")
plt.legend()
plt.grid(True)
output_dir = "output_images/FOD3_5fold"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "preprocessed_strain_data.png"))
plt.close()

# %% [markdown]
# # 4. MODEL, HELPERS, AND CROSS-VALIDATION

# %%
class EdgeAttrGNN(nn.Module):
    def __init__(self, num_node_features, edge_feature_dim, hidden_dim, output_dim, num_gnn_layers, dropout_p):
        super().__init__()
        self.dropout_p, self.num_gnn_layers = dropout_p, num_gnn_layers
        self.node_emb = nn.Linear(num_node_features, hidden_dim)
        self.convs, self.norms, self.dropouts = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(pyg_nn.GENConv(                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggr="add",
                msg_norm=True,          # Often beneficial
                learn_msg_scale=True,   # Often beneficial
                num_layers=2,           # Internal MLP depth
                norm=None,              # Using external BatchNorm/LayerNorm
                edge_dim=edge_feature_dim # Specify input edge dim
            ))
            self.norms.append(pyg_nn.BatchNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.pool = pyg_nn.global_mean_pool
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            #nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_emb(x)
        for i in range(self.num_gnn_layers):
            residual = x
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x)
            x = x + residual
            x = F.leaky_relu(x)
            x = self.dropouts[i](x)
        return self.readout(self.pool(x, batch))

def weighted_mse_loss(pred, target, weight_range, weight_value):
    mse_loss = F.mse_loss(pred, target, reduction='none')
    weight_mask = (target >= weight_range[0]) & (target <= weight_range[1])
    weights = torch.ones_like(target)
    weights[weight_mask] = weight_value
    return (mse_loss * weights).mean()

def collate_fn(data_list):
    augmented_data_list = []
    use_subsampling = CONFIG.get("use_subsampling", False)
    use_shuffling = CONFIG.get("use_shuffling", False)
    min_nodes = CONFIG.get("min_nodes_subsampling", 16)

    for d in data_list:
        num_nodes = d.num_nodes
        
        # Start with an ordered list of nodes
        node_indices = torch.arange(num_nodes)

        # Apply shuffling if enabled
        if use_shuffling:
            node_indices = node_indices[torch.randperm(num_nodes)]

        # Apply subsampling if enabled
        if use_subsampling and num_nodes > min_nodes:
            k = torch.randint(min_nodes, num_nodes + 1, (1,)).item()
            subset = node_indices[:k]
        else:
            subset = node_indices

        edge_index, _ = pyg.utils.subgraph(subset, d.edge_index, relabel_nodes=True, num_nodes=num_nodes)
        x = d.x[subset]
        row, col = edge_index
        edge_attr = (x[row] - x[col]).view(-1, 1)
        augmented_data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=d.y))
        
    return pyg.data.Batch.from_data_list(augmented_data_list)

def train_model(model, optimizer, scheduler, train_loader, val_loader, device, config, fold_key):
    best_val_loss, epochs_no_improve = float('inf'), 0
    best_model_path = os.path.join(output_dir, f"best_model_{fold_key}.pth")
    train_losses, val_losses = [], []
    criterion = nn.MSELoss()

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            # --- DATA AUGMENTATION ---
            if config.get("use_augmentation", False) and model.training:
                noise = torch.randn_like(data.x) * config.get("noise_level", 0.01)
                data.x = data.x + noise
            # -------------------------
            optimizer.zero_grad()
            pred = model(data)
            loss = weighted_mse_loss(pred, data.y, config["weight_range"], config["weight_value"]) if config["use_weighted_loss"] else criterion(pred, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["clip_norm"])
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                total_val_loss += criterion(model(data), data.y).item() * data.num_graphs
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch:03d} | Fold: {fold_key} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = avg_val_loss, 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= config["patience"]:
            print(f"Early stopping at epoch {epoch} for fold {fold_key}.")
            break
            
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model, train_losses, val_losses

def inference(model, loader, device, norm_params):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for data_batch in loader:
            data_batch = data_batch.to(device)
            pred_norm = model(data_batch)
            true_val_norm = data_batch.y.cpu().numpy().flatten()
            pred_val_norm = pred_norm.cpu().numpy().flatten()
            true_val_unnorm = unnormalize_target(true_val_norm, norm_params)
            pred_val_unnorm = unnormalize_target(pred_val_norm, norm_params)
            all_true.extend(true_val_unnorm)
            all_pred.extend(pred_val_unnorm)
    all_true, all_pred = np.array(all_true), np.array(all_pred)
    mse = np.mean((all_true - all_pred) ** 2)
    rmse = np.sqrt(mse)
    nonzero_mask = all_true != 0
    mape = np.mean(np.abs((all_true[nonzero_mask] - all_pred[nonzero_mask]) / all_true[nonzero_mask])) * 100 if np.any(nonzero_mask) else float('inf')
    return all_true, all_pred, mse, rmse, mape

def plot_predictions(true_values, predicted_values, metrics, title, key):
    mse, rmse, mape = metrics
    plt.figure(figsize=(16, 9))
    plt.plot(np.arange(len(true_values)), true_values, label="True Values", color="b")
    plt.plot(np.arange(len(predicted_values)), predicted_values, label="Predicted Values", color="r", linestyle='--')
    metrics_text = f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%"
    plt.annotate(metrics_text, xy=(0.95, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8, ec='gray'),
                 ha='right', va='top', fontsize=18, family='monospace')
    plt.xlabel("Cycles"); plt.ylabel("Stiffness (%)"); plt.title(title)
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"predictions_{key}.png"))
    plt.close()

def unnormalize_target(y_norm, p):
    return y_norm * (p['target_max'] - p['target_min']) + p['target_min']

# --- Helper function to enable dropout layers during test-time ---
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

# --- Monte Carlo Dropout Inference Function ---
def mc_dropout_inference(model, loader, device, norm_params, num_samples):
    model.eval()
    enable_dropout(model)
    
    all_true_unnormalized = []
    mc_predictions_unnormalized = []
    
    with torch.no_grad():
        for data_batch in loader:
            data_batch = data_batch.to(device)
            true_val_norm = data_batch.y.cpu().numpy()
            
            # Un-normalize and store the true values for this batch
            true_unnorm = unnormalize_target(true_val_norm, norm_params)
            all_true_unnormalized.extend(true_unnorm.flatten())

            # Perform MC forward passes for the batch
            batch_mc_preds = []
            for _ in range(num_samples):
                pred_norm = model(data_batch).cpu().numpy()
                pred_unnorm = unnormalize_target(pred_norm, norm_params)
                batch_mc_preds.append(pred_unnorm.flatten())
            
            # Transpose to get [num_points, num_samples] and append
            mc_predictions_unnormalized.extend(np.array(batch_mc_preds).T)
            
    all_true = np.array(all_true_unnormalized)
    mc_predictions = np.array(mc_predictions_unnormalized)

    mean_preds = mc_predictions.mean(axis=1)
    std_preds = mc_predictions.std(axis=1)

    # Compute final metrics
    mse = np.mean((all_true - mean_preds) ** 2)
    rmse = np.sqrt(mse)
    
    # Corrected MAPE calculation
    nonzero_mask = all_true != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((all_true[nonzero_mask] - mean_preds[nonzero_mask]) / all_true[nonzero_mask])) * 100
    else:
        mape = float('inf')
        
    return all_true, mean_preds, std_preds, mse, rmse, mape

# --- Plotting Functions ---
def plot_mc_predictions(true_values, mean_preds, std_preds, metrics, title, key, confidence_level=1.96):
    mse, rmse, mape = metrics
    plt.figure(figsize=(16, 9))
    x_axis = strain_x_rescaled[key]
    plt.plot(x_axis, true_values, label="True Values", color="b")
    plt.plot(x_axis, mean_preds, label="Predicted Mean", color="r", linestyle='--')
    plt.fill_between(x_axis, mean_preds - confidence_level * std_preds,
                     mean_preds + confidence_level * std_preds,
                     color="r", alpha=0.2, label=f"95% CI")
    metrics_text = f"MSE:  {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%"
    plt.annotate(metrics_text, xy=(0.95, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8, ec='gray'),
                 ha='right', va='top', fontsize=18, family='monospace')
    plt.xlabel("Cycles"); plt.ylabel("Stiffness (%)"); plt.title(title)
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"mc_predictions_{key}.png"))
    plt.close()

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

# %%
# ===================================================================================
# 5-FOLD CROSS-VALIDATION LOOP
# ===================================================================================
all_fold_metrics = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

for fold_key in DF_INDICES.keys():
    print(f"\n===== Processing Fold: {fold_key} =====")
    CONFIG['test_fold_key'] = fold_key
    
    # 1. Data Splitting
    test_idx = DF_INDICES[fold_key]
    train_indices = [i for i in DF_INDICES.values() if i != test_idx]
    
    # 2. Normalization (based on training data of the current fold)
    train_inputs_flat = np.concatenate([strain_data[i].flatten() for i in train_indices])
    train_targets_flat = np.concatenate([stiffness_data[i].flatten() for i in train_indices])
    
    norm_params = {
        'input_mean': train_inputs_flat.mean(),
        'input_std': train_inputs_flat.std(),
        'target_min': train_targets_flat.min(),
        'target_max': train_targets_flat.max()
    }

    def normalize_input(x, p): return (x - p['input_mean']) / (p['input_std'] + 1e-8)
    def normalize_target(y, p): return (y - p['target_min']) / (p['target_max'] - p['target_min'] + 1e-8)

    # 3. Graph Construction
    specimen_data_fold = []
    for i in range(len(strain_data)):
        num_nodes = strain_data[i].shape[1]
        edge_index = torch.tensor(list(itertools.permutations(range(num_nodes), 2)), dtype=torch.long).t().contiguous()
        hi_tensor = normalize_input(torch.tensor(strain_data[i], dtype=torch.float), norm_params)
        stiffness_tensor = normalize_target(torch.tensor(stiffness_data[i], dtype=torch.float), norm_params)
        data_list = []
        for t in range(hi_tensor.shape[0]):
            x = hi_tensor[t].reshape(-1, 1)
            row, col = edge_index
            edge_attr = (x[row] - x[col]).view(-1, 1)
            y = stiffness_tensor[t].reshape(1, 1)
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        specimen_data_fold.append(data_list)

    train_data = [graph for i in train_indices for graph in specimen_data_fold[i]]
    val_data = specimen_data_fold[test_idx]
    test_data = specimen_data_fold[test_idx]

    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn if CONFIG.get("use_subsampling", False) or CONFIG.get("use_shuffling", False) else None)
    val_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=CONFIG["batch_size"], shuffle=False)

    # 4. Model Initialization and Training
    model = EdgeAttrGNN(
        num_node_features=1, edge_feature_dim=1,
        hidden_dim=CONFIG["hidden_dim"], output_dim=CONFIG["output_dim"],
        num_gnn_layers=CONFIG["num_gnn_layers"], dropout_p=CONFIG["dropout_p"]
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)

    start_time = time.time()
    model, train_losses, val_losses = train_model(model, optimizer, scheduler, train_loader, val_loader, device, CONFIG, fold_key)
    print(f"Training for fold {fold_key} took: {time.time() - start_time:.2f} seconds")

    # 5. Plotting Loss Curves for the fold
    plt.figure(figsize=(10, 5))
    plt.semilogy(train_losses, label='Training Loss')
    plt.semilogy(val_losses, label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Log Loss'); plt.title(f'Loss vs. Epochs for Fold {fold_key}')
    plt.legend(); plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(output_dir, f"loss_vs_epochs_{fold_key}.png"))
    plt.close()

    # 6. Inference and Evaluation
    true_vals, pred_vals, mse, rmse, mape = inference(model, test_loader, device, norm_params)
    all_fold_metrics[fold_key] = {'MSE': mse, 'RMSE': rmse, 'MAPE': mape}
    
    fod_name = f"FOD{int(fold_key.split('f')[-1]) + 3}"
    plot_predictions(true_vals, pred_vals, (mse, rmse, mape), f"GNN Prediction on Test Fold: {fod_name}", fold_key)

    # --- Monte Carlo Dropout Inference ---
    if fold_key == CONFIG['test_fold_key'] and CONFIG['test_fold_key'] == 'df0':
        print(f"--- Running Monte Carlo Dropout for fold {fold_key} ---")
        mc_true, mc_mean, mc_std, mc_mse, mc_rmse, mc_mape = mc_dropout_inference(
            model, test_loader, device, norm_params, num_samples=CONFIG["mc_dropout_samples"]
        )
        plot_mc_predictions(
            mc_true, mc_mean, mc_std, (mc_mse, mc_rmse, mc_mape),
            "FOD3 - Cross Valdiation Fold", fold_key
        )

# %% [markdown]
# # 6. AGGREGATE AND SAVE RESULTS

# %%
# Calculate mean and std dev for each metric
mean_metrics = {metric: np.mean([all_fold_metrics[key][metric] for key in all_fold_metrics]) for metric in ['MSE', 'RMSE', 'MAPE']}
std_metrics = {metric: np.std([all_fold_metrics[key][metric] for key in all_fold_metrics]) for metric in ['MSE', 'RMSE', 'MAPE']}

final_results = {
    'mean_metrics': mean_metrics,
    'std_dev_metrics': std_metrics,
    'per_fold_metrics': all_fold_metrics
}

# Save results to JSON
results_path = os.path.join(output_dir, 'FOD3_5fold_results.json')
final_results_converted = convert_numpy_types(final_results)
with open(results_path, 'w') as f:
    json.dump(final_results_converted, f, indent=4)

print("\n===== Cross-Validation Finished =====")
print(f"Final results saved to: {results_path}")
print("\nMean Metrics:")
for metric, value in mean_metrics.items():
    print(f"  {metric}: {value:.4f}")
print("\nStandard Deviation of Metrics:")
for metric, value in std_metrics.items():
    print(f"  {metric}: {value:.4f}")

# %%
