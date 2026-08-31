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
    # "stiffness_drop_threshold" will be set in the loop

    # Leave-One-Out Cross-Validation Setup
    "test_fold_key": 'df2', # Options: 'df1', 'df2', 'df3', 'df4'

    # Model Architecture
    "hidden_dim": 8,
    "output_dim": 1,
    "num_gnn_layers": 3,
    "dropout_p": 0.3,

    # Training
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "epochs": 2000,
    "patience": 40,
    "batch_size": 128,
    "clip_norm": 1.0,

    # Weighted Loss
    "use_weighted_loss": True,
    "weight_range": (0.7, 0.90),
    "weight_value": 2.0,
    
    # Inference
    "mc_dropout_samples": 100,
    "use_augmentation": False,
    "noise_level": 0.01,
    "use_subsampling": False,
    "use_shuffling": False,
    "min_nodes_subsampling": 16
}
# Map dataframe names to indices for fold selection
# We exclude 'df0' as in the original script
DF_INDICES = {'df1': 0, 'df2': 1, 'df3': 2, 'df4': 3}

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
# # UTILITY FUNCTIONS

# %%
def resample_stiffness_to_match_strain(strain_df, stiffness_df):
    strain_length = len(strain_df)
    stiffness_length = len(stiffness_df)
    if stiffness_length == 0: return pd.DataFrame(np.nan, index=strain_df.index, columns=[0])
    if strain_length > stiffness_length:
        x_old, x_new = np.linspace(0, 1, stiffness_length), np.linspace(0, 1, strain_length)
        return pd.DataFrame(np.interp(x_new, x_old, stiffness_df))
    elif strain_length < stiffness_length:
        x_old, x_new = np.linspace(0, 1, stiffness_length), np.linspace(0, 1, strain_length)
        idx_new = np.clip(np.searchsorted(x_old, x_new), 0, stiffness_length - 1)
        return stiffness_df.iloc[idx_new].reset_index(drop=True)
    return stiffness_df.reset_index(drop=True)

def percentage_change_from_max(stiffness_df):
    if not isinstance(stiffness_df, pd.Series) or stiffness_df.empty or stiffness_df.isnull().all(): return stiffness_df
    max_index, max_value = stiffness_df.idxmax(), stiffness_df.max()
    if pd.isna(max_value) or max_value == 0: return pd.Series(np.nan, index=stiffness_df.index)
    percentage_change_df = (stiffness_df / max_value) * 100
    percentage_change_df.loc[:max_index] = 100
    return percentage_change_df

def find_closest_index(array, target): return np.abs(array - target).argmin()

def convert_numpy_types(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
    return obj

def unnormalize_target(y_norm, p): return y_norm * (p['target_max'] - p['target_min']) + p['target_min']

# %% [markdown]
# # MODEL, TRAINING, AND INFERENCE

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

def train_model(model, optimizer, scheduler, train_loader, val_loader, device, config, fold_key, output_dir):
    best_val_loss, epochs_no_improve = float('inf'), 0
    os.makedirs(output_dir, exist_ok=True)
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

    # Filter for fair comparison: only consider true values between 85 and 100
    fair_comparison_mask = (all_true >= 85) & (all_true <= 100)
    
    if np.any(fair_comparison_mask):
        true_filtered = all_true[fair_comparison_mask]
        pred_filtered = all_pred[fair_comparison_mask]

        mse = np.mean((true_filtered - pred_filtered) ** 2)
        rmse = np.sqrt(mse)
        
        # For MAPE, avoid division by zero
        nonzero_mask = true_filtered != 0
        if np.any(nonzero_mask):
            mape = np.mean(np.abs((true_filtered[nonzero_mask] - pred_filtered[nonzero_mask]) / true_filtered[nonzero_mask])) * 100
        else:
            mape = float('nan')
    else:
        # Handle case where no data points are in the desired range
        mse, rmse, mape = float('nan'), float('nan'), float('nan')

    return all_true, all_pred, mse, rmse, mape

def plot_predictions(true, pred, metrics, title, key, output_dir, strain_x):
    mse, rmse, mape = metrics
    plt.figure(figsize=(16, 9))
    plt.plot(strain_x[key], true, label="True Values", color="b")
    plt.plot(strain_x[key], pred, label="Predicted Values", color="r", linestyle='--')
    metrics_text = f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%"
    plt.annotate(metrics_text, (0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                 bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8, ec='gray'))
    plt.xlabel("Cycles"); plt.ylabel("Stiffness (%)"); plt.title(title)
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"predictions_{key}.png"))
    plt.close()

def plot_predictions(true, pred, metrics, title, key, output_dir, strain_x):
    mse, rmse, mape = metrics
    plt.figure(figsize=(16, 9))
    plt.plot(strain_x[key], true, label="True Values", color="b")
    plt.plot(strain_x[key], pred, label="Predicted Values", color="r", linestyle='--')
    metrics_text = f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%"
    plt.annotate(metrics_text, (0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                 bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8, ec='gray'))
    plt.xlabel("Cycles"); plt.ylabel("Stiffness (%)"); plt.title(title)
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"predictions_{key}.png"))
    plt.close()

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
    return all_true, mc_predictions

# %% [markdown]
# # SENSITIVITY ANALYSIS MAIN SCRIPT

# %%
def main():
    thresholds_to_search = [85, 80, 70, 60]
    all_threshold_results = {}
    search_results_dir = "Threshold_search/results"
    os.makedirs(search_results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Base Data Loading ---
    stiffness_data_path = '../Data/Stiffness_Reduction'
    strain_data_path = '../Data/Strain'
    stiff_file_paths = sorted([f.path for f in os.scandir(stiffness_data_path) if f.path.endswith('.h5')])
    base_stiffness_dfs = {f'df{i}': pd.read_hdf(file_path)['Stiffness'] for i, file_path in enumerate(stiff_file_paths)}
    strain_file_paths = sorted([f.path for f in os.scandir(strain_data_path) if f.path.endswith('.h5')])
    base_strain_dfs = {f'df{i}': pd.read_hdf(file_path) for i, file_path in enumerate(strain_file_paths)}

    for threshold in thresholds_to_search:
        print(f"\n{'='*60}\n===== PROCESSING THRESHOLD: {threshold} =====\n{'='*60}\n")
        CONFIG["stiffness_drop_threshold"] = threshold
        threshold_output_dir = os.path.join(search_results_dir, f"threshold_{threshold}")
        os.makedirs(threshold_output_dir, exist_ok=True)

        # --- Per-Threshold Data Preprocessing ---
        stiffness_dfs = {k: v.copy() for k, v in base_stiffness_dfs.items()}
        strain_dfs = {k: v.copy() for k, v in base_strain_dfs.items()}
        last_cycle = {key: len(stiffness_dfs[key]) for key in stiffness_dfs.keys()}

        stiffness_post, strain_post = {}, {}
        for key, strain_df in strain_dfs.items():
            if key == 'df0': continue
            if key == 'df2': strain_df = strain_df.iloc[:,:-8]
            
            strain_resampled = strain_df.resample("200s").mean().rolling(10).mean().dropna()
            strain_temp = pd.DataFrame(np.cumsum(abs(np.diff(strain_resampled, axis=0)), axis=0), columns=strain_resampled.columns)
            strain_temp.index = strain_resampled.index[1:]
            strain_post[key] = strain_temp
            
            stiffness_df = percentage_change_from_max(stiffness_dfs[key].rolling(50).mean().dropna())
            stiffness_resampled = resample_stiffness_to_match_strain(strain_post[key], stiffness_df)
            stiffness_post[key] = pd.DataFrame(stiffness_resampled)
            stiffness_post[key].index = strain_post[key].index

        target_indexes = {}
        for key, values in stiffness_post.items():
            stiffness_values = np.array(values).flatten()
            if len(stiffness_values) < 2:
                target_indexes[key] = {99: 0, 95: 0, 90: 0, threshold: 0}; continue
            closest_index_99 = find_closest_index(stiffness_values, 99)
            filtered_values = stiffness_values[closest_index_99 + 1:]
            offset = closest_index_99 + 1
            if len(filtered_values) == 0:
                last_idx = len(stiffness_values) - 1
                target_indexes[key] = {99: closest_index_99, 95: last_idx, 90: last_idx, threshold: last_idx}
            else:
                target_indexes[key] = {
                    99: closest_index_99,
                    95: find_closest_index(filtered_values, 95) + offset,
                    90: find_closest_index(filtered_values, 90) + offset,
                    threshold: find_closest_index(filtered_values, threshold) + offset
                }

        strain_x_rescaled = {}
        for key in list(stiffness_post.keys()):
            if key not in target_indexes or len(stiffness_post[key]) <= target_indexes[key][threshold]:
                if key in stiffness_post: del stiffness_post[key]
                if key in strain_post: del strain_post[key]
                continue
            cut_index = target_indexes[key][threshold]
            stiffness_post[key] = stiffness_post[key][:cut_index+1]
            strain_post[key] = strain_post[key].iloc[:cut_index+1]
            time_x = strain_post[key].index.total_seconds()
            strain_x_rescaled[key] = time_x * (last_cycle[key] / time_x.max()) if time_x.max() > 0 else time_x

        strain_data, stiffness_data = [], []
        for key in sorted(strain_post.keys()):
            if key not in DF_INDICES: continue
            strain_data.append(strain_post[key].values)
            stiffness_values = stiffness_post[key].values
            stiffness_data.append(stiffness_values.reshape(-1, 1) if stiffness_values.ndim == 1 else stiffness_values)

        # --- 4-FOLD CROSS-VALIDATION ---
        all_fold_metrics = {}
        for fold_key in DF_INDICES.keys():
            print(f"\n----- Processing Fold: {fold_key} for Threshold: {threshold} -----")
            test_idx = DF_INDICES[fold_key]
            train_indices = [i for i in DF_INDICES.values() if i != test_idx]
            
            train_inputs = np.concatenate([strain_data[i].flatten() for i in train_indices])
            train_targets = np.concatenate([stiffness_data[i].flatten() for i in train_indices])
            norm_params = {'input_mean': train_inputs.mean(), 'input_std': train_inputs.std(), 'target_min': train_targets.min(), 'target_max': train_targets.max()}

            def normalize_input(x, p): return (x - p['input_mean']) / (p['input_std'] + 1e-8)
            def normalize_target(y, p): return (y - p['target_min']) / (p['target_max'] - p['target_min'] + 1e-8)

            specimen_data_fold = []
            for i in range(len(strain_data)):
                num_nodes = strain_data[i].shape[1]
                edge_index = torch.tensor(list(itertools.permutations(range(num_nodes), 2)), dtype=torch.long).t().contiguous()
                hi = normalize_input(torch.tensor(strain_data[i], dtype=torch.float), norm_params)
                stiffness = normalize_target(torch.tensor(stiffness_data[i], dtype=torch.float), norm_params)
                specimen_data_fold.append([Data(x=hi[t].reshape(-1, 1), edge_index=edge_index, edge_attr=(hi[t][edge_index[0]] - hi[t][edge_index[1]]).view(-1, 1), y=stiffness[t].reshape(1, 1)) for t in range(hi.shape[0])])

            train_data, val_data = [g for i in train_indices for g in specimen_data_fold[i]], specimen_data_fold[test_idx]
            train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

            model = EdgeAttrGNN(1, 1, **{k: CONFIG[k] for k in ['hidden_dim', 'output_dim', 'num_gnn_layers', 'dropout_p']}).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=10)

            model, train_losses, val_losses = train_model(model, optimizer, scheduler, train_loader, val_loader, device, CONFIG, fold_key, threshold_output_dir)

            plt.figure(figsize=(10, 5))
            plt.semilogy(train_losses, label='Training Loss'); plt.semilogy(val_losses, label='Validation Loss')
            plt.xlabel('Epochs'); plt.ylabel('Log Loss'); plt.title(f'Loss vs. Epochs: Fold {fold_key} (Thresh: {threshold})')
            plt.legend(); plt.grid(True, which="both", ls="--")
            plt.savefig(os.path.join(threshold_output_dir, f"loss_vs_epochs_{fold_key}.png")); plt.close()

            true_vals, pred_vals, mse, rmse, mape = inference(model, test_loader, device, norm_params)
            all_fold_metrics[fold_key] = {'MSE': mse, 'RMSE': rmse, 'MAPE': mape}
            fod_name = f"FOD{int(fold_key.split('f')[-1]) + 3}"
            plot_predictions(true_vals, pred_vals, (mse, rmse, mape), f"GNN on Test Fold: {fod_name} (Thresh: {threshold})", fold_key, threshold_output_dir, strain_x_rescaled)

        # --- AGGREGATE FOLD METRICS FOR CURRENT THRESHOLD ---
        mean_metrics = {metric: np.mean([m[metric] for m in all_fold_metrics.values()]) for metric in ['MSE', 'RMSE', 'MAPE']}
        all_threshold_results[threshold] = {'mean_metrics': mean_metrics, 'fold_metrics': all_fold_metrics}
        print(f"\n--- Mean Metrics for Threshold {threshold} ---")
        print(json.dumps(convert_numpy_types(mean_metrics), indent=4))

    # --- SAVE FINAL AGGREGATED RESULTS ---
    final_results_path = os.path.join("Threshold_search", "threshold_search_results.json")
    with open(final_results_path, 'w') as f:
        json.dump(convert_numpy_types(all_threshold_results), f, indent=4)
    print(f"\n\nFINAL RESULTS SAVED TO: {final_results_path}")

if __name__ == '__main__':
    main()