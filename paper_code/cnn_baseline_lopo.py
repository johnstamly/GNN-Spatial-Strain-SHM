# %% [markdown]
# # IMPORTS

# %%
import os
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import torch
import json
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime
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
    "mc_dropout_samples": 100,
    "use_augmentation": False,
    "noise_level": 0.01
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
# Reshape data into 4x4 grids and create a single tensor
all_strain_data = []
all_stiffness_data = []

for key in sorted(strain_post.keys()):
    if key not in DF_INDICES:
        continue

    strain_df = strain_post[key]
    stiffness_df = stiffness_post[key]

    # Ensure the number of sensors is 16
    if strain_df.shape[1] == 16:
        # Reshape each time step into a 4x4 grid and add a channel dimension
        reshaped_strain = strain_df.values.reshape(-1, 1, 4, 4)
        all_strain_data.append(torch.tensor(reshaped_strain, dtype=torch.float32))
        
        # Ensure stiffness data is correctly shaped
        stiffness_values = stiffness_df.values.reshape(-1, 1)
        all_stiffness_data.append(torch.tensor(stiffness_values, dtype=torch.float32))

# %% [markdown]
# # VISUALIZE PREPROCESSED STRAIN DATA
#
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
output_dir = "output_images/CNN_5fold"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "preprocessed_strain_data.png"))
plt.close()

# %% [markdown]
# # 4. MODEL, HELPERS, AND CROSS-VALIDATION

# %%
class CNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout_p):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        
        # To determine the size of the flattened layer, we can do a dummy forward pass
        # Or calculate it manually. A 4x4 input with two 2x2 convolutions (and no padding/stride)
        # will result in a 2x2 feature map.
        # (4 - 2 + 1) = 3x3 after conv1
        # (3 - 2 + 1) = 2x2 after conv2
        # So, the flattened size will be 32 * 2 * 2 = 128
        self.readout = nn.Sequential(
            nn.Linear(32 * 2 * 2, hidden_dim), # Adjusted input size
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input 'x' is expected to be the raw tensor from the DataLoader,
        # not a PyG Data object.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.readout(x)
        return x

def weighted_mse_loss(pred, target, weight_range, weight_value):
    mse_loss = F.mse_loss(pred, target, reduction='none')
    weight_mask = (target >= weight_range[0]) & (target <= weight_range[1])
    weights = torch.ones_like(target)
    weights[weight_mask] = weight_value
    return (mse_loss * weights).mean()


def train_model(model, optimizer, scheduler, train_loader, val_loader, device, config, fold_key):
    best_val_loss, epochs_no_improve = float('inf'), 0
    best_model_path = os.path.join(output_dir, f"best_model_{fold_key}.pth")
    train_losses, val_losses = [], []
    criterion = nn.MSELoss()

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            pred = model(inputs)
            loss = weighted_mse_loss(pred, targets, config["weight_range"], config["weight_value"]) if config["use_weighted_loss"] else criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["clip_norm"])
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                pred = model(inputs)
                total_val_loss += criterion(pred, targets).item() * inputs.size(0)
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
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            pred_norm = model(inputs)
            true_val_norm = targets.cpu().numpy().flatten()
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
    return y_norm * p['target_std'] + p['target_mean']

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
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            true_val_norm = targets.cpu().numpy()

            # Un-normalize and store the true values for this batch
            true_unnorm = unnormalize_target(true_val_norm, norm_params)
            all_true_unnormalized.extend(true_unnorm.flatten())

            # Perform MC forward passes for the batch
            batch_mc_preds = []
            for _ in range(num_samples):
                pred_norm = model(inputs).cpu().numpy()
                pred_unnorm = unnormalize_target(pred_norm, norm_params)
                batch_mc_preds.append(pred_unnorm.flatten())
            
            # Transpose to get [num_points, num_samples] and append
            mc_predictions_unnormalized.extend(np.array(batch_mc_preds).T)
            
    all_true = np.array(all_true_unnormalized)
    mc_predictions = np.array(mc_predictions_unnormalized)
    return all_true, mc_predictions
# %%
# ===================================================================================
# 4-FOLD CROSS-VALIDATION LOOP
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

    # Create datasets for the current fold
    X_train_list = [all_strain_data[i] for i in train_indices]
    y_train_list = [all_stiffness_data[i] for i in train_indices]
    X_test = all_strain_data[test_idx]
    y_test = all_stiffness_data[test_idx]

    # Concatenate training data
    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)

    # Further split training data for validation
    num_train_samples = X_train.shape[0]
    split_idx = int(num_train_samples * 0.8)
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]

    # 2. Normalization (based on training data of the current fold)
    feature_mean = X_train.mean()
    feature_std = X_train.std()
    target_mean = y_train.mean()
    target_std = y_train.std()

    norm_params = {
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'target_mean': target_mean,
        'target_std': target_std
    }

    # Apply normalization
    X_train = (X_train - feature_mean) / (feature_std + 1e-8)
    X_val = (X_val - feature_mean) / (feature_std + 1e-8)
    X_test = (X_test - feature_mean) / (feature_std + 1e-8)
    y_train = (y_train - target_mean) / (target_std + 1e-8)
    y_val = (y_val - target_mean) / (target_std + 1e-8)
    y_test = (y_test - target_mean) / (target_std + 1e-8)

    # 3. Create PyTorch Datasets and DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # 4. Model Initialization and Training
    model = CNN(
        hidden_dim=CONFIG["hidden_dim"],
        output_dim=CONFIG["output_dim"],
        dropout_p=CONFIG["dropout_p"]
    ).to(device)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
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
    
    plot_predictions(true_vals, pred_vals, (mse, rmse, mape), f"GNN Prediction on Test Fold: {fold_key}", fold_key)

# %% [markdown]
# # 5. AGGREGATE AND SAVE RESULTS

# %%
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

# Calculate mean and std dev for each metric
mean_metrics = {metric: np.mean([all_fold_metrics[key][metric] for key in all_fold_metrics]) for metric in ['MSE', 'RMSE', 'MAPE']}
std_metrics = {metric: np.std([all_fold_metrics[key][metric] for key in all_fold_metrics]) for metric in ['MSE', 'RMSE', 'MAPE']}

final_results = {
    'mean_metrics': mean_metrics,
    'std_dev_metrics': std_metrics,
    'per_fold_metrics': all_fold_metrics
}

# Convert numpy types in results
final_results = convert_numpy_types(final_results)

# Save results to JSON
output_dir = "output_images/CNN_4Fold"
results_path = os.path.join(output_dir, '4Fold_results.json')
with open(results_path, 'w') as f:
    json.dump(final_results, f, indent=4)

print("\n===== Cross-Validation Finished =====")
print(f"Final results saved to: {results_path}")
print("\nMean Metrics:")
for metric, value in final_results['mean_metrics'].items():
    print(f"  {metric}: {value:.4f}")
print("\nStandard Deviation of Metrics:")
for metric, value in final_results['std_dev_metrics'].items():
    print(f"  {metric}: {value:.4f}")

# %%
