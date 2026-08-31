# %% [markdown]
# # MLP baseline (LOPO CV)
#
# The multi-layer perceptron baseline: the same point-wise task as GENEA, on
# the same HI features and the same truncation, but with no spatial structure
# at all -- the 16 sensor HI values enter as a flat vector.
#
# Assembled from `MLP_Paper_Plots_Stiffness.ipynb`, which ran one fold per
# execution with the fold chosen by hand. Here that is a loop over the four
# 16-sensor panels, with the model, optimizer and scheduler rebuilt inside the
# loop so folds cannot leak weights into each other. The preprocessing block is
# spliced from `genea_stiffness_lopo.py` with exactly one change: the HI step is
# wrapped in `if CONFIG["use_hi"]`. At the default True, the HI features and the
# 70% truncation are identical to the GENEA runs.
#
# Everything model-side follows the notebook rather than the GENEA scripts, and
# they genuinely differ -- see the caveat table in paper_code/README.md.
#
# CONFIG["use_hi"] = False feeds resampled, smoothed strain instead of the HI.
# That approximates the paper's with/without-HI ablation but does not reproduce
# it: measured here it costs ~4% RMSE, where the paper reports ~70%. See the
# "use_hi toggle" section of paper_code/README.md before relying on it.
#
# Run from the repository root:  python paper_code/mlp_baseline_lopo.py

# %% [markdown]
# # IMPORTS

# %%
import os
import numpy as np
import pandas as pd
import torch
import json
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    # Feed the HI (cumulative absolute first derivative of strain). Set False
    # for the no-HI ablation, which feeds the smoothed strain directly.
    "use_hi": True,

    # Model Architecture (MLP: input_dim, width, depth, activation)
    "input_dim": 16,
    "output_dim": 1,
    "width": 16,
    "depth": 3,
    "dropout_p": 0.2,

    # Training
    "learning_rate": 0.01,
    "weight_decay": 1e-3,
    "epochs": 1000,
    "patience": 80,
    "batch_size": 128,

    # Weighted Loss -- taken from the notebook's call site, not the function
    # defaults, which were never the values actually used.
    "use_weighted_loss": True,
    "weight_range": (0.0, 0.80),
    "weight_value": 6.0,

    # Inference
    "mc_dropout_samples": 100,
}

# The MLP takes a fixed 16-dimensional input, so only the 16-sensor panels
# (FOD4-FOD7) can be used. Values index into strain_data / stiffness_data.
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
# # DATA LOADING & UTILITY FUNCTIONS & PREPROCESSING
#
# Spliced verbatim from genea_stiffness_lopo.py.

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
    # The Health Indicator: cumulative absolute first derivative of strain.
    # CONFIG["use_hi"] = False skips it, giving the no-HI ablation input.
    if CONFIG["use_hi"]:
        strain_temp = np.cumsum(abs(np.diff(strain_resampled, axis=0)), axis=0)
        strain_temp = pd.DataFrame(strain_temp, columns=strain_resampled.columns)
        strain_temp.index = strain_resampled.index[1:]
        strain_resampled = strain_temp
    else:
        strain_resampled = strain_resampled.iloc[1:]
    
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
# # MODEL, HELPERS, AND CROSS-VALIDATION

# %%
class MLP(nn.Module):
    """Flat MLP over the 16 sensor HI values. No spatial structure."""

    def __init__(self, input_dim, output_dim, width, depth, activation, dropout_p):
        super(MLP, self).__init__()
        self.activation = activation

        self.input_layer = nn.Linear(input_dim, width, bias=True)
        self.dropout = nn.Dropout(dropout_p)

        # NOTE: the notebook appends the *same* dropout module instance after
        # every hidden linear layer. Preserved as-is; nn.Dropout is stateless,
        # so sharing the instance is equivalent to separate ones, and
        # enable_dropout() still finds it for MC dropout.
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(nn.Linear(width, width, bias=True))
            self.hidden_layers.append(self.dropout)

        self.output_layer = nn.Linear(width, output_dim, bias=True)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        return self.output_layer(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weighted_mse_loss(pred, target, weight_range, weight_value):
    mse_loss = F.mse_loss(pred, target, reduction='none')
    weight_mask = (target >= weight_range[0]) & (target <= weight_range[1])
    weights = torch.ones_like(target)
    weights[weight_mask] = weight_value
    return (mse_loss * weights).mean()


def validation(model, loader, device):
    model.eval()
    total = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item() * x.size(0)
    return total / len(loader.dataset)


def train_model(model, optimizer, scheduler, train_loader, val_loader, device,
                config, fold_key, writer, output_dir):
    """Notebook training loop, unchanged in substance.

    Two quirks are deliberate, not oversights:
      * the scheduler steps on the *training* loss, not the validation loss
        (the GENEA scripts step on validation);
      * the first three epochs are skipped for best-model tracking and early
        stopping.
    """
    best_val_loss, epochs_no_improve = float('inf'), 0
    train_losses, val_losses = [], []
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f"best_model_{fold_key}.pth")

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = (weighted_mse_loss(pred, y, config["weight_range"], config["weight_value"])
                    if config["use_weighted_loss"] else F.mse_loss(pred, y))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_loader.dataset)
        train_losses.append(total_loss)

        val_mse = validation(model, val_loader, device)
        val_losses.append(val_mse)
        print(f"Epoch {epoch:03d} | Fold: {fold_key} | Loss: {total_loss:.6f} | Val MSE: {val_mse:.6f}")

        if writer is not None:
            writer.add_scalar("loss", total_loss, epoch)
            writer.add_scalar("val_mse", val_mse, epoch)

        scheduler.step(total_loss)

        if epoch < 3:
            continue

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config["patience"]:
            print(f"Early stopping at epoch {epoch} for fold {fold_key}. Best Val MSE: {best_val_loss:.6f}")
            break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model, train_losses, val_losses


def unnormalize_target(y_norm, p):
    return y_norm * (p['target_max'] - p['target_min']) + p['target_min']


def inference(model, loader, device, norm_params):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_true.extend(unnormalize_target(y.cpu().numpy().flatten(), norm_params))
            all_pred.extend(unnormalize_target(pred.cpu().numpy().flatten(), norm_params))
    all_true, all_pred = np.array(all_true), np.array(all_pred)
    mse = np.mean((all_true - all_pred) ** 2)
    rmse = np.sqrt(mse)
    nonzero = all_true != 0
    mape = (np.mean(np.abs((all_true[nonzero] - all_pred[nonzero]) / all_true[nonzero])) * 100
            if np.any(nonzero) else float('inf'))
    return all_true, all_pred, mse, rmse, mape


def enable_dropout(model):
    """Keep dropout active at test time, which is what makes MC dropout work."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_inference(model, loader, device, norm_params, num_samples):
    model.eval()
    enable_dropout(model)
    all_true, mc_runs = [], []
    with torch.no_grad():
        for _ in range(num_samples):
            run = []
            for x, _ in loader:
                run.extend(model(x.to(device)).cpu().numpy().flatten())
            mc_runs.append(unnormalize_target(np.array(run), norm_params))
        for _, y in loader:
            all_true.extend(unnormalize_target(y.numpy().flatten(), norm_params))

    all_true = np.array(all_true)
    mc = np.stack(mc_runs, axis=1)          # [num_points, num_samples]
    mean_preds, std_preds = mc.mean(axis=1), mc.std(axis=1)

    mse = np.mean((all_true - mean_preds) ** 2)
    rmse = np.sqrt(mse)
    nonzero = all_true != 0
    mape = (np.mean(np.abs((all_true[nonzero] - mean_preds[nonzero]) / all_true[nonzero])) * 100
            if np.any(nonzero) else float('inf'))
    return all_true, mean_preds, std_preds, mse, rmse, mape


def plot_predictions(true_values, pred_values, metrics, title, key, output_dir, x_axis):
    mse, rmse, mape = metrics
    plt.figure(figsize=(16, 9))
    plt.plot(x_axis[:len(true_values)], true_values, label="True Values", color="b")
    plt.plot(x_axis[:len(pred_values)], pred_values, label="Predicted Values", color="r", linestyle='--')
    plt.annotate(f"MSE:  {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%",
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8, ec='gray'),
                 ha='right', va='top', fontsize=18, family='monospace')
    plt.xlabel("Cycles"); plt.ylabel("Stiffness (%)"); plt.title(title)
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"predictions_{key}.png"))
    plt.close()


def plot_mc_predictions(true_values, mean_preds, std_preds, metrics, title, key,
                        output_dir, x_axis, confidence_level=1.96):
    mse, rmse, mape = metrics
    x = x_axis[:len(true_values)]
    plt.figure(figsize=(16, 9))
    plt.plot(x, true_values, label="True Values", color="b")
    plt.plot(x, mean_preds, label="Predicted Mean", color="r", linestyle='--')
    plt.fill_between(x, mean_preds - confidence_level * std_preds,
                     mean_preds + confidence_level * std_preds,
                     color="r", alpha=0.2, label="95% CI")
    plt.annotate(f"MSE:  {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%",
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8, ec='gray'),
                 ha='right', va='top', fontsize=18, family='monospace')
    plt.xlabel("Cycles"); plt.ylabel("Stiffness (%)"); plt.title(title)
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"mc_predictions_{key}.png"))
    plt.close()


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


# %% [markdown]
# # FOUR-FOLD LEAVE-ONE-PANEL-OUT CROSS-VALIDATION

# %%
output_dir = "output_images/MLP_4fold" if CONFIG["use_hi"] else "output_images/MLP_4fold_no_HI"
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"HI features: {'ON' if CONFIG['use_hi'] else 'OFF (no-HI ablation)'}")

all_fold_metrics = {}

for fold_key in DF_INDICES.keys():
    print(f"\n===== Processing Fold: {fold_key} =====")
    test_idx = DF_INDICES[fold_key]
    train_indices = [i for i in DF_INDICES.values() if i != test_idx]

    # --- Normalisation, fitted on the training panels only ---------------
    # Per-feature standardisation of the inputs (the notebook's choice); the
    # GENEA scripts instead use a single global scalar mean/std.
    train_x = np.concatenate([strain_data[i] for i in train_indices], axis=0)
    train_y = np.concatenate([stiffness_data[i] for i in train_indices], axis=0)

    eps = 1e-8
    input_mean = np.mean(train_x, axis=0, keepdims=True)
    input_std = np.std(train_x, axis=0, keepdims=True) + eps
    target_min = np.min(train_y, axis=0)
    target_max = np.max(train_y, axis=0)
    target_range = target_max - target_min + eps

    norm_params = {
        'input_mean': input_mean.flatten(),
        'input_std': input_std.flatten(),
        'target_min': target_min,
        'target_max': target_max,
    }

    def to_dataset(idx_list):
        xs = np.concatenate([strain_data[i] for i in idx_list], axis=0)
        ys = np.concatenate([stiffness_data[i] for i in idx_list], axis=0)
        xs = (xs - input_mean) / input_std
        ys = (ys - target_min) / target_range
        return TensorDataset(torch.tensor(xs, dtype=torch.float32),
                             torch.tensor(ys, dtype=torch.float32))

    train_dataset = to_dataset(train_indices)
    # As in every script here, the held-out panel serves as both the
    # early-stopping validation set and the reported test set. See the
    # reproducibility notes in paper_code/README.md.
    test_dataset = to_dataset([test_idx])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # --- A fresh model per fold. Rebuilding here is essential: the notebook
    #     kept model/optimizer at module level, so a naive loop would carry
    #     the previous fold's weights forward. -------------------------------
    model = MLP(input_dim=CONFIG["input_dim"], output_dim=CONFIG["output_dim"],
                width=CONFIG["width"], depth=CONFIG["depth"],
                activation=nn.ReLU(), dropout_p=CONFIG["dropout_p"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                            weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    print(f"Trainable parameters: {count_parameters(model):,}")

    writer = SummaryWriter(os.path.join("log", "mlp", f"{fold_key}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))

    start_time = time.time()
    model, train_losses, val_losses = train_model(
        model, optimizer, scheduler, train_loader, val_loader, device,
        CONFIG, fold_key, writer, output_dir)
    print(f"Training for fold {fold_key} took: {time.time() - start_time:.2f} seconds")
    writer.close()

    plt.figure(figsize=(10, 5))
    plt.semilogy(train_losses, label='Training Loss')
    plt.semilogy(val_losses, label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Log Loss'); plt.title(f'Loss vs. Epochs for Fold {fold_key}')
    plt.legend(); plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(output_dir, f"loss_vs_epochs_{fold_key}.png"))
    plt.close()

    true_vals, pred_vals, mse, rmse, mape = inference(model, test_loader, device, norm_params)
    all_fold_metrics[fold_key] = {'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

    fod_name = f"FOD{int(fold_key.split('f')[-1]) + 3}"
    x_axis = strain_x_rescaled[fold_key]
    plot_predictions(true_vals, pred_vals, (mse, rmse, mape),
                     f"MLP Prediction on Test Fold: {fod_name}", fold_key, output_dir, x_axis)

    mc_true, mc_mean, mc_std, mc_mse, mc_rmse, mc_mape = mc_dropout_inference(
        model, test_loader, device, norm_params, num_samples=CONFIG["mc_dropout_samples"])
    plot_mc_predictions(mc_true, mc_mean, mc_std, (mc_mse, mc_rmse, mc_mape),
                        f"{fod_name} - Cross Validation Fold", fold_key, output_dir, x_axis)

# %% [markdown]
# # AGGREGATE AND SAVE RESULTS

# %%
mean_metrics = {m: np.mean([all_fold_metrics[k][m] for k in all_fold_metrics])
                for m in ['MSE', 'RMSE', 'MAPE']}
std_metrics = {m: np.std([all_fold_metrics[k][m] for k in all_fold_metrics])
               for m in ['MSE', 'RMSE', 'MAPE']}

final_results = {
    'config': {k: list(v) if isinstance(v, tuple) else v for k, v in CONFIG.items()},
    'mean_metrics': mean_metrics,
    'std_dev_metrics': std_metrics,
    'per_fold_metrics': all_fold_metrics,
}

results_path = os.path.join(output_dir, 'mlp_4fold_results.json')
with open(results_path, 'w') as f:
    json.dump(convert_numpy_types(final_results), f, indent=4)

print("\n===== Cross-Validation Finished =====")
print(f"Final results saved to: {results_path}")
print("\nMean Metrics:")
for m, v in mean_metrics.items():
    print(f"  {m}: {v:.4f}")
print("\nStandard Deviation of Metrics:")
for m, v in std_metrics.items():
    print(f"  {m}: {v:.4f}")
