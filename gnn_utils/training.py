"""
Training and evaluation utilities for the GNN stiffness prediction model.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorboardX import SummaryWriter


def train_epoch(model, loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = model.loss(pred, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, device):
    """Validate the model on the validation set."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            loss = model.loss(pred, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def run_training(model, train_loader, val_loader, device, writer, optimizer, scheduler,
                 epochs, patience=20, model_save_path="best_model/best_model_state.pth"):
    """Run the training loop with early stopping and model saving."""
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Scheduler Step (based on validation loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']  # Get current LR
        writer.add_scalar("LearningRate", current_lr, epoch)

        # Early Stopping and Best Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_save_path)
            print(f"    -> New best model saved with Val Loss: {val_loss:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"    -> No improvement. Patience: {epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs. Best Val Loss: {best_val_loss:.4f}")
            break

    # Load the best model state after training completes or stops early
    if best_model_state is not None:
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        print(f"\nLoaded best model state from: {model_save_path}")
    else:
        print("\nWarning: No best model state was saved during training.")

    return model, train_losses, val_losses


def run_inference(model, loader, device, unnormalize_target_fn):
    """Runs inference, unnormalizes predictions, and calculates metrics."""
    model.eval()
    all_true_norm = []
    all_pred_norm = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            all_true_norm.extend(data.y.cpu().numpy().flatten())
            all_pred_norm.extend(pred.cpu().numpy().flatten())

    all_true_norm = np.array(all_true_norm)
    all_pred_norm = np.array(all_pred_norm)

    # Unnormalize
    all_true_unnorm = unnormalize_target_fn(all_true_norm)
    all_pred_unnorm = unnormalize_target_fn(all_pred_norm)

    # Calculate Metrics (on unnormalized values)
    mse = np.mean((all_true_unnorm - all_pred_unnorm) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE (handle potential division by zero)
    nonzero_mask = all_true_unnorm != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((all_true_unnorm[nonzero_mask] - all_pred_unnorm[nonzero_mask]) / all_true_unnorm[nonzero_mask])) * 100
    else:
        mape = float('inf')  # Or 0, or NaN, depending on desired behavior

    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return all_true_unnorm, all_pred_unnorm, mse, rmse, mape


def plot_predictions(true_values, predicted_values, x_values, mse, rmse, mape, title):
    """Plots true vs. predicted values against cycles or time."""
    plt.figure(figsize=(12, 6))
    
    plt.plot(x_values, true_values, label="True Values", color="blue", marker='o', linestyle='-', markersize=0.5, alpha=0.7)
    plt.plot(x_values, predicted_values, label="Predicted Values", color="red", marker='x', linestyle='--', markersize=0.5, alpha=0.7)
    
    # Metrics text box
    metrics_text = (
        f"MSE:  {mse:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"MAPE: {mape:.2f}%"
    )
    plt.annotate(metrics_text, xy=(0.97, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=0.5, alpha=0.8),
                 ha='right', va='top', fontsize=10, family='monospace')
    
    plt.xlabel("Cycles")
    plt.ylabel("Stiffness (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def setup_tensorboard():
    """Set up TensorBoard writer with timestamped log directory."""
    log_dir = os.path.join("log", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved in: {log_dir}")
    return writer, log_dir