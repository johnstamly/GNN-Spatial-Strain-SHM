import os
import sys
import json
import optuna
from optuna.trial import TrialState
import subprocess

def objective(trial):
    # Define hyperparameter search space
    params = {
        "--num-gnn-layers": trial.suggest_int("num_gnn_layers", 2, 5),
        "--hidden-dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "--dropout": trial.suggest_float("dropout", 0.2, 0.6),
        "--batch-size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "--patience": 20,
        "--optimizer": "adamw"
    }

    try:
        # Execute training command
        cmd = ["python", "run_loocv.py"] 
        for param, value in params.items():
            cmd.append(f"{param}={value}")
        cmd.append("--save-plots")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )

        # Load results from JSON
        with open(os.path.join("results", "loocv_results.json")) as f:
            metrics = json.load(f)
            return metrics["summary"]["mean_mse"]

    except subprocess.CalledProcessError as e:
        print(f"Trial failed with error: {e.output}")
        return float("inf")  # Return high loss for failed trials

if __name__ == "__main__":
    # Check if Optuna is installed
    try:
        import optuna
    except ImportError:
        print("Error: Optuna not installed. Use 'pip install optuna' first.")
        sys.exit(1)

    # Create study with SQLite storage for resumability
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///hpo_study.db",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=15),
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(objective, n_trials=300, show_progress_bar=True)

    # Print results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (MSE): {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")