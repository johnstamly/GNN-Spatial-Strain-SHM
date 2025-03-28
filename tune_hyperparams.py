import os
import json
import optuna
import subprocess

def objective(trial):
    # Define hyperparameter search space
    params = {
        '--num-gnn-layers': trial.suggest_int('num_gnn_layers', 1, 7),
        '--hidden-dim': trial.suggest_categorical('hidden_dim', [8, 16, 32, 64, 128]),
        '--dropout': trial.suggest_float('dropout', 0.0, 0.6),
        '--batch-size': trial.suggest_categorical('batch_size', [32, 64, 128]),
    }

    # Build command with current parameters
    cmd = [
        'python', 'run_loocv.py',
        *[f"{k}={v}" for k,v in params.items()],
        '--patience=20',
        '--optimizer=adamw',
        '--save-plots'
    ]
    
    # Execute training and get results
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Load results from JSON
    with open(os.path.join('results', 'loocv_results.json')) as f:
        metrics = json.load(f)['summary']
    
    return metrics['mean_mse']

if __name__ == '__main__':
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")