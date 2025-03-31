import os
import json
import optuna
import plotly
from plotly import graph_objects as go

def load_study():
    """Load Optuna study with study selection"""
    try:
        storage = optuna.storages.RDBStorage("sqlite:///hpo_study.db")
        studies = storage.get_all_studies()
        study_names = [s.study_name for s in studies]
        
        if not study_names:
            print("No studies found in database")
            return None
            
        print("Available studies:")
        for i, name in enumerate(study_names, 1):
            print(f"{i}. {name}")
            
        selection = int(input("Enter study number: ")) - 1
        study_name = study_names[selection]
        
        return optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        
    except Exception as e:
        print(f"Error loading study: {e}")
        return None

def create_visualization_dir():
    """Create directory for saving visualizations"""
    os.makedirs("visualizations", exist_ok=True)

def plot_optimization_history(study):
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html("visualizations/optimization_history.html")

def plot_param_importances(study):
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html("visualizations/parameter_importance.html")

def plot_contour(study):
    fig = optuna.visualization.plot_contour(study, params=[
        "num_gnn_layers", "hidden_dim", "dropout", "batch_size"
    ])
    fig.write_html("visualizations/contour_plot.html")

def plot_slice(study):
    fig = optuna.visualization.plot_slice(study, params=[
        "num_gnn_layers", "hidden_dim", "dropout", "batch_size"
    ])
    fig.write_html("visualizations/slice_plot.html")

def plot_parallel_coordinate(study):
    # Get parameters dynamically from the best trial
    # This ensures we only use parameters that exist in the study
    if study.best_trial:
        # Get parameter names from best trial
        params = list(study.best_trial.params.keys())
        
        # Ensure num_gnn_layers (layer depth) is included if it exists
        if "num_gnn_layers" in params:
            # Move num_gnn_layers to the beginning for better visualization
            params.remove("num_gnn_layers")
            params.insert(0, "num_gnn_layers")
            
        fig = optuna.visualization.plot_parallel_coordinate(study, params=params)
        fig.update_layout(coloraxis_colorbar_title="Objective Loss")
        fig.write_html("visualizations/parallel_coordinate.html")
    else:
        print("Warning: No completed trials found, skipping parallel coordinate plot")

def save_best_params(study):
    """Save best parameters to JSON file"""
    os.makedirs("results", exist_ok=True)
    best_params = {
        "best_value": study.best_value,
        "params": study.best_params
    }
    
    with open("results/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

def print_best_trial_summary(study):
    best_trial = study.best_trial
    print(f"\nBest trial value (MSE): {best_trial.value:.4f}")
    print("Best parameters:")
    for key, value in best_trial.params.items():
        print(f"- {key}: {value}")

if __name__ == "__main__":
    try:
        study = load_study()
        if not study:
            exit(1)
            
        create_visualization_dir()
        
        plot_optimization_history(study)
        plot_param_importances(study)
        plot_contour(study)
        plot_slice(study)
        plot_parallel_coordinate(study)
        
        save_best_params(study)
        print_best_trial_summary(study)
        
        print("\nVisualization saved to 'visualizations' directory")
        print("Best parameters saved to 'results/best_params.json'")
        
    except ImportError as e:
        print(f"Missing dependency: {e}\nInstall with: pip install plotly kaleido")