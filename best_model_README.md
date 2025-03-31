# Running the Best Model from Comparison Study

This document explains how to use the `run_best_comparison_model.py` script to run the best model identified from the model comparison study with more epochs and generate detailed visualizations.

## Overview

After running the model comparison scripts (`compare_models_fixed.py` and `compare_models_no_edges.py`), you can use this script to:

1. Automatically identify the best-performing model from the comparison results
2. Run this model with more epochs for better convergence
3. Generate detailed visualizations and performance metrics
4. Compare models with and without edge attributes

## Usage

```bash
python run_best_comparison_model.py [options]
```

### Command-line Arguments

The script accepts several command-line arguments to customize the run:

- `--stiffness-path`: Path to stiffness data directory (default: 'Data/Stiffness_Reduction')
- `--strain-path`: Path to strain data directory (default: 'Data/Strain')
- `--drop-level`: Stiffness reduction level for truncation (default: 85)

- `--model-type`: Type of model to run (default: 'with_edges')
  - `with_edges`: Run only the best model with edge attributes
  - `no_edges`: Run only the best model without edge attributes
  - `both`: Run both best models and compare them

- `--epochs`: Maximum number of epochs (default: 2000)
- `--patience`: Patience for early stopping (default: 50)
- `--batch-size`: Batch size for training (default: 64)

- `--no-visualize`: Disable visualization
- `--save-plots`: Save plots to files instead of displaying them
- `--output-dir`: Directory to save results and plots (default: 'best_comparison_model')

### Examples

1. Run the best model with edge attributes with 3000 epochs:

```bash
python run_best_comparison_model.py --model-type=with_edges --epochs=3000 --save-plots
```

2. Run both best models (with and without edge attributes) and compare them:

```bash
python run_best_comparison_model.py --model-type=both --epochs=2500 --patience=100 --save-plots
```

3. Run only the best model without edge attributes:

```bash
python run_best_comparison_model.py --model-type=no_edges --epochs=2000 --save-plots
```

## Output

The script will generate the following outputs in the specified output directory:

1. **Model Results**: For each model, a subdirectory containing:
   - LOOCV results (MSE, RMSE, MAPE) for each fold
   - Training and validation loss plots
   - Prediction plots
   - Detailed visualizations (scatter plots, residual plots, time series plots)

2. **Summary Results**: A JSON file (`best_model_results.json`) with the performance metrics for the best model(s).

3. **Detailed Visualizations**:
   - Scatter plots of true vs. predicted values with regression lines
   - Residual plots showing prediction errors
   - Time series plots with error bands

## Interpreting the Results

The script provides several visualizations to help interpret the model's performance:

1. **Scatter Plots**: Show the relationship between true and predicted values. A perfect model would have all points on the diagonal line.

2. **Residual Plots**: Show the difference between predicted and true values. Ideally, residuals should be randomly distributed around zero.

3. **Time Series Plots**: Show how the model's predictions track the true values over time, with error bands indicating prediction uncertainty.

4. **Performance Metrics**:
   - **MSE (Mean Squared Error)**: Lower is better. This is the primary metric for regression tasks.
   - **RMSE (Root Mean Squared Error)**: Lower is better. This is in the same units as the target variable.
   - **MAPE (Mean Absolute Percentage Error)**: Lower is better. This gives the percentage error.
   - **R²**: Higher is better. This indicates how well the model explains the variance in the data.

If you run both models (with and without edge attributes), the script will also calculate the percentage improvement that edge attributes provide, which can be valuable for your study on the impact of edge attributes in GNN models.