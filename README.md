# Graph Neural Network for Stiffness Prediction

This project demonstrates the use of Graph Neural Networks (GNNs) for predicting stiffness reduction in materials based on strain sensor data.

## Project Structure

- `run_loocv.py`: Main Python script to run Leave-One-Out Cross-Validation
- `Cline_test.ipynb`: Original Jupyter notebook containing the full code
- `Cline_test_simplified.ipynb`: Simplified notebook that uses the `gnn_utils` package
- `Cline_test_loocv.ipynb`: Notebook implementing Leave-One-Out Cross-Validation (LOOCV)
- `gnn_utils/`: Python package containing utility functions and classes
  - `data_preprocessing.py`: Basic data preprocessing functions
  - `data_processing.py`: Advanced data processing functions for the pipeline
  - `model.py`: GNN model definition
  - `training.py`: Training and evaluation utilities
  - `graph_data.py`: Graph data preparation utilities
  - `visualization.py`: Plotting and visualization functions
  - `loocv.py`: Leave-One-Out Cross-Validation utilities
  - `__init__.py`: Package initialization
  - `README.md`: Package documentation
- `Data/`: Directory containing the input data
  - `Stiffness_Reduction/`: Stiffness reduction data
  - `Strain/`: Strain sensor data

## Purpose

The main goal of this project is to reduce the lines of code in the notebook to the bare minimum and keep only lines of code that are acting as inputs. This helps to reduce the token size when editing the code with LLM models.

The original notebook has been refactored to move all utility functions and reusable code into the `gnn_utils` package, making the notebook cleaner and more focused on the workflow rather than implementation details.

Additionally, a Python script (`run_loocv.py`) has been created to run the LOOCV process from the command line, further reducing the need for large notebooks.

## Workflow

1. **Data Loading**: Load stiffness reduction and strain data from HDF5 files
2. **Data Preprocessing**: Process the data to prepare it for the GNN
   - Resample strain data to a fixed frequency
   - Apply rolling mean smoothing
   - Apply custom feature engineering
   - Normalize stiffness data
   - Align timestamps between strain and stiffness data
3. **Data Truncation**: Truncate the data at specific stiffness reduction levels
4. **GNN Data Preparation**: Prepare the data for the GNN
   - Define train/validation split
   - Compute normalization parameters
   - Construct graph data objects
5. **Model Definition**: Define the GNN architecture
6. **Training**: Train the model on the prepared data
7. **Evaluation**: Evaluate the model on validation data

## Cross-Validation

The `run_loocv.py` script implements Leave-One-Out Cross-Validation (LOOCV) for more robust model evaluation. Since we have 4 available specimens (FOD panels), we use 3 for training and 1 for validation in each fold, rotating through all specimens. This approach:

1. Provides a more reliable estimate of model performance
2. Helps identify which specimens are more difficult to predict
3. Allows for better generalization to unseen data

## Usage

### Running the Python Script

```bash
# Run with default parameters
python run_loocv.py

# Run with custom parameters
python run_loocv.py --stiffness-path Data/Stiffness_Reduction --strain-path Data/Strain --num-nodes 16 --batch-size 128 --hidden-dim 64 --num-gnn-layers 4 --dropout 0.5 --epochs 1000 --patience 50 --drop-level 85

# Run without visualization
python run_loocv.py --no-visualize

# Save plots to files instead of displaying them (useful for headless environments)
python run_loocv.py --save-plots --output-dir results
```

#### Command-Line Options

- `--stiffness-path`: Path to stiffness data directory
- `--strain-path`: Path to strain data directory
- `--num-nodes`: Number of nodes in each graph
- `--batch-size`: Batch size for training
- `--hidden-dim`: Hidden dimension for the GNN model
- `--num-gnn-layers`: Number of GNN layers
- `--dropout`: Dropout probability
- `--epochs`: Maximum number of epochs
- `--patience`: Patience for early stopping
- `--drop-level`: Stiffness reduction level for truncation
- `--no-visualize`: Disable visualization
- `--save-plots`: Save plots to files instead of displaying them
- `--output-dir`: Directory to save results and plots

### Running the Notebooks

To run the simplified notebook:

```bash
jupyter notebook Cline_test_simplified.ipynb
```

To run the LOOCV notebook:

```bash
jupyter notebook Cline_test_loocv.ipynb
```

## Handling Display Issues

If you encounter display issues when running the script (e.g., "Could not load the Qt platform plugin"), use the `--save-plots` option to save plots to files instead of displaying them:

```bash
python run_loocv.py --save-plots
```

This is particularly useful when running the script in headless environments or over SSH connections without X11 forwarding.

## Dependencies

- Python 3.6+
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Matplotlib
- TensorBoardX
- scienceplots (optional, for enhanced plotting styles)

### Installing Dependencies

```bash
pip install torch torch-geometric numpy pandas matplotlib tensorboardx
pip install scienceplots  # Optional, for enhanced plotting styles
```

If scienceplots is not installed, the code will fall back to standard matplotlib styles.