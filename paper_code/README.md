# `paper_code/` — the scripts that produced the published results

These are the scripts used for the results reported in the published paper. All
but one are included **verbatim**, exactly as they were run, so that the record
is faithful; the exception is `mlp_baseline_lopo.py`, which was assembled from a
notebook and is documented as such below. They are standalone: they do *not*
import `gnn_utils/`, and each one re-implements loading, preprocessing, the
model and the cross-validation loop.

> **Read this before anything else:** the code at the repository root
> (`run_loocv.py`, `gnn_utils/`) is the earlier exploratory implementation used
> during the model-selection phase. It is a different, larger model and it does
> **not** reproduce the numbers in the paper's main tables. See
> "Root code vs `paper_code/`" below.

Run every script from the **repository root**, so that the relative paths
`Data/Stiffness_Reduction` and `Data/Strain` resolve:

```bash
python paper_code/genea_stiffness_lopo.py
```

Each script is in `# %%` cell format, so it also opens directly as a notebook in
VS Code or Jupyter (via jupytext).

## What each script produces

| Script | Protocol | Corresponding paper result |
|---|---|---|
| `genea_stiffness_lopo.py` | LOPO CV, GENEA, stiffness estimation | Main stiffness results; GENEA column of the baseline comparison |
| `genea_stiffness_lopo_with_fod3.py` | Five-fold LOPO **including** the 6-sensor FOD3 panel, plus MC-dropout uncertainty | "Generalization to varying geometries: incorporating FOD3"; MC-dropout confidence bands |
| `cnn_baseline_lopo.py` | LOPO CV, 2-D CNN baseline (16 HI values reshaped to a 4×4 grid) | CNN baseline in the MLP/CNN comparison |
| `mlp_baseline_lopo.py` | Four-fold LOPO, flat MLP over the 16 HI values, plus a `use_hi` toggle | MLP baseline in the MLP/CNN comparison (the toggle approximates, but does not reproduce, the HI ablation — see below) |
| `threshold_search/threshold_search.py` | Four-fold LOPO repeated at truncation thresholds 85 / 80 / 70 / 60 % | Truncation-threshold selection table |

Reference outputs from the actual runs are kept alongside:

* `reference_results/genea_4fold_results.json`
* `reference_results/genea_5fold_with_fod3_results.json`
* `reference_results/threshold_search_results.json`
* `reference_results/mlp_4fold_results.json`
* `reference_results/notebook_recovered_metrics.json`
* `trained_models/genea_4fold/*.pth` — four checkpoints, folds FOD4–FOD7
* `trained_models/genea_5fold_with_fod3/*.pth` — three checkpoints, folds FOD5–FOD7

The checkpoints are ~25 kB each (3,495 parameters) and load into the
`EdgeAttrGNN` class defined inside `genea_stiffness_lopo.py`, so the published
models can be evaluated without retraining. Two caveats, both verified by
loading the files:

* **The `genea_4fold` checkpoints need the readout dropout re-enabled.** Their
  `state_dict` has `readout.0` and `readout.3`, i.e. a four-entry readout
  `Linear → LeakyReLU → Dropout → Linear`. In the script as shipped that
  `nn.Dropout` line is commented out, giving `readout.0`/`readout.2`, and a
  strict `load_state_dict` fails. Uncomment

  ```python
  #nn.Dropout(p=dropout_p),
  ```

  in `EdgeAttrGNN.__init__` and all four load strictly. The `genea_5fold_*`
  checkpoints match the script as shipped and need no change. In other words
  the four-fold run predates that edit and the five-fold run postdates it.

* **The five-fold FOD3 and FOD4 checkpoints no longer exist.** On the original
  machine, `cnn_baseline_lopo.py` wrote to the same `output_images/FOD3_5fold/`
  directory as the five-fold GENEA script, and a later, partial CNN run
  overwrote `best_model_df0.pth` and `best_model_df1.pth` about 70 minutes
  after the GENEA run finished. Those two files were CNN weights, not GENEA,
  so they have been left out rather than shipped under a misleading name. The
  metrics for all five folds survive intact in
  `reference_results/genea_5fold_with_fod3_results.json`, which was written
  before the overwrite. To get the missing checkpoints back you must re-run.

### Changes made to the original scripts

The scripts are otherwise verbatim. Every change below was needed to make them
run from the repository root on a current environment; none touches the model,
the data, the hyperparameters or the metrics. All four scripts were then run end
to end (with `epochs` cut to 1) to confirm they complete.

| Script | Change | Why |
|---|---|---|
| `cnn_baseline_lopo.py` | outputs moved to `output_images/CNN_5fold/` and `output_images/CNN_4Fold/` | it shared output directories with the GENEA scripts — the collision that destroyed the two checkpoints above |
| `cnn_baseline_lopo.py` | `DF_INDICES` restricted to the four 16-sensor panels | the CNN reshapes 16 HI values into a 4×4 grid, so the 6-sensor FOD3 raised `IndexError` before training started |
| `cnn_baseline_lopo.py` | `float()` coercion in `unnormalize_target` | `target_mean`/`target_std` are 0-dim torch tensors and `y_norm` is a NumPy array; torch 2.x refuses the multiplication, so no fold could finish. Arithmetic is unchanged |
| `cnn_baseline_lopo.py`, `genea_stiffness_lopo.py` | `os.makedirs` after the late `output_dir` reassignment | the final results JSON write failed with `FileNotFoundError` unless the directory happened to exist |
| `threshold_search/threshold_search.py` | data paths `../Data/...` → `Data/...`; results now under `output_images/threshold_search/` | it expected to be run from inside its own subdirectory, and wrote a stray `Threshold_search/` at whatever the working directory was |

Note in particular that `cnn_baseline_lopo.py` as originally saved could not
complete a single fold on torch 2.x. If you are comparing against the paper's
CNN numbers, they came from a run predating those edits.

## The MLP baseline — assembled, not verbatim

`mlp_baseline_lopo.py` is the one script here that was not simply copied. Its
source is `notebooks/mlp_baseline_stiffness.ipynb`, which ran **one fold per
execution** with the fold chosen by editing a cell. Turning that into a
reproducible four-fold protocol required real changes:

* The fold loop is new. The notebook kept `model`, `opt` and `scheduler` at
  module level, so a naive loop would have carried each fold's trained weights
  into the next. All three are rebuilt inside the loop.
* The preprocessing block is spliced from `genea_stiffness_lopo.py` with
  exactly one change: the HI step is wrapped in `if CONFIG["use_hi"]`, with an
  `else` branch that drops the first sample so the lengths still line up. At
  the default `use_hi=True` the HI features and the 70 % truncation are
  identical to the GENEA runs.
* Checkpoints go to `output_images/MLP_4fold/`. The notebook wrote to
  `best_model/best_model_state.pth`, which in this repository is the tracked
  directory holding released checkpoints.
* `CONFIG["use_hi"]` was added: `False` skips the cumulative-absolute-derivative
  step and feeds smoothed strain instead. This approximates the with/without-HI
  ablation but does not reproduce its published numbers — see below.

Everything model-side is the notebook's, including three choices that differ
from the GENEA scripts and were kept deliberately:

| | `mlp_baseline_lopo.py` | GENEA scripts |
|---|---|---|
| input normalisation | per-feature (`axis=0`) standardisation | one global scalar mean/std |
| weighted loss | range `(0.0, 0.80)`, weight ×6.0 | range `(0.2, 0.95)`, weight ×2.0 |
| LR scheduler steps on | **training** loss | validation loss |
| early-stopping patience | 80 | 100 |
| best-model tracking | skips the first 3 epochs | from epoch 0 |

The weighted-loss values are taken from the notebook's *call site*, not the
function's defaults — the defaults were never the values actually used.

### Does it reproduce the paper?

Yes, as closely as an unseeded pipeline can. A full run of this script gives:

| Test panel | RMSE here | RMSE in paper | MAPE here | MAPE in paper |
|---|---|---|---|---|
| FOD4 | 4.51 | 4.75 | 3.35 | 3.71 |
| FOD5 | 4.46 | 5.41 | 3.98 | 3.85 |
| FOD6 | 1.75 | 1.76 | 1.68 | 1.09 |
| FOD7 | 1.10 | 1.17 | 0.82 | 0.89 |
| **Mean** | **2.96** | **3.27** | **2.46** | **2.38** |

Per-fold ordering and magnitudes track the published table throughout. The full
output is in `reference_results/mlp_4fold_results.json`. Nothing is seeded, so
your run will differ again by a similar margin.

The paper quotes the MLP twice from what appear to be separate runs — 3.27 /
2.38 in the MLP/CNN comparison table, and 3.14 / 2.39 as the "with the proposed
HI" row of the ablation table. The 2.96 / 2.46 above sits just below both.

### The `use_hi` toggle does *not* reproduce the ablation

This was run and checked, and it does not come out where the paper does:

| | RMSE | MAPE (%) |
|---|---|---|
| with HI, here | 2.96 | 2.46 |
| with HI, paper | 3.14 | 2.39 |
| **without HI, here** | **3.08** | **2.52** |
| **without HI, paper** | **5.34** | **4.25** |

Removing the HI degrades this pipeline by about 4 % RMSE. The paper reports
roughly 70 %. Matching the with-HI row while missing the without-HI row by that
margin says the difference is in how the no-HI input was constructed, not in the
model or the protocol.

`CONFIG["use_hi"] = False` skips the cumulative-absolute-derivative step and
feeds the resampled, rolling-mean-smoothed strain instead, which is the most
literal reading of "without HI (strain)". Per-feature standardisation then
removes the scale difference between the two representations, and what remains
still carries enough of the degradation signal for this small MLP.

So: **treat the toggle as an approximation of the ablation, not a reproduction
of it.** The paper's no-HI input was evidently prepared some other way — for
instance without the 200 s resampling or the rolling mean, or normalised
globally rather than per feature. `MLP_raw_strain_Stiffness_v1.ipynb` on the
authors' machine is named for the raw-strain run but its preprocessing cell
still computes the HI, so the configuration behind the 5.34 is not recorded
anywhere we could find. Output is in
`reference_results/mlp_4fold_no_hi_results.json`.

## `notebooks/` — the original notebooks, outputs stripped

Two notebooks are included as the underlying record. Execution outputs were
removed (4.8 MB → 44 kB and 19 MB → 62 kB); the metrics that lived only in those
outputs were extracted first into
`reference_results/notebook_recovered_metrics.json`.

* **`mlp_baseline_stiffness.ipynb`** — the source of `mlp_baseline_lopo.py`.
  Its stored run was fold df1 (FOD4): RMSE 5.23, MAPE 4.23, against the paper's
  4.75 / 3.71 for that panel. Use the script; this is here for provenance.

* **`rul_prototype_gcnconv.ipynb`** — a RUL prototype, and **not** the code
  behind the paper's RUL results. Two independent reasons:

  1. It uses two `GCNConv` layers with `edge_attr` set to all zeros. The paper
     states RUL uses "separate models, sharing this GENEA architecture" — this
     model discards the edge attributes that are the paper's central claim.
  2. Its stored fold (FOD7) gives RMSE 70 cycles at a 99 % truncation, while
     the paper reports RUL RMSE "typically ranging from 100 to 400 cycles".

  The code that produced the published RUL figures could not be located. The
  paper's RUL figure filenames encode a weighted loss of range `(0, 0.03)` at
  ×6, a configuration that appears in no surviving script. Treat this notebook
  as an early prototype only.

## Selecting four-fold vs five-fold — a manual step

`genea_stiffness_lopo.py` and `genea_stiffness_lopo_with_fod3.py` both ship with

```python
DF_INDICES = {'df0': 0, 'df1': 1, 'df2': 2, 'df3': 3, 'df4': 4}
```

which runs **five** folds. The four-panel results (FOD4–FOD7, needed wherever a
fixed 16-sensor input size is required, e.g. the MLP/CNN comparison) were
produced by removing the `'df0'` entry before running — this is why
`reference_results/genea_4fold_results.json` and
`trained_models/genea_4fold/` contain four folds, not five.

There is no command-line switch for this yet. If you want the four-fold
protocol, edit `DF_INDICES` to:

```python
DF_INDICES = {'df1': 1, 'df2': 2, 'df3': 3, 'df4': 4}
```

`threshold_search/threshold_search.py` is already fixed at four folds and uses a
different, self-contained indexing convention
(`{'df1': 0, 'df2': 1, 'df3': 2, 'df4': 3}` — positions in its own list).

Note also that `output_dir` is assigned twice in the GENEA scripts: intermediate
per-fold plots and checkpoints go to `output_images/FOD3_5fold/`, while the
aggregated JSON is written to `output_images/4Fold/`. Both GENEA scripts share
those two directories, so **running one overwrites the other's checkpoints**.
Move or rename the output before switching between them. All of
`output_images/` is gitignored.

## The GENEA architecture as published

Verified against `trained_models/genea_4fold/best_model_df1.pth`
(`node_emb.weight` is `[16, 1]`, three `convs.*` blocks):

```
x  : [num_sensors, 1]           node feature  = HI of that sensor at time t
e  : [num_edges, 1]             edge feature  = HI_i - HI_j
     fully connected graph, no self-loops, rebuilt per panel from its own
     sensor count (so 6-, 16- and 24-sensor panels all work unchanged)

Linear(1 -> 16)
3 x [ GENConv(16 -> 16, aggr='add', msg_norm=True, learn_msg_scale=True,
              num_layers=2, edge_dim=1)
      -> BatchNorm(16)
      -> residual add
      -> LeakyReLU
      -> Dropout(p=0.3) ]           <- nn.Dropout modules, not F.dropout:
                                        required for MC dropout at inference
global_mean_pool
Linear(16 -> 8) -> LeakyReLU -> Linear(8 -> 1)
```

Training configuration (the `CONFIG` dict at the top of each script):

| | `genea_stiffness_lopo*.py` | `threshold_search.py` |
|---|---|---|
| hidden dim | 16 | 8 |
| GNN layers | 3 | 3 |
| dropout | 0.3 | 0.3 |
| optimizer | AdamW, lr 0.01, weight decay 5e-4 | same |
| scheduler | ReduceLROnPlateau, factor 0.8, patience 10 | same |
| batch size | 128 | 128 |
| max epochs | 2000 | 2000 |
| early-stopping patience | 100 | 40 |
| gradient clipping | max-norm 1.0 | same |
| training loss | weighted MSE, ×2.0 on targets in [0.2, 0.95] (normalised) | same |
| validation loss | plain MSE | same |
| truncation threshold | 70 % | swept: 85 / 80 / 70 / 60 |
| MC-dropout samples | 100 | — |

The threshold sweep therefore used a **smaller** model (hidden dim 8, patience
40) than the final reported runs (hidden dim 16, patience 100). It was a
selection experiment, not a headline result.

## Reproducibility caveats — please read

1. **No random seeds are set.** Neither `torch.manual_seed` nor
   `np.random.seed` is called anywhere. Weight initialisation, `DataLoader`
   shuffling and MC dropout are all unseeded, so **re-running will not
   reproduce the published numbers exactly**. The reference JSONs in
   `reference_results/` are the output of one such run and themselves differ
   from the paper's tables by roughly the run-to-run spread. Expect to match
   the reported values in trend and magnitude, not digit for digit. If you
   build on this work, seed everything and report a mean over several seeds.

2. **The held-out panel is also the early-stopping validation set.** In the
   fold loop, `val_data` and `test_data` are the same panel:

   ```python
   train_data = [g for i in train_indices for g in specimen_data_fold[i]]
   val_data   = specimen_data_fold[test_idx]
   test_data  = specimen_data_fold[test_idx]
   ```

   This is the protocol described in the paper, which states that
   hyperparameter optimisation minimised "the MSE on the validation set for
   each fold". With only four to five panels there is no room for a third
   split. It does mean the reported per-fold metrics are model-selection
   optimistic, and it is the first thing to change if you extend this work to a
   larger panel population.

3. **Normalisation is fitted on training folds only** — input mean/std and
   target min/range are computed from the training panels and then applied to
   the held-out panel. That part is leak-free.

4. **MAPE is computed on stiffness expressed as a percentage of initial
   stiffness**, a quantity that lives in roughly the 70–100 % band. A MAPE
   around 1 % reflects that narrow dynamic range; compare RMSE across models,
   and do not read the MAPE as a general-purpose accuracy figure.

## Root code vs `paper_code/`

| | `paper_code/` | root (`gnn_utils/`, `run_loocv.py`) |
|---|---|---|
| Role | produced the published results | earlier model-selection / exploration phase |
| Hidden dim | 16 | 64 (checkpoints in `best_model/`) |
| GNN layers | 3 | 3 (checkpoints); CLI default is 4 |
| Residual connections | yes | no |
| Pooling | `global_mean_pool` | `global_add_pool` |
| Dropout | `nn.Dropout` per layer | `F.dropout`, skipped on last layer |
| MC dropout | yes | no |
| Variable sensor counts | yes (per-panel graph) | no (fixed `--num-nodes 16`) |
| Training loss | weighted MSE + grad clipping | plain MSE |
| Truncation | 70 % via `stiffness_drop_threshold` | 70 %, but reached via the `--drop-level 85` key |

The root code is still worth keeping: it is what produced the GNN-architecture
comparison (GENConv / GATv2 / GCN / EdgeConv / SAGPool, and the no-edge
variants) under `Comparison/`, which is how GENConv was chosen as the core
layer in the first place.
