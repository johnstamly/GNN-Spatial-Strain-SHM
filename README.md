# Graph Neural Networks for SHM: Exploiting Spatial Interdependencies of Strain Data

[![DOI](https://img.shields.io/badge/DOI-10.1177%2F14759217251386802-blue.svg)](https://doi.org/10.1177/14759217251386802)
[![Data DOI](https://img.shields.io/badge/Data%20DOI-10.5281%2Fzenodo.14627730-blue.svg)](https://doi.org/10.5281/zenodo.14627730)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-green.svg)](Data/LICENSE)

Code and data accompanying:

> Stamatelatos G, Galanopoulos G, Zarouchas D and Loutas T.
> **Graph neural networks for SHM: exploiting spatial interdependencies of
> strain data for diagnostics and prognostics.**
> *Structural Health Monitoring*, 2025. Advance online publication.
> [doi:10.1177/14759217251386802](https://doi.org/10.1177/14759217251386802)

The paper introduces **GENEA** (GENConv with Edge Attributes), a graph neural
network that treats a set of spatially distributed strain sensors as a graph and
learns from the *relationships between* sensors rather than from each sensor in
isolation. It is paired with a custom **Health Indicator (HI)** that decouples
damage signatures from operational load. Together they estimate structural
stiffness reduction (diagnostics) and Remaining Useful Life (prognostics) on
aeronautical composite panels under fatigue.

---

## Quick start

```bash
git clone https://github.com/johnstamly/GNN-Spatial-Strain-SHM.git
cd GNN-Spatial-Strain-SHM

# Install PyTorch first, matching your CUDA version: https://pytorch.org
pip install -r requirements.txt

# Reproduce the paper's stiffness-estimation experiment
python paper_code/genea_stiffness_lopo.py
```

Always run from the repository root — the scripts use the relative paths
`Data/Strain` and `Data/Stiffness_Reduction`.

Trained checkpoints from the published runs are in `paper_code/trained_models/`
(~25 kB each) if you want to evaluate the models without retraining — read the
two loading caveats in [`paper_code/README.md`](paper_code/README.md) first.

---

## The two ideas, in short

### 1. The Health Indicator decouples damage from load

Raw strain is dominated by the applied fatigue load, which swamps the far
subtler signature of accumulating damage. The HI is the **cumulative absolute
first derivative of strain over time**, computed per sensor:

```python
strain_resampled = strain.resample("200s").mean().rolling(10, min_periods=1).mean()
HI = np.cumsum(np.abs(np.diff(strain_resampled.values, axis=0)), axis=0)
```

Load cycling produces oscillations that largely cancel in the mean but
accumulate in the absolute derivative only when the underlying stiffness
changes. The result is a monotonically increasing, load-independent quantity per
sensor. Feeding the HI to a plain MLP instead of raw strain improves RMSE by
roughly 40 % in the paper's ablation, so most of the benefit is in the feature,
before any graph is involved.

### 2. The graph exploits spatial interdependency

At each time step, one panel becomes one graph:

| | |
|---|---|
| **Nodes** | one per strain sensor (6, 16 or 24 depending on the panel) |
| **Node feature** `x` | that sensor's HI at time `t`, a single scalar |
| **Edges** | fully connected, no self-loops |
| **Edge feature** `e_ij` | `HI_i − HI_j`, the *difference* in HI between the two sensors |
| **Target** `y` | graph-level scalar: stiffness as % of initial, or RUL |

Encoding the pairwise HI *difference* on the edge is what makes the model
spatially aware: damage between two sensors shows up as a divergence in their
HI accumulation rates, and that divergence is exactly what the edge carries.

Because the graph is rebuilt from each panel's own sensor count, the same
trained model runs unchanged on a panel with a different number of sensors —
which is how the paper tests generalisation on the 6-sensor FOD3 panel using a
model trained on 16-sensor panels. A CNN or MLP cannot do this without padding.

Prediction is point-wise: the HI at a single time step predicts stiffness at
that same time step. No temporal model (LSTM, Transformer) is used, deliberately
— it isolates the spatial contribution being measured.

---

## Repository layout

```
paper_code/          ← the scripts that produced the published results  ★ start here
  genea_stiffness_lopo.py             GENEA, LOPO CV, stiffness estimation
  genea_stiffness_lopo_with_fod3.py   five-fold LOPO incl. 6-sensor FOD3 + MC dropout
  cnn_baseline_lopo.py                2-D CNN baseline (HI reshaped to a 4×4 grid)
  mlp_baseline_lopo.py                MLP baseline + with/without-HI ablation toggle
  threshold_search/                   truncation-threshold sweep (85/80/70/60 %)
  notebooks/                          original notebooks, outputs stripped
  reference_results/                  metrics JSON from the actual paper runs
  trained_models/                     per-fold checkpoints from the paper runs
  README.md                           architecture, config, reproducibility caveats

Data/                ← strain and stiffness measurements (CC BY 4.0)
  Strain/                             FBG strain time series, FOD3–FOD7
  Stiffness_Reduction/                MTS stiffness per fatigue block
  README.md                           provenance, panel↔key mapping, sensor counts

gnn_utils/           ← earlier exploratory implementation (model-selection phase)
run_loocv.py                          LOOCV driver for the exploratory model
run_best_comparison_model.py          re-runs the best architecture from Comparison/
hyperparameter_tuning.py              Optuna study driver
visualize_hpo_results.py              Optuna plots → visualizations/
hpo_study.db                          the Optuna study (408 completed trials)
best_model/                           checkpoints from the exploratory model
Comparison/          ← GNN architecture comparison + MLP baseline (research phase)
results/                              per-panel HI curves as CSV; best_params.json
plot_HIs.ipynb                        HI / strain curve plots
```

### Which code should I use?

**Use `paper_code/`.** It contains the model and protocol behind the published
numbers, and its README documents the architecture, every hyperparameter, and
the reproducibility caveats.

The root-level code (`gnn_utils/`, `run_loocv.py`, `Comparison/`) is the earlier
**model-selection phase**: it is how GENConv was chosen over GATv2, GCN,
EdgeConv, SAGPool, GIN, GraphSAGE, SGConv and ChebConv, and how the value of
edge attributes was quantified. It is a different, larger model (hidden
dimension 64, sum pooling, no residual connections) and it does **not**
reproduce the paper's main tables. It is kept because the architecture study is
part of the paper's argument. `paper_code/README.md` has a side-by-side
comparison of the two.

---

## Reproducing the paper

See **[`paper_code/README.md`](paper_code/README.md)** for the full detail. The
essentials:

| Paper result | Command |
|---|---|
| Stiffness estimation, GENEA | `python paper_code/genea_stiffness_lopo.py` |
| Generalisation to the 6-sensor FOD3 panel + MC-dropout uncertainty | `python paper_code/genea_stiffness_lopo_with_fod3.py` |
| CNN baseline | `python paper_code/cnn_baseline_lopo.py` |
| MLP baseline | `python paper_code/mlp_baseline_lopo.py` |
| With/without-HI ablation | `python paper_code/mlp_baseline_lopo.py` with `CONFIG["use_hi"] = False` |
| Truncation-threshold selection | `python paper_code/threshold_search/threshold_search.py` |
| GNN architecture comparison (with / without edge attributes) | `python Comparison/compare_models.py`, `python Comparison/compare_models_no_edges.py` |

Three things to know before you run anything:

* **Four-fold vs five-fold is currently a manual edit.** Both GENEA scripts ship
  with all five panels in `DF_INDICES`. The four-panel protocol (FOD4–FOD7, used
  wherever a fixed 16-sensor input is required) was produced by deleting the
  `'df0'` entry. There is no CLI flag for it yet.
* **Nothing was seeded in the published runs.** Neither codebase called
  `torch.manual_seed` or `np.random.seed`, so re-running gives different numbers
  each time. Expect agreement in trend and magnitude, not digit for digit.
  `run_loocv.py` now takes a `--seed`; it is off by default so that the original
  behaviour is preserved. If you are building on this work, seed everything and
  average over several runs — see the reproducibility section of
  [`paper_code/README.md`](paper_code/README.md).
* **`--drop-level` is a key, not a percentage.** In the root scripts,
  `--drop-level 85` truncates at roughly **70 %** of initial stiffness, which is
  the level used for the published results. Only 99, 95, 90 and 85 are valid;
  the flag now rejects anything else rather than failing with a `KeyError`
  deep in preprocessing.

`Comparison/run_mlp_loocv.py` also runs an MLP, but it is the **exploratory
phase** MLP, not the paper's baseline. Use `paper_code/mlp_baseline_lopo.py`
for the reported comparison, and `Comparison/` for the architecture study.

### Not included in this repository

**The RUL prediction models.** The code that produced the paper's RUL figures
could not be located. `paper_code/notebooks/rul_prototype_gcnconv.ipynb` is an
early prototype and is *not* that code — it uses GCNConv layers with the edge
attributes zeroed out, whereas the paper's RUL model shares the GENEA
architecture. It is included for provenance, clearly labelled, and
[`paper_code/README.md`](paper_code/README.md) sets out the evidence. Everything
the RUL model builds on — the HI, the graph construction, the LOPO protocol,
MC-dropout uncertainty — is fully present; only the RUL target and its training
script are missing.

---

## Data

Five hybrid composite–metal "FOD panels" (FOD3–FOD7) fatigued to failure, each
instrumented with fibre Bragg grating strain sensors, from the H2020 **MORPHO**
project.

The panel-to-key mapping is **positional** (`df0`=FOD3 … `df4`=FOD7) and the
sensor counts differ per panel (FOD3 has 6, FOD5 has 24 truncated to 16). Both
are easy to trip over — see **[`Data/README.md`](Data/README.md)** before
modifying anything under `Data/`.

Raw dataset: Paunikar S, Galanopoulos G and Rébillat M, Zenodo, 2025,
[doi:10.5281/zenodo.14627730](https://doi.org/10.5281/zenodo.14627730) (CC BY
4.0). Experimental campaign: Galanopoulos G et al., *Aerospace* 2025; 12(11):
963, [doi:10.3390/aerospace12110963](https://doi.org/10.3390/aerospace12110963).

---

## Requirements

Pinned in [`requirements.txt`](requirements.txt) to the environment last
verified: Python 3.8.18, PyTorch 2.1.0, PyTorch Geometric 2.6.1, pandas 1.5.3,
NumPy 1.24.3. Newer versions will very likely work.

`tables` is required (pandas reads the `.h5` data through it) and
`scienceplots` is required by the `paper_code/` scripts, which call
`plt.style.use(['science', 'no-latex'])`. A GPU is optional — the published
GENEA model has only 3,495 parameters (counted from
`paper_code/trained_models/genea_4fold/best_model_df1.pth`), so the bottleneck
is the number of graphs per epoch rather than model size.

---

## Citation

If you use this code or data, please cite the paper:

```bibtex
@article{stamatelatos2025gnnshm,
  title   = {Graph neural networks for {SHM}: exploiting spatial
             interdependencies of strain data for diagnostics and prognostics},
  author  = {Stamatelatos, Giannis and Galanopoulos, Georgios and
             Zarouchas, Dimitrios and Loutas, Theodoros},
  journal = {Structural Health Monitoring},
  year    = {2025},
  doi     = {10.1177/14759217251386802},
  note    = {Advance online publication}
}
```

APA 7:

> Stamatelatos, G., Galanopoulos, G., Zarouchas, D., & Loutas, T. (2025). Graph
> neural networks for SHM: Exploiting spatial interdependencies of strain data
> for diagnostics and prognostics. *Structural Health Monitoring*. Advance
> online publication. https://doi.org/10.1177/14759217251386802

The article is currently SAGE OnlineFirst (article number 14759217251386802) and
has no volume, issue or page range yet. Once an issue is assigned, update the
BibTeX entry above and [`CITATION.cff`](CITATION.cff).

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff) — GitHub renders
it as a "Cite this repository" button.

If you use the measurements, please also cite the Zenodo dataset (see
[`Data/README.md`](Data/README.md)).

---

## Authors

| | |
|---|---|
| **Giannis Stamatelatos** ([ORCID](https://orcid.org/0009-0009-3560-6639)) | Applied Mechanics Laboratory, University of Patras, Greece |
| **Georgios Galanopoulos** ([ORCID](https://orcid.org/0000-0003-4998-1308)) | Structural Integrity and Composites Group, TU Delft, The Netherlands |
| **Dimitrios Zarouchas** | Center of Excellence in AI for Structures, TU Delft, The Netherlands |
| **Theodoros Loutas** | Applied Mechanics Laboratory, University of Patras, Greece |

Corresponding author: Giannis Stamatelatos — johnstamly@gmail.com

## Acknowledgements

Safran Composites for the FOD panels, Fraunhofer IFAM for the printed PZT
sensors, and FiSens GmbH for the optical fibres and interrogation systems.

Funded by the European Union's Horizon 2020 research and innovation programme
under grant agreement **No 101006854** (MORPHO).

## License

Code: [MIT](LICENSE). Data: [CC BY 4.0](Data/LICENSE).
