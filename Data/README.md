# Data

This directory holds the preprocessed strain and stiffness measurements used by
every script in this repository. Both `.csv` and `.h5` copies are provided; the
code reads the **`.h5`** files (`pandas.read_hdf`, which requires `tables`).

## Provenance and attribution

The measurements derive from the fatigue campaign of the H2020 **MORPHO**
project (grant agreement No 101006854) on hybrid composite–metal "FOD panels"
representative of an aircraft engine fan blade substructure.

* **Raw dataset** — Paunikar S, Galanopoulos G and Rébillat M. *An experimental
  data set for the SHM of a substructure of an engine fan blade from the MORPHO
  project.* Zenodo, 2025. <https://doi.org/10.5281/zenodo.14627730> (CC BY 4.0)
* **Experimental campaign** — Galanopoulos G, Paunikar S, Stamatelatos G,
  Loutas T, Mechbal N, Rébillat M and Zarouchas D. *SHM for Complex Composite
  Aerospace Structures: A Case Study on Engine Fan Blades.* Aerospace 2025;
  12(11): 963. <https://doi.org/10.3390/aerospace12110963>

If you use the data, cite the Zenodo record in addition to this repository's
accompanying paper.

## Contents

```
Data/
├── Strain/                     # FBG strain sensor time series, TimedeltaIndex
│   ├── Strains_FOD3.csv.h5     # 125 124 rows ×  6 sensors
│   ├── Strains_FOD4.csv.h5     # 164 146 rows × 16 sensors
│   ├── Strains_FOD5.csv.h5     # 186 405 rows × 24 sensors
│   ├── Strains_FOD6.csv.h5     # 286 481 rows × 16 sensors
│   └── Strains_FOD7.csv.h5     # 258 652 rows × 16 sensors
└── Stiffness_Reduction/        # MTS machine stiffness per fatigue block
    ├── MTSStiffness_exp_FOD3.h5   # 3 273 rows
    ├── MTSStiffness_exp_FOD4.h5   # 3 060 rows
    ├── MTSStiffness_exp_FOD5.h5   # 3 336 rows
    ├── MTSStiffness_exp_FOD6.h5   # 3 144 rows
    └── MTSStiffness_exp_FOD7.h5   # 3 233 rows
```

The stiffness files are read as `pd.read_hdf(path)['Stiffness']`.

## Panel ↔ `dfN` key mapping (important)

The scripts enumerate the `.h5` files with `os.scandir(...)` followed by
`sort()`, then name them `df0, df1, ...` **positionally**. With the files as
shipped this gives:

| key   | panel | strain sensors | notes |
|-------|-------|----------------|-------|
| `df0` | FOD3  | 6              | different geometry; used only in the five-fold run |
| `df1` | FOD4  | 16             | |
| `df2` | FOD5  | 24 → 16        | the last 8 columns are dropped in preprocessing |
| `df3` | FOD6  | 16             | |
| `df4` | FOD7  | 16             | |

This mapping is **positional, not looked up by name**. Adding, removing or
renaming a file in either directory silently shifts every `dfN` key and
therefore every fold. If you add panels, check this table first.

Two consequences worth knowing:

* **FOD5 is truncated to 16 sensors.** `preprocess_data` contains
  `if key == 'df2': strain_df = strain_df.iloc[:, :-8]`. That is FOD5's 24
  channels reduced to the 16 shared by FOD4/6/7, so that the fixed-input MLP and
  CNN baselines can be compared on a consistent four-panel dataset.
* **FOD3 has only 6 sensors.** It is excluded from the four-fold comparison
  (which requires a fixed input size) and included only in the five-fold GENEA
  run that tests generalisation across geometries.

## License

The data in this directory is licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
inherited from the Zenodo record above. See `Data/LICENSE`.

The *code* in this repository is MIT-licensed — see the top-level `LICENSE`.
