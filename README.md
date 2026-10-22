# LA-MDSI: Fangshan landslide susceptibility experiments

Point attributes and runnable Python code for the Fangshan case study, empirical sensitivity analyses, and controlled spatial simulations. The package supports leakage-aware spatial diagnostics alongside predictive performance.

## Contents

| Folder | Contents |
|---|---|
| `01_Data` | Landslide and sampled pseudo-absence points with attributes, the attributed candidate pool, fixed fold assignments, cluster centres, and the simulation footprint |
| `02_Config` | Analysis protocol, legacy simulation parameters, and source revision map |
| `03_Code` | Numbered run scripts and shared Python functions |
| `04_Documentation` | Data dictionary, detailed run order, scope, and test notes |
| `05_Outputs` | Created only when code is run; excluded from Git |

No original TIFF rasters, trained models, archived predictions, experimental intermediate files, virtual environments, or manuscript drafts are distributed. Point attributes have already been extracted. Running the analyses creates new outputs locally.

## Quick start

Use Python 3.12 (64-bit). Open a terminal in the downloaded repository.

On Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe 03_Code/00_Check_and_demo.py
```

On Linux or macOS:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 03_Code/00_Check_and_demo.py
```

The demonstration fits one logistic-regression model on the original training fold and evaluates held-out observations, including the six spatial diagnostics and 499-permutation response screen. It is a functional example, not the complete manuscript analysis. Use `--check-only` to skip the model fit.

After installation, scripts can be invoked from any working directory. Input paths are resolved from the script location. There are no hard-coded author-specific drive paths. If calling from another directory, provide the path to the desired script.

## Full benchmark

In the commands below, `python` means the Python executable from the environment created above.

```sh
python 03_Code/01_Train_benchmark.py --workers 2
python 03_Code/02_Diagnose_benchmark.py --workers 2
python 03_Code/03_Summarize_benchmark.py
```

The full benchmark contains 2,400 fits. It is substantially more expensive than the demonstration. See [validation and platform notes](04_Documentation/03_Validation_and_scope.md) for the tested scope and the legacy XGBoost wheel-tag warning. A reduced pipeline test is available:

```sh
python 03_Code/01_Train_benchmark.py --smoke --workers 2
python 03_Code/02_Diagnose_benchmark.py --smoke --workers 2
```

This test fits 48 models: one seed and one fold, both training-separation distances, both feature sets, three representations, and four classifiers. Smoke outputs are isolated from full-run outputs. The formal benchmark summarizer requires all five folds and ten seeds and will not report smoke runs as final results.

See [the run guide](04_Documentation/02_Run_guide.md) for clustering sweeps, sampling scales, feature importance, simulations, and the dependence extension. Run expensive stages explicitly; no full experiment starts during installation.

## Output location

The default is `05_Outputs` inside this repository. To save results elsewhere, set `LA_MDSI_OUTPUT_DIR` before starting Python. A relative value is resolved against the repository root. The default needs no configuration.

Generated run manifests record the resolved locations and hashes of their inputs. Move the repository before starting an experiment; do not relocate a partially completed output tree and resume its old checkpoints. Start a new output directory after changing code, data, or package versions.

## Data and release notes

The main table contains 8,806 observations: 4,403 landslides and 4,403 sampled pseudo-absence locations. Negative labels are not field-confirmed absence records. Coordinates use EPSG:32650. The candidate pool is needed only for the sampling-radius sensitivity experiment.

This is a portable analytical code release, not an automatic Word-manuscript or GIS-map builder. Scientific settings are preserved; packaging does not independently verify the remote sensing product provenance. Historical and primary simulation conventions are explicitly separated in the protocol.

The largest input is the compressed attributed candidate pool (about 57 MiB). For the initial GitHub upload, use Git or GitHub Desktop rather than dragging that file into the web uploader. No Git LFS pointer files are used in this package.

Before making a public release, the authors should confirm redistribution rights for the point data and choose suitable data and code licences. No new licence or ownership claim has been imposed by this packaging step.
