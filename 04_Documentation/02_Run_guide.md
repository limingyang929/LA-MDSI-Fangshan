# Run guide

## 1. Install and demonstrate

Create the Python 3.12 environment using the root README. All commands below use that environment's Python executable. `--workers 2` is a conservative starting point; increase it only when memory permits. Each classifier uses one model thread.

```sh
python 03_Code/00_Check_and_demo.py
```

The demonstration uses one full empirical validation fold, not random replacement data. It does not supply final manuscript numbers.

## 2. Empirical benchmark

```sh
python 03_Code/01_Train_benchmark.py --workers 2
python 03_Code/02_Diagnose_benchmark.py --workers 2
python 03_Code/03_Summarize_benchmark.py
```

Run in this order. The first command fits all 2,400 models. The second calculates the response screens and residual diagnostics. The third requires complete inputs and creates seed- and configuration-level summaries and paired uncertainty estimates. The threshold remains p < 0.05. Main results are under `05_Outputs/01_Full/01_Benchmark`.

For a reduced pipeline test, add `--smoke` to the first two commands. This uses 48 fits and writes under `05_Outputs/02_Smoke/01_Benchmark`. Do not run the formal summary on this partial experiment.

## 3. Empirical extensions

```sh
python 03_Code/04_Run_clustering_sweeps.py --workers 2
python 03_Code/05_Run_sampling_scales.py --workers 2
python 03_Code/06_Run_feature_importance.py --workers 2
python 03_Code/07_Summarize_extensions.py
python 03_Code/08_Plot_extensions.py
```

The clustering sweep has 700 model units over 100 fold designs. It uses the supplied, frozen clustering fold memberships. The sampling-scale stage has 400 model units and requires the attributed background candidate pool. Compatible freshly generated main-benchmark models are reused when available; otherwise sweep and scale models are fitted from scratch using the same settings. Feature importance requires the 50 main, unbuffered, size-weighted XGBoost models generated in stage 2; it does not read an archived model bundle.

The sweep, scale, and importance runners accept `--smoke` for limited runs. To test importance, first complete the benchmark smoke run. Extension summaries and plots require complete extension tables. The plot script recreates the analytical plots from these tables; it is not a pixel-for-pixel reconstruction of later manual artwork layouts.

Results are under `05_Outputs/01_Full/02_Empirical_extensions`.

## 4. Primary sampling simulation

```sh
python 03_Code/09_Run_sampling_simulation.py --freeze
python 03_Code/09_Run_sampling_simulation.py --run --workers 2
python 03_Code/16_Verify_sampling_simulation.py
python 03_Code/10_Summarize_sampling_simulation.py
python 03_Code/17_Inspect_weak_signal.py
```

This generates 600 independent worlds (100 per scenario). It evaluates N = 900 and 3,600, four sampling arms, five- and three-fold CV, and five capture fractions per world. The generator creates all synthetic populations at runtime. No population arrays or intermediate simulation tables are included in the repository.

For a separate six-world pipeline test:

```sh
python 03_Code/09_Run_sampling_simulation.py --smoke --workers 2
python 03_Code/16_Verify_sampling_simulation.py --smoke
python 03_Code/10_Summarize_sampling_simulation.py --smoke
```

Smoke repetition 0 is excluded from formal simulations. These are still real computations across all designed arms and sample sizes and can take several minutes. The weak-signal inspection command is intended for the full 100-world-per-scenario run.

## 5. Additional sizes, boundary check, and screening sensitivity

Complete the full primary simulation first, then run:

```sh
python 03_Code/11_Run_simulation_controls.py freeze
python 03_Code/11_Run_simulation_controls.py extend --workers 2
python 03_Code/11_Run_simulation_controls.py boundary --workers 2
python 03_Code/11_Run_simulation_controls.py summarize
```

This adds N = 400 and 8,100, computes the near-perfect-prediction boundary control, and summarizes threshold and fold-rule sensitivity. The primary five-fold, all-six-tests rule and p < 0.05 are unchanged. Boundary predictions are deliberately constructed from labels to test a mathematical property; they are not a predictive-validation result.

Use the full input tree for this extension. The primary simulation's six-world smoke tree is not a substitute for the required 600-world input. The extension's inherited `--smoke` option is a small batch of full-input worlds; it is not documented as a stand-alone smoke pipeline here.

Results are under `05_Outputs/01_Full/04_Simulation_controls`.

## 6. Separate dependence sensitivity

```sh
python 03_Code/12_Generate_dependence_baseline.py
python 03_Code/13_Run_dependence_sensitivity.py
python 03_Code/14_Summarize_six_settings.py
python 03_Code/15_Plot_dependence.py
```

This stage retains the inherited 30-by-30 baseline generator and its archived graph convention and 199-permutation screen. The baseline command generates the required labels, noise arrays, fold assignments, and component values from code; no old experimental outputs are distributed. The dependence extension checks that its paired baseline matches these regenerated inputs before varying template loading.

This experiment is distinct from the corrected 69-cell primary simulation. Do not combine their reportability denominators or rename one as the other. Dependence outputs are under `05_Outputs/01_Full/06_Dependence_sensitivity`.

## 7. Checkpoints and safe reruns

Outputs are created under `05_Outputs` by default and are excluded by `.gitignore`. Input files remain unchanged. Completed fold/world checkpoints can be reused with the same source code, data, and environment. If a run is interrupted, confirm no worker process is still running before removing that run's `RUNNING.lock` file. Do not change scientific code or inputs and then resume old checkpoints; choose a new output directory instead.

If you relocate the repository, start with a fresh output directory. Generated manifests contain runtime-resolved paths for auditing; they are not machine-specific source-code dependencies.

The root README explains the optional LA_MDSI_OUTPUT_DIR override. Full and smoke outputs are kept separate. Do not upload generated output trees or your virtual environment when releasing this source-and-input package.
