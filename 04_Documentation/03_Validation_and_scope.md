# Portability validation and scope

Validation date: 22 October 2026.

## Executed checks

Tests used a separately created Python 3.12.14 environment on 64-bit Windows. The environment was installed from the root requirements file; it did not import the author's vendored scientific dependencies.

| Check | Outcome |
|---|---|
| All 18 numbered entry points: help/startup | Passed |
| Repository copied to a different directory with spaces; invoked from a different working directory | Passed |
| Real-data demonstration: one full validation fold, logistic regression, training-fitted strata, and 499-permutation screening | Passed |
| Benchmark smoke run: both buffers, both feature sets, three representations, four classifiers | 48 fits completed |
| Spatial diagnostics on those 48 predictions | Passed |
| Spatial clustering sweep, seed 42, fold 1 | Six model units completed |
| Attribute clustering sweep, seed 42, fold 1 | Eight model units completed |
| Sampling-scale smoke run: gap, block width, exclusion radius | Three model units completed |
| Global and within-block feature-importance smoke run | Passed |
| Primary sampling simulation, one smoke world in each of six scenarios | Six worlds, 96 dataset/design records, and 480 capture-condition records completed |
| Independent simulation checks | 1,920 formula rows, 480 condition means, 384 graph checks, and 144 raw response-screen checks passed |
| Simulation summary tables | Generated successfully |
| Legacy baseline generator smoke run | Six worlds and 30 capture records completed |

The real-data demonstration used 6,964 training and 1,842 validation observations. Relocating the repository produced the same demonstration metrics. This tests path independence, not independence of scientific evidence.

Thirteen core scientific functions were compared structurally with the supplied source revisions and were unchanged: residuals, Moran's I, q, eta, joint theta, six-setting summaries, both training-only stratification routines, response permutation testing, corrected graphs, Holm adjustment, estimators, and predictive metrics.

The package was also checked for author-specific absolute drive paths, non-English folder/file names, non-English Markdown, and accidentally included raster/model/intermediate files. Test-generated results and bytecode were kept outside the release folder.

## Limits of this validation

This packaging task did not rerun all 2,400 benchmark fits, all 600 formal simulation worlds, the full dependence extension, or every full-run summary and plot. Those stages are supplied with their prerequisites and unchanged scientific settings, but the tests above are functional smoke tests rather than a new numerical reproduction of the complete paper. Linux and macOS were not executed in this validation.

The supplied Windows XGBoost 1.7.6 wheel installs and its models successfully fit and predict in the test environment. Its internal wheel tag is `cp39-cp39-win_amd64`, however, so `pip check` reports a platform-tag warning under Python 3.12. The package metadata was not altered, and the estimator version was not upgraded to suppress that warning. All other dependency checks passed. If a different system cannot load XGBoost, resolve its platform/runtime installation before running the full experiments; do not silently change estimator versions when reproducing the paper.

The package reproduces analyses from attributed points. It does not reconstruct raster processing, independently verify remote-sensing acquisition dates, render the manuscript, or guarantee pixel-identical later artwork. Input data redistribution rights and the final release licence remain author decisions.

## Release contents

Only attributed point/design inputs, necessary code/configuration, and English documentation are delivered. Original TIFFs, archived model bundles, prediction tables, raw simulation arrays, old audit folders, and test environments are excluded. Runtime outputs are ignored by Git. Conventional files such as `README.md` and `requirements.txt`, and importable helper modules, retain their standard names; data files, run scripts, documentation, and top-level folders are numbered.
