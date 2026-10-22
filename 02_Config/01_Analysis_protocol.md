# Analysis protocol

## Empirical benchmark

The input contains 4,403 landslide records and 4,403 sampled pseudo-absence locations. The latter are sampled negative labels, not field-confirmed stable sites. Coordinates are in metres (EPSG:32650). All raster-derived attributes needed by the analysis are already included in the point table.

The fixed benchmark uses 69 spatial groups, five folds, ten seeds (42 to 132 in steps of 10), training-separation distances of 0 and 800 m, and 16-factor and 20-factor designs. The three representations are all events, unweighted prototypes, and size-weighted prototypes. The four classifiers are logistic regression, random forest, radial-basis-function SVM, and XGBoost. Their original hyperparameters are fixed in `03_Code/00_Library/model_core.py`.

Aspect is represented by its supplied sine and cosine columns. Land cover, soil, and lithology (when included) are categorical. Preprocessors are fitted on uncompressed training observations. Prototypes are calculated from training positives within the fixed inventory clusters. Each class receives total weight N/2, where N is the retained, uncompressed training count. Validation observations remain unchanged. No grid search, inner model selection, or early stopping is introduced by this package.

## Spatial diagnostics and screening

The primary empirical and sampling-simulation graphs explicitly remove the query observation before selecting k = 4, 8, 16, and 32 neighbours. Directed links are combined by symmetric binary union. CART and the five-cluster K-means procedure use training observations only; validation observations are assigned using the fitted rules.

Each validation fold uses 499 unrestricted label permutations. Coordinates, graphs, strata, and class counts remain fixed. Two-sided extremeness is the absolute deviation from the permutation mean. The Monte Carlo p-value is (b + 1)/(499 + 1). The primary criterion remains strict p < 0.05 for all four Moran tests and both q tests. Holm adjustment is a separate sensitivity analysis. The primary five-fold rule requires all six tests to pass in all five folds; less strict rules are reported separately and do not replace it.

Raw, deviance, and Pearson residuals are crossed with CART and K-means strata. The original centred magnitude ratios and product-complement theta formula are retained. R-DSI is the Type-7 lower quartile of the six untruncated theta values; S-theta is one minus their interquartile range. Missing or non-finite required values propagate to summaries. Finite numerical diagnostics remain available with their screen status even when eligibility fails.

For clustering sweeps, AUC and Brier scores use complete pooled out-of-fold predictions within a seed, followed by equal seed averaging. R-DSI and S-theta use equal fold averaging within each seed, then equal seed averaging. Reference theta uses validation-size-weighted fold means, then equal seed averaging. Spatial diagnostics are never recomputed from pooled residuals. Summary field names distinguish these quantities. The primary benchmark summary retains its original diagnostic column names; `theta_tree_raw_mean` is an equal-fold diagnostic, not the size-weighted reference theta of the clustering sweeps.

## Simulations

The primary sampling simulation retains its 69-cell footprint, 110,400 candidate sites, six signal/prevalence scenarios, sample sizes 900 and 3,600, four sampling arms, five- and three-fold designs, five capture fractions, and 100 independently generated worlds per scenario. Controlled probability scores are not trained classifiers. The extension adds sample sizes 400 and 8,100, threshold sensitivity, and a label-constructed near-perfect-prediction boundary check. It does not change the primary threshold or formulas.

The legacy 30-by-30-grid generator is retained only because the separate dependence-sensitivity experiment replays its baseline labels and folds. It intentionally retains its archived graph convention and 199 permutations. It is not the corrected primary sampling experiment and must not be pooled with it. Legacy baseline files are generated at runtime, not distributed as precomputed intermediates.

## Portability changes

Input paths are resolved from the repository location, not the working directory. Optional output redirection uses LA_MDSI_OUTPUT_DIR. All model files, predictions, plots, checkpoints, and logs are generated under the output directory. Smoke and full runs use separate output trees. No scientific threshold, estimator hyperparameter, random seed, or score formula was selected or changed for packaging.

Original provenance assertions that referenced external Word manuscripts or unrelated archived audit tables were removed. Checks on the actual numerical inputs and source functions remain. The package contains extracted fixed clustering fold memberships, not the old predictions from which those memberships were recovered.
