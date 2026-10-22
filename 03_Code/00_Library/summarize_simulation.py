import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
import run_experiment as run
OUT=run.OUT
def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], float_precision="round_trip", low_memory=False)

def exact_interval(passed, total):
    return (0. if passed == 0 else float(beta.ppf(.025, passed, total - passed + 1)),
            1. if passed == total else float(beta.ppf(.975, passed + 1, total - passed)))

def finite_median(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan
def analyze(smoke=False):
    base = OUT / "smoke" if smoke else OUT
    verification = json.loads((base / "verification.json").read_text(encoding="utf-8"))
    assert verification["success"] and verification["smoke"] == smoke
    data, samples, folds, conditions = [read(base / f"{name}.csv") for name in ("datasets", "samples", "folds", "conditions")]
    expected = 1 if smoke else 100
    summary_rows = []
    for (n, scene, arm, cv), frame in data.groupby(["N", "scenario", "arm", "fold_count"], sort=False):
        assert len(frame) == expected
        passed = int(frame.reportable_at_capture1.sum())
        low, high = exact_interval(passed, len(frame))
        fs = folds.query("N == @n and scenario == @scene and arm == @arm and fold_count == @cv")
        eligible = frame.sampling_feasible
        summary_rows.append(dict(N=n, scenario=scene, arm=arm, fold_count=cv, worlds=len(frame),
            sampling_feasible=int(eligible.sum()), numerical_finite=int(frame.numeric_at_capture1.sum()),
            valid_complete_screens=int(frame.valid_all_fold_screens.sum()), all_fold_screen_pass=int(frame.response_all_folds_pass.sum()),
            fully_reportable=passed, reportability=passed / len(frame), ci_low=low, ci_high=high,
            holm_reportable=int((frame.numeric_at_capture1 & frame.holm_all_folds_pass).sum()),
            all_capture_finite=int(frame.numeric_all_captures.sum()), all_capture_reportable=int(frame.reportable_all_captures.sum()),
            median_min_validation_positive=finite_median(frame.min_validation_positive),
            min_validation_positive=float(frame.min_validation_positive.min()),
            median_min_validation_n=finite_median(frame.min_validation_n),
            single_stratum_worlds=int(frame.min_tree_H.eq(1).sum()),
            invalid_fold_screens=int((~fs.screen_valid).sum()),
            numerical_failure_folds=int((~fs.numeric_finite).sum()),
            constant_response_folds=int(fs.numerical_failure.eq("constant_validation_response").sum()),
            median_background_contamination=finite_median(frame.background_contamination),
            sampling_failures=int((~eligible).sum())))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(base / "reportability_summary.csv", index=False, encoding="utf-8-sig")
    lookup = summary.set_index(["N", "scenario", "arm", "fold_count"])
    paired_rows = []
    rng = np.random.default_rng(2026101841)
    for (n, scene, arm), frame in data.groupby(["N", "scenario", "arm"], sort=False):
        paired = frame.pivot(index="world_id", columns="fold_count", values="reportable_at_capture1")
        assert len(paired) == expected and not paired.isna().any().any()
        f3, f5 = paired[3].to_numpy(bool), paired[5].to_numpy(bool)
        difference = f3.astype(int) - f5.astype(int)
        bootstrap = difference[rng.integers(0, len(paired), size=(10000, len(paired)))].mean(axis=1)
        paired_rows.append(dict(N=n, scenario=scene, arm=arm, paired_worlds=len(paired),
            both_pass=int((f3 & f5).sum()), three_only=int((f3 & ~f5).sum()), five_only=int((f5 & ~f3).sum()),
            neither=int((~f3 & ~f5).sum()), difference_3minus5=float(difference.mean()),
            paired_bootstrap_ci_low=float(np.quantile(bootstrap, .025, method="linear")),
            paired_bootstrap_ci_high=float(np.quantile(bootstrap, .975, method="linear")),
            bootstrap_repetitions=10000))
    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(base / "paired_cv_comparison.csv", index=False, encoding="utf-8-sig")
    sampling_rows = []
    for (n, scene, cv), frame in data.groupby(["N", "scenario", "fold_count"], sort=False):
        paired = frame.pivot(index="world_id", columns="arm", values="reportable_at_capture1")
        for arm_a, arm_b in [("balanced_true_absence", "representative"),
                             ("balanced_background", "balanced_true_absence"),
                             ("balanced_background_buffer1000", "balanced_background")]:
            left, right = paired[arm_a].to_numpy(bool), paired[arm_b].to_numpy(bool)
            delta = left.astype(int) - right.astype(int)
            bootstrap = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
            sampling_rows.append(dict(N=n, scenario=scene, fold_count=cv, arm_A=arm_a, arm_B=arm_b,
                paired_worlds=len(paired), both_pass=int((left & right).sum()), A_only=int((left & ~right).sum()),
                B_only=int((right & ~left).sum()), neither=int((~left & ~right).sum()),
                difference_AminusB=float(delta.mean()), paired_bootstrap_ci_low=float(np.quantile(bootstrap, .025)),
                paired_bootstrap_ci_high=float(np.quantile(bootstrap, .975))))
    pd.DataFrame(sampling_rows).to_csv(base / "paired_sampling_comparison.csv", index=False, encoding="utf-8-sig")
    failure = folds.groupby(["N", "scenario", "arm", "fold_count", "numerical_failure"], dropna=False).size().reset_index(name="folds")
    failure.to_csv(base / "numerical_failure_summary.csv", index=False, encoding="utf-8-sig")
    score_rows = []
    for keys, frame in conditions.groupby(["N", "scenario", "arm", "fold_count", "capture"], sort=False):
        for metric in ["r_dsi", "s_theta", "roc_auc", "brier", *run.sc.THETA,
                       *[f"eta_a_{r}" for r in run.sc.RESIDUALS],
                       *[f"eta_h_{s}_{r}" for s in run.sc.STRATA for r in run.sc.RESIDUALS]]:
            finite = frame[metric].to_numpy(float)
            finite = finite[np.isfinite(finite)]
            score_rows.append(dict(zip(["N", "scenario", "arm", "fold_count", "capture"], keys)) |
                              dict(metric=metric, scheduled_worlds=expected, generated_conditions=len(frame), n_finite=len(finite),
                                   median=float(np.median(finite)) if len(finite) else np.nan,
                                   q25=float(np.quantile(finite, .25)) if len(finite) else np.nan,
                                   q75=float(np.quantile(finite, .75)) if len(finite) else np.nan,
                                   conditional_on_finite=True))
    pd.DataFrame(score_rows).to_csv(base / "score_response_summary.csv", index=False, encoding="utf-8-sig")
    score_cv_rows = []
    for (n, scene, arm), frame in conditions.groupby(["N", "scenario", "arm"], sort=False):
        # Controlled predictions are the same for both CV designs; only spatial
        # diagnostics and training-fitted strata depend on the fold allocation.
        for metric in ["roc_auc", "brier"]:
            paired_metric = frame.pivot(index=["world_id", "capture"], columns="fold_count", values=metric)
            assert np.allclose(paired_metric[3], paired_metric[5], equal_nan=True, atol=1e-14, rtol=1e-14)
        primary = frame.loc[frame.capture.eq(1)]
        values = primary.pivot(index="world_id", columns="fold_count", values="r_dsi")
        valid = np.isfinite(values[3]) & np.isfinite(values[5])
        differences = values.loc[valid, 3] - values.loc[valid, 5]
        stable = primary.pivot(index="world_id", columns="fold_count", values="s_theta")
        svalid = np.isfinite(stable[3]) & np.isfinite(stable[5])
        score_cv_rows.append(dict(N=n, scenario=scene, arm=arm, paired_finite=len(differences),
            median_r_dsi_difference_3minus5=finite_median(differences),
            median_absolute_r_dsi_difference=finite_median(abs(differences)),
            median_s_theta_difference_3minus5=finite_median(stable.loc[svalid,3]-stable.loc[svalid,5]),
            conditioning="both CV summaries numerically finite, screening pass not required"))
    score_cv_summary = pd.DataFrame(score_cv_rows)
    score_cv_summary.to_csv(base / "score_cv_comparison.csv", index=False, encoding="utf-8-sig")
    print("Simulation summary tables written to", base, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    analyze(args.smoke)

if __name__ == '__main__':
    main()
