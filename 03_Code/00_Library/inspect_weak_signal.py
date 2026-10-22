"""Read-only follow-up: locate weak-signal screening bottlenecks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from project_config import *
BASE = SIMULATION


def read(name, columns=None):
    return pd.read_csv(BASE / name, usecols=columns, keep_default_na=False,
                       na_values=[""], low_memory=False, float_precision="round_trip")


screen = read("screens.csv", ["world_id", "scenario", "N", "arm", "fold_count", "fold",
    "setting", "observed", "null_mean", "null_sd", "permutation_p"])
folds = read("folds.csv")
datasets = read("datasets.csv")
target = screen.loc[screen.scenario.eq("weak_mixed") & screen.arm.eq("balanced_background")].copy()
target["passed"] = np.isfinite(target.permutation_p) & target.permutation_p.lt(.05)
target["valid"] = np.isfinite(target.permutation_p)
target["absolute_z"] = abs(target.observed - target.null_mean) / target.null_sd
by_test = target.groupby(["N", "fold_count", "setting"], sort=True).agg(
    passed=("passed", "sum"), total=("passed", "size"), valid=("valid", "sum"),
    median_p=("permutation_p", "median"), median_absolute_z=("absolute_z", "median"))
by_test["pass_percent"] = 100 * by_test.passed / by_test.total
print("PER-TEST WEAK SIGNAL, BALANCED BACKGROUND, NO BUFFER")
print(by_test.to_string())

print("\nFOLD AND DATASET STATUS")
for (n, count), frame in target.groupby(["N", "fold_count"], sort=True):
    per_fold = frame.groupby(["world_id", "fold"]).passed.sum()
    any_fold = per_fold.eq(6).groupby("world_id").any()
    all_folds = per_fold.eq(6).groupby("world_id").all()
    fs = folds.loc[folds.scenario.eq("weak_mixed") & folds.arm.eq("balanced_background") & folds.N.eq(n) & folds.fold_count.eq(count)]
    ds = datasets.loc[datasets.scenario.eq("weak_mixed") & datasets.arm.eq("balanced_background") & datasets.N.eq(n) & datasets.fold_count.eq(count)]
    print(json.dumps(dict(N=int(n), fold_count=int(count), unique_folds=len(per_fold),
        folds_all_six_pass=int(per_fold.eq(6).sum()), datasets_any_fold_pass=int(any_fold.sum()),
        datasets_all_folds_pass=int(all_folds.sum()), numerical_finite=int(ds.numeric_at_capture1.sum()),
        valid_folds=int(fs.screen_valid.sum()),
        histogram_tests_passing_in_fold={str(int(k)):int(v) for k,v in per_fold.value_counts().sort_index().items()},
        mean_validation_n=float(fs.validation_n.mean()), median_validation_positives=float(fs.validation_positive.median()),
        median_background_contamination=float(ds.background_contamination.median())), ensure_ascii=False))

print("\nTRUE-ABSENCE REFERENCE")
ref = datasets.loc[datasets.scenario.eq("weak_mixed") & datasets.arm.eq("balanced_true_absence")]
print(ref.groupby(["N", "fold_count"]).reportable_at_capture1.agg(["sum", "count"]).to_string())

print("\nGENERATING SIGNAL FRACTION OF BERNOULLI VARIANCE")
for scenario in ["weak_mixed", "mixed"]:
    shares, sds = [], []
    for repetition in range(1, 101):
        path = BASE / "worlds" / f"{scenario}_r{repetition:03d}" / "population.npz"
        with np.load(path, allow_pickle=False) as saved:
            p = saved["probability"]
        shares.append(float(np.var(p) / (p.mean() * (1 - p.mean()))))
        sds.append(float(np.std(p)))
    print(json.dumps(dict(scenario=scenario, median_probability_sd=float(np.median(sds)),
        median_structured_variance_share=float(np.median(shares)),
        q25_structured_variance_share=float(np.quantile(shares,.25)),
        q75_structured_variance_share=float(np.quantile(shares,.75))), ensure_ascii=False))
