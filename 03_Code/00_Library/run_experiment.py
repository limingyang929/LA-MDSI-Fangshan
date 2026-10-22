"""Frozen, paired sampling and spatial-CV experiment. Original files are read only."""
from __future__ import annotations

import os
import sys
for _thread_option in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_option] = "1"
sys.dont_write_bytecode = True

from pathlib import Path
from project_config import *
OUT = ensure_output(SIMULATION)

import argparse
import ast
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import scientific_core as sc

REFERENCE_SAMPLES = DATA / '06_Simulation_footprint.csv'
REFERENCE_GRAPH = CODE / 'graph_core.py'
REFERENCE_CORE = CODE / 'scientific_core.py'
SCENARIOS = {
    "null": (0.0, 0.0, .20),
    "weak_mixed": (.3, .3, .20),
    "autocorrelation_only": (1.2, 0.0, .20),
    "heterogeneity_only": (0.0, 1.2, .20),
    "mixed": (1.2, 1.2, .20),
    "mixed_rare": (1.2, 1.2, .05),
}
ARMS = ("representative", "balanced_true_absence", "balanced_background", "balanced_background_buffer1000")
SIZES = (900, 3600)
FOLDS = (5, 3)
CAPTURES = (0.0, .25, .5, .75, 1.0)
START = time.perf_counter()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def frame_csv(path, rows):
    path = Path(path)
    assert path.resolve().is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def log(value):
    print(f"[{time.perf_counter() - START:.1f}s] {value}", flush=True)


def fixed_graph_function():
    source = REFERENCE_GRAPH.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "correct_graphs")
    namespace = dict(np=np, sparse=sparse, NearestNeighbors=NearestNeighbors, sc=sc, hashlib=hashlib)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(REFERENCE_GRAPH), "exec"), namespace)
    return namespace["correct_graphs"]


correct_graphs = fixed_graph_function()


def standardized(a):
    return (a - np.mean(a)) / np.std(a)


def freeze():
    """This command must finish before any response generation or test is run."""
    if (OUT / "protocol.json").exists():
        protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
        assert protocol["runner_sha256"] == sha(__file__), "Frozen runner changed"
        for path, expected in protocol["protected_inputs"].items():
            assert sha(path) == expected, path
        return protocol
    ref = pd.read_csv(REFERENCE_SAMPLES, usecols=["group_x", "group_y", "spatial_block"])
    groups = sorted(ref.spatial_block.unique(), key=lambda s: tuple(map(int, s.split("_"))))
    assert len(groups) == 69
    origin = np.array([ref.group_x.min(), ref.group_y.min()], float)
    cells = np.array([list(map(int, s.split("_"))) for s in groups], int)
    sub = 40
    shape = tuple((cells.max(axis=0) + 1) * sub)
    mask = np.zeros(shape, bool)
    for row, col in cells:
        mask[row * sub:(row + 1) * sub, col * sub:(col + 1) * sub] = True
    iy, ix = np.where(mask)
    xy = origin + np.c_[ix + .5, iy + .5] * (5000 / sub)
    group = (iy // sub) * int(cells[:, 1].max() + 1) + ix // sub
    x_norm = (xy[:, 0] - origin[0]) / (shape[1] * 5000 / sub)
    band = np.minimum((x_norm * 5).astype(int), 4)
    heterogeneity = standardized(np.array([-1.3, .7, -.5, 1.4, -.3])[band])
    terrain_rng = np.random.default_rng(2026101807)
    elevation_grid = gaussian_filter(terrain_rng.normal(size=shape), sigma=24, mode="reflect")
    slope_grid = np.hypot(*np.gradient(elevation_grid))
    cover_grid = gaussian_filter(terrain_rng.normal(size=shape), sigma=12, mode="reflect")
    terrain_values = [elevation_grid[mask], slope_grid[mask], cover_grid[mask]]
    terrain_codes = [np.digitize(v, np.quantile(v, [.3333333333, .6666666667])) for v in terrain_values[:2]]
    terrain_codes.append(np.digitize(terrain_values[2], np.quantile(terrain_values[2], [.25, .5, .75])))
    sampling_stratum = terrain_codes[0] * 12 + terrain_codes[1] * 4 + terrain_codes[2]
    np.savez_compressed(OUT / "geometry.npz", coordinates=xy, groups=group, mask=mask,
                        row=iy, col=ix, heterogeneity=heterogeneity,
                        sampling_stratum=sampling_stratum, origin=origin)
    protected_paths = [REFERENCE_SAMPLES, REFERENCE_CORE, REFERENCE_GRAPH]
    protocol = {
        "frozen_utc": now(), "runner_sha256": sha(__file__),
        "authorization": "User agreed to sampling-matched simulations and paired five-versus-three-fold comparison",
        "main_alpha": .05, "p_comparison": "strict p < 0.05", "permutations_per_fold": 499,
        "formulas": "Original eta magnitude ratios, product-complement theta, Type-7 Q1, 1-IQR; negative scores retained",
        "epsilon": sc.EPS, "probability_clip": sc.P_CLIP,
        "scenarios": {k: dict(zip(["spatial_strength", "heterogeneity_strength", "population_prevalence"], v)) for k, v in SCENARIOS.items()},
        "independent_worlds_per_scenario": 100, "total_independent_worlds": 600,
        "seed_rule": "2026101800 + scenario_index*10000 + repetition (1..100); smoke repetition 0 excluded",
        "sizes": list(SIZES), "fold_counts": list(FOLDS), "capture_fractions": list(CAPTURES),
        "arms": {
            "representative": "N sites drawn without replacement from the finite Bernoulli population; observed labels are latent Bernoulli responses",
            "balanced_true_absence": "N/2 sampled true cases and N/2 known latent noncases; diagnostic reference, not field pseudo-absence practice",
            "balanced_background": "Same N/2 cases, with N/2 background sites labelled zero; unobserved latent cases can occur among background sites",
            "balanced_background_buffer1000": "Same cases; exclude background sites at Euclidean distance <=1000 m from observed cases before drawing N/2 background controls",
        },
        "source_population": {"candidate_sites": int(mask.sum()), "69_cell_footprint_source": str(REFERENCE_SAMPLES),
            "cell_width_m": 5000, "subgrid_per_cell": sub, "candidate_spacing_m": 125,
            "spatial_field": "Gaussian-smoothed independent normal grid, sigma 3000 m / 125 m = 24 pixels; standardized over 69-cell mask",
            "heterogeneity": "Five x bands on the masked domain, standardized fixed template [-1.3,0.7,-0.5,1.4,-0.3]",
            "response": "Conditional independent Bernoulli with scenario-specific logistic latent probability; intercept calibrated to generating prevalence",
            "sampling_strata": "36 fixed combinations of synthetic elevation tercile, gradient-magnitude tercile and land-cover quartile proxy; controls allocated proportionally to eligible candidates, largest remainder",
            "matching_limit": "Synthetic labels and four diagnostic covariates; not a replication of the 16 real factors, inventory cluster geometry, original fold memberships, or regional event process"},
        "pairing": "Arms, sizes and CV counts share a generating world. Balanced arms share observed cases at each size. Five capture scores share all response screens. These are paired, not independent repetitions.",
        "infeasibility": "Insufficient eligible background or cases is recorded without redrawing the generating world, reducing quotas, relaxing buffer, duplication, or replacement",
        "grouping": "Own synthetic point cell, using the footprint of the original 69 occupied cells; occupied synthetic groups are recorded, not forced to equal 69",
        "splitter": "StratifiedGroupKFold(shuffle=True,random_state=world_seed), separately for 5 and 3 folds; same group indivisibility, newly allocated folds",
        "stratification": "Original training-only CART and custom K-means with frozen parameters; four covariates (x,y,latent spatial field,heterogeneity template)",
        "graph": "Extracted corrected true-kNN function; self explicitly removed; symmetric union at k=4,8,16,32",
        "screen": "Four Moran and two q response tests in each validation fold; unrestricted label permutations; fixed coordinates/graphs/strata/class counts; original two-sided deviation about permutation mean",
        "gate": "All six valid nominal tests must pass in every contributing fold. Holm shown separately. Finite summaries also required; no fold/setting omitted or zero-imputed.",
        "controlled_scores": "Representative arm uses generating probabilities. Balanced arms use a global finite-population case-control prior adjustment, exact under ideal uniform class sampling but approximate with integer stratum quotas and not an exact probability model for spatially excluded or contaminated background samples. At capture c, scores are prior+c*(adjusted_latent_probability-prior). No classifiers are trained for these controlled scores.",
        "permutation_seed_rule": "world_seed*1000000 + N*100 + arm_index*1000 + fold_count*10 + fold; distinct scheduled fold seeds",
        "primary_reportability": "Complete finite R-DSI and S_theta at capture=1 AND complete-fold response screen; other capture conditions reported separately; independent denominator 100 worlds per scenario",
        "null_interpretation": "Null refers to latent population signal. Buffering and background sampling can induce observed-label structure, so buffered-null passes are not automatically called statistical false positives.",
        "comparison_outputs": ["numerical availability", "sampling feasibility", "six-test fold and all-fold passes", "fully reportable count", "paired 3-versus-5 changes", "control contamination", "minimum validation positives", "occupied strata", "score/component responses"],
        "sampling_failures_in_denominator": True, "replicates_discarded": False,
        "no_parameter_selection_by_pass_rate": True,
        "protected_inputs": {str(p.resolve()): sha(p) for p in protected_paths},
        "geometry_sha256": sha(OUT / "geometry.npz"),
        "environment": {"python": platform.python_version(), "executable": sys.executable,
                        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    write_json(OUT / "protocol.json", protocol)
    log(f"FROZEN: 600 worlds; 2 sample sizes; 4 sampling arms; 5/3 folds; 110400-site synthetic population")
    return protocol


def intercept(signal, target):
    lo, hi = -20., 20.
    for _ in range(80):
        mid = (lo + hi) / 2
        if expit(mid + signal).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def stratified_sample(candidates, size, strata, priority):
    if len(candidates) < size:
        return None
    codes, counts = np.unique(strata[candidates], return_counts=True)
    desired = counts * (size / counts.sum())
    quota = np.floor(desired).astype(int)
    remainder_order = np.argsort(-(desired - quota), kind="stable")
    quota[remainder_order[:size - int(quota.sum())]] += 1
    selected = []
    for code, count in zip(codes, quota):
        eligible = candidates[strata[candidates] == code]
        selected.append(eligible[np.argsort(priority[eligible], kind="stable")[:count]])
    result = np.concatenate(selected)
    assert len(result) == size and np.unique(result).size == size
    return result


def adjusted_latent_probability(p, latent_y):
    # Inclusion-odds correction under ideal uniform class sampling. Integer stratum
    # quotas make it approximate; both background arms keep this controlled score.
    ratio = np.sum(latent_y == 0) / np.sum(latent_y == 1)
    return ratio * p / (ratio * p + 1 - p)


def holm(rows):
    values = np.asarray([r["permutation_p"] for r in rows], float)
    adjusted = np.full(6, np.nan)
    if np.isfinite(values).all():
        order = np.argsort(values)
        adjusted[order] = np.minimum(1., np.maximum.accumulate(values[order] * (6 - np.arange(6))))
    for row, value in zip(rows, adjusted):
        row["holm_p"] = value
        row["holm_pass"] = bool(np.isfinite(value) and value < .05)
    return rows


def complete_mean(values):
    values = np.asarray(values, float)
    return float(values.mean()) if np.isfinite(values).all() else np.nan


def diagnostics(y, probability, graphs, levels):
    values = sc.theta_six(y, probability, graphs, levels)
    for residual in sc.RESIDUALS:
        vector = sc.residual(y, probability, residual)
        values[f"residual_rmse_{residual}"] = float(np.sqrt(np.mean(vector * vector)))
        values[f"residual_variance_{residual}"] = float(np.var(vector))
        # Explicit mathematical constant-vector guard: formula unchanged, no floating pseudo-q.
        if np.ptp(vector) == 0:
            for k in sc.K_VALUES:
                values[f"i_residual_k{k}_{residual}"] = np.nan
                values[f"eta_a_k{k}_{residual}"] = np.nan
            values[f"eta_a_{residual}"] = np.nan
            for name in levels:
                values[f"q_residual_{name}_{residual}"] = np.nan
                values[f"eta_h_{name}_{residual}"] = np.nan
                values[f"theta_{name}_{residual}"] = np.nan
    six = np.array([values[name] for name in sc.THETA])
    values["n_finite_theta"] = int(np.isfinite(six).sum())
    if not np.isfinite(six).all():
        values["r_dsi"] = values["s_theta"] = np.nan
    return values


def failure_reason(y, levels, values):
    if np.ptp(y) == 0:
        return "constant_validation_response"
    if any(len(np.unique(a)) < 2 for a in levels.values()):
        return "single_validation_stratum"
    n = len(y)
    if any(abs(values[f"i_y_k{k}"] + 1 / (n - 1)) <= sc.EPS for k in sc.K_VALUES):
        return "near_zero_response_baseline"
    for name, labels in levels.items():
        expected = (len(np.unique(labels)) - 1) / (n - 1)
        if abs((values[f"q_y_{name}"] - expected) / (1 - expected)) <= sc.EPS:
            return "near_zero_response_baseline"
    return "other_nonfinite" if values["n_finite_theta"] != 6 else ""


def world_task(scenario_index, repetition, smoke=False):
    scenario = list(SCENARIOS)[scenario_index]
    a, h, prevalence = SCENARIOS[scenario]
    seed = 2026101800 + scenario_index * 10000 + repetition
    key = f"{scenario}_r{repetition:03d}"
    base = OUT / ("smoke/worlds" if smoke else "worlds") / key
    base.mkdir(parents=True, exist_ok=True)
    if (base / "completion.json").exists():
        return {"key": key, "cached": True}
    with np.load(OUT / "geometry.npz", allow_pickle=False) as saved:
        geometry = {name: saved[name] for name in saved.files}
    xy, groups, mask, hy = [geometry[name] for name in ("coordinates", "groups", "mask", "heterogeneity")]
    rng = np.random.default_rng(seed)
    field = standardized(gaussian_filter(rng.normal(size=mask.shape), sigma=24, mode="reflect")[mask])
    signal = a * field + h * hy
    p = expit(intercept(signal, prevalence) + signal)
    latent_y = rng.binomial(1, p).astype(np.int8)
    np.savez_compressed(base / "population.npz", field=field, probability=p, latent_response=latent_y)
    features = np.c_[xy, field, hy]
    case_priority = rng.random(len(xy))
    control_priority = rng.random(len(xy))
    representative_priority = rng.permutation(len(xy))
    true_cases = np.flatnonzero(latent_y == 1)
    case_order = true_cases[np.argsort(case_priority[true_cases], kind="stable")]
    negatives = np.flatnonzero(latent_y == 0)
    adjusted = adjusted_latent_probability(p, latent_y)
    metadata, dataset_rows, condition_rows, folds, screens, components, graph_rows = [], [], [], [], [], [], []
    work_started = time.perf_counter()
    with threadpool_limits(limits=1):
        for n in SIZES:
            half = n // 2
            cases = case_order[:half]
            available_background = np.setdiff1d(np.arange(len(xy)), cases, assume_unique=True)
            distances = cKDTree(xy[cases]).query(xy[available_background], workers=1)[0]
            background_outside = available_background[distances > 1000.]
            for arm_index, arm in enumerate(ARMS):
                info = dict(world_id=key, scenario=scenario, repetition=repetition, world_seed=seed,
                            N=n, arm=arm, population_N=len(xy), population_positive=int(latent_y.sum()),
                            population_prevalence=float(latent_y.mean()), target_population_prevalence=prevalence,
                            spatial_strength=a, heterogeneity_strength=h, sampling_feasible=True,
                            sampling_failure="", eligible_controls=0, background_contamination=np.nan,
                            sampling_stratum_count=int(np.unique(geometry["sampling_stratum"]).size))
                if arm == "representative":
                    indices = representative_priority[:n]
                    observed_y = latent_y[indices].copy()
                    q = p[indices]
                    prior = prevalence
                else:
                    eligible = negatives if arm == "balanced_true_absence" else (
                        background_outside if arm.endswith("buffer1000") else available_background)
                    info["eligible_controls"] = len(eligible)
                    controls = stratified_sample(eligible, half, geometry["sampling_stratum"], control_priority)
                    if len(cases) < half or controls is None:
                        info.update(sampling_feasible=False, sampling_failure="insufficient_eligible_cases_or_controls")
                        metadata.append(info)
                        for count in FOLDS:
                            dataset_rows.append({**info, "fold_count": count, "response_all_folds_pass": False,
                                "holm_all_folds_pass": False, "numeric_at_capture1": False,
                                "reportable_at_capture1": False, "numeric_all_captures": False,
                                "reportable_all_captures": False, "valid_all_fold_screens": False,
                                "passing_folds": 0, "valid_folds": 0, "min_validation_positive": np.nan})
                        continue
                    indices = np.r_[cases, controls]
                    observed_y = np.r_[np.ones(half, np.int8), np.zeros(half, np.int8)]
                    q, prior = adjusted[indices], .5
                    info["background_contamination"] = float(latent_y[controls].mean())
                    info["control_min_distance_m"] = float(cKDTree(xy[cases]).query(xy[controls], workers=1)[0].min())
                    if arm.endswith("buffer1000"):
                        assert info["control_min_distance_m"] > 1000
                assert len(indices) == n and np.unique(indices).size == n
                info.update(observed_positive=int(observed_y.sum()), observed_prevalence=float(observed_y.mean()),
                            occupied_groups=int(np.unique(groups[indices]).size),
                            sample_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
                metadata.append(info)
                sample_key = f"N{n}_{arm}"
                np.savez_compressed(base / f"sample_{sample_key}.npz", candidate_indices=indices,
                    response=observed_y, latent_response=latent_y[indices], controlled_probability=q)
                for count in FOLDS:
                    common = dict(world_id=key, scenario=scenario, repetition=repetition, world_seed=seed,
                                  N=n, arm=arm, fold_count=count)
                    splits = list(StratifiedGroupKFold(count, shuffle=True, random_state=seed).split(features[indices], observed_y, groups[indices]))
                    assignment = np.zeros(n, np.int16)
                    tree_val = np.zeros(n, np.int32)
                    kmeans_val = np.zeros(n, np.int16)
                    prepared, fold_values = [], {capture: [] for capture in CAPTURES}
                    permutation_seed_base = seed * 1000000 + n * 100 + arm_index * 1000 + count * 10
                    for fold, (ti, vi) in enumerate(splits, 1):
                        assert not np.intersect1d(groups[indices[ti]], groups[indices[vi]]).size
                        assignment[vi] = fold
                        tree, km, rules = sc.fit_strata_matrix(features[indices[ti]], features[indices[vi]], observed_y[ti], seed + fold)
                        tree_val[vi], kmeans_val[vi] = tree, km
                        levels = dict(tree=tree, kmeans=km)
                        graphs, graph_audit = correct_graphs(xy[indices[vi]])
                        with np.errstate(divide="ignore", invalid="ignore"):
                            screen = holm(sc.identifiability(observed_y[vi], graphs, levels, 499, permutation_seed_base + fold))
                        valid = all(np.isfinite(row["permutation_p"]) for row in screen)
                        gate = bool(valid and all(row["permutation_p"] < .05 for row in screen))
                        holm_gate = bool(valid and all(row["holm_pass"] for row in screen))
                        for row in screen:
                            screens.append({**common, "fold": fold, **row})
                        for k, audit in graph_audit.items():
                            graph_rows.append({**common, "fold": fold, "k": k, **audit})
                        fold_row = {**common, "fold": fold, "training_n": len(ti), "validation_n": len(vi),
                            "validation_positive": int(observed_y[vi].sum()), "training_positive": int(observed_y[ti].sum()),
                            "tree_H": int(np.unique(tree).size), "kmeans_H": int(np.unique(km).size),
                            "tree_training_leaves": int(rules["tree"].get_n_leaves()),
                            "train_index_sha256": hashlib.sha256(indices[ti].tobytes()).hexdigest(),
                            "validation_index_sha256": hashlib.sha256(indices[vi].tobytes()).hexdigest(),
                            "screen_valid": valid, "six_tests_pass": gate, "holm_six_tests_pass": holm_gate,
                            "failed_response_tests": ";".join(str(row["setting"]) for row in screen if not np.isfinite(row["permutation_p"]) or row["permutation_p"] >= .05)}
                        for capture in CAPTURES:
                            prediction = np.clip(prior + capture * (q[vi] - prior), sc.P_CLIP, 1 - sc.P_CLIP)
                            values = diagnostics(observed_y[vi], prediction, graphs, levels)
                            fold_values[capture].append(values)
                            components.append({**common, "fold": fold, "capture": capture, **values})
                            if capture == 1.:
                                fold_row["numeric_finite"] = values["n_finite_theta"] == 6
                                fold_row["numerical_failure"] = failure_reason(observed_y[vi], levels, values)
                        folds.append(fold_row)
                        prepared.append(fold_row)
                    assert assignment.min() == 1 and assignment.max() == count
                    np.savez_compressed(base / f"folds_{sample_key}_cv{count}.npz", assignment=assignment, tree_labels=tree_val, kmeans_labels=kmeans_val)
                    response_pass = all(row["six_tests_pass"] for row in prepared)
                    holm_pass = all(row["holm_six_tests_pass"] for row in prepared)
                    numeric_all = True
                    at1 = None
                    for capture in CAPTURES:
                        parts = fold_values[capture]
                        numeric = all(row["n_finite_theta"] == 6 for row in parts)
                        numeric_all = numeric_all and numeric
                        prediction = np.clip(prior + capture * (q - prior), sc.P_CLIP, 1 - sc.P_CLIP)
                        predictive = dict(roc_auc=np.nan, brier=float(np.mean((observed_y - prediction) ** 2)))
                        if np.unique(observed_y).size == 2:
                            predictive.update(sc.predictive(observed_y, prediction))
                        row = {**common, "capture": capture, "numeric_finite": numeric,
                            "response_all_folds_pass": response_pass, "fully_reportable": bool(numeric and response_pass),
                            "holm_reportable": bool(numeric and holm_pass), **predictive}
                        for name in ["r_dsi", "s_theta", *sc.THETA,
                                     *[f"eta_a_{r}" for r in sc.RESIDUALS],
                                     *[f"eta_h_{s}_{r}" for r in sc.RESIDUALS for s in sc.STRATA]]:
                            row[name] = complete_mean([part[name] for part in parts])
                        row["gated_r_dsi"] = row["r_dsi"] if row["fully_reportable"] else np.nan
                        row["gated_s_theta"] = row["s_theta"] if row["fully_reportable"] else np.nan
                        condition_rows.append(row)
                        if capture == 1.:
                            at1 = row
                    dataset_rows.append({**info, **common, "response_all_folds_pass": response_pass,
                        "holm_all_folds_pass": holm_pass, "numeric_at_capture1": bool(at1["numeric_finite"]),
                        "reportable_at_capture1": bool(at1["fully_reportable"]), "numeric_all_captures": numeric_all,
                        "reportable_all_captures": bool(numeric_all and response_pass),
                        "valid_all_fold_screens": all(row["screen_valid"] for row in prepared),
                        "passing_folds": sum(row["six_tests_pass"] for row in prepared),
                        "valid_folds": sum(row["screen_valid"] for row in prepared),
                        "min_validation_positive": min(row["validation_positive"] for row in prepared),
                        "min_validation_n": min(row["validation_n"] for row in prepared),
                        "min_tree_H": min(row["tree_H"] for row in prepared),
                        "min_kmeans_H": min(row["kmeans_H"] for row in prepared),
                        "r_dsi_at_capture1": at1["r_dsi"], "s_theta_at_capture1": at1["s_theta"]})
    for name, rows in [("samples", metadata), ("datasets", dataset_rows), ("conditions", condition_rows),
                       ("folds", folds), ("screens", screens), ("components", components), ("graphs", graph_rows)]:
        frame_csv(base / f"{name}.csv", rows)
    write_json(base / "completion.json", {"success": True, "key": key, "world_seed": seed,
        "population_sha256": sha(base / "population.npz"), "sample_rows": len(metadata),
        "dataset_rows": len(dataset_rows), "condition_rows": len(condition_rows),
        "fold_rows": len(folds), "screen_rows": len(screens), "wall_seconds": time.perf_counter() - work_started})
    return {"key": key, "seconds": round(time.perf_counter() - work_started, 2), "sample_failures": sum(not row["sampling_feasible"] for row in metadata)}


def run(smoke=False, workers=8):
    protocol = freeze()
    destination = OUT / ("smoke" if smoke else "")
    destination.mkdir(exist_ok=True)
    tasks = [(i, rep, smoke) for i in range(6) for rep in ([0] if smoke else range(1, 101))]
    lock = destination / "RUNNING.lock"
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(world_task, *task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if smoke or done % 10 == 0:
                    log(f"WORLD {done}/{len(tasks)} {result}")
                    write_json(destination / "progress.json", {"worlds_complete": done, "worlds_required": len(tasks), "last": result, "utc": now()})
        for table in ("samples", "datasets", "conditions", "folds", "screens", "components", "graphs"):
            paths = sorted((destination / "worlds").glob(f"*/{table}.csv"))
            assert len(paths) == len(tasks), (table, len(paths))
            frame = pd.concat([pd.read_csv(path, keep_default_na=False, na_values=[""], float_precision="round_trip") for path in paths], ignore_index=True)
            frame.to_csv(destination / f"{table}.csv", index=False, encoding="utf-8-sig")
            log(f"MERGED {table}: {len(frame)} rows")
        for path, expected in protocol["protected_inputs"].items():
            assert sha(path) == expected, path
        write_json(destination / "completion.json", {"success": True, "worlds": len(tasks), "smoke": smoke,
            "runner_sha256": sha(__file__), "protocol_sha256": sha(OUT / "protocol.json"),
            "protected_inputs_unchanged": True, "wall_seconds": time.perf_counter() - START, "utc": now()})
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.smoke or args.run:
        run(args.smoke, args.workers)
    else:
        parser.error('Choose --freeze, --smoke or --run')

if __name__ == '__main__':
    main()
