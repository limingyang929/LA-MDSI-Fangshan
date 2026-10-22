"""Independent checks of sampling, grouping, formulas, graph membership and screens."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import run_experiment as run
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

OUT = run.OUT


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], float_precision="round_trip")


def q_direct(y, labels):
    y = np.asarray(y, float)
    den = np.sum((y - y.mean()) ** 2)
    if den <= 0 or len(np.unique(labels)) < 2:
        return np.nan
    within = sum(np.sum((y[labels == label] - y[labels == label].mean()) ** 2) for label in np.unique(labels))
    return float(1 - within / den)


def moran_direct(y, graph):
    z = np.asarray(y, float) - np.mean(y)
    den = np.dot(z, z)
    if den <= 0:
        return np.nan
    rows, cols = graph.nonzero()
    return float(len(z) * np.sum(z[rows] * z[cols]) / (len(rows) * den))


def residual(y, p, kind):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if kind == "raw":
        return y - p
    if kind == "pearson":
        return (y - p) / np.sqrt(p * (1 - p))
    return np.where(y == 1, 1., -1.) * np.sqrt(-2 * np.where(y == 1, np.log(p), np.log1p(-p)))


def close(actual, expected, label, atol=3e-9):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), (label, a, b)
    mask = np.isfinite(a)
    assert np.allclose(a[mask], b[mask], atol=atol, rtol=atol), (label, a, b)


def type7(a, fraction):
    a = np.sort(np.asarray(a, float))
    if not np.isfinite(a).all():
        return np.nan
    index = (len(a) - 1) * fraction
    left, right = int(np.floor(index)), int(np.ceil(index))
    return a[left] + (index - left) * (a[right] - a[left])


def eta(response, remainder, expectation, normalizer=1.):
    ay, ar = (response - expectation) / normalizer, (remainder - expectation) / normalizer
    return 1 - abs(ar) / abs(ay) if np.isfinite([ay, ar]).all() and abs(ay) > 1e-12 else np.nan


def verify(smoke=False):
    started = time.perf_counter()
    base = OUT / "smoke" if smoke else OUT
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    completion = json.loads((base / "completion.json").read_text(encoding="utf-8"))
    assert completion["success"] and completion["smoke"] == smoke
    assert run.sha(run.__file__) == protocol["runner_sha256"]
    with np.load(OUT / "geometry.npz", allow_pickle=False) as archive:
        xy, groups, strata, hy = [archive[name] for name in ("coordinates", "groups", "sampling_stratum", "heterogeneity")]
    frames = {name: read(base / f"{name}.csv") for name in ("samples", "datasets", "conditions", "folds", "screens", "components", "graphs")}
    worlds = frames["samples"].world_id.unique()
    assert len(worlds) == (6 if smoke else 600)
    assert len(frames["samples"]) == len(worlds) * 8
    assert len(frames["datasets"]) == len(worlds) * 16
    common = ["world_id", "N", "arm", "fold_count"]
    data = frames["datasets"]
    assert not data.duplicated(common).any()
    data_indexed = data.set_index(common)
    fold_indexed = frames["folds"].set_index(common + ["fold"])
    folds_by_config = {key: group for key, group in frames["folds"].groupby(common, sort=False)}
    tests = frames["screens"]
    assert tests.permutations.eq(499).all()
    assert not tests.duplicated(common + ["fold", "component", "setting"]).any()
    assert tests.groupby(common + ["fold"]).size().eq(6).all()
    # Scheduled fold seeds are unique across independent analysis configurations.
    unique_fold_seeds = tests.groupby(common + ["fold"]).permutation_seed.first()
    assert unique_fold_seeds.is_unique
    for key, subset in tests.groupby(common + ["fold"], sort=False):
        p = subset.permutation_p.to_numpy(float)
        row = fold_indexed.loc[key]
        expected_valid = bool(np.isfinite(p).all())
        assert bool(row.screen_valid) == expected_valid
        assert bool(row.six_tests_pass) == bool(expected_valid and (p < .05).all())
        adjusted = np.full(6, np.nan)
        if expected_valid:
            order = np.argsort(p)
            adjusted[order] = np.minimum(1, np.maximum.accumulate((6 - np.arange(6)) * p[order]))
        close(subset.holm_p, adjusted, (key, "Holm"), atol=1e-14)
    component = frames["components"]
    joined = component.merge(frames["folds"][[*common, "fold", "validation_n", "tree_H", "kmeans_H"]],
                             on=common + ["fold"], validate="many_to_one")
    n = joined.validation_n.to_numpy(float)
    rebuilt = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for r in run.sc.RESIDUALS:
            av = []
            for k in run.sc.K_VALUES:
                ay = joined[f"i_y_k{k}"].to_numpy(float) + 1 / (n - 1)
                ar = joined[f"i_residual_k{k}_{r}"].to_numpy(float) + 1 / (n - 1)
                av.append(np.where(np.isfinite(ay) & np.isfinite(ar) & (abs(ay) > 1e-12), 1 - abs(ar) / abs(ay), np.nan))
            a = np.mean(av, axis=0)
            close(joined[f"eta_a_{r}"], a, (r, "all etaA"))
            for s in run.sc.STRATA:
                h = joined["tree_H" if s == "tree" else "kmeans_H"].to_numpy(float)
                eq = (h - 1) / (n - 1)
                ay = (joined[f"q_y_{s}"].to_numpy(float) - eq) / (1 - eq)
                ar = (joined[f"q_residual_{s}_{r}"].to_numpy(float) - eq) / (1 - eq)
                b = np.where(np.isfinite(ay) & np.isfinite(ar) & (abs(ay) > 1e-12), 1 - abs(ar) / abs(ay), np.nan)
                theta = 1 - (1 - a) * (1 - b)
                close(joined[f"eta_h_{s}_{r}"], b, (r, s, "all etaH"))
                close(joined[f"theta_{s}_{r}"], theta, (r, s, "all theta"))
                rebuilt.append(theta)
    rebuilt = np.asarray(rebuilt).T
    ordered = np.sort(rebuilt, axis=1)
    valid = np.isfinite(rebuilt).all(axis=1)
    q1 = np.where(valid, .75 * ordered[:, 1] + .25 * ordered[:, 2], np.nan)
    q3 = np.where(valid, .25 * ordered[:, 3] + .75 * ordered[:, 4], np.nan)
    close(joined.r_dsi, q1, "all independent Type-7 Q1")
    close(joined.s_theta, 1 - (q3 - q1), "all independent IQR")
    assert np.array_equal(joined.n_finite_theta, np.isfinite(rebuilt).sum(axis=1))
    formula_checks = len(joined)
    component_by_config = {key: group for key, group in component.groupby(common, sort=False)}
    primary_indexed = component.loc[component.capture.eq(1)].set_index(common + ["fold"])
    screens_indexed = tests.set_index(common + ["fold"])
    graphs_indexed = frames["graphs"].set_index(common + ["fold", "k"])
    # Reconstruct complete means and NA propagation from all saved fold components.
    aggregate_checks = 0
    for key, subset in frames["conditions"].groupby(common, sort=False):
        result = data_indexed.loc[key]
        fold_set = folds_by_config[key]
        assert len(fold_set) == int(key[-1])
        assert bool(result.response_all_folds_pass) == bool(fold_set.six_tests_pass.all())
        finite_all = True
        for _, row in subset.iterrows():
            source_parts = component_by_config[key]
            parts = source_parts.loc[source_parts.capture.eq(row.capture)]
            finite = bool(parts.n_finite_theta.eq(6).all())
            assert bool(row.numeric_finite) == finite
            assert bool(row.fully_reportable) == bool(finite and result.response_all_folds_pass)
            for name in ["r_dsi", "s_theta", *run.sc.THETA]:
                values = parts[name].to_numpy(float)
                expected = values.mean() if np.isfinite(values).all() else np.nan
                close([row[name]], [expected], (key, row.capture, name))
            close([row.gated_r_dsi], [row.r_dsi if row.fully_reportable else np.nan], (key, row.capture, "gate"))
            finite_all = finite_all and finite
            if row.capture == 1:
                assert bool(result.numeric_at_capture1) == finite
                assert bool(result.reportable_at_capture1) == bool(row.fully_reportable)
            aggregate_checks += 1
        assert bool(result.numeric_all_captures) == finite_all
    direct_checks = 0
    graph_checks = 0
    raw_screen_checks = 0
    training_only_refits = 0
    max_null_difference = 0.
    # A prespecified direct subset, plus all-world sampling/group membership checks.
    direct_repetitions = {0} if smoke else {1, 11, 51, 100}
    for world in worlds:
        world_samples = frames["samples"].query("world_id == @world")
        world_data = data.query("world_id == @world")
        world_dir = base / "worlds" / world
        with np.load(world_dir / "population.npz", allow_pickle=False) as archive:
            latent = archive["latent_response"]
            field = archive["field"]
        features = np.c_[xy, field, hy]
        world_rep = int(world_samples.repetition.iloc[0])
        for n in run.SIZES:
            balanced_cases = None
            for arm in run.ARMS:
                meta = world_samples.query("N == @n and arm == @arm").iloc[0]
                if not meta.sampling_feasible:
                    assert meta.eligible_controls < n // 2 or int(meta.population_positive) < n // 2
                    assert not world_data.query("N == @n and arm == @arm").reportable_at_capture1.any()
                    continue
                sample_key = f"N{n}_{arm}"
                with np.load(world_dir / f"sample_{sample_key}.npz", allow_pickle=False) as archive:
                    indices, y, probability = [archive[name] for name in ("candidate_indices", "response", "controlled_probability")]
                assert len(indices) == n and len(np.unique(indices)) == n
                assert hashlib.sha256(indices.tobytes()).hexdigest() == meta.sample_index_sha256
                if arm == "representative":
                    assert np.array_equal(y, latent[indices])
                else:
                    half = n // 2
                    assert np.all(y[:half] == 1) and np.all(y[half:] == 0)
                    assert np.all(latent[indices[:half]] == 1)
                    if balanced_cases is None:
                        balanced_cases = indices[:half].copy()
                    else:
                        assert np.array_equal(balanced_cases, indices[:half])
                    close([meta.background_contamination], [latent[indices[half:]].mean()], (world, n, arm, "contamination"))
                    if arm == "balanced_true_absence":
                        assert np.all(latent[indices[half:]] == 0)
                    if arm.endswith("buffer1000"):
                        assert cKDTree(xy[indices[:half]]).query(xy[indices[half:]])[0].min() > 1000
                for count in run.FOLDS:
                    with np.load(world_dir / f"folds_{sample_key}_cv{count}.npz", allow_pickle=False) as archive:
                        assignment, tl, kl = [archive[name] for name in ("assignment", "tree_labels", "kmeans_labels")]
                    assert set(assignment) == set(range(1, count + 1))
                    assert pd.DataFrame(dict(group=groups[indices], fold=assignment)).groupby("group").fold.nunique().max() == 1
                    if world_rep not in direct_repetitions:
                        continue
                    # Fixed first validation fold, all arms, all sizes and both CV designs.
                    vi = np.flatnonzero(assignment == 1)
                    ti = np.flatnonzero(assignment != 1)
                    ref_tree, ref_km, _ = run.sc.fit_strata_matrix(features[indices[ti]], features[indices[vi]], y[ti], int(meta.world_seed) + 1)
                    assert np.array_equal(ref_tree, tl[vi]) and np.array_equal(ref_km, kl[vi])
                    stored_fold = fold_indexed.loc[(world, n, arm, count, 1)]
                    assert hashlib.sha256(indices[ti].tobytes()).hexdigest() == stored_fold.train_index_sha256
                    assert hashlib.sha256(indices[vi].tobytes()).hexdigest() == stored_fold.validation_index_sha256
                    training_only_refits += 1
                    labels = dict(tree=tl[vi], kmeans=kl[vi])
                    graphs, audits = run.correct_graphs(xy[indices[vi]])
                    nearest = cKDTree(xy[indices[vi]])
                    distance, _ = nearest.query(xy[indices[vi]], k=33)
                    for k, graph in graphs.items():
                        saved_graph = graphs_indexed.loc[(world, n, arm, count, 1, k)]
                        assert saved_graph.graph_sha256 == audits[k]["graph_sha256"]
                        assert int(saved_graph.edge_count) == graph.nnz
                        assert graph.diagonal().sum() == 0 and (graph != graph.T).nnz == 0
                        # Every symmetrized edge must be among a k-nearest directed
                        # neighborhood on at least one side, allowing distance ties.
                        rr, cc = graph.nonzero()
                        edge_distance = np.linalg.norm(xy[indices[vi[rr]]] - xy[indices[vi[cc]]], axis=1)
                        assert np.all((edge_distance <= distance[rr, k] + 1e-8) | (edge_distance <= distance[cc, k] + 1e-8))
                        graph_checks += 1
                    primary = primary_indexed.loc[(world, n, arm, count, 1)]
                    yv = y[vi]
                    if np.ptp(yv) > 0:
                        for k, graph in graphs.items():
                            close([primary[f"i_y_k{k}"]], [moran_direct(yv, graph)], (world, n, arm, count, k))
                        for s, values in labels.items():
                            close([primary[f"q_y_{s}"]], [q_direct(yv, values)], (world, n, arm, count, s))
                        for r in run.sc.RESIDUALS:
                            vector = residual(yv, probability[vi], r)
                            for k, graph in graphs.items():
                                close([primary[f"i_residual_k{k}_{r}"]], [moran_direct(vector, graph)], (world, n, arm, count, r, k))
                            for s, values in labels.items():
                                close([primary[f"q_residual_{s}_{r}"]], [q_direct(vector, values)], (world, n, arm, count, r, s))
                    direct_checks += 1
                    if n != 900 or count != 5:
                        continue
                    stored = screens_indexed.loc[[(world, n, arm, count, 1)]]
                    seed = int(stored.permutation_seed.iloc[0])
                    rng = np.random.default_rng(seed)
                    permutations = np.stack([rng.permutation(yv) for _ in range(499)])
                    for _, test in stored.iterrows():
                        if test.component == "autocorrelation":
                            graph = graphs[int(test.setting[1:])]
                            null = np.array([moran_direct(row, graph) for row in permutations])
                            observed = moran_direct(yv, graph)
                        else:
                            group = labels[test.setting]
                            null = np.array([q_direct(row, group) for row in permutations])
                            observed = q_direct(yv, group)
                        valid = bool(np.isfinite(null).all() and np.isfinite(observed))
                        assert valid == bool(np.isfinite(test.permutation_p))
                        if valid:
                            close([test.observed, test.null_mean, test.null_sd], [observed, null.mean(), null.std(ddof=1)], (world, n, arm, "screen raw"), atol=1e-11)
                            deviation = abs(null - null.mean())
                            observed_deviation = abs(observed - null.mean())
                            # Different algebraic orders can move exact binary-data
                            # ties by a few ulps. Bound that count explicitly.
                            lower = (1 + np.sum(deviation > observed_deviation + 1e-11)) / 500
                            upper = (1 + np.sum(deviation >= observed_deviation - 1e-11)) / 500
                            assert lower - 1e-12 <= test.permutation_p <= upper + 1e-12
                            max_null_difference = max(max_null_difference, abs(float(test.null_mean) - null.mean()))
                        raw_screen_checks += 1
    for path, expected in protocol["protected_inputs"].items():
        assert run.sha(path) == expected, path
    output = dict(success=True, smoke=smoke, worlds=len(worlds), formula_rows_checked=formula_checks,
        condition_means_checked=aggregate_checks, direct_fold_checks=direct_checks,
        graph_membership_checks=graph_checks, independent_raw_screen_checks=raw_screen_checks,
        training_only_stratification_refits=training_only_refits,
        maximum_null_mean_difference=max_null_difference, protected_inputs_unchanged=True,
        runner_matches_frozen_protocol=True, permutation_ties="Independent p-count bounded for floating ties; saved exact >= rule unchanged",
        wall_seconds=time.perf_counter() - started, utc=run.now())
    run.write_json(base / "verification.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    verify(args.smoke)

if __name__ == '__main__':
    main()
