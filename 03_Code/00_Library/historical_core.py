import numpy as np
import pandas as pd
from scipy import sparse
def logistic(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))

def calibrate_intercept(signal: np.ndarray, target: float) -> float:
    low, high = -20.0, 20.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if logistic(middle + signal).mean() < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)

def moran_i(values: np.ndarray, adjacency: sparse.csr_matrix) -> float:
    centered = np.asarray(values, dtype=float) - np.mean(values)
    denominator = float(centered @ centered)
    if denominator <= 0:
        return np.nan
    return float((len(values) / adjacency.sum()) * (centered @ adjacency @ centered) / denominator)

def q_stat(values: np.ndarray, strata: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    variance = np.var(values)
    if variance <= 0:
        return np.nan
    within = sum(np.sum(strata == level) * np.var(values[strata == level]) for level in np.unique(strata))
    return float(1 - within / (len(values) * variance))

def dsi(y: np.ndarray, probability: np.ndarray, adjacency: sparse.csr_matrix, strata: np.ndarray) -> dict:
    residual = y - probability
    n = len(y)
    i_y, i_r = moran_i(y, adjacency), moran_i(residual, adjacency)
    q_y, q_r = q_stat(y, strata), q_stat(residual, strata)
    expected_i = -1 / (n - 1)
    expected_q = (pd.Series(strata).nunique() - 1) / (n - 1)
    centered_i_y = i_y - expected_i
    centered_i_r = i_r - expected_i
    centered_q_y = (q_y - expected_q) / (1 - expected_q)
    centered_q_r = (q_r - expected_q) / (1 - expected_q)
    eta_a = 1 - abs(centered_i_r) / abs(centered_i_y) if abs(centered_i_y) > 1e-12 else np.nan
    eta_h = 1 - abs(centered_q_r) / abs(centered_q_y) if abs(centered_q_y) > 1e-12 else np.nan
    eta_a_original = 1 - abs(i_r) / abs(i_y) if abs(i_y) > 1e-12 else np.nan
    eta_h_original = 1 - abs(q_r) / abs(q_y) if abs(q_y) > 1e-12 else np.nan
    return {
        "i_y": i_y,
        "q_y": q_y,
        "eta_a": eta_a,
        "eta_h": eta_h,
        "theta_independent": 1 - (1 - eta_a) * (1 - eta_h),
        "eta_a_original": eta_a_original,
        "eta_h_original": eta_h_original,
        "theta_original": 1 - (1 - eta_a_original) * (1 - eta_h_original),
    }
