"""Time-series block bootstrap helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample overlapping moving blocks with replacement until length n."""
    if n <= 0:
        return np.array([], dtype=int)
    block_size = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
    return idx


def block_bootstrap_ic(
    signal: pd.Series,
    forward_return: pd.Series,
    *,
    n_bootstrap: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> dict[str, float]:
    """Moving-block bootstrap CI and one-sided p for Pearson IC > 0."""
    df = pd.concat({"s": signal, "r": forward_return}, axis=1).dropna()
    if len(df) < max(block_size + 5, 20):
        return {
            "ic": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value_one_sided": np.nan,
            "n_obs": int(len(df)),
        }
    s = df["s"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)
    n = len(s)
    point = float(np.corrcoef(s, r)[0, 1])
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = moving_block_indices(n, block_size, rng)
        ss, rr = s[idx], r[idx]
        if ss.std() == 0 or rr.std() == 0:
            samples[i] = np.nan
        else:
            samples[i] = float(np.corrcoef(ss, rr)[0, 1])
    samples = samples[np.isfinite(samples)]
    # one-sided: P*(IC* <= 0) under bootstrap centered at point (percentile)
    p_one = float(np.mean(samples <= 0.0)) if len(samples) else np.nan
    return {
        "ic": point,
        "ci_low": float(np.percentile(samples, 2.5)) if len(samples) else np.nan,
        "ci_high": float(np.percentile(samples, 97.5)) if len(samples) else np.nan,
        "p_value_one_sided": p_one,
        "n_obs": n,
    }
