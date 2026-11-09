"""OOS model comparison vs simple baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align(y_hat: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    df = pd.concat({"yhat": y_hat, "y": y}, axis=1).dropna()
    return df["yhat"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)


def regression_metrics(
    y_hat: pd.Series,
    y: pd.Series,
    *,
    train_mean: float | None = None,
) -> dict[str, float]:
    """OOS R² vs zero and vs train-mean; RMSE/MAE; predictive correlation."""
    yh, yt = _align(y_hat, y)
    if len(yt) < 5:
        return {
            "n_obs": float(len(yt)),
            "rmse": np.nan,
            "mae": np.nan,
            "corr": np.nan,
            "r2_vs_zero": np.nan,
            "r2_vs_train_mean": np.nan,
            "rmse_rel_zero": np.nan,
            "rmse_rel_train_mean": np.nan,
        }
    err = yh - yt
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_zero = float(np.sum(yt**2))
    mu = 0.0 if train_mean is None else float(train_mean)
    ss_mean = float(np.sum((yt - mu) ** 2))
    rmse_zero = float(np.sqrt(np.mean(yt**2)))
    rmse_mean = float(np.sqrt(np.mean((yt - mu) ** 2)))
    corr = float(np.corrcoef(yh, yt)[0, 1]) if yh.std() > 0 and yt.std() > 0 else np.nan
    return {
        "n_obs": float(len(yt)),
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "r2_vs_zero": 1.0 - ss_res / ss_zero if ss_zero > 0 else np.nan,
        "r2_vs_train_mean": 1.0 - ss_res / ss_mean if ss_mean > 0 else np.nan,
        "rmse_rel_zero": rmse / rmse_zero if rmse_zero > 0 else np.nan,
        "rmse_rel_train_mean": rmse / rmse_mean if rmse_mean > 0 else np.nan,
    }


def compare_predictions(
    predictions: dict[str, pd.Series],
    y: pd.Series,
    *,
    train_mean: float | None = None,
) -> pd.DataFrame:
    rows = []
    for name, pred in predictions.items():
        rows.append({"model": name, **regression_metrics(pred, y, train_mean=train_mean)})
    return pd.DataFrame(rows).set_index("model")
