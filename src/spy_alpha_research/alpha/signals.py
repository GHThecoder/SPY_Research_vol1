"""Alpha signal construction from selected features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spy_alpha_research.statistics.ic import series_ic


def expanding_zscore(series: pd.Series, min_periods: int = 63) -> pd.Series:
    mu = series.expanding(min_periods=min_periods).mean()
    sd = series.expanding(min_periods=min_periods).std().replace(0.0, np.nan)
    return ((series - mu) / sd).rename(series.name)


def oriented_signal(
    feature: pd.Series,
    forward_return: pd.Series,
    *,
    train_index: pd.Index,
    min_periods: int = 63,
) -> pd.Series:
    """Causal z-score with direction fixed from train-period IC sign."""
    z = expanding_zscore(feature, min_periods=min_periods)
    train_ic = series_ic(z.reindex(train_index), forward_return.reindex(train_index))
    sign = 1.0 if (not np.isfinite(train_ic) or train_ic >= 0) else -1.0
    out = (z * sign).rename(feature.name)
    return out


def build_alpha_signals(
    features: pd.DataFrame,
    alpha_names: list[str],
    forward_return: pd.Series,
    train_index: pd.Index,
    *,
    min_periods: int = 63,
) -> dict[str, pd.Series]:
    signals = {}
    for name in alpha_names:
        if name not in features.columns:
            continue
        signals[name] = oriented_signal(
            features[name],
            forward_return,
            train_index=train_index,
            min_periods=min_periods,
        )
    return signals
