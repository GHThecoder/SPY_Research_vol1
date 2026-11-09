import numpy as np
import pandas as pd

from spy_alpha_research.preprocessing import ExpandingScaler, RollingScaler


def test_expanding_scaler_ignores_future_in_fit_transform():
    idx = pd.bdate_range("2020-01-01", periods=100)
    x = pd.DataFrame({"a": np.arange(100, dtype=float)}, index=idx)
    scaler = ExpandingScaler(min_periods=10)
    z = scaler.fit_transform(x)
    t = 50
    window = x["a"].iloc[: t + 1]
    expected = (window.iloc[-1] - window.mean()) / window.std(ddof=1)
    assert abs(z["a"].iloc[t] - expected) < 1e-10


def test_rolling_scaler_oos_uses_train_history():
    idx = pd.bdate_range("2020-01-01", periods=80)
    x = pd.DataFrame({"a": np.arange(80, dtype=float)}, index=idx)
    train, test = x.iloc[:50], x.iloc[50:]
    scaler = RollingScaler(window=10, min_periods=5)
    scaler.fit(train)
    z_oos = scaler.transform(test)

    # Manual: prepend last 10 train rows then rolling
    combined = pd.concat([train.iloc[-10:], test])
    mu = combined["a"].rolling(10, min_periods=5).mean()
    sd = combined["a"].rolling(10, min_periods=5).std()
    expected = ((combined["a"] - mu) / sd).reindex(test.index)
    assert np.allclose(z_oos["a"], expected, equal_nan=True)

    # Without history, early OOS rolling would differ — ensure first OOS point is finite
    assert np.isfinite(z_oos["a"].iloc[0])
