import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def spy_like(rng):
    idx = pd.bdate_range("2016-01-01", periods=1500)
    rets = pd.Series(rng.normal(0.0003, 0.01, size=len(idx)), index=idx, name="ret")
    price = 100 * np.exp(rets.cumsum())
    ohlcv = pd.DataFrame(
        {
            "Open": price,
            "High": price * 1.01,
            "Low": price * 0.99,
            "Close": price,
            "Adj Close": price,
            "Volume": rng.integers(1e6, 5e6, size=len(idx)),
        },
        index=idx,
    )
    return ohlcv, rets
