import numpy as np
import pandas as pd

from spy_alpha_research.attribution import select_alpha_candidates, save_alpha_candidates, load_alpha_candidates
from spy_alpha_research.alpha import equal_weight_blend, expanding_zscore
from spy_alpha_research.statistics import series_ic, ic_with_nw
from spy_alpha_research.backtesting import run_timing_backtest


def test_consensus_top_k(tmp_path):
    imp = {
        "rf": pd.Series({"a": 0.5, "b": 0.4, "c": 0.1, "d": 0.05}),
        "pca": pd.Series({"a": 0.3, "b": 0.2, "c": 0.15, "d": 0.01}),
    }
    names = select_alpha_candidates(imp, top_k=2, method="borda")
    assert len(names) == 2
    path = tmp_path / "alphas.yaml"
    save_alpha_candidates(names, path)
    assert load_alpha_candidates(path) == names


def test_ic_positive_on_synthetic():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    signal = pd.Series(rng.normal(size=300), index=idx)
    noise = pd.Series(rng.normal(scale=0.5, size=300), index=idx)
    fwd = 0.5 * signal + noise
    assert series_ic(signal, fwd) > 0.2
    stats = ic_with_nw(signal, fwd, window=40, nw_lags=3)
    assert stats["ic"] > 0


def test_equal_weight_and_backtest_lag():
    idx = pd.bdate_range("2020-01-01", periods=10)
    s1 = pd.Series(1.0, index=idx, name="a")
    s2 = pd.Series(-1.0, index=idx, name="b")
    blend = equal_weight_blend({"a": s1, "b": s2})
    assert np.allclose(blend, 0.0)

    signal = pd.Series([0, 1, 1, -1, -1, 0, 0, 1, 1, 0], index=idx, dtype=float)
    rets = pd.Series([0.01] * 10, index=idx)
    bt = run_timing_backtest(signal, rets, signal_lag=1, position="sign", costs_bps=0.0)
    # day0 position from signal0=0; day1 uses pos0=0; day2 uses pos1=sign(1)=1
    assert bt.gross_returns.iloc[0] == 0.0
    assert abs(bt.gross_returns.iloc[2] - 0.01) < 1e-12
