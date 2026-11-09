"""Regression tests for the five critical research-protocol bugs."""

import numpy as np
import pandas as pd
import pytest

from spy_alpha_research.backtesting import (
    benchmark_bundle,
    run_timing_backtest,
    to_simple_returns,
    vol_matched_spy_scale,
)
from spy_alpha_research.statistics import block_bootstrap_ic
from spy_alpha_research.statistics.hypotheses import test_combo_vs_best as combo_vs_best_hypothesis


def test_turnover_uses_traded_not_target_pos():
    idx = pd.bdate_range("2020-01-01", periods=6)
    # target pos flips on day 1; with lag=1 the traded flip is on day 2
    signal = pd.Series([1, -1, -1, -1, -1, -1], index=idx, dtype=float)
    rets = pd.Series(0.01, index=idx)
    bt = run_timing_backtest(signal, rets, signal_lag=1, position="sign", costs_bps=10.0, return_type="simple")
    # traded: [0, 1, -1, -1, -1, -1] after shift+fillna
    assert list(bt.traded_positions.astype(float)) == [0.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    # cost day aligned with traded change (day2: | -1 - 1 | = 2)
    assert abs(bt.turnover.iloc[2] - 2.0) < 1e-12
    assert abs(bt.costs.iloc[2] - 2.0 * 10.0 / 1e4) < 1e-12
    # day1 traded change from 0→1 costs on day1, not on the unlagged pos flip day0/1 mismatch
    assert abs(bt.turnover.iloc[1] - 1.0) < 1e-12


def test_log_returns_converted_before_cumprod_wealth():
    idx = pd.bdate_range("2020-01-01", periods=5)
    signal = pd.Series(1.0, index=idx)  # tanh(1) ~ constant long
    log_rets = pd.Series(np.log1p(0.01), index=idx)  # ~1% simple
    bt_log = run_timing_backtest(
        signal, log_rets, signal_lag=0, position="tanh", costs_bps=0.0, return_type="log"
    )
    simple = to_simple_returns(log_rets, return_type="log")
    bt_simple = run_timing_backtest(
        signal, simple, signal_lag=0, position="tanh", costs_bps=0.0, return_type="simple"
    )
    np.testing.assert_allclose(bt_log.net_returns, bt_simple.net_returns, rtol=1e-10)
    # Wealth formula on net (already simple PnL) stays finite and matches (1+r).prod path
    w = (1.0 + bt_log.net_returns).prod()
    assert w > 1.0


def test_centered_bootstrap_p_not_naive_percentile_mass():
    idx = pd.bdate_range("2020-01-01", periods=250)
    rng = np.random.default_rng(7)
    signal = pd.Series(rng.normal(size=250), index=idx)
    # Strong positive IC — centered p should be small; naive P*(IC*<=0) also small,
    # so use a mild positive IC where they can diverge in interpretation.
    fwd = 0.15 * signal + pd.Series(rng.normal(scale=1.0, size=250), index=idx)
    out = block_bootstrap_ic(signal, fwd, n_bootstrap=400, block_size=10, seed=1)
    assert out["method"] == "centered_moving_block_bootstrap"
    assert 0.0 <= out["p_value_one_sided"] <= 1.0
    # Under H0-like noise (independent), centered p should not be tiny
    noise = pd.Series(rng.normal(size=250), index=idx)
    null = block_bootstrap_ic(signal, noise, n_bootstrap=400, block_size=10, seed=2)
    assert null["p_value_one_sided"] > 0.05


def test_h3_best_frozen_on_selection_not_final():
    idx = pd.bdate_range("2020-01-01", periods=200)
    sel = idx[:120]
    final = idx[120:]
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(scale=0.01, size=200), index=idx)
    # alpha_a wins on selection; alpha_b wins on final
    a = pd.Series(0.0, index=idx)
    b = pd.Series(0.0, index=idx)
    a.loc[sel] = r.loc[sel]  # perfect on selection
    b.loc[final] = r.loc[final]  # perfect on final
    combo = 0.5 * (a + b)
    out = combo_vs_best_hypothesis(
        combo.reindex(final),
        {"alpha_a": a, "alpha_b": b},  # full history required for selection scoring
        r,
        selection_index=sel,
        n_bootstrap=50,
        block_size=5,
        seed=0,
    )
    assert out["best_source"] == "selection"
    assert out["best_alpha"] == "alpha_a"


def test_vol_matched_requires_selection_strategy_coverage():
    idx = pd.bdate_range("2022-01-01", periods=300)
    sel = idx[:150]
    final = idx[150:]
    rng = np.random.default_rng(3)
    spy = pd.Series(rng.normal(scale=0.01, size=len(idx)), index=idx)
    # Final-only strategy — must raise (no silent scale=1)
    final_only = spy.reindex(final) * 0.5
    with pytest.raises(ValueError, match="selection"):
        vol_matched_spy_scale(final_only, spy, selection_index=sel, return_type="simple")

    sel_strat = spy.reindex(sel) * 0.4
    scale = vol_matched_spy_scale(sel_strat, spy, selection_index=sel, return_type="simple")
    assert scale != 1.0
    bmarks = benchmark_bundle(
        sel_strat, spy, selection_index=sel, eval_index=final, return_type="simple"
    )
    # vol-matched should not equal plain BH when scale != 1
    assert not np.allclose(bmarks["spy_vol_matched"], bmarks["spy_bh"])
