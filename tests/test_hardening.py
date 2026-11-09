"""Research-grade hardening: three-way split, bootstrap IC, ablation, benchmarks."""

import numpy as np
import pandas as pd
import pytest

from spy_alpha_research.attribution import group_members, grouped_ablation
from spy_alpha_research.backtesting import benchmark_bundle, buy_and_hold_returns, summarize_benchmarks
from spy_alpha_research.data import split_three_way
from spy_alpha_research.statistics import block_bootstrap_ic, ic_inference


def test_three_way_split_integrity():
    idx = pd.bdate_range("2016-01-01", "2024-12-31")
    train, sel, final = split_three_way(
        idx,
        train_end="2021-12-31",
        selection_start="2022-01-01",
        selection_end="2023-12-31",
        final_start="2024-01-01",
        end="2024-12-31",
    )
    assert train.max() <= pd.Timestamp("2021-12-31")
    assert sel.min() >= pd.Timestamp("2022-01-01")
    assert sel.max() <= pd.Timestamp("2023-12-31")
    assert final.min() >= pd.Timestamp("2024-01-01")
    assert final.max() <= pd.Timestamp("2024-12-31")
    assert set(train).isdisjoint(sel)
    assert set(sel).isdisjoint(final)
    assert set(train).isdisjoint(final)


def test_three_way_rejects_overlap():
    idx = pd.bdate_range("2016-01-01", "2024-12-31")
    with pytest.raises(ValueError):
        split_three_way(
            idx,
            train_end="2022-06-01",
            selection_start="2022-01-01",
            selection_end="2023-12-31",
            final_start="2024-01-01",
            end="2024-12-31",
        )


def test_block_bootstrap_ic_smoke():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(1)
    signal = pd.Series(rng.normal(size=200), index=idx)
    fwd = 0.4 * signal + pd.Series(rng.normal(scale=0.6, size=200), index=idx)
    out = block_bootstrap_ic(signal, fwd, n_bootstrap=200, block_size=10, seed=0)
    assert out["n_obs"] == 200
    assert out["ic"] > 0
    assert out["ci_low"] <= out["ic"] <= out["ci_high"]
    assert 0.0 <= out["p_value_one_sided"] <= 1.0

    inf = ic_inference(signal, fwd, n_bootstrap=100, block_size=10, seed=0)
    assert "p_value_bootstrap" in inf
    assert "nw_p_value" in inf
    assert "rolling_ic_mean" in inf


def test_grouped_ablation_shapes():
    idx = pd.bdate_range("2020-01-01", periods=260)
    rng = np.random.default_rng(2)
    cols = ["mom_10", "mom_21", "ret_lag_1", "ret_lag_2", "vol_21", "volume_z_21", "dow"]
    feats = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    target = pd.Series(rng.normal(scale=0.01, size=len(idx)), index=idx)
    rets = target.shift(1).fillna(0.0)
    train = idx[:180]
    eval_idx = idx[180:]
    abl = grouped_ablation(
        feats,
        cols,
        target,
        rets,
        train,
        eval_idx,
        min_periods=20,
        costs_bps=0.0,
    )
    assert "full" in abl.index
    assert any(i.startswith("drop_") for i in abl.index)
    assert {"ic", "sharpe", "delta_ic", "delta_sharpe"}.issubset(abl.columns)
    members = group_members(cols)
    assert "momentum" in members
    assert "ret_lags" in members


def test_benchmark_always_long_sharpe_finite():
    idx = pd.bdate_range("2022-01-01", periods=400)
    rng = np.random.default_rng(3)
    spy = pd.Series(rng.normal(scale=0.01, size=len(idx)), index=idx, name="spy")
    sel = idx[:250]
    final = idx[250:]
    strategy = spy.reindex(sel) * 0.5
    bmarks = benchmark_bundle(
        strategy,
        spy,
        selection_index=sel,
        eval_index=final,
        periods_per_year=252,
    )
    assert set(bmarks) == {"spy_bh", "cash", "spy_vol_matched"}
    bh = buy_and_hold_returns(spy.reindex(final))
    assert np.isfinite(bh).all()
    summary = summarize_benchmarks(bmarks)
    assert np.isfinite(summary.loc["spy_bh", "sharpe"]) or summary.loc["spy_bh", "ann_vol"] == 0
    assert summary.loc["cash", "ann_return"] == 0.0 or np.isnan(summary.loc["cash", "sharpe"])
