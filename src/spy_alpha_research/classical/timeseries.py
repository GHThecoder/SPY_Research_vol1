"""ACF/PACF and ARIMA baselines."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def acf_pacf_values(series: pd.Series, nlags: int = 40) -> tuple[np.ndarray, np.ndarray]:
    from statsmodels.tsa.stattools import acf, pacf

    x = series.dropna().to_numpy(dtype=float)
    acf_v = acf(x, nlags=nlags, fft=True)
    pacf_v = pacf(x, nlags=nlags)
    return acf_v, pacf_v


def adf_pvalue(series: pd.Series) -> float:
    from statsmodels.tsa.stattools import adfuller

    x = series.dropna()
    if len(x) < 20:
        return np.nan
    return float(adfuller(x, autolag="AIC")[1])


def fit_arima_grid(
    series: pd.Series,
    order_grid: Sequence[tuple[int, int, int]] | None = None,
    max_iter: int = 200,
) -> dict:
    """Fit ARIMA candidates; return best by AIC plus in-sample fitted values."""
    from statsmodels.tsa.arima.model import ARIMA

    order_grid = list(order_grid or [(1, 0, 0), (1, 0, 1), (2, 0, 1)])
    x = series.dropna()
    best = None
    for order in order_grid:
        try:
            model = ARIMA(x, order=order)
            res = model.fit(method_kwargs={"maxiter": max_iter})
            row = {
                "order": order,
                "aic": float(res.aic),
                "bic": float(res.bic),
                "result": res,
            }
            if best is None or row["aic"] < best["aic"]:
                best = row
        except Exception:
            continue
    if best is None:
        raise RuntimeError("No ARIMA model converged")
    fitted = best["result"].fittedvalues.reindex(series.index)
    return {
        "order": best["order"],
        "aic": best["aic"],
        "bic": best["bic"],
        "fitted": fitted.rename("arima_fitted"),
        "resid": best["result"].resid.reindex(series.index).rename("arima_resid"),
        "result": best["result"],
    }


def ljung_box_pvalue(resid: pd.Series, lags: int = 10) -> float:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = resid.dropna()
    if len(r) < lags + 5:
        return np.nan
    table = acorr_ljungbox(r, lags=[lags], return_df=True)
    return float(table["lb_pvalue"].iloc[0])
