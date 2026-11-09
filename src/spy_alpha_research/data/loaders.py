"""SPY OHLCV download and parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import resolve_path


def download_spy_ohlcv(
    start: str,
    end: str,
    *,
    ticker: str = "SPY",
    auto_adjust: bool = False,
    progress: bool = False,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download SPY OHLCV and optionally cache. End date is inclusive filter after download."""
    import yfinance as yf

    raw = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=progress,
        threads=False,
    )
    if raw.empty:
        raise ValueError("Empty download for SPY")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    frame = raw.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    # Hard cap: drop anything after configured end (yfinance end is exclusive-ish)
    end_ts = pd.Timestamp(end)
    frame = frame.loc[frame.index <= end_ts]
    if cache_path is not None:
        save_frame(frame, cache_path)
    return frame


def save_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    out = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out)
    return out


def load_frame(path: str | Path) -> pd.DataFrame:
    p = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    frame = pd.read_parquet(p)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame.index = frame.index.tz_localize(None)
    return frame.sort_index()


def ensure_spy(
    start: str,
    end: str,
    cache_path: str | Path,
    *,
    force: bool = False,
    **kwargs,
) -> pd.DataFrame:
    path = resolve_path(cache_path) if not Path(cache_path).is_absolute() else Path(cache_path)
    if path.exists() and not force:
        frame = load_frame(path)
        end_ts = pd.Timestamp(end)
        return frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= end_ts)]
    return download_spy_ohlcv(start, end, cache_path=path, **kwargs)
