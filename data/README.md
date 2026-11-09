# Data contract

SPY-only research. Market data is **not** committed (see `.gitignore`).

| Path | Contents | Git |
|---|---|---|
| `raw/` | yfinance OHLCV parquet | ignored |
| `processed/` | returns, feature panels, scaled splits | ignored |
| `processed/alpha_candidates.yaml` | Frozen Borda shortlist | **tracked** |

Hard date cap: **2024-12-31** (no 2025+ in the research sample).

Rebuild: run notebooks `01` → `02` (and later chapters as needed).
