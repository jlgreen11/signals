# Trailing 7-year view (2019-04-01 → 2026-04-01)

Seven years of daily equity-curve data for every top strategy, ending on April 1, 2026 (anchored to the nearest equity trading day to avoid weekend boundary effects). All strategies start at $10,000 on 2019-04-01.

**BTC calendar validation**: `scripts/validate_btc_calendar.py` confirms that the 4-asset portfolio captures every BTC tick, including weekends — Monday's portfolio bar carries the full Friday→Monday 3-day return, and the product of equity-calendar bars equals the product of all 365 native BTC bars to floating-point precision. No BTC price action is lost.

Annualization: 252/yr on the equity shared calendar. Risk-free rate ≈ 2.3% (2018-2024 historical average).

## Summary — sorted by Sharpe

| strategy | end value | total return | CAGR | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| `4-asset equal-weight portfolio` | $38,646 | +286.5% | +21.3% | +1.106 | -39.6% | +0.54 |
| `GLD buy-and-hold` | $36,026 | +260.3% | +20.1% | +1.009 | -22.0% | +0.91 |
| `BTC_HYBRID_PRODUCTION` | $156,643 | +1466.4% | +48.1% | +0.979 | -75.8% | +0.64 |
| `60/40 SP/GLD (gold bond)` | $28,958 | +189.6% | +16.4% | +0.965 | -22.5% | +0.73 |
| `BTC buy-and-hold (raw)` | $163,722 | +1537.2% | +49.1% | +0.918 | -76.6% | +0.64 |
| `SP500 buy-and-hold` | $22,933 | +129.3% | +12.6% | +0.583 | -33.9% | +0.37 |
| `60/40 SP/TLT (classic)` | $15,154 | +51.5% | +6.1% | +0.352 | -28.2% | +0.22 |
| `TLT buy-and-hold` | $6,936 | -30.6% | -5.1% | -0.372 | -51.8% | -0.10 |

## Year-end value progression

$10,000 initial on 2019-04-01. Each row shows the value at the final equity trading day of that calendar year (or the window end for 2026).

| year | 4-asset portfolio | SP B&H | BTC hybrid | TLT B&H | GLD B&H |
|---|---:|---:|---:|---:|---:|
| 2019-12-31 | $12,593 | $11,268 | $15,287 | $10,893 | $11,758 |
| 2020-12-31 | $18,675 | $13,100 | $35,142 | $12,682 | $14,676 |
| 2021-12-31 | $26,060 | $16,623 | $93,810 | $11,915 | $14,067 |
| 2022-12-30 | $18,701 | $13,391 | $41,756 | $8,005 | $13,959 |
| 2023-12-29 | $24,937 | $16,636 | $89,465 | $7,950 | $15,730 |
| 2024-12-31 | $34,676 | $20,514 | $221,741 | $7,022 | $19,923 |
| 2025-12-31 | $39,066 | $23,875 | $175,806 | $7,008 | $32,610 |
| 2026-04-01 | $38,646 | $22,933 | $156,643 | $6,936 | $36,026 |

## Raw data

- `scripts/data/trailing_7y_view.parquet` — daily values for all strategies
- Reproduce: `python scripts/trailing_7y_view.py`
- BTC calendar validation: `python scripts/validate_btc_calendar.py`

