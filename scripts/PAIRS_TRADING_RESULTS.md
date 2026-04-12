# Pairs Trading Evaluation

**Window**: 2019-04-01 -> 2026-04-01 (trailing 7 years)
**Universe**: 20 tickers -- AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, BRK-B, UNH, JNJ, JPM, V, PG, XOM, LLY, AVGO, COST, NFLX, AMD, ADBE
**Parameters**: coint_pvalue=0.05, entry_z=2.0, exit_z=0.5, lookback=252, max_pairs=5
**Annualization**: 252/yr, rf = 2.26%

## Strategy Summary

- Average pairs found per re-discovery: **5.0**
- Total round-trip trades: **155**
- Win rate: **61.9%**
- Average holding period: **37 days**
- Correlation to SP500: **-0.108**

The strategy exhibits **low correlation** to SP500, confirming approximate market-neutrality.

## Full 7-Year Comparison

| strategy | end value | total return | CAGR | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| `Pairs Trading` | $6,562 | -34.4% | -5.8% | -0.474 | -47.2% | -0.12 |
| `SP500 B&H` | $22,933 | +129.3% | +12.6% | +0.583 | -33.9% | +0.37 |
| `EW 20-stock B&H` | $49,130 | +391.3% | +25.5% | +1.032 | -29.4% | +0.87 |

## Multi-Seed Evaluation (5 seeds, non-overlapping 6-month windows)

- Avg Sharpe delta (pairs - SP): **-1.062**
- Seeds where pairs beats SP on Sharpe: **0/5**

| seed | pairs Sharpe | SP Sharpe | delta |
|---:|---:|---:|---:|
| 7 | -0.349 | +1.609 | -1.713 |
| 42 | -0.349 | +1.115 | -0.980 |
| 100 | +0.035 | +1.115 | -0.784 |
| 999 | -0.075 | +1.099 | -0.853 |
| 1337 | -0.109 | +1.115 | -0.980 |

## Key Insight

Pairs trading is structurally different from the directional models (Markov chains, trend filters, hybrid vol-routers) that failed on individual stocks. It is market-neutral by construction: each position consists of a long leg and a short leg, so the portfolio does not rely on the market going up. The question is whether cointegration relationships among major US stocks are strong and stable enough to generate risk-adjusted returns after transaction costs.

## Raw data

- `scripts/data/pairs_trading.parquet` -- daily equity curves
- Reproduce: `python scripts/pairs_trading_eval.py`

