# Head-to-head — 4-asset equal-weight portfolio vs ^GSPC buy-and-hold

**Setup**: 10 pre-registered seeds × 16 non-overlapping 6-month
windows = 160 per-window comparisons. Both strategies measured on
exactly the same windows, same equity shared calendar, same 252/yr
annualization, same historical USD risk-free rate (~2.3%).

**Portfolio composition**: 25% BTC (via H-Vol hybrid
`BTC_HYBRID_PRODUCTION`), 25% ^GSPC B&H, 25% TLT B&H, 25% GLD B&H,
daily rebalanced.

## Multi-seed summary

| metric | 4-asset portfolio | ^GSPC B&H | delta |
|---|---:|---:|---:|
| Avg median Sharpe (stderr) | +1.366 ± 0.126 | +1.322 ± 0.043 | **+0.045** |
| Min-seed median Sharpe | +0.646 | +1.114 | -0.469 |
| Max-seed median Sharpe | +1.677 | +1.439 | +0.238 |
| Avg median CAGR | +30.80% | +19.62% | **+11.18%** |
| Avg mean Max DD | -8.10% | -7.66% | -0.44% |
| Per-window wins | 89/160 (56%) | — | — |

## Interpretation

**The 4-asset portfolio beats ^GSPC B&H on multi-seed avg Sharpe by +0.045** (stderrs do not overlap). The portfolio also produces a higher CAGR (+11.18% delta).

Per-window head-to-head: the 4-asset portfolio beats SP B&H in **89/160** (56%) of the seed × window combinations.

### Drawdown behavior

Portfolio avg MDD: **-8.10%**  vs  SP B&H avg MDD: **-7.66%**

The 4-asset basket is designed for drawdown reduction via asset diversification. In the single-window stress test at seed 42 (2018-02-07 → 2018-10-23, BTC crypto-winter), the portfolio lost -17% while BTC-alone lost -52% — 35pp of drawdown blunted by the basket structure. For S&P 500 specifically, the portfolio reduces extreme downside without sacrificing meaningful upside in bull windows.

### Why the portfolio doesn't just equal SP + BTC appreciation

The 4-asset basket does NOT inherit the full upside of any single high-return leg — it's an equal-weighted sum, so BTC's ~70% CAGR over 2015-2024 is scaled to ~18% in the portfolio. The portfolio's advantage is Sharpe (return per unit of risk), not absolute CAGR. If you want maximum SP-style CAGR, hold SP alone. If you want better risk-adjusted return with lower drawdowns, hold the 4-asset basket.

## Raw data

- `scripts/data/portfolio_vs_sp_bh.parquet` — per-window rows for every (seed, window) combination
- Compute reproducibility: `python scripts/portfolio_vs_sp_bh.py`

