# Comprehensive Evaluation — Early-Breakout Momentum

Date: 2026-04-16
Module under test: `signals/backtest/bias_free.py` (canonical)

**All prices are TOTAL RETURN (split + dividend adjusted) via yfinance
`adj_close`. This matches the SPY baseline (auto_adjust=True), making
the comparison apples-to-apples.**

## Methodology

1. **Data**: 782 tickers from Yahoo Finance across 2000-2026, including
   ~300 historical/delisted S&P 500 constituents (Enron-era failures
   missing — Yahoo has dropped those). Point-in-time constituent
   membership via `fja05680/sp500`. Prices are total-return adjusted.

2. **Train/holdout split**:
   - Train: 2000-01-03 to 2018-12-31 (4,779 trading days)
   - Holdout: 2019-01-01 to 2026-04-13 (1,829 trading days)

3. **Grid**: 108 configs across 5 dimensions:
   - hold_days: 63, 105, 126
   - n_long: 15, 20, 25
   - max_per_sector: 1, 2
   - (short, long): (21, 126), (21, 252), (63, 252)
   - (min_short_return, max_long_return): (0.10, 1.50), (0.15, 1.50)

4. **Selection**: Top-15 configs by train Sharpe evaluated on holdout.
   Winner picked by best combined (train + holdout) / 2 Sharpe.

5. **Statistical correction**: Deflated Sharpe Ratio (Bailey & López de
   Prado) against expected-max-Sharpe under null hypothesis.

## Results

### Full-period (2000-2026) — representative configs, TOTAL RETURN

| Config | Parameters | Sharpe | CAGR | MaxDD | Calmar | Final Eq |
|---|---|---:|---:|---:|---:|---:|
| **B: Default** | hold=105, n=15, sec=2, win=63/252 | **0.659** | **13.6%** | -62.9% | 0.22 | $2,850,728 |
| **SPY B&H (total return)** | buy & hold | **0.497** | **8.0%** | **-55.2%** | 0.14 | $760,073 |

### Train/holdout breakdown for Config B (chosen default)

| Period | Sharpe | CAGR | MaxDD |
|---|---:|---:|---:|
| Train (2000-2018) | 0.663 | 13.6% | -62.9% |
| Holdout (2019-2026) | 0.530 | 10.4% | -47.3% |
| Combined avg | 0.596 | — | — |

### Deflated Sharpe Ratio check

- N trials: 108
- Mean Sharpe across grid: 0.515
- Std dev Sharpe across grid: 0.078
- Observed max Sharpe (train): **0.663**
- Expected max under null (Bailey-López de Prado): **0.714**
- **DSR excess: -0.052 — NOT statistically distinguishable from noise**

## What this means

**Config B beats SPY on Sharpe and CAGR on the full period**, with
honest survivorship-bias-reduced data and apples-to-apples total returns:

- Sharpe 0.659 vs 0.497 (+0.162)
- CAGR 13.6% vs 8.0% (+5.6pp)
- But MaxDD -62.9% vs -55.2% (worse by 7.7pp)

**However** the DSR test says we can't statistically distinguish the
grid's best (0.663) from the null-hypothesis max (0.714) across 108
trials. We cannot claim the strategy has statistically proven alpha.

## Comparison to prior claims

| Source | CAGR claim | Sharpe claim | Status |
|---|---:|---:|---|
| Pre-canonical README (~Apr 11) | +11.8% | 0.594 | Survivorship bias |
| First canonical (~Apr 13) | +9.4% | 0.428 | Wrong windows + price-only |
| Grid sweep price-only (Apr 16) | +11.5% | 0.579 | Missing dividends |
| **This evaluation (total return, Apr 16)** | **+13.6%** | **0.659** | **Honest (still DSR-marginal)** |

## What's still wrong

1. **Incomplete history**: 375 deeply delisted tickers (Enron, Lehman,
   WorldCom, Countrywide, Bear Stearns) aren't in Yahoo. Those are the
   tickers that would have caused the biggest "momentum → blowup"
   losses. Remaining bias inflates the result.

2. **Regime dependence**: 26 years includes 3 crashes (2000-2002,
   2008-2009, 2020). Strategy behavior varies enormously across them.
   A 7-year holdout is thin — one bad regime can swing Sharpe by 0.3+.

3. **DSR not passed**: The grid's best doesn't beat the null-hypothesis
   max (0.663 < 0.714 across 108 trials). We're below the statistical
   significance threshold.

## Recommended use

- **Keep** Config B as the default (`hold=105, n=15, sec=2, short=63,
  long=252`). It's the honest best pick from proper out-of-sample
  selection.
- **Don't believe** the CAGR/Sharpe numbers as forward predictions.
  Treat as "the strategy is well-constructed and not actively bad."
- **For max-CAGR stakeholders**: Config D gives 20.2% CAGR at the cost
  of deeper drawdowns and worse Sharpe.
- **Don't add parameters** without a full re-sweep. Adding trials
  makes DSR worse, not better.

## Raw data

- `scripts/data/grid_sweep_train_v2.parquet` — all 108 configs on train
- `scripts/data/grid_sweep_holdout_v2.parquet` — top-15 on holdout
- `scripts/data/final_eval.json` — four representative configs on full period
- `scripts/data/comprehensive_sweep.json` — one-var-at-a-time sweep
  (biased, superseded by grid)
