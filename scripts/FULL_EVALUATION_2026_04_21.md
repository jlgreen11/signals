# Full Project Evaluation — 2026-04-21

## What This Project Is

An equity momentum research platform built on a 26-year survivorship-bias-free
backtest of S&P 500 stocks. The headline strategy is **early-breakout momentum
acceleration** — ranking stocks by short-term return minus long-term pace,
catching moves at the start instead of the top.

Also contains an exhausted BTC Markov-chain track (tied with buy-and-hold on
Sharpe, useful only as a risk management overlay) and a multi-account Alpaca
automation layer for paper trading.

## The Numbers That Matter

### Early-Breakout Momentum (Canonical Config: 63/252/105/15/2)

| Period | Sharpe | CAGR | Max DD | Final Equity ($100K start) |
|--------|--------|------|--------|---------------------------|
| Full (2000-2026) | 0.659 | 13.6% | -62.9% | $2,850,728 |
| Train (2000-2018) | 0.663 | 13.6% | -62.9% | — |
| Holdout (2019-2026) | 0.530 | 10.4% | -47.3% | — |
| SPY B&H (total return) | 0.497 | 8.0% | -55.2% | $760,073 |

Edge: +0.16 Sharpe, +5.6pp CAGR vs SPY. Worse max drawdown by 7.7pp.

### Deflated Sharpe Ratio

| n_trials | E[max SR] | DSR (full) | DSR (holdout) | Verdict |
|----------|-----------|------------|---------------|---------|
| 1 | 0.00 | 1.000 | 1.000 | PASS (trivially) |
| 2 | 0.56 | 1.000 | ~0.00 | Marginal |
| 3 | 0.85 | 0.000 | 0.000 | FAIL |
| 6 | 1.30 | 0.000 | 0.000 | FAIL |
| 108 | 2.56 | 0.000 | 0.000 | FAIL |

**Interpretation**: DSR passes at n_trials <= 2 and fails at >= 3. The config
was selected from a 108-grid with ~6 effectively independent strategy families.
At the honest trial count of 6, DSR fails decisively. The 0.66 Sharpe is
normal for equity momentum but DSR needs > 1.3 to clear at 6 trials.

This does NOT mean the signal is zero. It means: a Sharpe of 0.66 from
a 6-strategy search is statistically consistent with chance. The holdout
(0.53) confirms mild decay, not collapse — also consistent with a weak
but real signal that DSR is underpowered to detect at this Sharpe level.

### BTC Markov Chain (Exhausted Track)

| Test | Result |
|------|--------|
| Multi-seed avg Sharpe | 1.00 (tied with B&H) |
| vs NaiveVolFilter | +0.37 Sharpe (Markov chain adds value) |
| Clean holdout (2023-2024) | B&H 1.96 > Hybrid 1.66 > VolFilter 1.49 |
| Parameter tuning | Exhausted (all dimensions at plateau) |
| S&P strategies | All lose to B&H |

## What Was Done Today (2026-04-21)

### New Code
- `signals/model/vol_filter.py` — NaiveVolFilter (null hypothesis baseline)
- `signals/backtest/engine.py` — vol_filter model type, BacktestConfig validation
- `scripts/null_hypothesis_eval.py` — 3-experiment BTC comparison
- `scripts/dsr_single_config.py` — Single-config DSR test
- `scripts/dsr_effective_trials.py` — DSR at intermediate trial counts
- `tests/test_vol_filter.py` — 12 tests for vol filter
- `tests/test_backtest.py` — 5 new config validation tests
- `tests/test_bias_free.py` — 14 tests for bias-free module

### Fixes
- `.env` permissions: 644 -> 600 (Alpaca keys were world-readable)
- `engine.py:405`: assert -> RuntimeError (production safety)
- `paper_trade_log.py:138`: added chmod 600 (log file permissions)
- BacktestConfig.__post_init__: bounds checking on all critical params
- Documentation: fixed window size mismatches, hold period docs

### Test Count: 305 -> 336 passing, ruff clean

## Where This Should Go

### The Decision Framework

The DSR result creates a fork:

**If you accept DSR as the bar**: The momentum signal doesn't survive.
Reframe as a research platform, publish negative results, expand universe
to Russell 1000+ where momentum effects are stronger (more small/mid-cap).

**If you treat DSR as one input among several**: The signal is consistent
across train (0.66) and holdout (0.53) with no catastrophic decay. It
beats SPY by 5.6pp CAGR over 26 years including 3 crashes. That's a
useful portfolio component even if DSR doesn't certify it as alpha.

### Concrete Next Steps (Ordered by Impact)

**1. Universe expansion** (highest expected value)
- Current: ~500 S&P stocks. Academic momentum uses 1000-3000.
- The project already has 1,194 unique tickers from SP500 history.
- Norgate Data (~$50/mo) gives Russell 1000/3000 with survivorship handling.
- Momentum effects are STRONGER in smaller stocks (less analyst coverage).
- This is the single change most likely to push Sharpe above DSR@6 threshold.

**2. Lock the config and paper trade** (immediate)
- The canonical config (63/252/105/15/2) is the honest best.
- Run `signals auto trade --account momentum` daily for 6 months.
- Compare realized vs backtested returns (the paper_trade_log reconciliation
  framework is already built).
- Real execution data > any amount of historical backtesting.

**3. Stop sweeping parameters** (discipline)
- Every new config tested increases n_trials and makes DSR harder.
- The FUTURE_IMPROVEMENTS.md items that add parameters (adaptive hold,
  staggered entry, multi-timeframe) should be tested ONE AT A TIME with
  pre-registered hypotheses, not swept.

**4. Investigate the remaining survivorship bias**
- ~375 deeply delisted tickers (Enron, Lehman, etc.) missing from Yahoo.
- These are exactly the stocks momentum would have bought and lost on.
- Norgate Data has delisting returns. This is the same subscription as #1.

**5. Reframe the BTC track honestly**
- It's a vol-regime risk management tool, not alpha.
- The 40/60 BTC/SP portfolio (Sharpe 1.16 across seeds) is the only
  genuine cross-asset finding. Consider deploying this as a separate
  allocation strategy for the baseline account.

## Files Changed

```
signals/model/vol_filter.py          (NEW — null hypothesis model)
signals/backtest/engine.py           (vol_filter type, config validation, assert fix)
signals/backtest/bias_free.py        (docstring fixes)
signals/model/momentum.py            (docstring fixes)
signals/broker/paper_trade_log.py    (chmod 600)
tests/test_vol_filter.py             (NEW — 12 tests)
tests/test_backtest.py               (5 new validation tests)
tests/test_bias_free.py              (NEW — 14 tests)
scripts/null_hypothesis_eval.py      (NEW — BTC null hypothesis eval)
scripts/NULL_HYPOTHESIS_RESULTS.md   (NEW — BTC results)
scripts/dsr_single_config.py         (NEW — single-config DSR)
scripts/dsr_effective_trials.py      (NEW — effective trial DSR)
FUTURE_IMPROVEMENTS.md               (window size doc fix)
.env                                 (permissions fix)
```
