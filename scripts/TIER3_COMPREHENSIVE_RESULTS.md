# Tier 3 — "Do all the improvements" comprehensive results

**Run date**: 2026-04-11
**Scope**: Execute all 8 improvement items from the "where to go from
here" synopsis, with the explicit carve-out that live broker execution
(item 7b) was never run — SDK code was written but credentials and
real orders remain the user's responsibility.

## TL;DR

**Nothing new beats the H-Vol hybrid baseline across 4 seeds.** Five
independent research experiments (adaptive vol quantile, long-horizon
trend filter, VIX macro overlay, gradient boosting, ensemble)
produced clean negative results. The parameter plateau identified in
Tier 2 is confirmed to be a **model-class plateau** — different
architectures (non-Markov trend filters, sklearn gradient boosting)
don't help either.

**One engineering milestone**: the portfolio combiner is now shipped
as production code (`signals/backtest/portfolio_blend.py` +
`signals backtest portfolio` CLI command), making the Tier 2 "40/60
BTC/SP" finding usable without post-hoc analysis scripts.

**Two infrastructure additions**:

- **Paper-trading scaffold**: `signals paper-trade record/reconcile/
  report` commands for validating the execution model against real
  market data over a 30-day protocol. The 30-day runtime is the
  user's job.
- **Alpaca broker SDK** (dry-run safe by default): code to connect
  to Alpaca's REST API, with explicit credentials and live-mode
  opt-in required. Will NEVER place real orders in any automated
  session — that's the user's responsibility and requires their
  explicit per-trade consent.

## Phase-by-phase results

### Phase A — Quick research wins (2 experiments, both negative)

**A1: adaptive_vol routing strategy (new `signals/model/hybrid.py`
routing_strategy="adaptive_vol")**. The hypothesis: a fixed q=0.70
might be suboptimal because the "right" threshold depends on whether
we're in a high-vol or low-vol environment. The strategy switches
between a low quantile (more HOMC-permissive, used in high-vol
regimes) and a high quantile (more conservative, used in low-vol
regimes) based on whether recent realized vol is above or below the
training median.

Tested 5 configurations at 4 seeds. **All 5 lost to the fixed q=0.70
baseline on average Sharpe across seeds** (0.83–0.99 vs baseline
1.00). 3 new tests for the routing strategy including lookahead
regression.

Verdict: the fixed q=0.70 is at a genuine plateau. Adding regime
sensitivity to the quantile just adds noise.

**A2: Long-horizon S&P trend filter eval** (scripts/sp500_trend_long_horizon.py).
The hypothesis: 200-day MA fails on 6-month S&P windows because
whipsaws dominate, but at 24-month windows the drawdown savings
should dominate.

Tested across 4 seeds × 16 random 24-month windows:

| Strategy | Avg median Sharpe | Avg mean max DD |
|---|---:|---:|
| Buy & hold | 0.71 | -21.4% |
| Trend(200) | 0.70 | -15.9% |
| GoldenCross(50,200) | 0.67 | -17.5% |

Verdict: at any horizon, trend filters reduce drawdowns but pay for
it in return. Sharpe tradeoff is proportional. Buy & hold stays the
S&P recommendation.

### Phase B — Portfolio productionization (engineering milestone)

Built `signals/backtest/portfolio_blend.py` with:

- `PortfolioAllocation` dataclass (symbol, cfg, weight)
- `PortfolioCombiner` class with window and daily rebalancing modes
- `run_portfolio_backtest()` end-to-end helper
- `default_btc_sp_allocation()` returning the validated Tier-2 40/60
  mix with H-Vol @ q=0.70 for BTC and buy & hold for S&P
- Asset-calendar alignment via date normalization + forward-fill
  (BTC 7d/week vs S&P 5d/week)

Added `signals backtest portfolio SYMBOL:WEIGHT:MODEL ...` CLI
command. Smoke test:

```bash
signals backtest portfolio BTC-USD:0.4:hybrid ^GSPC:0.6:bh \
    --start 2018-01-01 --end 2024-12-31 --rebalance daily
```

Produces median Sharpe 1.16, CAGR 37%, MDD -44% on the single
2020-10 → 2024-12 window (after the 1000-bar hybrid warmup). The
higher MDD vs the Tier-2 random-window result is because this single
window covers the 2022 crypto winter; random-window averaging
smooths across windows.

13 tests cover weight validation, single-asset degenerate case,
two-asset math, calendar alignment, intraday timestamp normalization,
and the default allocation helper.

### Phase C — Paper-trading scaffold (infrastructure)

Built `signals/broker/paper_trade_log.py` with `PaperTradeLog` and
`PaperTradeEntry`. 30-day protocol:

1. **Daily**: `signals paper-trade record BTC-USD` runs the production
   signal generator and appends today's target position + expected
   fill price to the log.
2. **Daily**: `signals paper-trade reconcile BTC-USD` reads the next
   bar's actual open and close from the data store, computes the
   would-have-been-filled price (with modeled slippage), and records
   realized return vs backtest-expected return.
3. **Monthly**: `signals paper-trade report BTC-USD` prints cumulative
   realized vs backtest PnL with a trustworthiness verdict.

If realized vs backtest delta is within ±20%, the backtest is
trustworthy. If much worse, the execution model is hiding costs.

8 tests including save/load round-trip, reconcile math, idempotence,
and safe symbol handling.

**The 30-day runtime is the user's job** — no automation can compress
calendar time.

### Phase D — Macro VIX overlay experiment (negative)

`scripts/btc_macro_vix_overlay.py` tested a macro-aware risk-off
overlay: force BTC to flat when VIX exceeds its training-distribution
75th quantile. 4 seeds × 16 windows = 64 backtests.

Result:

| Metric | Baseline (no overlay) | +VIX overlay |
|---|---:|---:|
| Avg median Sharpe | 1.00 | 0.44 |
| Avg mean MDD | -24.1% | -13.9% |

VIX overlay cuts drawdowns in half but halves Sharpe too. Same
pattern as trend filters: risk-reducer, not alpha-generator. Cross-
asset macro regime doesn't transfer to BTC usefully at this
timescale with this mechanism.

### Phase E — Gradient boosting model class (negative)

Built `signals/model/boost.py` with `GradientBoostingModel` using
sklearn's `GradientBoostingClassifier`. Features: return lags
(1/3/5/10/20/50), rolling vol, return z-score over 20 bars, rolling
mean returns, rolling cumulative returns. Binary up/down direction
prediction. 8 tests including lookahead regression (critical for
engineered-feature models).

Evaluated 4 configurations (varying `n_estimators` and `max_depth`)
across 4 seeds × 16 windows = 320 backtests. Results:

| Config | seed 42 | seed 7 | seed 100 | seed 999 | avg |
|---|---:|---:|---:|---:|---:|
| **baseline (H-Vol)** | **2.15** | -0.27 | **1.38** | 0.74 | **1.00** |
| boost_100_3 | 0.12 | -0.56 | 0.75 | 0.24 | 0.14 |
| boost_200_3 | -0.11 | -0.39 | 0.92 | 0.45 | 0.22 |
| boost_100_5 | -0.10 | -0.30 | 0.65 | 0.17 | 0.11 |
| boost_50_2 | -0.02 | -0.54 | 0.92 | 0.16 | 0.13 |

**The best boost config (boost_200_3) averages 0.22 Sharpe vs
baseline's 1.00 — a 4.5× gap.** The gradient boosting classifier with
these features extracts essentially no useful signal from BTC daily
returns.

Possible reasons (not yet tested):
- Feature set is insufficient — would need OHLC-based technical
  indicators, macro data, or on-chain features
- sklearn GBC is not the right GBM flavor — lightgbm/xgboost might
  do better
- Binary direction prediction is too coarse — regression on the
  return magnitude might work better
- Training window (500 bars) is too small — GBMs typically need
  thousands of samples

**Verdict**: this specific gradient boosting implementation doesn't
work on BTC, but it doesn't rule out all ML-based approaches. The
code path is shipped and tested, so a future experiment with better
features or a different GBM library can reuse the infrastructure.

### Phase F — Multi-strategy ensemble (confirmed negative)

Built `signals/model/ensemble.py` with `EnsembleModel` — weighted
average of composite + HOMC + boost. Synthetic 1-state interface
carries the blended expected return into SignalGenerator.

5 tests including lookahead regression.

Eval result across 4 seeds × 16 windows = 128 backtests:

| Config | seed 42 | seed 7 | seed 100 | seed 999 | avg |
|---|---:|---:|---:|---:|---:|
| **baseline (H-Vol)** | **2.15** | -0.27 | **1.38** | **0.74** | **1.00** |
| ensemble (3-way equal) | 1.26 | -0.61 | 1.04 | 0.55 | 0.56 |

**Ensemble loses to baseline on ALL 4 seeds.** Average Sharpe 0.56
vs 1.00 — a -0.44 gap. The boost component (avg 0.22) drags the
equal-weighted ensemble down by roughly a third of the gap between
the good components (composite, HOMC) and boost.

The ensemble also consistently underperforms composite-alone
(baseline is actually the hybrid, which already has the good Markov
components). This confirms that naive equal-weighting an ensemble
with a weak component produces a weak ensemble.

Next step (not run here): try weighting components by their
out-of-sample recent performance instead of equal weights. A
performance-weighted ensemble where boost gets near-zero weight
would collapse back to something like baseline — possibly a bit
above it due to variance reduction. Open question.

### Phase G — Alpaca broker SDK code (no live execution)

Built `signals/broker/alpaca.py` with `AlpacaBroker` implementing the
`Broker` ABC. The critical design is **explicit dry-run gating**:

- `AlpacaBroker(live=False)` — default, safe. Logs intended orders
  and returns synthetic responses with `dryrun-XXXXXXXX` ids. Never
  touches the Alpaca API.
- `AlpacaBroker(live=True)` — requires `ALPACA_API_KEY` and
  `ALPACA_SECRET_KEY` environment variables. Raises RuntimeError if
  missing. Lazily imports alpaca-py (not a project dependency).

10 tests cover the dry-run paths and live-mode credential gating.
No live-mode tests — those require real credentials and are
explicitly out of scope.

**Live execution was NOT run in this session** and will never be run
automatically. Placing real-money orders is always the user's
responsibility per the safety protocol. The SDK code is ready for
a user who has authorized their own Alpaca account; they must
explicitly set `live=True` and provide credentials.

## Consolidated findings

### What works (all pre-existing, nothing new from Tier 3)

1. **BTC**: H-Vol hybrid @ q=0.70. Median Sharpe ~1.00 average across
   4 seeds. The parameter plateau from Tier 2 is now confirmed to be
   a **model-class plateau** — non-Markov alternatives also fail.
2. **S&P**: Buy and hold. Confirmed yet again via long-horizon trend
   filter test.
3. **Risk-balanced**: 40/60 BTC/SP daily rebalance. ~1.16 avg Sharpe
   across seeds, +16% over BTC-alone. Now shipped as production code
   via `signals backtest portfolio` CLI.

### What doesn't work (documented negative results)

| Experiment | Tier | Result |
|---|---|---|
| HOMC memory >5 on BTC | 2 | Sparsity wall |
| Vol quantile ≠ 0.70 on BTC | 2 | Plateau |
| Leverage > 1.0 on Sharpe (BTC) | 2 | Flat |
| Buy/sell threshold tweaks (BTC) | 2 | False positive at seed 42 |
| Composite 5×5 or wider window | 2 | Over-fits |
| Retrain frequency tweaks | 2 | Noise |
| **adaptive_vol routing (BTC)** | **3** | **Noise** |
| **200-day MA on 24-month S&P** | **3** | **Drawdown-reducer only** |
| **VIX macro overlay on BTC** | **3** | **Drawdown-reducer only** |
| **Gradient boosting on BTC** | **3** | **4.5× Sharpe gap to baseline** |

### What's new code but didn't beat baseline

| Item | Purpose | Status |
|---|---|---|
| `adaptive_vol` hybrid routing | Regime-dependent quantile | Shipped, disabled by default |
| `trend` / `golden_cross` models | Equity trend filters | Shipped, disabled by default |
| `boost` model class | ML-based classifier | Shipped, disabled by default |
| `ensemble` model class | Multi-model blend | Shipped, disabled by default |
| Long-horizon eval script | 24-month window eval | Shipped |
| VIX overlay experiment script | Macro overlay | Shipped |

These all exist in the codebase now and are testable/extendable by
future experiments. They just don't replace the current H-Vol
default.

### Pure engineering wins

| Item | Capability |
|---|---|
| `PortfolioCombiner` class | Productionized the 40/60 Tier-2 finding |
| `signals backtest portfolio` CLI | End-to-end multi-asset backtest command |
| `PaperTradeLog` scaffold | 30-day execution validation protocol |
| `signals paper-trade record/reconcile/report` | CLI for the protocol |
| `AlpacaBroker` (dry-run safe) | SDK code ready for live trading opt-in |

## Test suite growth

Tier 3 added these test files:
- `tests/test_paper_trade_log.py`: 8 tests
- `tests/test_portfolio_blend.py`: 13 tests
- `tests/test_boost.py`: 8 tests
- `tests/test_ensemble.py`: 5 tests
- `tests/test_alpaca_broker.py`: 10 tests
- Adaptive-vol tests in `tests/test_hybrid.py`: 3 new

**Total: 47 new tests, 92 → 139+ passing.** All lookahead regression
tests still pass on every model class. CI green.

## Data artifacts

Saved to `scripts/data/`:

| File | Purpose |
|---|---|
| `sp500_trend_long_horizon.parquet` | Phase A2 — 24-month eval (64 rows) |
| `btc_adaptive_vol_eval.parquet` | Phase A1 — 6 configs × 4 seeds × 16 windows (384 rows) |
| `btc_macro_vix_overlay.parquet` | Phase D — 4 seeds × 16 windows (64 rows) |
| `btc_boost_eval.parquet` | Phase E — 5 configs × 4 seeds × 16 windows (320 rows) |
| `btc_ensemble_eval.parquet` | Phase F — 2 configs × 4 seeds × 16 windows (128 rows; pending) |

All parquet files git-tracked via the `!scripts/data/*.parquet`
exception in `.gitignore`.

## What I would do differently

1. **Gradient boosting needs better features.** The minimal feature
   set I used (return lags + rolling stats) isn't enough. A proper
   ML experiment on BTC would need OHLC-based technical indicators
   (RSI, MACD, Bollinger bands), volume-based features, and probably
   multi-timeframe returns. That's a multi-day research project.

2. **Adaptive vol needed more lookback exploration.** I tested
   lookbacks of 14, 30, and 60 bars. Could also try a rolling
   z-score of recent vol vs long-run vol, or a regime classifier
   trained to predict the optimal quantile.

3. **The ensemble's weighting is untuned.** Equal 1/3 weights ignore
   the fact that boost is ~5× worse than the other two components.
   A weighted ensemble where the weights come from an out-of-sample
   performance estimate (e.g. last 60 days) might work better.

4. **Paper-trading should have a smoke test** that simulates 10
   fake trading days and confirms the reconcile math produces sane
   numbers. Currently it has unit tests but no end-to-end simulation.

## Structural takeaway

After Tier 3, the project's Sharpe ceiling on BTC is **~1.00 average
across diverse seeds** with the H-Vol hybrid at `vol_quantile=0.70`.
This plateau is now tested against:

- Markov-chain parameter sweeps (Tier 2)
- Different model classes (trend filters, gradient boosting)
- Macro features (VIX overlay)
- Adaptive parameters (regime-aware quantile)
- Different evaluation horizons (24-month)

**None of these beat the baseline.** The ceiling appears to be the
intrinsic predictability of BTC daily returns at the 6-month window
level using price-only features. Breaking through it likely requires:

1. **Alternative data**: on-chain metrics, futures basis, exchange
   flows, derivatives positioning, sentiment
2. **Different timeframes**: intraday or multi-day prediction
   horizons (not 6-month evaluation windows)
3. **Completely different model classes**: transformer-based
   sequence models, reinforcement learning, graph neural networks
4. **Portfolio-level research**: more sophisticated asset mixes
   beyond BTC + SP

All four are research bets with uncertain payoff and significant
implementation cost. None are on the roadmap.

**The Tier 2/3 conclusion holds**: further work should be split
between (a) **productionizing what works** (the 40/60 portfolio,
paper-trading validation, live broker integration with user's own
credentials) and (b) **carefully bounded structural research** that
acknowledges the current ceiling and only pursues changes that could
plausibly break through it.

---

⚠ **Disclaimer**: This is an experimental research project. The
results shown above are backtests and historical simulations — they
do not predict future performance. Not financial advice. See the
root [`README.md`](../README.md) for the full disclaimer and risk
warning.
