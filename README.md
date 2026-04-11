# Signals

[![CI](https://github.com/jlgreen11/signals/actions/workflows/ci.yml/badge.svg)](https://github.com/jlgreen11/signals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-188%20passing-brightgreen.svg)](./tests)

A multi-asset portfolio research project that started out chasing a
Markov-chain trading edge on Bitcoin and ended — after five rounds of
adversarial review — recommending a **4-asset equal-weight risk-parity
basket**. The Markov code still runs (it powers the BTC leg of the
basket) but it is **sunset** as a standalone strategy: repeated
controlled experiments could not show that any Markov variant extracts
directional insight beyond what a one-line volatility filter provides.

The current recommendation is not a Markov-chain strategy. The current
recommendation is portfolio math.

## ⚠ Disclaimer

Experimental research project. Nothing here is financial advice.
Backtest results are historical and **do not predict future
performance** — especially for cryptocurrency, which has seen multiple
~80% drawdowns in the past decade. MIT-licensed with **no warranty of
any kind**. See [`LICENSE`](./LICENSE). Conduct your own due diligence
and start with amounts you can afford to lose entirely.

## Current production recommendation

### 1. 4-asset equal-weight risk-parity basket

The **first-choice** production configuration. See
[`scripts/risk_parity_4asset.py`](./scripts/risk_parity_4asset.py)
and [`scripts/RISK_PARITY_4ASSET_RESULTS.md`](./scripts/RISK_PARITY_4ASSET_RESULTS.md).

| Leg | Allocation | Strategy |
|---|---:|---|
| BTC-USD | 25% | H-Vol hybrid (`BTC_HYBRID_PRODUCTION`) |
| ^GSPC | 25% | Buy & hold |
| TLT (long Treasuries) | 25% | Buy & hold |
| GLD (gold) | 25% | Buy & hold |

Daily rebalance. Inner-joined on the US equity calendar (~252 bars/year)
so weekend BTC bars don't desync the portfolio.

**Measured multi-seed performance** (10 pre-registered seeds × 16
non-overlapping 6-month windows, correctly annualized at 252/yr for
the equity shared calendar, risk-free rate ≈ 2.3%):

| | avg Sharpe | stderr | min seed | max seed |
|---|---:|---:|---:|---:|
| **4-asset equal-weight** | **+1.366** | 0.126 | +0.646 | +1.677 |
| 4-asset inverse-vol (21d) | +1.118 | 0.036 | +0.837 | +1.284 |
| 4-asset inverse-vol (63d) | +1.040 | 0.063 | +0.762 | +1.362 |
| BTC alone (hybrid, 365/yr) | +1.188 | 0.025 | +1.152 | +1.239 |

Equal weighting beats inverse-vol because inverse-vol down-weights
BTC, the highest-vol and highest-return leg. The +0.178 Sharpe
improvement over BTC-alone is structural — it comes from the
portfolio-math benefit of combining assets with low average pairwise
correlation, not from any new alpha signal.

**Stress behavior**: see [`scripts/data/portfolio_trace_180d.md`](./scripts/data/portfolio_trace_180d.md)
for a day-by-day trace of the portfolio through its worst-case window
(2018-02-07 → 2018-10-23, the BTC 2018 crash into crypto winter):

- BTC leg alone over that window: **−52%**
- Equal-weight 4-asset portfolio: **−17%** — 35 pp of drawdown blunted
  by the diversification math.

### 2. BTC alone, if you only want one asset

`BacktestConfig(**BTC_HYBRID_PRODUCTION)` from
[`signals/backtest/engine.py`](./signals/backtest/engine.py) — the
Round-3 parameter bundle:

```
hybrid_vol_quantile  = 0.50
retrain_freq         = 14
train_window         = 750
vol_window           = 10
routing_strategy     = "vol"
periods_per_year     = 365  (BTC trades 365 days/year)
```

Multi-seed avg Sharpe **+1.188 ± 0.025** on 10 seeds × 16
non-overlapping 6-month windows, stderr 0.025 (the tightest reading in
the project). Stable across seeds. Beats a pure vol-filter baseline
(~1.15) by ~0.04 and the legacy q=0.70 config by ~0.30.

### 3. ^GSPC (S&P 500) alone

**Use buy & hold.** No active strategy in this project beats B&H on
S&P 500 across 4 different model classes tested (composite, HOMC,
trend filters, golden cross). See
[`scripts/HOMC_TIER0E_BTC_SP500.md`](./scripts/HOMC_TIER0E_BTC_SP500.md)
and [`scripts/SP500_TREND_AND_HOMC_MEMORY.md`](./scripts/SP500_TREND_AND_HOMC_MEMORY.md).
The Markov-chain backbone is the wrong tool for a secular-uptrend
equity index.

## ⚠️ Markov model sunset

All four Markov model classes — `composite`, `homc`, `hmm`, `hybrid` —
are **sunset** as standalone strategies. Direct instantiation emits a
`DeprecationWarning`. They are **not deleted** because
`HybridRegimeModel` composes them internally and the hybrid is the BTC
leg of the production basket above.

The closure of the "is the Markov class doing anything?" question came
from two pre-registered experiments plus a regime ablation:

| Test | Result |
|---|---|
| [`regime_ablation.py`](./scripts/regime_ablation.py) | Pure vol filter matches full hybrid within 0.2 Sharpe. The Markov components are decorative in the production config. |
| [`absolute_encoder_eval.py`](./scripts/absolute_encoder_eval.py) | 9 pre-registered configs of HOMC at absolute-width bins (the Nascimento paper's actual granularity). **Zero trades across the entire grid** due to k-tuple sparsity. |
| [`rule_based_eval.py`](./scripts/rule_based_eval.py) | Nascimento-style rule extraction with P(direction)≥0.60 gate. 16 configs. Winner at 0.567 ± 0.072 in-sample, 0.618 on pristine holdout — positive but roughly half the pure-vol-filter baseline. |

The hybrid's edge comes from the **parameter bundle** (q, retrain_freq,
train_window) interacting with the vol router, not from the chain's
state transitions. We keep the code because it produces the best BTC
Sharpe in the project, and we're attached to returns, not technology.

## Install

Python 3.11+ required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Daily workflow

```bash
# Fetch price data (one time, or incremental refresh)
signals data fetch BTC-USD --start 2015-01-01
signals data fetch ^GSPC --start 2015-01-01
signals data fetch TLT --start 2015-01-01
signals data fetch GLD --start 2015-01-01

# Next-day signal for BTC alone (legacy CLI)
signals signal next BTC-USD

# Backtest the production bundle
signals backtest run BTC-USD --model hybrid \
  --start 2018-01-01 --end 2024-12-31 \
  --states 5 --order 5 --train-window 750
```

For the 4-asset portfolio, use the Python API:

```python
from signals.backtest.engine import BTC_HYBRID_PRODUCTION, BacktestConfig
# See scripts/risk_parity_4asset.py for the full harness.
```

## Architecture

```
[yfinance / CoinGecko]
        │
        ▼
[DataSource] → [DataPipeline] → [DataStore (parquet + SQLite)]
        │
        ▼
[Features: log returns, rolling volatility]
        │
        ▼
[Model: hybrid (sunset internals: composite / HMM / HOMC)]
        │       [or] TrendFilter / DualMovingAverage / VolFilter-only
        ▼
[SignalGenerator: BUY/SELL/HOLD + sized target_position]
        │
        ├─▶ [Portfolio] → [BacktestEngine + metrics]
        └─▶ [signal next] → CLI: tomorrow's action
```

For multi-asset use:

```
[BTC hybrid equity] ─┐
[^GSPC B&H equity]  ─┤
[TLT B&H equity]    ─┼─▶ [equal-weight risk-parity blender] ─▶ [portfolio equity]
[GLD B&H equity]    ─┘       (daily rebalance)
```

## Tests

```bash
pytest --cov=signals
```

188 tests across:

- `test_lookahead.py` — strict no-lookahead regression, asserts equity
  curves up to bar N are bit-identical regardless of how much future
  data is in the input. Covers composite, HOMC, HMM, and hybrid.
- `test_sunset_warnings.py` — verifies Markov models warn on direct
  instantiation but remain silent inside the hybrid.
- `test_hybrid.py`, `test_trend.py`, `test_homc.py`, `test_composite.py`,
  `test_hmm.py`, `test_boost.py`, `test_ensemble.py`, `test_signals.py`,
  `test_states.py` — per-model component coverage.
- `test_backtest.py`, `test_holdout.py`, `test_portfolio_blend.py`,
  `test_vol_target.py`, `test_data.py`, `test_paper_trade_log.py`,
  `test_excel_report.py`, `test_alpaca_broker.py`, `test_rule_signals.py`,
  `test_absolute_encoder.py` — engine, portfolio, metrics, data, and
  new-model coverage.

CI runs `ruff check` + `pytest --cov=signals` on Python 3.11 and 3.12
on every push and PR. See `.github/workflows/ci.yml`.

## Methodology discipline

This repo has been through five adversarial review rounds. The
discipline rules that emerged, in rough order of importance:

1. **No single-seed headlines.** Every reported Sharpe quotes
   `mean ± stderr` across ≥ 10 pre-registered seeds, or is explicitly
   labeled `(seed=42, in-sample)`.
2. **Non-overlapping windows.** The shared sampler at
   [`scripts/_window_sampler.py`](./scripts/_window_sampler.py)
   guarantees `spacing ≥ window_len` between every pair of random
   starts. Never use `random.sample` directly.
3. **Correct annualization per calendar.** BTC 365/yr, equities and
   equity-calendar portfolios 252/yr. Explicit `periods_per_year` on
   every `compute_metrics` call, no legacy index-inference.
4. **Non-zero risk-free rate.** Use
   [`historical_usd_rate(window)`](./signals/backtest/risk_free.py)
   for a period-exact T-bill average — the 2018–2024 window averaged
   ~2.3%, the 2023–2024 window averaged ~5%.
5. **Pre-registered grids.** Every sweep script declares its parameter
   grid at the top in a docstring comment. Grids are not expanded when
   a sweep fails.
6. **Project-level DSR.** The deflated Sharpe correction counts
   every trial the project has ever run, not just the one sweep.
   See [`scripts/project_level_dsr.py`](./scripts/project_level_dsr.py).
7. **Pristine holdout.** The 2023–2024 BTC slice is held out of any
   new parameter search. See
   [`scripts/pristine_holdout.py`](./scripts/pristine_holdout.py).

See [`SKEPTIC_REVIEW.md`](./SKEPTIC_REVIEW.md) for the external
teardown that drove these rules, and
[`IMPROVEMENTS_PROGRESS.md`](./IMPROVEMENTS_PROGRESS.md) for the full
round-by-round history of what shipped and what was rejected.

## History in one paragraph

The project started in early 2026 as a first-order Markov chain over
BTC return × volatility states. It expanded through higher-order chains
(Nascimento 2022), an HMM regime detector, a vol-routed hybrid,
parameter sweeps, a deep 1,616-config grid search, and a multi-asset
portfolio experiment. A skeptic review in mid-April caught the original
"Sharpe 2.15" headline as a seed-42 artifact of a buggy non-overlap
sampler; the corrected number is ~1.19 across 10 seeds. Two follow-up
experiments (absolute-granularity HOMC and Nascimento rule extraction)
closed the theoretical Markov-edge question as a negative. A 4-asset
risk-parity experiment shipped as the current recommendation. Along
the way, the project built an opinionated stack of methodology scripts
(non-overlap sampler, multi-seed eval, bootstrap CIs, permutation
tests, DSR, pristine holdout, cost sensitivity, regime ablation, etc.)
that together form a reusable adversarial-review toolkit. That toolkit
is the part of the project most likely to be useful to other quant
research repositories.

## Project layout

```
signals/
├── data/            # DataSource, DataPipeline, DataStore (parquet + SQLite)
├── features/        # returns, volatility, indicators
├── model/
│   ├── states.py          # QuantileStateEncoder, CompositeStateEncoder,
│   │                      # AbsoluteGranularityEncoder (Round-4 addition)
│   ├── composite.py       # CompositeMarkovChain (sunset)
│   ├── hmm.py             # HiddenMarkovModel (sunset)
│   ├── homc.py            # HigherOrderMarkovChain (sunset, houses
│   │                      # the sunset-warning helper)
│   ├── hybrid.py          # HybridRegimeModel (sunset internals but
│   │                      # still the BTC production path)
│   ├── trend.py           # TrendFilter, DualMovingAverage
│   ├── boost.py           # GradientBoostingModel (research)
│   ├── ensemble.py        # EnsembleModel (research)
│   ├── signals.py         # SignalGenerator
│   └── rule_signals.py    # RuleBasedSignalGenerator (Round-4 addition)
├── backtest/
│   ├── engine.py          # BacktestEngine + BTC_HYBRID_PRODUCTION constant
│   ├── portfolio.py       # Long/short portfolio w/ sizing + stops
│   ├── portfolio_blend.py # 2-asset portfolio combiner
│   ├── metrics.py         # Sharpe, DSR, CAGR, drawdown
│   ├── risk_free.py       # historical_usd_rate helper (Round-2 addition)
│   └── vol_target.py      # Vol-targeting overlay (available but REJECTED
│                          # by Round-4 sweep — use for single-asset only)
├── broker/                # Broker ABC, PaperBroker, AlpacaBroker (dry-run default)
└── cli.py                 # Typer CLI

scripts/
├── _window_sampler.py     # Shared non-overlap sampler — use this!
├── risk_parity_4asset.py  # Current production recommendation
├── portfolio_trace_180d.py  # Day-by-day 180-day portfolio trace
├── multi_seed_eval.py     # 10-seed × 6-quantile BTC validation
├── explore_improvements.py  # 144-config parameter search
├── block_bootstrap.py     # Moving-block bootstrap 95% CI
├── permutation_test.py    # Monte-Carlo null test
├── cost_sensitivity.py    # 2-D cost × deadband surface
├── trivial_baselines_btc.py  # Vol filter vs Trend vs Golden Cross vs hybrid
├── regime_ablation.py     # Component ablation
├── pristine_holdout.py    # Tier-A4 clean OOS evaluation
├── project_level_dsr.py   # Cross-tier DSR + binomial tests
├── absolute_encoder_eval.py  # Experiment 1 (Markov closure)
├── rule_based_eval.py     # Experiment 2 (Markov closure)
├── vol_target_sweep.py    # Round-4 #1 (rejected)
├── hysteresis_sweep.py    # Round-4 #4 (rejected)
├── confirm_vol_filter_winner.py  # 10-seed vol filter confirmation
├── plot_per_window.py     # PNG plots from eval parquets
└── data/                  # Persisted per-sweep results (parquet + md)
```
