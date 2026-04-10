# Signals

[![CI](https://github.com/jlgreen11/signals/actions/workflows/ci.yml/badge.svg)](https://github.com/jlgreen11/signals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-92%20passing-brightgreen.svg)](./tests)

Markov-chain market signal generator with four swappable model backends,
walk-forward backtesting, sized long/short execution, and a regime-routed
**hybrid** default that achieves **median Sharpe 2.15 across 16 random
6-month BTC windows** — the highest result in the project's history.

## ⚠ Disclaimer

This is an **experimental research project**. Nothing in this repo is
financial advice. Backtest results are historical and **do not predict
future performance** — they are especially unreliable for cryptocurrency,
which has seen multiple ~80% drawdowns in the past decade. The authors
assume no liability for losses from use of this software. Before risking
capital, conduct your own due diligence, understand the model's
limitations, and start with amounts you can afford to lose entirely.

The project is provided under the MIT License with **no warranty of any
kind, express or implied**. See [`LICENSE`](./LICENSE).

**Production scope**: BTC-USD and ^GSPC (S&P 500).

**TL;DR per asset**:

- **BTC-USD** → use the `hybrid` model (`H-Vol` routing at `vol_quantile=0.70`).
  Median Sharpe 2.15 on random-window evaluation, holdout Sharpe 2.21 on
  2023-2024 BTC bull, 0.99 on BTC bear-stress.
- **^GSPC (S&P 500)** → **use buy & hold.** No strategy in this project beats
  B&H on S&P. The Markov-chain backbone is the wrong tool for a secular-
  uptrend equity index. See `scripts/HOMC_TIER0E_BTC_SP500.md` for the full
  comparison.

## Models

| Model | What it is | Status |
|---|---|---|
| `composite` | 1st-order discrete Markov chain over a 2D (return × volatility) state grid (default 3×3 = 9 states). The Phase-1 model — solid bear-defense. | Baseline |
| `hmm` | Gaussian Hidden Markov Model (`hmmlearn`) over standardized continuous features. Hidden regimes discovered via Baum-Welch. | Research |
| `homc` | Higher-order Markov chain over quantile-binned returns, inspired by Nascimento et al. (2022). Good bull participation on BTC. | Research |
| **`hybrid`** | **Regime-routed ensemble of composite + HOMC. Default routing is vol-based: top 30% of training-vol days → composite (bear defense), rest → HOMC (bull participation). Also supports `hmm` and `blend` routing strategies.** | **BTC production default** |
| `trend` | Classic single-MA trend filter: long when close > MA(200), flat otherwise. Faber (2007). Tested on S&P — reduces drawdowns but doesn't beat B&H on Sharpe. | Research (S&P) |
| `golden_cross` | Dual-MA crossover: long when MA(50) > MA(200). Smoother than `trend` but pays more lag. Tested on S&P — strictly worse than `trend`. | Research (S&P) |

All six implement a common interface (`fit`, `predict_state`, `predict_next`,
`state_returns_`, `label`, `save`/`load`) so the engine, signal generator,
and CLI work with any of them transparently.

## Headline result — BTC-USD, 16 random 6-month windows, seed 42

The random-window evaluation is the most robust validation methodology in the
project: it samples windows across bull, bear, and chop regimes and reports
per-window performance + head-to-head counts.

| Metric | B&H | Composite | HOMC | H-Vol @ 0.70 | H-Blend |
|---|---:|---:|---:|---:|---:|
| **Median Sharpe** | 1.18 | 1.44 | 1.83 | **2.15** 🏆 | 2.06 |
| Mean Sharpe | 1.03 | 1.10 | 1.39 | 1.59 | 1.48 |
| Median CAGR | +101% | +44% | +118% | **+156%** | +141% |
| Mean Max DD | -28% | -21% | -23% | -21% | -22% |
| Sharpe capture vs oracle | — | 16% | 22% | **23%** | 21% |
| Positive CAGR windows | 9/16 | 11/16 | 10/16 | **12/16** | 13/16 |

**H-Vol @ q=0.70 is the BTC production default.** H-Blend is a close second on
the median but ~0.08 Sharpe behind on the best-case pair. See
`scripts/HOMC_TIER0F_SIZING_BLEND.md` for the full blend sweep.

## Headline result — ^GSPC, 16 random 6-month windows, seed 42

| Metric | **B&H** | Composite | HOMC@5 | H-Vol | H-Blend | Trend(200) | GCross(50,200) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean Sharpe | **1.07** | 0.70 | 1.04 | 0.46 | 0.34 | 0.60 | 0.85 |
| **Median Sharpe** | **0.77** | 0.49 | 0.66 | -0.03 | -0.40 | 0.57 | 0.54 |
| Median CAGR | **+12.2%** | +4.2% | +8.3% | -1.2% | -5.4% | +6.2% | +6.6% |
| Mean Max DD | -15.3% | -14.5% | -14.5% | -15.3% | -15.0% | **-9.4%** | -14.3% |
| Positive CAGR | — | 8/16 | 12/16 | 8/16 | 6/16 | — | — |

**Nothing beats buy & hold on S&P 500.** This has been tested four ways:

1. **Markov chains** (composite, HOMC, H-Vol, H-Blend) — all lose. Tier 0e.
2. **Vol quantile tuning** on the hybrid — no quantile rescues it. Tier 0e.
3. **Classic trend filters** (200-day MA, 50/200 golden cross) — reduce
   drawdowns as expected but give up too much return to compensate. Tier 1S.
4. **HOMC memory depth sweep** (orders 1–9) — one apparent winner at
   order=6 (seed 42, median Sharpe 0.90) failed a 4-seed robustness check:
   it beat B&H on 2/4 seeds and catastrophically underperformed on 1/4
   (seed 7: 0.23 vs B&H 1.42). Data-mining artifact. Tier 1S.

Full S&P analysis (Markov + trend + memory sweep + robustness) is in
`scripts/SP500_TREND_AND_HOMC_MEMORY.md`. Tier-0e analysis is in
`scripts/HOMC_TIER0E_BTC_SP500.md`.

**Recommendation**: hold SPY (or equivalent) directly. Don't run signals on S&P.

Why does S&P resist all the strategies in this project? The Markov
backbone assumes the underlying process has meaningfully distinct
regimes to detect. S&P has rare sharp drawdowns but spends most of its
time in a single slow-uptrend regime with nothing to detect. Trend
filters work as advertised (lower drawdowns, ~40% return cost) but the
tradeoff doesn't improve Sharpe on 6-month windows. A genuine S&P
strategy needs model classes that don't exist in this project yet —
see `SP500_TREND_AND_HOMC_MEMORY.md` "What's next for S&P" for
candidates (multi-asset portfolio construction, macro-feature regime
models, factor rotation).

## Leverage note (BTC)

`max_long` is a clean risk/return dial with **flat Sharpe across [1.0, 2.0]**:

| `max_long` | Median Sharpe | Median CAGR | Mean Max DD |
|---:|---:|---:|---:|
| 1.00 (current default) | 2.15 | +156% | -21% |
| 1.25 | 2.15 | +216% | -26% |
| **1.50** | **2.16** | **+288%** | -30% |
| 2.00 | 2.13 | +480% | -38% |

The alpha is fully expressed at `max_long=1.0`; scaling up just scales the
expression. **The default is `max_long=1.0` (conservative).** If you can
stomach a -30% drawdown profile, passing `--max-long 1.5` gives you roughly
1.85× the return for the same Sharpe. This is a personal risk-tolerance
decision, not a methodology decision — the default was deliberately left
conservative.

See `scripts/HOMC_TIER0F_SIZING_BLEND.md` for the full sizing sweep.

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
[StateEncoder: composite (2D) / quantile / continuous]
        │
        ▼
[Model: composite | HMM | HOMC | hybrid] ──── shared interface ────┐
        │                                                            │
        ▼                                                            │
[SignalGenerator: BUY/SELL/HOLD + sized target_position]             │
        │                                                            │
        ├─▶ [Portfolio] → [BacktestEngine + metrics]                 │
        └─▶ [signal next] → CLI: tomorrow's action  ◀────────────────┘
```

## Install

Python 3.11+ required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Daily workflow — tomorrow's signal

```bash
# One command. Refreshes the latest bars, retrains the hybrid on a
# rolling 1000-bar window, and prints the action for the next bar's open.
signals signal next BTC-USD

# With a specific model (composite is still available as a simpler
# fallback, HOMC and HMM for research)
signals signal next BTC-USD --model hybrid     # new default
signals signal next BTC-USD --model composite  # legacy default

# Lower level: just decode the current state from the saved model
signals signal now BTC-USD --model hybrid
```

## Full CLI

```bash
# Data
signals data fetch BTC-USD --start 2015-01-01 --interval 1d
signals data fetch ^GSPC --start 2015-01-01
signals data refresh BTC-USD                  # incremental, append-only
signals data list

# Model — train any of composite | hmm | homc | hybrid
signals model train BTC-USD --model composite
signals model train BTC-USD --model hmm --states 4
signals model train BTC-USD --model homc --states 5 --order 5
signals model inspect BTC-USD --model composite
signals model plot BTC-USD --model composite

# Backtest — H-Vol hybrid is the default
signals backtest run BTC-USD --model hybrid \
  --start 2018-01-01 --end 2024-12-31 \
  --train-window 1000 --states 5 --order 5

# Holdout validation (trailing 20% of data reserved from the sweep)
signals backtest run BTC-USD --model hybrid \
  --start 2018-01-01 --end 2024-12-31 \
  --holdout-frac 0.2

# 4-way head-to-head with the same strategy params
signals backtest compare BTC-USD --start 2018-01-01 --end 2024-12-31

# Grid search thresholds and stops, with deflated Sharpe + holdout
signals backtest sweep BTC-USD --model hybrid \
  --start 2018-01-01 --end 2024-12-31 \
  --buy-grid "10,15,20,25,30" --sell-grid "-10,-15,-20,-25,-30" \
  --stop-grid "0" --holdout-frac 0.2 --rank-by sharpe

signals backtest list
signals backtest show 1
```

## Strategy layer

The `Portfolio` is target-driven: `set_target(ts, price, fraction)` reconciles
by trading the delta. `fraction = +1.0` is 100% long, `-1.0` is 100% short,
`0` is flat. Features:

- **Sized longs and shorts** (`--max-long`, `--max-short`)
- **Per-bar stop loss** with cooldown bars (`--stop-loss`, `--stop-cooldown`)
- **Min-trade-fraction deadband** to suppress rebalance churn (`--min-trade`)
- **Hold preserves position** so trends aren't churned out by neutral signals
- **Asymmetric thresholds** — `--buy-bps` and `--sell-bps` independent
- **Risk-free rate in Sharpe** — `BacktestConfig.risk_free_rate` (annualized)
- **Deflated Sharpe on sweep output** — multi-trial correction per
  Bailey & López de Prado (2014)
- **Holdout validation** — `--holdout-frac 0.2` reserves trailing 20% from
  the sweep and re-runs the best config on the held-out portion

## Project layout

```
signals/
├── data/            # DataSource, DataPipeline, DataStore (parquet + SQLite)
├── features/        # returns, volatility, indicators
├── model/
│   ├── states.py          # QuantileStateEncoder, CompositeStateEncoder
│   ├── composite.py       # CompositeMarkovChain (1st-order, 2D states)
│   ├── hmm.py             # HiddenMarkovModel (Gaussian HMM)
│   ├── homc.py            # HigherOrderMarkovChain
│   ├── hybrid.py          # HybridRegimeModel (production default)
│   └── signals.py         # SignalGenerator + SignalDecision
├── backtest/        # Portfolio (long/short/sized/stops), Engine, metrics
├── broker/          # Broker ABC + PaperBroker (live execution stub)
└── cli.py           # Typer entry point
```

## Tests

```bash
pytest --cov=signals
```

**92 tests** across:

- `test_lookahead.py` — strict no-lookahead regression. Asserts equity
  curves up to bar N are bit-identical regardless of how much future data
  is in the input. Covers composite, HOMC, HMM, and hybrid (both routing
  strategies).
- `test_hybrid.py` — 17 tests for the hybrid: fit/predict round-trip,
  routing validation, both `vol` and `blend` strategies, lookahead
  regression, save/load.
- `test_trend.py` — 16 tests for TrendFilter and DualMovingAverage:
  fit/predict, above/below-MA detection, lookahead regression on both
  models, engine integration.
- `test_backtest.py`, `test_holdout.py` — engine, portfolio, metrics,
  deflated Sharpe, risk-free rate.
- `test_composite.py`, `test_hmm.py`, `test_homc.py`, `test_signals.py`,
  `test_states.py`, `test_data.py` — component-level tests.

CI runs `ruff check` + `pytest --cov=signals` on Python 3.11 and 3.12 on
every push and PR. See `.github/workflows/ci.yml`.

## The investigation (chronological, with result docs)

The current defaults are the output of a ~2-day investigation that went
through multiple dead ends. Each tier has a pinned result doc.

| Tier | Question | Result | Doc |
|---|---|---|---|
| **0** | Does HOMC at order=7 (Nascimento paper default) work on BTC walk-forward? | **No.** Broken — 78k-cell 7-tuple table can't be filled in 252-bar windows. All 25 sweep configs DSR=0, 8 trades in 5 years, holdout collapses to 1 trade. | `scripts/HOMC_ORDER7_RESULTS.md` |
| **0a** | Does HOMC work with order=5 and a 1000-bar training window? | **Surprise** — in-sample Sharpe 0.42 but holdout Sharpe 1.76 on BTC 2023-2024. 4× Sharpe expansion on the same data composite couldn't exploit. Inconclusive without more evidence. | `scripts/HOMC_ORDER5_W1000_RESULTS.md` |
| **0b** | Is HOMC@5/w1000 a real edge or period-cherry-picked? | **Regime-biased.** Random-window eval: HOMC wins 11/16 on BTC. But fails on the 30% holdout bear stress (Sharpe 1.76 → 0.48) and catastrophically on ETH. HOMC is a bull-regime specialist. | `scripts/HOMC_TIER0B_COMPREHENSIVE.md` |
| **0c** | Can a regime-routed hybrid combine composite's bear defense with HOMC's bull participation? | **Yes — with vol-based routing, not HMM.** HMM routing whipsaws on ambiguous regimes (median Sharpe 0.37 — worse than any single model). Vol-based routing produces median Sharpe 1.92 — the first model to beat both singles. | `scripts/HOMC_TIER0C_HYBRID_RESULTS.md` |
| **0e** | Tune `hybrid_vol_quantile`, add continuous blending, add S&P 500. | BTC optimum is q=**0.70** (not 0.75) — median Sharpe **2.15**. Continuous blend at default ramp scores 2.06. **S&P 500 finding: no strategy beats B&H.** | `scripts/HOMC_TIER0E_BTC_SP500.md` |
| **0f** | Tune sizing and blend ramp parameters. | **Sharpe plateau at 2.15.** `target_scale_bps` is inert (magnitude saturates at max_long). `max_long` is a clean risk/return dial but not a Sharpe lever. Best blend pair (0.40, 0.90) scores 2.07 — loses to hard switch. | `scripts/HOMC_TIER0F_SIZING_BLEND.md` |
| **1S** | Can anything beat B&H on S&P 500? Tests classic trend filters (200-day MA, 50/200 golden cross) AND a HOMC memory-depth sweep (orders 1-9) with 4-seed robustness check. | **No.** Trend filters reduce drawdowns ~40% but cost ~40% of return. HOMC order=6 looked promising on seed 42 (median Sharpe 0.90 vs B&H 0.77) but failed 4-seed robustness (2/4 seeds beat B&H, 1 catastrophic miss at 0.23 vs 1.42). Buy & hold remains the S&P recommendation. | `scripts/SP500_TREND_AND_HOMC_MEMORY.md` |
| **2** | Deep multi-dimensional BTC sweep: 1,616 backtests across 5 hyperparameter dimensions (HOMC order×states, composite grid/window/alpha, hybrid quantile×leverage, retrain frequency, buy/sell thresholds). All raw data saved to parquet. | **No parameter tweak robustly improves baseline.** Apparent seed-42 winners (sell_bps=-20 at 2.40 Sharpe vs 2.15 baseline) all failed 4-seed robustness. On average across seeds, the "winners" are actively worse than baseline (0.94 vs 1.00). **Production defaults are at a seed-robust plateau.** | `scripts/BTC_DEEP_SWEEP_RESULTS.md` |
| **2p** | Multi-asset portfolio: BTC H-Vol hybrid + S&P B&H at various weights with daily or window rebalancing. Tests the "diversification lunch" hypothesis. | **Soft win.** 40/60 BTC/SP with daily rebalancing scored median Sharpe **2.44** at seed 42 and averaged **1.16** across 4 seeds — a +16% improvement over BTC-alone (1.00). Beats baseline on 3/4 seeds (loses at seed 100 where BTC landed on strong periods). First "genuine alpha" in the project that isn't from parameter tuning. | `scripts/BTC_SP500_PORTFOLIO_RESULTS.md` |

Earlier foundational work:

- `scripts/RANDOM_WINDOW_EVAL.md` — the original 16-window evaluation
  methodology, composite vs buy & hold vs perfect-foresight oracle
  (before hybrid existed).
- `IMPROVEMENTS.md` — the forward-looking roadmap with completed items
  marked and future candidates listed.

## Current defaults (what ships in `BacktestConfig`)

| Field | Default | Notes |
|---|---|---|
| `model_type` | `"composite"` | Factory default; `hybrid` is recommended via CLI |
| `train_window` | `252` | For composite; hybrid overrides to 1000 for HOMC component |
| `retrain_freq` | `21` | ~1 month |
| `return_bins × vol_bins` | `3 × 3` | 9 composite states |
| `vol_window` | `10` | Rolling volatility window |
| `laplace_alpha` | `0.01` | Composite smoothing |
| `buy_threshold_bps` | `25` | Buy when E[r_next] ≥ 25 bps |
| `sell_threshold_bps` | `-35` | Sell when E[r_next] ≤ -35 bps |
| `target_scale_bps` | `20` | Effectively inert for HOMC magnitudes |
| `max_long` | `1.0` | **Conservative. Pass 1.5 for ~1.85× return at same Sharpe.** |
| `allow_short` | `False` | BTC's secular uptrend punishes shorts |
| `stop_loss_pct` | `0.0` | Empirically unhelpful — sell signal exits faster |
| `hybrid_routing_strategy` | `"vol"` | `"vol"` (default), `"hmm"`, or `"blend"` |
| `hybrid_vol_quantile` | `0.70` | Tuned from sweep (was 0.75) |
| `hybrid_blend_low` | `0.50` | H-Blend ramp low |
| `hybrid_blend_high` | `0.85` | H-Blend ramp high |

## Roadmap

Active improvements (see `IMPROVEMENTS.md` for the full list):

- [x] Data pipeline (yfinance + parquet/SQLite)
- [x] Composite Markov chain
- [x] Hidden Markov Model backend
- [x] Higher-order Markov chain backend
- [x] Walk-forward backtest with lookahead regression tests
- [x] Long/short portfolio with sizing and stop losses
- [x] Holdout validation + deflated Sharpe
- [x] `signal next` daily workflow
- [x] `backtest sweep` grid search with holdout
- [x] CI workflow (pytest + ruff on Python 3.11/3.12)
- [x] HybridRegimeModel with vol and blend routing
- [x] Vol quantile tuning (BTC optimum q=0.70)
- [x] Sizing sweep (Sharpe plateau at 2.15)
- [x] BTC + S&P 500 multi-asset scope (BTC hybrid wins; S&P use B&H)
- [ ] Multi-asset portfolio construction (BTC hybrid + SPX B&H + treasuries)
- [ ] Macro features in composite (VIX, yield curve, BTC dominance)
- [ ] Absolute-granularity encoder for composite (Nascimento-style bins)
- [ ] Time-varying hyperparameters (recheck optimum q across epochs)
- [ ] Real broker integration (PaperBroker → Alpaca/Coinbase)
- [ ] Hourly / intraday bars
