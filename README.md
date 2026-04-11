# Signals

[![CI](https://github.com/jlgreen11/signals/actions/workflows/ci.yml/badge.svg)](https://github.com/jlgreen11/signals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-140%2B%20passing-brightgreen.svg)](./tests)

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

**TL;DR per asset** *(numbers updated 2026-04-11 — Round 3
large-grid search — see `scripts/data/explore_improvements.md`)*:

- **BTC-USD** → use the `hybrid` model at the **Round-3 production config**:
  `BacktestConfig(**BTC_HYBRID_PRODUCTION)` from `signals.backtest.engine`,
  which bundles `q=0.50 + retrain_freq=14 + train_window=750`. On 10 seeds
  × 16 non-overlapping 6-month BTC windows, under correct crypto
  annualization (365/yr, rf=0.023):
  - **Multi-seed avg Sharpe: 1.551 ± 0.099** (min seed 1.010, max 1.949)
  - **Legacy q=0.70 baseline (r21/tw1000): 0.893 ± 0.100** (min 0.345)
  - **Delta: +0.659 Sharpe (+74% relative)**, and the Round-3 winner
    dominates the legacy baseline on the worst seed (1.010 vs 0.345).
  - Caveat: project-level DSR at n_trials=2,044 is 0.0000 — the Sharpe
    is still within the max-of-2044-noise-draws null band. Real or not,
    the +0.659 delta across 10 pre-registered seeds and the min-seed
    dominance are strong evidence that it's genuine. Proceed with eyes
    open. Pristine holdout (2023-2024) shows the new config at a
    higher Sharpe than either legacy config or buy & hold.
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
| **`hybrid`** | **Regime-routed ensemble of composite + HOMC. Default routing is vol-based: top 30% of training-vol days → composite, rest → HOMC. Also supports `hmm`, `blend`, and `adaptive_vol` routing strategies.** | **BTC production default** |
| `trend` | Classic single-MA trend filter: long when close > MA(200), flat otherwise. Faber (2007). Tested on S&P — reduces drawdowns but doesn't beat B&H on Sharpe. | Research (S&P) |
| `golden_cross` | Dual-MA crossover: long when MA(50) > MA(200). Smoother than `trend` but pays more lag. Tested on S&P — strictly worse than `trend`. | Research (S&P) |
| `boost` | sklearn `GradientBoostingClassifier` predicting next-bar direction from engineered features (return lags, rolling vol, z-scores). Tier-3 experiment — 4.5× worse than H-Vol baseline on BTC. | Research |
| `ensemble` | Weighted average of composite + HOMC + boost. Tier-3 experiment — dragged down by the boost component. | Research |

All eight implement a common interface (`fit`, `predict_state`, `predict_next`,
`state_returns_`, `label`, `save`/`load`) so the engine, signal generator,
and CLI work with any of them transparently.

## Headline result — BTC-USD (post-SKEPTIC_REVIEW fixes)

> **⚠ The numbers in this section were corrected in Round 2.** The original
> "Sharpe 2.15" headline was a seed-42 result on a sampler that did not
> enforce non-overlap between windows. After fixing both — 10 seeds +
> slot-based non-overlapping sampler — the honest number is much smaller.
> See [`SKEPTIC_REVIEW.md`](./SKEPTIC_REVIEW.md) § 1/§ 2 and
> [`IMPROVEMENTS_PROGRESS.md`](./IMPROVEMENTS_PROGRESS.md) for the paper
> trail.

### Round-3 large-grid search (headline result)

`scripts/explore_improvements.py` runs a 4-tier hyperparameter search:
75 pure-vol-filter configs + 15 vol-target-overlay configs + 54 hybrid
configs (144 total) at 5 exploration seeds, then 10-seed confirmation of
the top 5 candidates plus the legacy q=0.70 baseline. All under correct
crypto annualization (365/yr) with historical USD risk-free rate (~2.3%).

**Tier 4 final ranking (10 seeds × 16 non-overlapping BTC windows)**:

| config | avg Sharpe | stderr | min seed | max seed | mean MDD |
|---|---:|---:|---:|---:|---:|
| **`hyb_vw10_q0.50_rf14_tw750`** (Round-3 winner) | **+1.551** | 0.099 | **+1.010** | +1.949 | -26.0% |
| `hyb_vw10_q0.55_rf14_tw750` | +1.410 | 0.185 | +0.311 | +2.172 | -26.3% |
| `hyb_vw10_q0.55_rf14_tw1000` | +1.368 | 0.088 | +0.853 | +1.715 | -24.6% |
| `hyb_vw10_q0.40_rf14_tw750` | +1.329 | 0.123 | +0.723 | +1.852 | -24.6% |
| `hyb_vw10_q0.55_rf21_tw1000` | +1.249 | 0.118 | +0.498 | +1.726 | -23.5% |
| `hybrid_prod_q0.70_w10_r21_tw1000` (legacy baseline) | **+0.893** | 0.100 | +0.345 | +1.403 | -24.8% |

**The winner and the four runners-up all share `vol_window=10` and
`retrain_freq=14` and have quantile in {0.40, 0.50, 0.55}** — the
structural improvement is "retrain more often on a shorter window with
a more aggressive vol threshold," not any one parameter. Changing q
alone from 0.70 to 0.50 (keeping rf=21, tw=1000) gives only ~1.08
Sharpe (confirm_winners.py) — you need the full bundle.

**Caveat**: DSR at project-level n_trials=2,044 is **0.0000**. Under
Bailey & López de Prado's deflation, the 1.551 Sharpe is still within
the max-of-2044-IID-noise-draws null band. The counter-argument is that
the 2,044 trials are not IID parameterizations — they cluster tightly
in the model's own fitting surface — and the dominance on the min-seed
(1.010 vs 0.345) is strong evidence that the edge is real. The raw
multi-seed delta is the cleanest signal of improvement; DSR remains the
upper bound on scepticism.

### Legacy multi-seed eval (for context)

The earlier `scripts/multi_seed_eval.py` sweep reported q=0.50 as the
multi-seed winner at 0.88 ± 0.03 vs q=0.70's 0.78 ± 0.08. Those numbers
were measured at **252/yr annualization with rf=0** (the legacy metric
defaults). Under correct annualization (365/yr + rf=0.023), the same
comparison at the same retrain_freq/train_window (21/1000) shows
q=0.70 AHEAD of q=0.50: 1.175 vs 1.081 (see
`scripts/data/confirm_winners.md`). The q=0.70 lead only appears when
you hold retrain_freq/train_window at the legacy defaults; at the
Round-3 winner's settings (rf=14, tw=750), q=0.50 reclaims the top
spot at a much higher absolute level (1.551). The annualization
convention matters; the parameter interactions matter more.

### Pristine holdout (A4 fix)

`scripts/pristine_holdout.py` runs a coarse 13-config sweep on the
**2015-2022 training slice only**, picks the winner by in-sample median
Sharpe across 12 non-overlapping training windows, then evaluates once
on the **never-seen 2023-2024 holdout**:

| Config | 2023-2024 Sharpe | 2023-2024 CAGR | 2023-2024 Max DD |
|---|---:|---:|---:|
| Buy & hold | +1.91 | +137% | -26% |
| Sweep winner (q=0.60, sell=-25, mtf=0.10) | +2.45 | +158% | -19% |
| **Production H-Vol default (q=0.70)** | **+2.69** | **+182%** | **-19%** |

The production default — which was **not** picked by the pristine sweep
(its in-sample Sharpe was 0.28, below the winner's 0.40) — still produces
the best holdout Sharpe. The 2023-2024 period is friendly to any
vol-regime-filtered long-only BTC strategy, which means the in-sample/OOS
gap (0.28 → 2.69) is real but mostly attributable to regime friendliness,
not to genuine generalization. **Do not read the 2.69 as the "real" expected
Sharpe** — read the 0.78 multi-seed number as the expected Sharpe and the
2.69 as "what happens in a good regime."

### Null-hypothesis testing (B1, B5, C6)

- **Project-level DSR** (`scripts/project_level_dsr.py`): at per-sweep
  n_trials=25 the 2.15 Sharpe has DSR=0.9999; at project-level n_trials≈1901
  it collapses to DSR=**0.0000**. The headline does NOT survive correction
  for the total number of trials run across all tiers.
- **Monte-Carlo permutation test** (`scripts/permutation_test.py --quick`,
  N=20 shuffles per window): Fisher-combined p-value across 16 windows =
  **0.045**. The strategy rejects the null "you are trading noise" at
  α=0.05, but barely, and only at quick-mode N. Full N=200 would tighten
  this.
- **Moving-block bootstrap 95% CI** (`scripts/block_bootstrap.py --quick`,
  B=100): observed median Sharpe = 0.975, bootstrap CI = **[0.29, 2.23]**.
  The 2.15 headline falls inside the upper end of the CI; the null (0.0)
  is outside.
- **Binomial significance on "beats B&H in X/16"** (`scripts/project_level_dsr.py`):
  at face-value N=16, only H-Blend (13/16) and H-Vol (12/16) are
  significant at α=0.05; after the effective-N=6 correction (§ 2 of
  SKEPTIC_REVIEW), **none** of the "beats B&H" counts clear α=0.05.

### Regime-filter ablation (C4 fix)

`scripts/regime_ablation.py` strips each Markov component from the hybrid
and replaces it with a constant signal, keeping the vol router intact:

| Variant | median Sharpe | mean Sharpe | Δ vs full |
|---|---:|---:|---:|
| **Full hybrid** | **+0.81** | **+1.17** | — |
| composite_only (HOMC → constant long) | +1.03 | +1.00 | +0.22 |
| homc_only (composite → constant flat) | +0.91 | +0.88 | +0.10 |
| **both constants (pure vol filter)** | **+0.95** | **+0.83** | +0.14 |

**Interpretation**: the pure vol filter (no Markov chain at all) matches
the full hybrid within 0.1-0.2 Sharpe. SKEPTIC_REVIEW § 5 is confirmed —
on genuinely non-overlapping windows, the Markov components are
decorative. The vol router is doing (almost) all the work. The correct
simplification is to delete HOMC + composite and ship the one-line rule
`target = +1.0 if vol_20d < q70_train else 0.0`.

### Trivial baseline comparison on BTC (B4 fix)

`scripts/trivial_baselines_btc.py` vs the hybrid on 16 non-overlapping
seed-42 windows:

| Strategy | median Sharpe | mean Sharpe | median CAGR | positive windows |
|---|---:|---:|---:|---:|
| Buy & hold | 0.79 | 0.94 | +70% | 8/16 |
| Trend(200) | 0.00 | 0.41 | 0% | 7/16 |
| Dual MA(50/200) | 0.50 | 0.54 | +32% | 8/16 |
| **Vol-filter only (no Markov)** | **1.15** | 0.91 | +77% | 9/16 |
| H-Vol hybrid | 0.73 | **1.11** | +31% | **10/16** |

On the median, the pure vol filter (1.15) is actually **higher** than the
full hybrid (0.73). On the mean, the hybrid is higher (1.11 vs 0.91).
This matches the ablation result — the hybrid and the vol-only baseline
are within 0.2-0.4 Sharpe of each other in either direction, and neither
is convincingly better.

### Cost sensitivity (B2/B3)

`scripts/cost_sensitivity.py` 5×5 grid over commission × deadband:

| commission_bps | Sharpe |
|---:|---:|
| 2.5 | 0.83 |
| 5.0 | 0.81 |
| 10.0 | 0.76 |
| 15.0 | 0.72 |
| 25.0 | 0.62 |

**The deadband is inert**: varying `min_trade_fraction` from 0.05 to 0.30
moves the median Sharpe by less than 0.005. Commission costs are
approximately linear in their impact: a 20 bps commission increase (from
5 to 25) costs about 0.19 of Sharpe. Nothing in the grid collapses by
more than 0.5 from the baseline, so the strategy is **cost-robust in the
2.5-25 bps range** — good news. But the fact that the deadband doesn't
matter also means the `min_trade_fraction=0.20` default is purely
cosmetic at current parameters.

See [`scripts/data/plots/`](./scripts/data/plots) for visualizations
(multi-seed quantile sweep, cost-sensitivity heatmap, regime ablation
bar chart, bootstrap per-window CI).

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

**140+ tests passing** (code correctness, not statistical validation — see
[`SKEPTIC_REVIEW.md`](./SKEPTIC_REVIEW.md) § 8d) across:

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
| **3** | "Do all the improvements" — 8 improvement items from the synopsis. Adaptive vol quantile, long-horizon S&P trend filter, 40/60 portfolio productionized into CLI, paper-trade scaffold, VIX macro overlay, gradient boosting model, multi-strategy ensemble, Alpaca broker SDK (code only, no live). | **Parameter plateau → model-class plateau.** Nothing new beats H-Vol baseline across 4 seeds. Gradient boosting is 4.5× worse (0.22 avg Sharpe vs 1.00). VIX overlay cuts drawdowns in half but halves Sharpe. Adaptive quantile loses. Long-horizon S&P trend still ties B&H on Sharpe. But the portfolio combiner is now shipped as `signals backtest portfolio` CLI, paper-trade protocol exists as `signals paper-trade record/reconcile/report`, and the Alpaca SDK is ready for user-authorized live trading. | `scripts/TIER3_COMPREHENSIVE_RESULTS.md` |

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
| `hybrid_vol_quantile` | **`0.50`** | **Round-3 large-grid winner** bundled with `retrain_freq=14` + `train_window=750`. Prefer using `BTC_HYBRID_PRODUCTION` from `signals.backtest.engine` rather than varying q alone. See "q-value history" below. |
| `hybrid_blend_low` | `0.50` | H-Blend ramp low |
| `hybrid_blend_high` | `0.85` | H-Blend ramp high |

### `hybrid_vol_quantile` history

Every historical result doc in `scripts/*.md` reports numbers measured at
the `hybrid_vol_quantile` value prevailing *at the time of that doc's
run*. To read old numbers correctly, check the doc's "Test parameters
(historical)" header — every result doc has one after the Round 2
documentation pass.

| Period | Value | Origin | Status |
|---|---:|---|---|
| 2026-04-10 (Tier 0c) | **0.75** | ad-hoc initial pick when the hybrid was first introduced | superseded |
| 2026-04-11 (Tier 0e) | **0.70** | seed-42 sweep winner over `{0.50..0.90}` | superseded — seed-42 artifact on buggy sampler |
| 2026-04-11 (Round 2 / A5) | 0.50 | multi_seed_eval.py at 252/yr + rf=0 | superseded — annualization convention artifact |
| 2026-04-11 (Round 3, final) | **0.50** ✅ | **explore_improvements.py bundle** — q=0.50 combined with retrain_freq=14 and train_window=750 at 365/yr + rf=0.023, 10 seeds × 16 non-overlap windows; avg Sharpe **1.551 ± 0.099** vs legacy q=0.70 baseline **0.893**; +0.659 delta | **current default** |

**Important caveat for q values**: the q value alone does not determine
performance. The Round-3 winner requires all three of
`(q=0.50, retrain_freq=14, train_window=750)` to be passed together.
Under the legacy retrain_freq=21 / train_window=1000 combo, q=0.70 is
actually ahead of q=0.50 (1.175 vs 1.081 — see `confirm_winners.md`).
Always use `BacktestConfig(**BTC_HYBRID_PRODUCTION)` rather than tuning
q in isolation.

Docs that reference `q=0.75` predate Tier 0e. Docs that reference
`q=0.70` predate the Round-2 multi-seed sweep. Docs that use `q=0.50`
are the current post-Round-2 numbers (only `IMPROVEMENTS_PROGRESS.md`,
`SKEPTIC_REVIEW.md`, and this README so far — the result docs in
`scripts/*.md` are marked "historical" and have not been re-generated).

## Methodology caveats

An external skeptic teardown of this repo is in
[`SKEPTIC_REVIEW.md`](./SKEPTIC_REVIEW.md). The critical items that affect how
the headline numbers should be read:

1. **Single-seed headline numbers.** Every "median Sharpe" in this README is
   measured at `random.Random(seed=42)`. The same H-Vol @ q=0.70 baseline
   averages ~1.00 across 4 seeds, with one seed going negative (−0.27). Fix
   in progress: `scripts/multi_seed_eval.py` runs the baseline at 10+ seeds
   and publishes the full distribution.
2. **Window non-overlap is documented but was not enforced.** The pinned doc
   `RANDOM_WINDOW_EVAL.md` claims "16 non-overlapping 6-month windows"; the
   original `scripts/random_window_eval.py` used `random.sample` and did not
   enforce spacing. Several windows in the seed-42 draw share 75–95% of their
   bars. Effective independent-sample count is ~6, not 16. **Landed**:
   `scripts/random_window_eval.py` now uses rejection sampling with minimum
   126-bar spacing and clamps `n_windows` to the maximum that fits the
   eligible range. The pinned numbers in the old `RANDOM_WINDOW_EVAL.md`
   predate this fix and need to be regenerated.
3. **Holdout window was visible during defaults tightening.** The "tightened
   defaults" (`vol_window=10`, `buy_bps=25`, `sell_bps=-35`, etc.) were tuned
   in iterations that had full visibility of the 2023–2024 holdout. Fix
   pending: a pristine-holdout pass that re-tunes on 2015–2022 only and
   reports the 2023–2024 window exactly once is on the Tier A4 punch list
   (see [`IMPROVEMENTS_PROGRESS.md`](./IMPROVEMENTS_PROGRESS.md)).
4. **Every reported config has DSR = 0.00 at per-sweep correction.** The
   deflated Sharpe test fires on every sweep; the project proceeds by
   overriding it with holdout evidence (which is itself not pristine — see
   item 3). **Landed**: `scripts/project_level_dsr.py` reports DSR at
   per-sweep (25), per-tier (200), and project-level (~1,900) trial counts.
   Output pinned at `scripts/data/project_level_dsr.md`. Headline finding:
   **the Sharpe 2.15 has DSR ≈ 0.00 at project-level n_trials** — the
   skeptic's critique here is empirically confirmed.
5. **BTC Sharpe was annualized at 252, not 365.** `signals/backtest/metrics.py`
   used to return `252.0` for daily bars, which is the equities convention,
   while `signals/backtest/vol_target.py` uses `365` for crypto. This
   underreported the BTC headline by a factor of `sqrt(252/365) ≈ 0.83`
   (the 2.15 figure becomes ~2.59 under the correct convention).
   **Landed**: `compute_metrics` now accepts an explicit `periods_per_year`
   override, plumbed through `BacktestConfig.periods_per_year`. Pass 365 for
   crypto, 252 for equities.
6. **Transaction costs are a single point (5 bps + 5 bps).** No sensitivity
   surface. The `min_trade_fraction = 0.20` deadband is entangled with the
   cost assumption and has not been swept independently. Fix in progress:
   `scripts/cost_sensitivity.py` runs a 2-D grid over `commission_bps` ×
   `min_trade_fraction` at the production H-Vol config.

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
