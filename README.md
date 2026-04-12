# Signals

[![CI](https://github.com/jlgreen11/signals/actions/workflows/ci.yml/badge.svg)](https://github.com/jlgreen11/signals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-254%20passing-brightgreen.svg)](./tests)

A quant research project that spent 10 rounds of adversarial review
testing every approach we could find — Markov chains, trend filters,
vol-regime routing, hybrid ensembles, pairs trading — and discovered
that **the signal class matters more than the model complexity**. Three
strategies beat buy-and-hold on US equities; the original Markov
approach was not one of them.

## ⚠ Disclaimer

Experimental research project. Nothing here is financial advice.
Backtest results are historical and **do not predict future
performance**. MIT-licensed with **no warranty of any kind**. See
[`LICENSE`](./LICENSE). Conduct your own due diligence and start with
amounts you can afford to lose entirely.

---

## What works (and what doesn't)

After testing 5 model classes across 20 major US stocks (top-15 SP500 +
top-10 NASDAQ), here's the honest scorecard. All numbers are trailing
7-year (2019-04-01 → 2026-04-01), 252/yr equity annualization,
risk-free rate ~2.3%.

### Strategies that BEAT buy-and-hold

| Strategy | Sharpe | CAGR | Max DD | Signal class |
|---|---:|---:|---:|---|
| **Cross-sectional momentum** | **+1.030** | **+34.7%** | −34.9% | Relative stock ranking |
| **PEAD earnings drift** | **+0.960** | **+23.4%** | −26.6% | Fundamental (earnings surprise) |
| **TSMOM multi-asset** | **+0.947** | +9.8% | **−8.9%** | Macro asset-class trends |

### Strategies that LOST to buy-and-hold

| Strategy | Sharpe | CAGR | Max DD | Why it failed |
|---|---:|---:|---:|---|
| SP500 buy-and-hold | +0.583 | +12.6% | −33.9% | *(benchmark)* |
| Markov hybrid (best of 5 variants) | avg −0.18 delta | — | — | Wrong signal class: daily price/vol patterns on single stocks contain no exploitable edge at retail |
| Trend filter (200-day MA) | 0/20 stocks | — | — | Individual stocks don't trend-follow like asset classes do |
| Golden cross (50/200 MA) | 0/20 stocks | — | — | Same failure mode as trend filter |
| Pairs trading (stat arb) | −0.474 | −5.8% | −47.2% | Edge too small for costs; mega-cap cointegration is unstable |

### The meta-lesson

The project spent 10 rounds trying to extract alpha from **time-series
patterns on single-stock daily bars** using Markov chains and trend
filters. That approach failed across 20 stocks × 5 model variants =
100 comparisons, with a **5% win rate** against buy-and-hold.

The three strategies that work all use a **different information axis**:

1. **Momentum** — *which stock is winning relative to others?*
   (cross-sectional, not time-series)
2. **TSMOM** — *is this entire asset class trending?*
   (macro, not single-stock)
3. **PEAD** — *did the company beat earnings expectations?*
   (fundamental, not price-pattern)

**The signal class was the problem all along, not the evaluation
methodology or the model complexity.**

---

## The three winning strategies in detail

### 1. Cross-sectional momentum — the standout winner

[`signals/model/momentum.py`](./signals/model/momentum.py) |
[`scripts/CROSS_SECTIONAL_MOMENTUM_RESULTS.md`](./scripts/CROSS_SECTIONAL_MOMENTUM_RESULTS.md)

Jegadeesh & Titman (1993): rank 20 stocks by trailing 12-month return
(excluding the most recent month), go long the top 5, rebalance
monthly. 5+5 bps transaction costs.

**Trailing 7 years** ($10,000 initial):

| | End value | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| **Momentum Top-5** | **$80,665** | **+34.7%** | **+1.030** | −34.9% |
| Equal-weight 20 B&H | $61,994 | +29.8% | +0.961 | −43.4% |
| SP500 B&H | $22,933 | +12.6% | +0.583 | −33.9% |

**First strategy class in the entire project to beat buy-and-hold on
equities.** Beats both benchmarks on Sharpe AND CAGR, with shallower
drawdown than equal-weight B&H. The single most-replicated anomaly in
the academic finance literature, and it worked here on the simplest
possible implementation.

Multi-seed validation (5 seeds × 12 windows): momentum's avg CAGR is
+47.4% vs EW-20 B&H's +29.4%. Momentum wins 47% of 6-month windows on
Sharpe — not dominant in every sub-period, but the compounding over
longer horizons makes the difference.

### 2. Post-Earnings Announcement Drift (PEAD)

[`signals/model/pead.py`](./signals/model/pead.py) |
[`signals/data/earnings.py`](./signals/data/earnings.py) |
[`scripts/PEAD_RESULTS.md`](./scripts/PEAD_RESULTS.md)

Buy stocks after positive earnings surprises (≥3% beat), hold for 30
days. Uses yfinance earnings data with a YoY EPS growth fallback when
consensus estimates are unavailable.

**Trailing 5 years** (2021-04-01 → 2026-04-01, $10,000 initial):

| Config | Sharpe | CAGR | Max DD | Trades | Win rate |
|---|---:|---:|---:|---:|---:|
| **PEAD 3% / 30-day hold** | **+0.960** | **+23.4%** | −26.6% | 234 | 59.4% |
| PEAD 5% / 90-day hold | +0.790 | +20.6% | −28.6% | 186 | 63.4% |
| SP500 B&H | +0.535 | +10.3% | −25.4% | — | — |
| EW-20 B&H | +0.929 | +20.5% | −29.4% | — | — |

The drift concentrates in the first month — 30-day holds consistently
outperform 60- and 90-day holds. Higher surprise thresholds (10%)
produce higher win rates (66%) but fewer trades (104). The 3%/30-day
variant has the best overall Sharpe.

### 3. Time-Series Momentum (multi-asset trend-following)

[`signals/model/tsmom.py`](./signals/model/tsmom.py) |
[`scripts/TSMOM_MULTI_ASSET_RESULTS.md`](./scripts/TSMOM_MULTI_ASSET_RESULTS.md)

Moskowitz, Ooi & Pedersen (2012): apply trend signals across 8 asset
classes (BTC, SP500, TLT, GLD, USO, UUP, EFA, IEF), weight by
inverse realized volatility. Combined signal averages 1-month, 3-month,
and 12-month lookbacks.

**Trailing 7 years** ($10,000 initial):

| | Sharpe | CAGR | Max DD | Calmar |
|---|---:|---:|---:|---:|
| **TSMOM Combined** | **+0.947** | +9.8% | **−8.9%** | **1.10** |
| EW multi-asset B&H | +0.881 | +13.6% | −22.8% | 0.60 |
| SP500 B&H | +0.583 | +12.6% | −33.9% | 0.37 |

TSMOM's value is **risk-adjusted**: it beats SP on Sharpe (+0.364) and
dramatically on drawdown (−8.9% vs −33.9%) but trails on raw CAGR.
Best used as a **defensive overlay** combined with a B&H equity core,
not as a standalone return maximizer.

---

## What we tried and sunset

### Markov chain models (sunset)

All four Markov model classes — `composite`, `homc`, `hmm`, `hybrid` —
are **sunset**. Direct instantiation emits a `DeprecationWarning`. They
are retained (not deleted) because `HybridRegimeModel` internally
composes them for the BTC leg of legacy portfolio experiments.

The closure came from:
- **100 (ticker × model) tests** across 20 major stocks: 5% win rate vs B&H
- **AbsoluteGranularityEncoder experiment**: zero trades across 9 pre-registered configs
- **Rule-based signal generator experiment**: +0.567 Sharpe, below the 1.30 materiality threshold
- **Regime ablation**: pure vol filter matched the full hybrid within 0.2 Sharpe

See [`SKEPTIC_REVIEW.md`](./SKEPTIC_REVIEW.md) for the external
teardown that drove 5 rounds of corrections, and
[`scripts/MULTI_STOCK_ALGO_EVAL.md`](./scripts/MULTI_STOCK_ALGO_EVAL.md)
for the definitive 20-stock × 5-model failure.

### 4-asset portfolio (context-dependent)

The equal-weight BTC/SP/TLT/GLD basket from earlier rounds
(see [`scripts/TRAILING_7Y_VIEW.md`](./scripts/TRAILING_7Y_VIEW.md))
produced strong trailing numbers (+21.3% CAGR, Sharpe 1.106 over 7
years) but those numbers are **dominated by BTC's 2019–2024 bull run**.

Under conservative forward BTC assumptions:

| BTC forward CAGR | 4-asset basket CAGR | vs SP alone |
|---:|---:|---:|
| 0% (bear) | 4.0% | loses to SP |
| 15% | 7.8% | roughly ties SP |
| 20% | 9.0% | barely beats SP |

The basket only beats SP if you believe BTC's forward CAGR exceeds
~15%. The structural diversification benefits (drawdown blunting, ruin
insurance) are real regardless, but they are risk reduction, not return
enhancement.

---

## Install

### Option 1 — Install from GitHub (any machine, no clone needed)

```bash
pip install git+https://github.com/jlgreen11/signals.git
```

### Option 2 — Clone and install locally (for development)

```bash
git clone https://github.com/jlgreen11/signals.git
cd signals
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Set up API keys

Create a `.env` file in the project root (this file is gitignored and
will never be committed):

```bash
# .env
ALPACA_API_KEY=PK...your key...
ALPACA_SECRET_KEY=...your secret...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Get your Alpaca keys at [alpaca.markets](https://alpaca.markets) →
sign up → Trading API → Paper Trading → API Keys.

The `.env` file auto-loads when you run any `signals auto` command.
Without it, the local paper broker works fine — Alpaca is only needed
for real paper trading with live market fills.

### Fetch price data

```bash
# Fetch SP500 constituents via Alpaca data API (fast, ~500 stocks in 2 min):
signals data fetch ^GSPC --start 2015-01-01

# Or fetch individual tickers via yfinance:
signals data fetch AAPL --start 2015-01-01
signals data fetch NVDA --start 2015-01-01
# ... etc
```

The full SP500 dataset (~498 tickers) can be bulk-fetched via the
Alpaca data API if you have credentials set up — see
`scripts/universe_analysis.py` for the batch fetch code.

### Run the daily automation

```bash
# Generate signals + place paper trades on Alpaca:
signals auto trade --broker alpaca

# Or use the local paper broker (no API keys needed):
signals auto trade

# View positions, performance, signal history:
signals auto positions --broker alpaca
signals auto performance --broker alpaca
signals auto history LITE --days 30
```

### Automate (optional cron job)

```bash
# Add to crontab — runs Mon-Fri at 4:35pm ET:
crontab -e
# Paste this line:
35 20 * * 1-5 cd /path/to/signals && .venv/bin/signals auto trade --broker alpaca >> data/auto.log 2>&1
```

## Quick start — momentum strategy (Python API)

```python
from signals.model.momentum import CrossSectionalMomentum
from signals.data.storage import DataStore
from signals.config import SETTINGS

store = DataStore(SETTINGS.data.dir)

# Load the full SP500 (after fetching data)
import pandas as pd
sp500 = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
tickers = sp500["Symbol"].str.replace(".", "-").tolist()
prices = {t: store.load(t, "1d") for t in tickers if len(store.load(t, "1d")) > 500}

# Run momentum: rank 498 stocks, go long top 10
mom = CrossSectionalMomentum(lookback_days=252, skip_days=21, n_long=10)
equity = mom.backtest(prices, "2022-04-01", "2026-04-01")
print(f"Final: ${equity.iloc[-1]:,.0f}")  # ~$66k from $10k
```

## Tests

```bash
pytest --cov=signals
```

**254 tests** across 26 test modules covering all model classes
(momentum, TSMOM, PEAD, pairs, Markov sunset, hybrid, trend, boost,
ensemble, composite, HMM, HOMC, lookahead regression, sunset warnings,
absolute encoder, rule-based signals) plus engine, portfolio, metrics,
data, and broker infrastructure.

## Methodology discipline

10 rounds of adversarial review produced these rules:

1. **No single-seed headlines.** Every Sharpe quotes `mean ± stderr`
   across ≥ 5 pre-registered seeds, or is labeled `(seed=42)`.
2. **Non-overlapping windows.**
   [`scripts/_window_sampler.py`](./scripts/_window_sampler.py)
   guarantees `spacing ≥ window_len`.
3. **Correct annualization per calendar.** Equities 252/yr, crypto
   365/yr. Explicit `periods_per_year` on every `compute_metrics` call.
4. **Non-zero risk-free rate.**
   [`historical_usd_rate()`](./signals/backtest/risk_free.py) for
   period-exact T-bill averages.
5. **Pre-registered grids.** Every sweep script declares its grid in
   the docstring. Grids are not expanded on failure.
6. **Project-level DSR.** Deflated Sharpe counts every trial ever run
   (~2,000+), not just the current sweep.
7. **Forward-expectation honesty.** Trailing CAGR is NOT a forecast.

## History in one paragraph

Started in April 2026 as a Markov-chain BTC signal generator inspired
by Nascimento et al. (2022). A skeptic review tore down the "Sharpe
2.15" headline as a seed-42 artifact on overlapping windows. Five
rounds of corrections fixed annualization, samplers, and DSR. A
4-asset portfolio experiment showed diversification math works but
depends on BTC's forward return. The Markov approach was conclusively
sunset after failing 100/100 ticker-model tests on equities. The
project then pivoted to testing signal classes the academic literature
identifies as persistent: cross-sectional momentum (Jegadeesh-Titman
1993), time-series momentum (Moskowitz et al. 2012), and post-earnings
announcement drift. Three of three worked. The reusable output is
both the strategies and the adversarial-review methodology toolkit
(non-overlap sampler, multi-seed evaluator, DSR, bootstrap CIs,
permutation tests, cost sensitivity, regime ablation).

## Project layout

```
signals/
├── model/
│   ├── momentum.py        # Cross-sectional momentum (WINNER)
│   ├── tsmom.py           # Time-series momentum multi-asset
│   ├── pead.py            # Post-earnings announcement drift
│   ├── pairs.py           # Statistical arbitrage / pairs trading
│   ├── hybrid.py          # HybridRegimeModel (sunset internals)
│   ├── composite.py       # CompositeMarkovChain (sunset)
│   ├── homc.py            # HigherOrderMarkovChain (sunset)
│   ├── hmm.py             # HiddenMarkovModel (sunset)
│   ├── trend.py           # TrendFilter, DualMovingAverage
│   ├── boost.py           # GradientBoostingModel (research)
│   ├── ensemble.py        # EnsembleModel (research)
│   ├── signals.py         # SignalGenerator
│   ├── rule_signals.py    # RuleBasedSignalGenerator
│   └── states.py          # State encoders (quantile, absolute, composite)
├── data/
│   ├── earnings.py        # Earnings data fetcher (yfinance + YoY fallback)
│   └── ...                # DataSource, DataPipeline, DataStore
├── backtest/
│   ├── engine.py          # BacktestEngine + BTC_HYBRID_PRODUCTION
│   ├── portfolio.py       # Long/short portfolio with sizing + stops
│   ├── metrics.py         # Sharpe, DSR, CAGR, drawdown
│   ├── risk_free.py       # historical_usd_rate helper
│   └── vol_target.py      # Vol-targeting overlay
├── broker/                # PaperBroker, AlpacaBroker (dry-run default)
└── cli.py                 # Typer CLI

scripts/
├── cross_sectional_momentum_eval.py  # WINNER — beats B&H on equities
├── tsmom_multi_asset_eval.py         # Defensive overlay — best Sharpe/DD ratio
├── pead_eval.py                      # Earnings drift — event-driven alpha
├── pairs_trading_eval.py             # Stat arb — didn't work on mega-caps
├── multi_stock_algo_eval.py          # 20-stock × 5-model comprehensive test
├── broad_comparison.py               # All strategies in one table
├── trailing_7y_view.py               # 7-year single-window comparison
├── drawdown_tolerant.py              # BTC-heavy allocation analysis
├── portfolio_vs_sp_bh.py             # 4-asset basket vs SP head-to-head
├── risk_parity_4asset.py             # BTC/SP/TLT/GLD basket
├── explore_improvements.py           # 144-config parameter search
├── _window_sampler.py                # Shared non-overlap sampler
└── data/                             # Persisted results (parquet + markdown)
```
