# Signals

[![CI](https://github.com/jlgreen11/signals/actions/workflows/ci.yml/badge.svg)](https://github.com/jlgreen11/signals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-276%20passing-brightgreen.svg)](./tests)

A quant research project that spent 10 rounds of adversarial review
testing every approach we could find — Markov chains, trend filters,
vol-regime routing, hybrid ensembles, pairs trading — and discovered
that **most backtest results are inflated by survivorship bias**. On a
bias-free basis (26 years, 1,081 historical SP500 constituents), the
best strategy edges out SPY by ~3.9% CAGR with worse drawdowns. The
original Markov approach failed outright.

## ⚠ Disclaimer

Experimental research project. Nothing here is financial advice.
Backtest results are historical and **do not predict future
performance**. MIT-licensed with **no warranty of any kind**. See
[`LICENSE`](./LICENSE). Conduct your own due diligence and start with
amounts you can afford to lose entirely.

---

## What works (and what doesn't)

After testing 7 model classes across 498 SP500 stocks, here's the
honest scorecard. Momentum and multi-factor numbers are on the full
SP500 universe; TSMOM is on 8 asset classes. All correctly annualized
at 252/yr for equities, risk-free rate ~2.3%.

### The honest numbers (survivorship-bias-free)

All headline results below are from a **26-year survivorship-bias-free
backtest** (2000-2026) using the
[fja05680/sp500](https://github.com/fja05680/sp500) dataset of daily
SP500 constituent lists — 1,081 unique tickers including Enron, Lehman,
Countrywide, and 585 other companies that were later delisted, acquired,
or went bankrupt.

| Strategy | $100K became | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| **Early-breakout momentum (production)** | **$1,881,447** | **+11.8%** | **0.594** | −59.0% |
| Classic 12-month momentum | $1,519,771 | +10.9% | 0.520 | −64.4% |
| SPY buy & hold | $743,656 | +7.9% | 0.492 | −55.2% |
| Biased momentum (today's SP500) | $64,240,548 | +27.9% | 0.910 | −66.8% |

**Survivorship bias inflates Sharpe by ~80% and CAGR by ~18pp.** The
biased test's $64M result is a fantasy built on always picking from a
list of future survivors. Our production model's real edge is **+3.9%
CAGR over SPY** with comparable drawdowns — meaningful but modest, not
the 50%+ CAGR the biased tests suggest.

Key findings:
- **59% win rate** on early-breakout model (vs 49% for classic momentum)
- **Extreme entry momentum predicts blowups**: QCOM at +2563% momentum
  lost 59.5%; SMCI at +850% lost 52.0% — the early-breakout model
  avoids these by filtering stocks with >150% trailing returns
- **20% win rate in 2008**, avg trade −11.9% — no momentum strategy
  avoids crash years
- A **26-rule exit-rule sweep** found no profit target, stop loss, or
  trailing stop improves the strategy
- A **122-parameter sweep** optimized the acceleration windows, hold
  periods, sector caps, and entry filters

See [`scripts/SURVIVORSHIP_FREE_RESULTS.md`](./scripts/SURVIVORSHIP_FREE_RESULTS.md)
for the full era-by-era breakdown.

### Strategies that failed

| Strategy | Result | Why |
|---|---|---|
| Biased momentum backtests | Sharpe 1.3+, CAGR 60%+ | **Survivorship bias** — using today's SP500 at all historical dates |
| Markov hybrid (5 variants) | 5% win rate vs B&H (100 tests) | Wrong signal class: daily price/vol patterns contain no exploitable edge |
| Trend filter / golden cross | 0/20 stocks beat B&H | Individual stocks don't trend-follow like asset classes |
| Pairs trading (stat arb) | Sharpe −0.47, CAGR −5.8% | Edge too small for costs; cointegration is unstable |

### Other models (not yet bias-tested)

These models showed positive results on biased backtests but have **not
been validated survivorship-bias-free**. Treat these numbers as upper
bounds until proven otherwise.

| Strategy | Sharpe | CAGR | Max DD | Caveat |
|---|---:|---:|---:|---|
| PEAD earnings drift | +0.960 | +23.4% | −26.6% | 20-stock universe, 5-year test only |
| TSMOM multi-asset | +0.947 | +9.8% | −8.9% | 8 asset classes (ETFs), not SP500 stocks |
| Multi-factor composite | ~+1.2 | ~+45% | ~−35% | Uses today's SP500 — likely biased |

### The meta-lesson

The project spent 10 rounds trying to extract alpha from **time-series
patterns on single-stock daily bars** using Markov chains and trend
filters. That approach failed across 20 stocks × 5 model variants =
100 comparisons, with a **5% win rate** against buy-and-hold.

The four strategies that work all use a **different information axis**:

1. **Momentum** — *which stock is winning relative to others?*
   (cross-sectional, not time-series)
2. **Multi-factor** — *momentum + is it cheap + is it profitable?*
   (composite scoring with value + quality filters)
3. **TSMOM** — *is this entire asset class trending?*
   (macro, not single-stock)
4. **PEAD** — *did the company beat earnings expectations?*
   (fundamental, not price-pattern)

**The signal class was the problem all along, not the evaluation
methodology or the model complexity.**

---

## The winning strategies in detail

### 1. Early-breakout momentum — production model

[`signals/model/momentum.py`](./signals/model/momentum.py) |
[`scripts/survivorship_free_test.py`](./scripts/survivorship_free_test.py)

Instead of classic 12-month momentum (which buys already-extended
winners), the early-breakout model ranks by **momentum acceleration**:
1-month return minus the annualized 6-month pace. This catches stocks
at the start of a move. Stocks with >150% trailing 6-month return are
filtered to avoid buying at the top. Max 2 per GICS sector.

**Bias-free results** (2000-2026, $100K, point-in-time SP500 constituents):

| Config | Sharpe | CAGR | Max DD | $100K → |
|---|---:|---:|---:|---:|
| **Early breakout (15 stocks, 2/sector)** | **0.594** | **+11.8%** | −59.0% | **$1,881,447** |
| Classic 12m momentum (10 stocks) | 0.520 | +10.9% | −64.4% | $1,519,771 |
| SPY B&H | 0.492 | +7.9% | −55.2% | $743,656 |

The edge is real but modest: **+3.9% CAGR over SPY**, with worse
drawdowns (−59% vs −55%). Optimized via a 122-parameter sweep across
acceleration windows, hold periods, position counts, sector caps, and
entry filters. See
[`scripts/survivorship_free_test.py`](./scripts/survivorship_free_test.py).

### 1b. Multi-factor composite (momentum + value + quality + news filter)

[`signals/model/multifactor.py`](./signals/model/multifactor.py) |
[`signals/model/news_filter.py`](./signals/model/news_filter.py) |
[`scripts/MULTIFACTOR_RESULTS.md`](./scripts/MULTIFACTOR_RESULTS.md)

Blends three factors into a composite percentile-rank score:

```
score = 40% × momentum_rank + 30% × value_rank (inverse P/E) + 30% × quality_rank (ROE)
```

Then excludes the top-25% highest-volatility stocks before picking the
top 10. After ranking, a **news sentiment filter** scans recent yfinance
headlines for geopolitical, regulatory, one-off, and sector-shock risks
— flagged tickers get weight reduced (CAUTION) or removed (SKIP).

The result is a **diversified** top-10 across 7+ sectors instead of
pure momentum's concentration in semiconductors. Lower expected CAGR
but much less sector risk.

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

The system supports **multiple parallel Alpaca accounts** for A/B testing
strategies. Add prefixed keys to `.env` for each account:

```bash
# .env — three parallel paper accounts
ALPACA_API_KEY=PK...              # Account 1: momentum
ALPACA_SECRET_KEY=...
ALPACA_MULTIFACTOR_KEY=PK...      # Account 2: multi-factor
ALPACA_MULTIFACTOR_SECRET=...
ALPACA_BASELINE_KEY=PK...         # Account 3: SPY B&H baseline
ALPACA_BASELINE_SECRET=...
```

```bash
# Trade a specific account:
signals auto trade --account momentum
signals auto trade --account multifactor

# Side-by-side performance of all accounts:
signals auto performance --account all

# Positions for a specific account:
signals auto positions --account momentum

# Account health check:
signals auto config
signals auto history LITE --days 30
```

### Automate (cron — runs unattended)

```bash
# Two cron entries, one per algo account (Mon-Fri 4:35pm ET):
crontab -e
# Paste:
35 20 * * 1-5 cd /path/to/signals && .venv/bin/signals auto trade --account momentum >> data/momentum.log 2>&1
36 20 * * 1-5 cd /path/to/signals && .venv/bin/signals auto trade --account multifactor >> data/multifactor.log 2>&1
```

**Monthly rebalancing is automatic.** The system tracks the last
rebalance date in SQLite and only submits orders every 21 trading days.
On the other ~20 days it generates signals and records equity but does
not trade. Baseline (SPY B&H) has no cron entry — it just sits.

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

# Early-breakout momentum (default): acceleration-ranked, sector-diversified
mom = CrossSectionalMomentum()  # defaults: 1m/6m accel, 15 stocks, 2/sector
weights = mom.rank(prices, as_of_date=pd.Timestamp("2026-04-12", tz="UTC"))
print({t: w for t, w in weights.items() if w > 0})

# Classic mode for comparison
mom_classic = CrossSectionalMomentum(mode="classic", lookback_days=252, n_long=10)
equity = mom_classic.backtest(prices, "2022-04-01", "2026-04-01")
# NOTE: backtest on today's SP500 is survivorship-biased — real edge is ~3.9% CAGR over SPY
```

## Tests

```bash
pytest --cov=signals
```

**283 tests** across 28 test modules covering all model classes
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
by Nascimento et al. (2022). A skeptic review exposed the "Sharpe
2.15" headline as a seed-42 artifact on overlapping windows. Five
rounds of corrections fixed annualization, samplers, and DSR. The
Markov approach was sunset after failing 100/100 ticker-model tests.
The project pivoted to cross-sectional momentum and discovered that
survivorship bias inflated all backtest results by ~80%. A 26-year
bias-free test using 1,081 historical SP500 constituents showed
classic momentum barely edges SPY (CAGR 10.9% vs 7.9%). An early-
breakout variant — ranking by momentum acceleration instead of raw
trailing return, with sector diversification — improved to 11.8% CAGR
bias-free, validated across a 122-parameter sweep. The real edge over
SPY is ~3.9% CAGR, not the 50%+ CAGR that biased backtests suggest.
Three Alpaca paper trading accounts run automated rebalancing for
live forward testing. 283 tests, 28 test modules.

## Project layout

```
signals/
├── model/
│   ├── momentum.py        # Cross-sectional momentum (PRODUCTION — account 1)
│   ├── multifactor.py     # Multi-factor composite (PRODUCTION — account 2)
│   ├── news_filter.py     # Post-signal headline risk scanner
│   ├── tsmom.py           # Time-series momentum multi-asset
│   ├── pead.py            # Post-earnings announcement drift
│   ├── pairs.py           # Statistical arbitrage / pairs trading
│   ├── hybrid.py          # HybridRegimeModel (sunset)
│   ├── composite.py       # CompositeMarkovChain (sunset)
│   ├── homc.py            # HigherOrderMarkovChain (sunset)
│   ├── hmm.py             # HiddenMarkovModel (sunset)
│   ├── trend.py           # TrendFilter, DualMovingAverage
│   ├── boost.py           # GradientBoostingModel (research)
│   ├── ensemble.py        # EnsembleModel (research)
│   ├── signals.py         # SignalGenerator
│   ├── rule_signals.py    # RuleBasedSignalGenerator
│   └── states.py          # State encoders (quantile, absolute, composite)
├── automation/
│   ├── signal_store.py    # SQLite signal persistence
│   ├── cash_overlay.py    # Multi-model portfolio blender
│   ├── insights_engine.py # Daily runner: data → models → signals → report
│   ├── paper_runner.py    # Alpaca execution + monthly rebalance gating
│   └── cli.py             # `signals auto` subcommands
├── data/
│   ├── earnings.py        # Earnings data fetcher (yfinance + YoY fallback)
│   └── ...                # DataSource, DataPipeline, DataStore
├── backtest/
│   ├── engine.py          # BacktestEngine + BTC_HYBRID_PRODUCTION
│   ├── portfolio.py       # Long/short portfolio with sizing + stops
│   ├── metrics.py         # Sharpe, DSR, CAGR, drawdown
│   ├── risk_free.py       # historical_usd_rate helper
│   └── vol_target.py      # Vol-targeting overlay
├── broker/
│   ├── paper.py           # In-memory PaperBroker
│   ├── alpaca.py          # Alpaca Trading API (paper + live)
│   └── paper_trade_log.py # Trade logging
└── cli.py                 # Typer CLI entry point

scripts/
├── cross_sectional_momentum_eval.py  # Full SP500 momentum evaluation
├── multifactor_eval.py               # Multi-factor composite evaluation
├── universe_analysis.py              # Optimal universe size + sector analysis
├── tsmom_multi_asset_eval.py         # 8-asset trend-following
├── pead_eval.py                      # Earnings drift
├── pairs_trading_eval.py             # Stat arb (didn't work)
├── multi_stock_algo_eval.py          # 20-stock × 5-model Markov failure test
├── survivorship_free_test.py         # 26-year bias-free backtest (1,081 tickers)
├── exit_rules_sweep.py              # 26-rule exit rule optimization
├── historical_stress_test.py         # 25-year era-by-era stress test
├── optimizer_eval.py                 # Portfolio optimizer comparison
├── automation_demo.py                # End-to-end automation demo
├── _window_sampler.py                # Shared non-overlap sampler
└── data/                             # Persisted results (parquet + markdown)
```
