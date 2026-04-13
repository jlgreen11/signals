# Signals

[![CI](https://github.com/jlgreen11/signals/actions/workflows/ci.yml/badge.svg)](https://github.com/jlgreen11/signals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-305%20passing-brightgreen.svg)](./tests)

Quant research project that tested 7 model classes on US equities and
discovered most backtest results are inflated by survivorship bias. On
a **26-year bias-free backtest** (1,081 historical SP500 constituents
including dead companies), our best strategy beats SPY by ~3.9% CAGR
— real but modest.

## Disclaimer

Experimental research. Not financial advice. Backtest results are
historical and **do not predict future performance**. MIT-licensed,
**no warranty**. See [`LICENSE`](./LICENSE).

---

## Results (survivorship-bias-free)

All numbers below are from a 26-year backtest (2000-2026) using the
[fja05680/sp500](https://github.com/fja05680/sp500) dataset of daily
SP500 constituent lists — 1,081 unique tickers including Enron, Lehman,
Countrywide, and 585 other delisted/bankrupt companies. No lookahead.

| Strategy | CAGR | Sharpe | Max DD | $100K became |
|---|---:|---:|---:|---:|
| **Early-breakout momentum** | **+11.8%** | **0.594** | -59.0% | **$1,881,447** |
| Classic 12-month momentum | +10.9% | 0.520 | -64.4% | $1,519,771 |
| SPY buy & hold | +7.9% | 0.492 | -55.2% | $743,656 |

With $10K/month DCA (fully invested, 0% cash reserve):

| Strategy | 10-year | 20-year | 26-year |
|---|---:|---:|---:|
| **Early-breakout momentum** | **$2.69M** | **$11.8M** | **$29.5M** |
| SPY buy & hold | $1.98M | $6.0M | $18.0M |
| Total invested | $1.29M | $2.49M | $3.21M |

Statistical significance: p=0.0002 (t-test, excess returns vs SPY).

### How the model works

Instead of buying stocks with the highest trailing 12-month return
(classic momentum — which buys at the top), the early-breakout model
ranks by **momentum acceleration**: 1-month return minus the annualized
6-month pace. This catches stocks at the start of a move.

- **Signal**: 1-month/6-month acceleration (optimized via 122-parameter sweep)
- **Filter**: min 10% 1-month return, max 150% 6-month return
- **Diversification**: max 2 per GICS sector, 15 positions
- **Hold**: 105 trading days (~5 months)
- **Fully invested**: 0% cash reserve, contributions deployed immediately

### What we tested and rejected

| Approach | Result |
|---|---|
| Classic 12-month momentum | Works but survivorship bias inflates results ~80% |
| 26 exit rules (profit targets, stop losses, trailing stops) | None beat doing nothing |
| Regime filters (golden cross, drawdown limits) | Reduce drawdown but cost too much CAGR |
| Markov chains (5 variants, 100 tests) | 5% win rate vs buy & hold |
| Trend filters / golden cross on individual stocks | 0/20 beat buy & hold |
| Pairs trading (stat arb) | Negative returns after costs |
| Portfolio optimizers (risk parity, mean-variance, etc.) | No improvement over equal weight |

### Models not yet bias-tested

These showed positive results on survivorship-biased backtests. Treat
as upper bounds until validated on historical constituents.

| Strategy | Biased Sharpe | Caveat |
|---|---:|---|
| PEAD earnings drift | +0.96 | 20 stocks, 5-year test only |
| TSMOM multi-asset | +0.95 | 8 ETFs, not SP500 stocks |
| Multi-factor composite | ~+1.2 | Uses today's SP500 — likely biased |

---

## Install

```bash
# From GitHub
pip install git+https://github.com/jlgreen11/signals.git

# Or clone for development
git clone https://github.com/jlgreen11/signals.git
cd signals
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### API keys

Create a `.env` file (gitignored):

```bash
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Get keys at [alpaca.markets](https://alpaca.markets) → Trading API →
Paper Trading → API Keys.

## Usage

```python
from signals.model.momentum import CrossSectionalMomentum

# Early-breakout momentum (default)
mom = CrossSectionalMomentum()
weights = mom.rank(prices_dict, as_of_date=pd.Timestamp("2026-04-12", tz="UTC"),
                   sectors=sector_map)
# {ticker: weight} for selected stocks, 0.0 for others

# Classic mode
mom = CrossSectionalMomentum(mode="classic", lookback_days=252, n_long=10)
```

### Automated trading (Alpaca)

```bash
signals auto trade --account momentum
signals auto performance --account all
signals auto positions --account momentum
```

Supports multiple parallel accounts. Monthly rebalancing tracked in
SQLite — runs unattended via cron.

## Tests

```bash
pytest --cov=signals
```

305 tests across 28 modules.

## History

Started April 2026 as a Markov-chain BTC signal generator. A skeptic
review exposed inflated results. The Markov approach failed 100/100
equity tests. Pivoted to cross-sectional momentum and discovered
survivorship bias inflated all results ~80%. A 26-year bias-free test
on 1,081 historical SP500 constituents showed classic momentum barely
edges SPY. An early-breakout variant — ranking by acceleration with
sector diversification — improved to 11.8% CAGR (vs SPY 7.9%),
validated across 122 parameter combinations and statistically
significant at p=0.0002. Three Alpaca paper accounts run live forward
testing.
