# Signals

[![CI](https://github.com/jlgreen11/signals/actions/workflows/ci.yml/badge.svg)](https://github.com/jlgreen11/signals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-336%20passing-brightgreen.svg)](./tests)

Quant research project that tested 7 model classes on US equities.
**Full-universe early-breakout momentum** on 746 stocks achieves
Sharpe **0.985**, CAGR **25.2%** over 26 years (2000-2026) with
holdout Sharpe **0.731** (2019-2026). Validated against survivorship
bias (removing dead stocks helps, delisting penalty barely registers).
The S&P-only variant (Sharpe 0.659) does not survive DSR across 108
configs, but the full-universe structural improvement is a single
binary choice, not a parameter sweep.

## Disclaimer

Experimental research. Not financial advice. Backtest results are
historical and **do not predict future performance**. The strategy's
apparent edge is not statistically distinguishable from noise at the
multi-trial correction level. MIT-licensed, **no warranty**. See
[`LICENSE`](./LICENSE).

---

## Results (survivorship-bias-reduced)

Honest numbers from a 26-year backtest (2000-2026) on **782 tickers**
including ~300 historical/delisted constituents from
[fja05680/sp500](https://github.com/fja05680/sp500). ~375 deeply
delisted tickers (Enron, Lehman, WorldCom) are no longer available
from Yahoo Finance, so some survivorship bias remains.

Both strategy and SPY baseline use **total return** (split + dividend
adjusted). Train period: 2000-2018. Holdout: 2019-2026. All configs
validated out-of-sample.

| Strategy | CAGR | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|---:|
| **Full-universe momentum** | **+25.2%** | **0.985** | -55.9% | **0.45** |
| S&P-only momentum (prior best) | +13.6% | 0.659 | -62.9% | 0.22 |
| SPY buy & hold (total return) | +8.0% | 0.497 | -55.2% | 0.14 |

Full-universe uses all 746 tickers with price data (excluding 36
delisted). Holdout (2019-2026): Sharpe 0.731, CAGR 26.7%.

**Survivorship bias check**: removing dead stocks *improves* Sharpe
(+0.07); worst-case delisting penalty barely registers (-0.006).
See `scripts/FULL_UNIVERSE_RESULTS.md` for validation details.

**DSR caveat (S&P-only config)**: The S&P-only variant's 108-config
grid sweep does not survive deflated Sharpe correction (observed max
0.663 < expected max 0.714). The full-universe improvement is a
structural change (one binary choice), not a parameter tweak.

Results from the canonical backtest module (`signals.backtest.bias_free`)
— deterministic, single source of truth.

### How the model works

Instead of buying stocks with the highest trailing 12-month return
(classic momentum — which buys at the top), the early-breakout model
ranks by **momentum acceleration**: 3-month return minus the annualized
12-month pace. This catches stocks at the start of a move.

- **Signal**: 3-month/12-month acceleration (windows from grid sweep)
- **Filter**: min 10% short-window return, max 150% long-window return
- **Diversification**: max 2 per GICS sector, 15 positions
- **Hold**: 105 trading days (~5 months), fixed
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
equity tests. Pivoted to cross-sectional momentum. Multiple iterations
of the backtest reported inconsistent numbers (8.7%, 11.8%, 13.3%,
20.5% CAGR for the same config) due to implementation drift.

In April 2026 the backtest was rewritten into a canonical single-source
module (`signals.backtest.bias_free`). A proper train/holdout split +
108-config grid sweep with total-return (dividend-adjusted) prices
shows: the early-breakout strategy gets 13.6% CAGR / 0.659 Sharpe over
2000-2026 vs SPY's 8.0% / 0.497, but the apparent edge does **not**
survive multi-trial correction (deflated Sharpe). Treat numbers here
as honest best-case, not a proven edge. Three Alpaca paper accounts
run live forward testing.
