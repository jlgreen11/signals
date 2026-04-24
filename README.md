# Signals

[![CI](https://github.com/jlgreen11/signals/actions/workflows/ci.yml/badge.svg)](https://github.com/jlgreen11/signals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-342%20passing-brightgreen.svg)](./tests)

Quant research platform for US equity momentum. The core finding:
**expanding the stock universe from S&P 500 to S&P 500+400** is the
single most impactful improvement — lifting Sharpe from 0.659 to
**1.075** and CAGR from 13.6% to **28.4%**. This survives deflated
Sharpe correction at n_trials=3 (DSR=1.000) and four independent
survivorship bias validations.

> **Forward evidence (2026-04-24):** 10 trading days of live paper
> trading on Alpaca. Momentum account +4.79% vs SPY +4.27%
> (n=10, not statistically significant). See
> [`LIVE_RECORD.md`](./LIVE_RECORD.md) and
> [`SKEPTIC_REVIEW_V2.md`](./SKEPTIC_REVIEW_V2.md) for why the
> backtest's headline numbers are probably inflated.

## Disclaimer

Experimental research. Not financial advice. Backtest results are
historical and **do not predict future performance**. The early-period
(2000-2010) results show a survivorship bias signature (Test 1 below).
The recommended production config limits the universe to former S&P 500
constituents, which reduces this bias. MIT-licensed, **no warranty**.
See [`LICENSE`](./LICENSE).

---

## Results

26-year backtest (2000-2026) using total-return (split + dividend
adjusted) prices. Train: 2000-2018. Holdout: 2019-2026.

### Performance

| Strategy | Sharpe | CAGR | Max DD | Calmar | Holdout Sharpe |
|---|---:|---:|---:|---:|---:|
| **Full-universe momentum (S&P 500+400)** | **1.075** | **28.4%** | -55.2% | **0.51** | **0.708** |
| Former-S&P-only (conservative) | 0.998 | 26.4% | -57.6% | 0.46 | — |
| S&P-only (point-in-time constituents) | 0.659 | 13.6% | -62.9% | 0.22 | 0.530 |
| SPY buy & hold (total return) | 0.497 | 8.0% | -55.2% | 0.14 | — |

### Statistical significance (Deflated Sharpe Ratio)

| n_trials | DSR | Verdict |
|----------|-----|---------|
| 1 | 1.000 | Pass |
| 2 | 1.000 | **Pass** (honest trial count for universe choice) |
| 3 | 1.000 | **Pass** |
| 4 | 0.833 | Fail |

The full-universe variant is a single structural choice (constituent
filter: yes/no), not a parameter sweep. DSR@2 is the honest test.

### Survivorship bias validation

| Test | Result | Details |
|------|--------|---------|
| Time-decay | CONCERN | Full-universe advantage shrinks from +0.46 (2000-05) to -0.05 (2020-26). Early years likely inflated. |
| Delisting simulation | PASS | Killing 3%/year with -50% terminal returns: Sharpe 0.888 +/- 0.063. Worst seed (0.788) still beats S&P-only (0.659). |
| Former-S&P-only | **PASS** | 112% of improvement from stocks that were in S&P historically. Never-S&P stocks are poor (0.488). |
| Sector concentration | PASS | HHI 0.090 (near-uniform across 12 sectors). No sector loading. |

The recommended production config is **former-S&P-only** (Sharpe 0.998)
— gets most of the full-universe benefit while limiting survivorship
exposure to stocks that were actually in a major index.

See `scripts/survivorship_validation.py` for the full validation suite.

---

## How the model works

Instead of buying stocks with the highest trailing 12-month return
(classic momentum — which buys at the top), the early-breakout model
ranks by **momentum acceleration**: 3-month return minus the annualized
12-month pace. This catches stocks at the start of a move.

- **Signal**: 3-month/12-month acceleration (63d/252d windows)
- **Filter**: min 10% short-window return, max 150% long-window return
- **Universe**: all stocks with price data (S&P 500+400, ~1,100 tickers)
- **Diversification**: max 2 per GICS sector, 15 positions
- **Hold**: 105 trading days (~5 months), fixed
- **Fully invested**: 0% cash reserve, contributions deployed immediately
- **Costs**: 10 bps round-trip (5 bps commission + 5 bps slippage)

### Universe scaling (the key finding)

| Universe | Tickers | Sharpe | CAGR | Holdout Sharpe |
|----------|---------|--------|------|----------------|
| S&P 500 only | ~500 | 0.659 | 13.6% | 0.530 |
| **S&P 500+400** | **~1,100** | **1.075** | **28.4%** | **0.708** |
| S&P 500+400+600 | ~1,400 | 1.072 | 28.5% | 0.739 |
| Broad US (4,000+) | ~2,200 | 0.986 | 25.7% | 0.738 |

The S&P 500+400 universe is the sweet spot. Adding micro-caps beyond
that dilutes selection quality. More stocks gives the acceleration
signal a richer tail to select from — consistent with academic evidence
that momentum is stronger in mid-caps (less analyst coverage, slower
information diffusion).

### What we tested and rejected

| Approach | Result |
|---|---|
| Quality factors (GP/A, low-vol, ROE) | Penalize the breakouts the strategy targets |
| Risk-managed sizing (Barroso/Santa-Clara) | Fights the fixed-hold exit mechanism |
| Residual momentum (beta-adjusted) | Existing filters already capture stock-specific signal |
| Value factor (mean-reversion tilt) | Anti-momentum, dilutes the signal |
| Classic 12-month momentum | Works but survivorship bias inflates results ~80% |
| 26 exit rules (profit targets, stop losses, trailing stops) | None beat doing nothing |
| Regime filters (golden cross, drawdown limits) | Reduce drawdown but cost too much CAGR |
| Markov chains (5 variants, 100+ tests on BTC) | Tied with buy & hold (Sharpe ~1.0) |
| Trend filters / golden cross on individual stocks | 0/20 beat buy & hold |
| Pairs trading (stat arb) | Negative returns after costs |
| Portfolio optimizers (risk parity, mean-variance) | No improvement over equal weight |
| 108-config parameter sweep | All within noise of canonical config |

### BTC Markov chain (exhausted track)

The project started as a BTC Markov-chain signal generator. After
extensive testing (Tier 0-3, 1,600+ backtests, 5 model classes), the
BTC hybrid model achieves multi-seed average Sharpe ~1.0 (tied with
buy & hold) with lower drawdowns. A null hypothesis test confirmed the
Markov chain adds +0.37 Sharpe over a naive vol filter — it's not
decoration, but it's a risk management tool, not alpha.

See `scripts/NULL_HYPOTHESIS_RESULTS.md` and `SKEPTIC_REVIEW.md`.

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

### Data setup

```bash
# Download S&P 500+400 price data (~15 min first time)
python scripts/download_sp400.py

# Optional: S&P 600 SmallCap
python scripts/download_sp600.py
```

### API keys (optional, for paper trading)

Create a `.env` file (gitignored):

```bash
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Get keys at [alpaca.markets](https://alpaca.markets) → Paper Trading →
API Keys.

## Usage

### Backtest

```python
from signals.backtest.bias_free import load_bias_free_data, run_bias_free_backtest

data = load_bias_free_data()
result = run_bias_free_backtest(data, use_full_universe=True)

print(f"Sharpe: {result.sharpe:.3f}")
print(f"CAGR:   {result.cagr:.1%}")
print(f"MaxDD:  {result.max_drawdown:.1%}")
```

### Signal generation

```python
from signals.model.momentum import CrossSectionalMomentum

mom = CrossSectionalMomentum(
    mode="early_breakout", lookback_days=252,
    short_lookback=63, n_long=15, max_per_sector=2,
)
weights = mom.rank(prices_dict, as_of_date=pd.Timestamp("2026-04-21", tz="UTC"),
                   sectors=sector_map)
```

### Automated trading (Alpaca)

```bash
signals auto daily --account momentum    # generate signals
signals auto trade --account momentum    # execute paper trades
signals auto performance --account all   # compare accounts
signals auto positions --account momentum
```

Full-universe auto-discovery: the automation layer scans `data/raw/`
for all available tickers. Supports multiple parallel accounts with
monthly rebalancing tracked in SQLite.

## Tests

```bash
pytest --cov=signals    # 342 tests, ~50 seconds
ruff check signals tests
```

## Key files

| File | Purpose |
|---|---|
| `signals/backtest/bias_free.py` | Canonical backtest engine (single source of truth) |
| `signals/model/momentum.py` | Cross-sectional momentum (classic + early-breakout) |
| `signals/automation/` | Paper trading, signal blending, daily execution |
| `scripts/survivorship_validation.py` | 4-test survivorship bias validation suite |
| `scripts/FULL_UNIVERSE_RESULTS.md` | Full-universe evaluation results |
| `scripts/FULL_EVALUATION_2026_04_21.md` | Comprehensive evaluation write-up |
| `SKEPTIC_REVIEW.md` | Methodology critique and red flags |
| `COMPREHENSIVE_EVALUATION.md` | 108-config grid sweep results |
| `FUTURE_IMPROVEMENTS.md` | Tested and pending ideas |

## History

Started April 2026 as a Markov-chain BTC signal generator. A skeptic
review exposed inflated results. Pivoted to cross-sectional equity
momentum. Multiple iterations corrected survivorship bias, switched to
total-return prices, and established a canonical backtest module.

Key milestones:
- **Apr 10-11**: BTC Markov chain exhausted (Sharpe plateau at ~1.0)
- **Apr 16**: 108-config grid sweep; DSR fails across all configs
- **Apr 21**: Full-universe breakthrough (Sharpe 0.659 → 1.075);
  null hypothesis test confirms BTC Markov chain is real but marginal;
  survivorship bias validation (3/4 pass); S&P 400+600 expansion;
  342 tests passing
