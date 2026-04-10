# Signals

Markov-chain market signal generator with three swappable model backends, walk-forward
backtesting, sized long/short execution, and a one-shot "what should I do tomorrow?" CLI.

## Models

| Model | What it is |
|---|---|
| `composite` | 1st-order discrete Markov chain over a 2D (return × volatility) state grid (default 3×3 = 9 states). The original phase-1 model — currently the best performer on BTC. |
| `hmm` | Gaussian Hidden Markov Model (`hmmlearn`) over standardized continuous features. Hidden regimes discovered via Baum-Welch. |
| `homc` | Higher-order Markov chain over quantile-binned returns, inspired by Nascimento et al. (2022). Captures *k* steps of memory. |

All three implement a common interface (`fit`, `predict_state`, `predict_next`,
`state_returns_`, `label`, `save`/`load`) so the engine, signal generator, and
CLI work with any of them transparently.

## Best result so far

BTC-USD walk-forward, 2018-09 → 2024-12 (composite, optimized via sweep):

| Metric | Composite (3×3) | HMM (4s) | HOMC (5s, o3) | Buy & Hold |
|---|---|---|---|---|
| Final equity | **$378,959** | $112,596 | $167,749 | $141,013 |
| CAGR | **78.83%** | 47.28% | 56.98% | 52.68% |
| Sharpe | **1.13** | 0.84 | 0.97 | 0.81 |
| Max drawdown | -55.7% | -58.6% | -54.9% | -76.6% |
| Calmar | **1.41** | 0.81 | 1.04 | 0.69 |
| # trades | 115 | 19 | 245 | — |

Optimized config: `--buy-bps 25 --sell-bps -30 --target-scale-bps 20 --stop-loss 0.25
--no-short --train-window 252`. Asymmetric thresholds (faster entries, slower exits)
matter; shorts hurt on BTC's secular uptrend.

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
[Model: composite | HMM | HOMC] ──── shared interface ────┐
        │                                                  │
        ▼                                                  │
[SignalGenerator: BUY/SELL/HOLD + sized target_position]   │
        │                                                  │
        ├─▶ [Portfolio] → [BacktestEngine + metrics]       │
        └─▶ [signal next] → CLI: tomorrow's action  ◀──────┘
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
# One command. Refreshes the latest bars, retrains the composite model on a
# rolling 252-bar window, and prints the action for the *next* bar's open.
signals signal next BTC-USD

# Other supported markets — anything yfinance returns
signals signal next ETH-USD
signals signal next ^GSPC
signals signal next AAPL --target-scale-bps 15

# Lower level: just decode the current state from the saved model
signals signal now BTC-USD
```

## Full CLI

```bash
# Data
signals data fetch BTC-USD --start 2015-01-01 --interval 1d
signals data refresh BTC-USD                  # incremental, append-only
signals data list

# Model — train any of composite | hmm | homc
signals model train BTC-USD --model composite --states 9
signals model train BTC-USD --model hmm --states 4
signals model train BTC-USD --model homc --states 5 --order 3
signals model inspect BTC-USD --model composite
signals model plot BTC-USD --model composite

# Backtest
signals backtest run BTC-USD --model composite \
  --start 2018-01-01 --end 2024-12-31 \
  --train-window 252 --target-scale-bps 20 \
  --buy-bps 25 --sell-bps -30 --stop-loss 0.25 --no-short

# 3-way head-to-head with the same strategy params
signals backtest compare BTC-USD --start 2018-01-01 --end 2024-12-31

# Grid search thresholds + stops
signals backtest sweep BTC-USD --model composite \
  --start 2018-01-01 --end 2024-12-31 \
  --buy-grid "10,15,20,25,30,40" \
  --sell-grid "-10,-15,-20,-25,-30,-40" \
  --stop-grid "0,0.15,0.20,0.25,0.30" \
  --rank-by calmar --top 10

signals backtest list
signals backtest show 1
```

## Strategy layer

The `Portfolio` is target-driven: `set_target(ts, price, fraction)` reconciles by
trading the delta. `fraction = +1.0` is 100% long, `-1.0` is 100% short, `0` is flat.
Features:

- **Sized longs and shorts** (`--max-long`, `--max-short`)
- **Per-bar stop loss** with cooldown bars (`--stop-loss`, `--stop-cooldown`)
- **Min-trade-fraction deadband** to suppress rebalance churn (`--min-trade`)
- **Hold preserves position** so trends aren't churned out by neutral signals
- **Asymmetric thresholds** — `--buy-bps` and `--sell-bps` independent

## Project layout

```
signals/
├── data/        # DataSource, DataPipeline, DataStore (parquet + SQLite)
├── features/    # returns, volatility, indicators
├── model/
│   ├── states.py      # QuantileStateEncoder, CompositeStateEncoder
│   ├── composite.py   # CompositeMarkovChain (1st-order, 2D states)
│   ├── hmm.py         # HiddenMarkovModel (Gaussian HMM, standardized features)
│   ├── homc.py        # HigherOrderMarkovChain
│   └── signals.py     # SignalGenerator + SignalDecision
├── backtest/    # Portfolio (long/short/sized/stops), Engine, metrics
├── broker/      # Broker ABC + PaperBroker (live execution stub)
└── cli.py       # Typer entry point
```

## Tests

```bash
pytest --cov=signals
```

44 tests covering data pipeline, all 3 model classes, signal generation,
portfolio (longs/shorts/stops), and walk-forward engine.

## Roadmap

- [x] Data pipeline (yfinance + parquet/SQLite)
- [x] Composite Markov chain (return × volatility states)
- [x] Hidden Markov Model backend
- [x] Higher-order Markov chain backend
- [x] Walk-forward backtest with no-lookahead enforcement
- [x] Long/short portfolio with position sizing and stop losses
- [x] `signal next` daily workflow command
- [x] Grid-search optimization (`backtest sweep`)
- [ ] Hourly / intraday bars
- [ ] CoinGecko fallback wired into the pipeline
- [ ] PaperBroker scheduled live signals
- [ ] Real broker integration
