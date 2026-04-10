# Random window evaluation — composite vs perfect oracle vs buy & hold

**Run date**: 2026-04-10
**Script**: `scripts/random_window_eval.py`
**Seed**: `random.Random(42)` (deterministic)
**Universe**: BTC-USD daily, 2018-01-01 → 2024-12-31 (2,557 bars)
**Windows**: 16 random non-overlapping 6-month (126 trading-bar) windows
**Strategy**: composite-3×3 with the tightened defaults (`vol_window=10`, `alpha=0.01`, `train_window=252`, `retrain_freq=21`, `buy_bps=25`, `sell_bps=-35`, `target_scale_bps=20`, no shorts, no stop)
**Oracle**: perfect knowledge of next bar's open→close direction, executes at next open with the same 5 bps slippage + 5 bps commission as the engine

## Per-window results

| Window | B&H CAGR | Strategy CAGR | Oracle (L/F) CAGR | Strategy Sharpe |
|---|---:|---:|---:|---:|
| 2019-01-05 → 2019-05-10 | +338.9% | **+581.3%** | +2,796% | 4.18 |
| 2019-01-11 → 2019-05-16 | **+821.5%** | +332.4% | +5,447% | 3.54 |
| 2019-01-25 → 2019-05-30 | **+1056.3%** | +864.4% | +9,185% | 3.69 |
| 2019-02-02 → 2019-06-07 | **+1017.9%** | +474.6% | +11,071% | 2.84 |
| 2019-09-16 → 2020-01-19 | -38.4% | **-26.7%** | +2,461% | -0.30 |
| 2019-10-13 → 2020-02-15 | +65.6% | **+95.6%** | +4,038% | 1.42 |
| 2019-11-18 → 2020-03-22 | **-64.5%** | **+32.0%** | +5,947% | 0.76 |
| 2019-12-25 → 2020-04-28 | +22.9% | **+114.1%** | +15,348% | 1.72 |
| 2020-04-18 → 2020-08-21 | **+292.9%** | +30.3% | +4,970% | 0.73 |
| 2021-03-08 → 2021-07-11 | **-70.9%** | +0.8% | +19,091% | 0.31 |
| 2021-03-27 → 2021-07-30 | **-56.1%** | +0.5% | +22,658% | 0.30 |
| 2021-05-04 → 2021-09-06 | -3.8% | **+5.1%** | +34,981% | 0.35 |
| 2021-06-24 → 2021-10-27 | **+361.1%** | +242.8% | +31,507% | 1.88 |
| 2021-10-25 → 2022-02-27 | **-77.7%** | **-41.8%** | +2,427% | -0.82 |
| 2023-06-19 → 2023-10-22 | +38.2% | +11.7% | +620% | 0.43 |
| 2024-05-25 → 2024-09-27 | -14.0% | -26.0% | +2,240% | -0.42 |

(**Bold** = winner of the B&H vs Strategy column for that row.)

## Aggregate

|  | Buy & Hold | **Strategy** | Oracle (long/flat) | Oracle (long/short) |
|---|---:|---:|---:|---:|
| Mean CAGR | +230.6% | +168.2% | +10,924% | +1,867,464% |
| **Median CAGR** | **+30.5%** | **+31.1%** | +5,697% | +97,782% |
| Min CAGR | -77.7% | -41.8% | +620% | +3,499% |
| Max CAGR | +1056.3% | +864.4% | +34,981% | +9,010,031% |
| Mean Sharpe | 1.01 | **1.29** | 8.19 | 13.31 |
| Mean Max DD | -31.3% | **-21.2%** | -0.6% | — |

- Strategy beats buy & hold in **9 / 16** windows.
- Strategy positive CAGR in **13 / 16** windows.
- Buy & hold positive CAGR in **9 / 16** windows.

## Capture ratio (strategy / oracle)

| Capture (CAGR basis) | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| vs long/flat oracle | +2.7% | +0.7% | -1.7% | +20.8% |
| vs long/short oracle | +0.4% | +0.0% | -0.0% | +3.3% |

The CAGR capture ratio looks catastrophic, but **the oracle's CAGR is a mathematical fantasy** — 126 bars of perfect daily directional bets compound to absurd numbers. CAGR is the wrong axis for this comparison.

The fairer capture metric is **Sharpe**:

> Strategy mean Sharpe **1.29** / Oracle long/flat mean **8.19** = **~16% Sharpe capture**

The 6.9-Sharpe gap between the strategy and the oracle is the room for model improvement. The 0.3-Sharpe gap between the strategy and buy & hold is what the strategy currently captures.

## What the data says about the strategy

1. **Defensive overlay, not alpha engine.** Median CAGR is *tied* with B&H (31.1% vs 30.5%). The Sharpe edge (+28%) and the drawdown reduction (-32%) come from staying out of losing periods, not from capturing extra returns in winning ones.
2. **Direction is right far more often than magnitude is.** When the strategy and B&H disagree on a winning window, the strategy is usually too small. 2019-01-11 → 2019-05-16: B&H +822%, Strategy +332% — same direction, half the size.
3. **Bull rally underperformance is the structural cost.** 2020-04 → 2020-08 missed +263pp during the post-COVID rally. The defensive `sell_bps=-35` flattens out too eagerly when realized vol is low and price is grinding higher.
4. **Bear and crash windows are where the strategy earns its keep.** COVID crash window (2019-11 → 2020-03): B&H **-64.5%**, Strategy **+32.0%**. That's the kind of asymmetric payoff that makes the regime model worth running.

## What to try next

Untested ideas suggested by this profile:

1. **More aggressive position sizing.** Try `target_scale_bps=10` (lower scale → larger fractional positions for the same expected return) and `max_long=1.5` to lever up high-confidence states without touching the defensive thresholds.
2. **Remove the unit-position cap on confidence-weighted targets.** Current `min(1, |E[r]|/scale) * confidence` caps every signal at 100% notional. Letting it run to `max_long=2.0` would let the strongest signals get more allocation.
3. **Stack the HMM regime as an extra feature in the composite encoder.** The HMM detects multi-day regimes the per-bar composite state can't see. Neither model has been tested in a stacked configuration.

## How to reproduce

```bash
python scripts/random_window_eval.py
```

Change `rng = random.Random(42)` in the script for fresh draws on a different seed. The full output (including aggregates and capture stats) is regenerated deterministically.

---

⚠ **Disclaimer**: This is an experimental research project. The results
shown above are backtests and historical simulations — they do not
predict future performance. Not financial advice. See the root
[`README.md`](../README.md) for the full disclaimer and risk warning.
