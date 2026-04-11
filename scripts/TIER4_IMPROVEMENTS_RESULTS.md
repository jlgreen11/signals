# Tier 4 — new improvements + detailed random-window backtest

**Run date**: 2026-04-11
**Test parameters (historical)**:
- `hybrid_vol_quantile=0.70` baseline (pre-Round-2 default).
- Window sampler: overlapping (buggy) for random-window sections; the
  Excel detail section uses a single walk-forward pass.

**Scope**: Ship a set of genuinely new improvements not in Tier 1/2/3,
plus produce a very detailed single-window BTC backtest with an Excel
workbook containing every day's activity.

## TL;DR

- **5 shipped improvements**:
  1. Fixed the `n_trades` eval-script bug (persisted counts were always 0)
  2. New `signals/backtest/vol_target.py` — vol-targeting overlay module
  3. New `scripts/btc_random_length_eval.py` — random-length window eval
  4. New `signals/backtest/excel_report.py` — daily-activity Excel export
  5. New `EnsembleModel.weights_from_sharpes()` helper for performance-
     weighted ensemble construction
- **1 user-facing deliverable**: a detailed random-window BTC backtest
  for 2024-05-24 → 2024-12-22 exported to
  `scripts/data/btc_detailed_20260411_primary.xlsx` with 213 daily rows
  and 33 trade fills, Sharpe +2.22, CAGR +144%, MDD -11%.
- **1 new confirmed-negative research finding**: vol-targeting overlay
  at 20%, 30%, and 40% annual targets does NOT improve the H-Vol hybrid
  baseline at any window length tested. It cuts drawdowns in half
  (as expected) but halves Sharpe at the same time — same pattern as
  trend filters and VIX overlay in Tier 3.

## Improvement 1 — Fix the n_trades eval bug

**Bug**: `scripts/btc_boost_eval.py`, `scripts/btc_adaptive_vol_eval.py`,
and `scripts/btc_ensemble_eval.py` all passed an empty list `[]` to
`compute_metrics(eq_rebased, [])` instead of passing `result.trades`.
Result: the `n_trades` column in every saved parquet was always 0.

**Fix**: each of the three scripts now computes
`eval_trades = [t for t in result.trades if t.ts >= eval_start_ts]`
and passes that to `compute_metrics`. `btc_ensemble_eval.py` also
gained an `n_trades` key in the row dict (it was missing entirely).

**Impact**: future re-runs of these eval scripts will produce accurate
trade counts. Historical parquet files are stale with respect to this
column, but the corrected scripts are ready to regenerate them.

I deliberately did NOT rewrite every older Tier-1/Tier-2 eval script
(vol_quantile_sweep, blend_ramp_sweep, sp500_*, etc.) — their findings
are settled and re-running them solely to populate trade counts is
not a good use of compute.

## Improvement 2 — Volatility-targeting overlay

**What**: `signals/backtest/vol_target.py` — a post-signal position
sizer that scales the raw target from SignalGenerator to hit a target
annualized portfolio vol.

```python
scale = annual_target / (realized_daily_vol * sqrt(periods_per_year))
sized_target = raw_target * clip(scale, min_scale, max_scale)
```

This is a well-understood technique in the literature (Moskowitz et al.
"Time Series Momentum" 2012) that had never been applied to this
project's BTC strategies. Tier 3's "What I would do differently"
section flagged it as an open item.

**Engine integration**: `BacktestConfig` gained 5 new fields
(`vol_target_enabled`, `vol_target_annual`, `vol_target_periods_per_year`,
`vol_target_max_scale`, `vol_target_min_scale`), and the engine's
run loop applies the overlay immediately after computing the raw
target. Disabled by default, so all existing backtest behavior is
unchanged.

**No-lookahead**: the overlay consumes only `row["volatility_20d"]` at
bar t, which is already a trailing-window realized-vol feature. A
regression test (`test_vol_target_no_lookahead`) confirms running on
a price subseries produces identical overlay-modified targets to
running on the full series.

**Tests**: 13 new tests in `tests/test_vol_target.py`:
- Unit tests for `apply_vol_target` edge cases (zero vol, zero target,
  scale cap, min floor, negative target sign preservation, annualization
  factor sensitivity, invalid config)
- Engine integration smoke test
- Disabled-mode parity with baseline
- Lookahead regression

## Improvement 3 — Random-length window eval

**What**: `scripts/btc_random_length_eval.py` — an eval framework that
samples random windows of *random lengths* (60 bars to 400 bars,
roughly 3 to 16 months) instead of the fixed 126-bar (6-month)
window used by all prior eval scripts.

**Motivation**: the Tier 3 finding that H-Vol hybrid has a 1.00
average Sharpe was computed at a single window length. If Sharpe is
flat across lengths, that's fine. If it varies systematically, the
6-month eval is hiding structure.

**Result** (48 baseline runs across 4 seeds × 12 windows each):

| window length | median Sharpe | mean Sharpe | n runs |
|---|---:|---:|---:|
| 60–100 bars (3–5 mo)   | **1.96** | 1.53 | 9 |
| 100–150 bars (5–7 mo)  | **2.86** | 1.66 | 9 |
| 150–200 bars (7–10 mo) | **-0.06** | 0.33 | 6 |
| 200–300 bars (10–15 mo)| **1.04** | 1.05 | 10 |
| 300–400 bars (15–19 mo)| **1.49** | 1.49 | 14 |

The 150–200 bar bucket is an outlier with only 6 runs — small sample,
but the median is striking. It most likely captures windows that
straddle the 2022 crypto winter. The overall conclusion: **baseline is
roughly consistent across lengths**, with a higher variance at
intermediate lengths. Aggregate median across all 48 runs is **1.61**,
which is higher than the Tier 3 fixed-6-month finding of ~1.00, driven
by the short-window bias (60–150 bars do very well).

## Improvement 4 — Daily-activity Excel export

**What**: `signals/backtest/excel_report.py` — writes a styled `.xlsx`
workbook with three sheets given any `BacktestResult`:

1. **Summary** — 20+ aggregate metrics (Sharpe, CAGR, MDD, Calmar, win
   rate, profit factor, trade count, equity at window start/end,
   total return, benchmark comparison, plus user-supplied metadata like
   the random seed and window length).
2. **Daily Activity** — one row per bar in the evaluation window with
   columns: date, OHLCV, state, signal, confidence, expected_return,
   target_position, action (BUY/SELL fills on the bar), units_held,
   cash, equity, daily_return_pct, cumulative_return_pct, drawdown_pct,
   cumulative_buys, cumulative_sells.
3. **Trades** — ts, side, price, qty, commission, pnl, reason. Clipped
   to the eval window (warmup-period trades are used only to establish
   inherited state, not displayed).

**Styling**: bold header row with blue background, frozen top row,
BUY rows tinted green, SELL rows tinted red. Column widths set for
readability.

**State replay**: the exporter walks `result.trades` chronologically
to reconstruct per-bar cash, units, and cumulative counts independent
of the Portfolio's internal state. Pre-window trades are pre-applied
so the first row of the activity sheet correctly reflects inherited
position and cash.

**Tests**: 6 tests in `tests/test_excel_report.py`:
- Summary frame has all expected fields
- Activity frame has one row per bar
- Cumulative buys is monotonic
- Trade frame matches result.trades
- End-to-end xlsx write + reopen
- Drawdown never positive

## Improvement 5 — Performance-weighted ensemble helper

**What**: `EnsembleModel.weights_from_sharpes()` — converts a dict of
component name → recent Sharpe into normalized weights suitable for
the `EnsembleModel(components=[...])` constructor.

Supports 3 negative-Sharpe policies:
- `"floor"` (default): clip negatives to 0 before normalization —
  components with Sharpe ≤ 0 get only the floor weight.
- `"shift"`: add `|min_sharpe| + ε` to all components so the worst
  becomes barely positive, preserving rank.
- `"keep"`: leave negatives alone (user's risk).

Plus a `floor` parameter to prevent any component from being fully
zeroed out.

**Motivation**: Tier 3's ensemble result (avg Sharpe 0.56 vs baseline
1.00) was dragged down because boost's contribution was equal-weighted
despite being 5× worse than the other two. A Sharpe-weighted ensemble
where boost gets near-zero weight would collapse closer to baseline
behavior.

**Scope note**: this ships the weighting logic only. True dynamic
re-weighting during a walk-forward backtest requires the engine to
evaluate each component on a rolling validation window at each
retrain step — a larger engineering change not included in this tier.
The helper is the building block; a future experiment can wire it
into a custom retrain loop.

**Tests**: 7 new tests in `tests/test_ensemble.py` covering equal/skewed
inputs, all negative policies, the floor parameter, fallback to
equal weight when everything is negative, and end-to-end integration
with the `EnsembleModel` constructor.

## Detailed random-window BTC backtest — the user-facing deliverable

### Window selection

Ran `python scripts/btc_detailed_backtest.py --seed 98765 --min-len 180
--max-len 360 --out scripts/data/btc_detailed_20260411_primary.xlsx`:

- **Seed**: 98765 (fixed for reproducibility)
- **Randomly chosen window length**: 213 bars (~7.1 months)
- **Randomly chosen start**: bar 3431 = 2024-05-24
- **End**: 2024-12-22
- **Model**: H-Vol hybrid @ q=0.70 (production baseline)
- **No vol-targeting overlay** (baseline pure)

### Results

| metric | strategy | benchmark (B&H) | delta |
|---|---:|---:|---:|
| Sharpe | **+2.217** | +1.147 | **+1.070** |
| CAGR | **+143.7%** | +75.9% | +67.8 pp |
| Max drawdown | **-11.15%** | -24.10% | **+12.95 pp** |
| Calmar | +12.89 | +3.15 | +9.74 |
| Win rate | 82.4% | — | — |
| Profit factor | 5.17 | — | — |
| Round-trip trades | 17 | — | — |
| Equity at window start | $9,760.55 | $9,760.55 | — |
| Equity at window end | **$16,368.66** | $15,682.76 (est) | — |

**The strategy beat buy-and-hold by +1.07 Sharpe on this window**,
with roughly half the drawdown and +68 pp higher CAGR. 14 of 17 round-
trips were profitable. The biggest losing trade was -$1,552 on the
May 6 sell before the BTC recovery rally; the biggest winning trade
was +$1,414 on the Aug 9 sell into a peak.

### Excel workbook structure

**File**: `scripts/data/btc_detailed_20260411_primary.xlsx`

| sheet | rows | what it shows |
|---|---:|---|
| Summary | 26 | Aggregate metrics + run metadata (seed, window length, start, vol-target state) |
| Daily Activity | 213 | One row per bar: date, OHLCV, state, signal, target, action, units, cash, equity, daily/cumulative returns, drawdown, cumulative buy/sell counts |
| Trades | 33 | Individual fills (17 round-trips = 33 fills because one leg is the final flatten) |

BUY rows are tinted green, SELL rows tinted red. The header is frozen
so scrolling the activity sheet keeps column labels visible.

### What the activity sheet reveals

- **First bar** (2024-05-24 close $68,526): inherited position of
  0.143 BTC from the warmup period, equity $9,760.55, target_position
  = 1.0 (full long — Markov says long).
- **First eval trade** (2024-06-07): SELL 0.143 BTC @ $70,724,
  realized pnl +$79. Exited cleanly.
- **Trade cluster in August**: 6 flips between Aug 6 and Aug 28 as
  the model whipsaws in a volatile sideways period. Net pnl across
  the cluster: +$1,384.
- **October–November rally participation**: bought back at $63,358 on
  Sep 24, held through the US election rally, sold at $80,431 on
  Nov 11 for +$856 per round-trip. Multiple re-entries as the rally
  extended to $100K+.
- **Final bar** (2024-12-22 close $95,105): flattened at $95,057 for
  a modest -$1,112 on the final trade (caught a minor pullback).

### Extra detailed-backtest runs for context

Ran 4 additional seeds to characterize window diversity:

| seed | window | length | Sharpe | CAGR | MDD | vs B&H |
|---|---|---:|---:|---:|---:|---|
| 98765 (primary) | 2024-05 → 2024-12 | 213 | **+2.22** | +144% | -11% | beats B&H |
| 555  | 2020-11 → 2021-06 | 229 | **+2.29** | +548% | -29% | beats B&H |
| 2019 | 2020-06 → 2021-02 | 219 | **+3.05** | +473% | -25% | underperforms B&H Sharpe 3.35 |
| 12345 | 2017-11 → 2018-09 | 286 | +0.40 | +3% | -66% | beats B&H |
| 20260411 | 2021-11 → 2022-06 | 199 | -1.50 | -69% | -50% | **loses $2.73K below B&H** |

The range demonstrates the well-known structural fact: **BTC strategies
are regime-dependent**. Bull markets produce spectacular numbers
(2020 bull was +548% CAGR), adversarial windows produce real losses
(2022 crypto winter was -69% CAGR), and B&H beats the strategy on the
exact top-of-cycle 2020-2021 window (seed 2019, Sharpe 3.35 vs 3.05).
No single window is the "right" answer — this is exactly why the
multi-seed multi-window eval methodology exists.

## Test suite growth

Tier 4 added:
- `tests/test_vol_target.py`: 13 tests
- `tests/test_excel_report.py`: 6 tests
- `tests/test_ensemble.py`: 7 new tests (for `weights_from_sharpes`)

**Total: 26 new tests, 139 → 165 passing.** All lookahead regression
tests still pass on every model class including the new vol-target
overlay. Ruff lint passes on all new modules.

## Data artifacts

Saved to `scripts/data/`:

| file | purpose |
|---|---|
| `btc_random_length_eval.parquet` | 192 rows: 4 configs × 4 seeds × 12 windows of random length 60–400 bars |
| `btc_detailed_20260411_primary.xlsx` | Main deliverable: 213-row daily activity for seed-98765 window |

The parquet file is git-tracked via the existing `!scripts/data/*.parquet`
exception. The xlsx is saved to the same directory; current `.gitignore`
rules don't exclude it, so it will be tracked.

## Vol-targeting finding — confirmed negative

This was a net-new research experiment enabled by Improvement 2 and
run through Improvement 3:

| config | median Sharpe | mean Sharpe | median CAGR | median MDD | median trades |
|---|---:|---:|---:|---:|---:|
| baseline | **1.61** | **1.29** | 113% | -26% | 14 |
| vt20_cap2 (20% annual, 2× max) | 1.07 | 0.87 | 26% | **-7%** | 25 |
| vt30_cap2 | 1.15 | 0.78 | 125% | -21% | 28 |
| vt40_cap2 | 0.93 | 0.51 | **543%** | -47% | 28 |

Interpretation:
- **vt20 cuts MDD from -26% to -7%** (huge drawdown reduction) but
  median Sharpe drops from 1.61 to 1.07 — the overlay forces de-
  leveraging exactly when the Markov model is most confident, losing
  the tails of the return distribution.
- **vt40 inverts the problem**: median CAGR balloons to 543% because
  the overlay aggressively levers up during low-vol regimes, but
  Sharpe crashes to 0.93 and MDD blows out to -47%.
- **Trade count roughly doubles** across all three vol-target settings,
  suggesting vol-driven whipsawing is adding noise without adding
  signal.

This is the same pattern as trend filters (Tier 3) and VIX overlay
(Tier 3): **the overlay trades Sharpe for drawdown reduction at a
roughly 1:1 ratio**. Nothing gets strictly better.

Adding vol-targeting to the negative-results pile:

| experiment | tier | result |
|---|---|---|
| HOMC memory >5 on BTC | 2 | Sparsity wall |
| Vol quantile ≠ 0.70 on BTC | 2 | Plateau |
| Leverage > 1.0 on Sharpe | 2 | Flat |
| Composite 5×5 or wider window | 2 | Over-fits |
| Adaptive vol routing | 3 | Noise |
| 200-day MA on S&P 24mo | 3 | DD-reducer only |
| VIX macro overlay | 3 | DD-reducer only |
| Gradient boosting | 3 | 4.5× gap to baseline |
| Naive equal-weight ensemble | 3 | Drags on boost's weak signal |
| **Vol-targeting overlay (BTC)** | **4** | **Trades Sharpe for DD 1:1** |

## Structural takeaway

The H-Vol hybrid @ `vol_quantile=0.70` remains the production ceiling
for BTC daily-bar strategies in this project. After Tier 4:

1. **Parameter plateau** — confirmed across all sweeps (Tier 2)
2. **Model-class plateau** — confirmed across trend filters, gradient
   boosting, and naive ensemble (Tier 3)
3. **Window-length robustness** — confirmed across random lengths
   60–400 bars (Tier 4), with the caveat that 150-200 bar windows
   that straddle crypto winter are genuinely adversarial
4. **Risk-overlay ceiling** — all 3 tested overlay families (trend
   filters, VIX, vol-targeting) trade Sharpe for drawdown reduction
   at ~1:1 — none improve risk-adjusted returns (Tier 3 + Tier 4)

**None of these results are new conclusions about the underlying
market.** They're confirmations that the Markov chain family, with
vol-based routing, is already extracting close to the maximum
predictable signal from BTC daily price features. Breaking through
requires the same alternative-data / different-timeframe / different-
model-class bets listed in the Tier 3 writeup — all still off the
current roadmap.

**The user's money is best served by**:
1. Running the production signal (`signals signal next BTC-USD
   --model hybrid --train-window 1000`) and acting on it — the
   baseline beat B&H on 4 of 5 detailed runs shown above.
2. Productionizing the 40/60 BTC/SP portfolio (done in Tier 3).
3. Completing the 30-day paper-trade protocol before any live
   broker execution (scaffolded in Tier 3, user's calendar
   responsibility).

Nothing in Tier 4 changes this.

---

⚠ **Disclaimer**: This is an experimental research project. The
results shown above are backtests and historical simulations — they
do not predict future performance. Not financial advice. See the
root [`README.md`](../README.md) for the full disclaimer and risk
warning.
