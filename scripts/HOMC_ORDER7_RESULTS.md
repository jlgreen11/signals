# Tier 0 experiment — HOMC at order=7 on BTC

**Run date**: 2026-04-10
**Source paper**: Nascimento et al., *Extracting Rules via Markov Chains for
Cryptocurrencies Returns Forecasting*, Computational Economics (2022).
**Paper's claim**: BTC, ETH, and XRP have **7 steps of memory** when prices
are discretized at granularity 0.01 (1% return bins).
**Question**: Does that finding transfer to the signals execution model
(walk-forward, fees, slippage, real positions)?

## How it was run

```bash
signals backtest sweep BTC-USD --model homc \
  --start 2018-01-01 --end 2024-12-31 \
  --states 5 --order 7 \
  --buy-grid "10,15,20,25,30" \
  --sell-grid "-10,-15,-20,-25,-30" \
  --stop-grid "0" \
  --no-short \
  --rank-by sharpe \
  --top 10 \
  --holdout-frac 0.2
```

- 25 (buy × sell × stop) configurations searched
- Train portion: 2018-01-01 → 2023-08-07 (in-sample)
- Holdout: 2023-08-08 → 2024-12-31 (512 bars, never seen during sweep)
- Walk-forward retraining on rolling 252-bar windows, every 21 bars
- Quantile-binned returns (5 quantile states)
- Wall time: **30 seconds** for 25 configs

## Results

### In-sample sweep (top 10 by Sharpe)

| buy_bps | sell_bps | stop | Final | CAGR | Sharpe | DSR | Max DD | Calmar | trades |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | -10 | 0% | $22,879 | 20.9% | 0.54 | **0.00** | -54.9% | 0.38 | **8** |
| 30 | -15 | 0% | $22,879 | 20.9% | 0.54 | 0.00 | -54.9% | 0.38 | 8 |
| 30 | -20 | 0% | $22,879 | 20.9% | 0.54 | 0.00 | -54.9% | 0.38 | 8 |
| 30 | -25 | 0% | $22,879 | 20.9% | 0.54 | 0.00 | -54.9% | 0.38 | 8 |
| 30 | -30 | 0% | $22,879 | 20.9% | 0.54 | 0.00 | -54.9% | 0.38 | 8 |
| 20 | -10 | 0% | $19,832 | 16.6% | 0.48 | 0.00 | -51.6% | 0.32 | 11 |
| 20 | -15 | 0% | $19,832 | 16.6% | 0.48 | 0.00 | -51.6% | 0.32 | 11 |
| 20 | -20 | 0% | $19,832 | 16.6% | 0.48 | 0.00 | -51.6% | 0.32 | 11 |
| 20 | -25 | 0% | $19,832 | 16.6% | 0.48 | 0.00 | -51.6% | 0.32 | 11 |
| 20 | -30 | 0% | $19,832 | 16.6% | 0.48 | 0.00 | -51.6% | 0.32 | 11 |

Buy & hold benchmark over the same period: **$35,075, CAGR 34.9%, MDD -76.6%,
Calmar 0.46**.

### Holdout validation of the top config (buy=30, sell=-10)

| Metric | Train (in-sample) | Holdout (out-of-sample) |
|---|---:|---:|
| CAGR | 20.94% | **1.41%** |
| Sharpe | 0.54 | 0.66 |
| Max DD | -54.93% | **-0.13%** |
| Calmar | 0.38 | 10.72 |
| # trades | 8 | **1** |

## Verdict — the paper's claim does NOT transfer

This is a definitive negative result. Three independent tells:

1. **DSR = 0.00 on every config.** With 25 trials and N=1429 in-sample bars,
   the expected max Sharpe under H₀ (true SR = 0) is roughly 1.4. Every
   observed Sharpe is below 0.55. The deflated Sharpe correctly identifies
   that none of these results survive multi-trial deflation — they are
   indistinguishable from noise.

2. **All `sell_bps` rows in each `buy_bps` cluster are identical.** Rows 1-5
   (buy=30, sell varying from -10 to -30) all produce the same 8 trades, the
   same 20.9% CAGR, the same Sharpe. This means **the model never actually
   produces a sell signal within those threshold bands**. The HOMC's
   expected-return distribution at order=7 is too compressed to ever exceed
   even the loosest sell threshold of -10 bps. The strategy degenerates to a
   "buy occasionally and never sell" rule, with the few trades coming from
   thin buy crossings.

3. **Holdout shows the strategy effectively stops trading.** 1 trade in 512
   holdout bars vs 8 trades in 1429 train bars. The holdout MDD of -0.13% is
   not "the strategy avoided losses" — it's "the strategy did nothing".
   Holdout CAGR 1.41% is far below the train CAGR of 20.9% and below
   buy & hold over the same period. The Calmar of 10.72 on the holdout is a
   numerical artifact of dividing a small CAGR by a near-zero drawdown that
   happens because the position was almost always flat.

## Why the paper's finding doesn't carry over

The Nascimento paper used a **single 95/5 train/test split on 1,156
observations** (1,098 training, 58 test) and measured **forecast accuracy via
MAPE** — never trading. With 1,098 observations and a small absolute-bin
state space, even 7-tuples repeat enough that the empirical transition table
covers most observed histories.

This signals project uses **walk-forward retraining on 252-bar windows**
because that's required for an adaptive strategy. **252 / 5⁷ = 0.003 — there
are 78,125 possible 7-tuples and only 245 actual ones in the window.** Almost
every 7-tuple seen at inference time was unobserved during training, so the
HOMC falls back to its marginal distribution. The marginal distribution has
expected next-bar returns very close to zero, so the buy/sell thresholds
almost never fire. The model is **structurally undertrained at order=7
within a walk-forward 252-bar window** — there isn't enough data to estimate
the transition table densely enough to be useful.

The paper's headline finding is real *as a statement about offline forecast
accuracy on a static dataset*, but it does not survive contact with:

- **Walk-forward retraining** (training windows shrink the effective sample
  per fit by ~4×)
- **Quantile binning** (the bins drift between training and inference, so
  even the 7-tuples that *did* repeat in raw return-space now look different)
- **Magnitude-thresholded execution** (the marginal-fallback predictions are
  too compressed to clear any reasonable trading threshold)

## Implications for the roadmap

This result moves several Tier-1 items in `IMPROVEMENTS.md`:

- **Tier-1 #1 (AbsoluteGranularityEncoder)**: still worth building, but the
  expected upside is smaller. The dominant bottleneck is sample size per
  walk-forward window, not bin scheme. Building it is justified mainly to
  *test* whether the paper's finding holds at all (not because it's likely
  to ship a better model).
- **Tier-1 #2 (order/granularity sweep flags)**: still valuable, but the
  sweep result already says order ≥ 5 is untenable on 252-bar windows.
  Useful sweep range is now **order ∈ {1, 2, 3}** rather than {1, 3, 5, 7, 9}.
- **Tier-1 #5, #6, #7 (asymmetric sizing, cap removal, multi-asset eval)**:
  these become **higher priority**. The HOMC path is dead-end without much
  larger training windows; the composite + better sizing path is the
  realistic frontier.
- **Tier-2 #8 (purged k-fold cross-validation)**: more important now. The
  random-window eval already gave a single point estimate of strategy
  performance; CV gives a confidence interval, which is what we need to
  decide whether the composite-strategy improvements are real or chance.
- **Tier-3 #12 (sparse k-tuple representation)**: deprioritize. The fix
  doesn't help the actual problem (sample size); it just lets us run a model
  variant we've now shown is broken at this sample size.

## What I would test next

The honest follow-up experiment is: **does the HOMC start to work at all if
the training window is much larger?**

```bash
signals backtest sweep BTC-USD --model homc \
  --start 2015-01-01 --end 2024-12-31 \
  --states 5 --order 5 \
  --train-window 1000 \
  --buy-grid "10,15,20,25,30" \
  --sell-grid "-10,-15,-20,-25,-30" \
  --stop-grid "0" \
  --no-short \
  --rank-by sharpe \
  --top 10 \
  --holdout-frac 0.2
```

(Order 5 not 7, because even at 1000 bars the 7-tuple cardinality is
borderline. Train window 1000 not 252, to give the higher-order chain enough
samples per k-tuple to actually estimate transitions.)

If this run produces a DSR > 0.5 and a holdout Sharpe ≥ 0.5, the HOMC
backend is salvageable with a wider window and the right order. If it
doesn't, **HOMC should be permanently demoted to a research-only backend**
and the production default should stay composite-3×3 with the asymmetric-
sizing improvements from Tier-1 #5 and #6.

## Reproducibility

Re-run from the repo root with the command in the "How it was run" section
above. Wall time is ~30 seconds with the current dependency set on a 2024
M-series Mac. The same command on different hardware should produce
bit-identical numbers (the engine and HOMC are deterministic; Quantile bin
edges are computed from the training window only and have no random
component).

---

⚠ **Disclaimer**: This is an experimental research project. The results
shown above are backtests and historical simulations — they do not
predict future performance. Not financial advice. See the root
[`README.md`](../README.md) for the full disclaimer and risk warning.
