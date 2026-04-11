# Skeptic's Corner, No. 14 — `jlgreen11/signals`: The Markov-Chain BTC Bot That Quietly Benchmarks Against Itself

> "Median Sharpe 2.15 across 16 random 6-month BTC windows — the highest result in the project's history."
> — [`README.md`](./README.md), line 10

I write this column to examine the wave of AI-assisted quant repos that have
been landing on GitHub over the last year. Most of them follow a familiar arc:
plausible README, a hundred passing tests, a headline backtest number, and —
under the hood — a methodology that would not survive five minutes in front of
a sell-side review committee. `jlgreen11/signals` is more interesting than
most. The engine code is surprisingly clean, the author has clearly read
Bailey & López de Prado, and the repo includes the kind of self-incriminating
result docs that most projects bury. That's why it makes for such a good
teardown: the failures here are *methodological and epistemic*, not
coding-error failures, and they are failures in the patterns that ship the
most often in AI-generated quant work.

Let me be clear up front: I'm going to tear this apart, but the author has
done work most "AI-generated trading bot" repos don't even attempt — a
walk-forward engine with real lookahead regression tests, a deflated Sharpe
implementation, multi-seed robustness checks, and an explicit `SECURITY.md`.
I'll give credit where it's due at the end. The critique below is about the
distance between what the repo *claims* in its README and what the repo
*actually establishes* in its own data.

## TL;DR — the seven red flags

| # | Flag | Severity |
|---|---|---|
| 1 | **The 2.15 headline Sharpe is a single-seed result.** Across 4 seeds the same config averages **1.00**, with one seed (7) going **negative**. | 🔴 Critical |
| 2 | **"16 random non-overlapping 6-month windows" is false.** The sampler doesn't enforce non-overlap. Four of the 16 windows span Jan 5 – Feb 2, 2019 — they share 75–95% of their bars. Effective sample count is closer to 6–8. | 🔴 Critical |
| 3 | **Every reported config has DSR = 0.00.** The author overrides the test based on out-of-sample evidence, but the out-of-sample window is a 2023–2024 BTC uptrend that was *also* seen by the hyperparameter tuning. | 🔴 Critical |
| 4 | **The "tightened defaults" were tuned over a window that overlaps the holdout.** Memorialized in the project's own notes; never corrected in the README. | 🟠 High |
| 5 | **Asset scope is survivorship-biased.** ETH and SOL were "deprioritized" after the hybrid model produced a **-0.40 Sharpe** on ETH. The README frames this as "production scope," not as a failure. | 🟠 High |
| 6 | **Transaction costs are a single point (5 bps + 5 bps).** No sensitivity surface. A strategy whose edge lives or dies on a cost assumption has not been cost-tested. | 🟡 Medium |
| 7 | **BTC Sharpe is annualized at 252, not 365**, in direct contradiction to the project's own `vol_target.py`. Paradoxically this *understates* the headline, but it's sloppy and internally inconsistent. | 🟡 Low |

Now the long version.

---

## 1. The headline is the tail of the distribution

This is the single most important thing in this critique, so I'll be as
concrete as possible.

The README ([`README.md:10`](./README.md)) advertises:

> H-Vol @ q=0.70 … median Sharpe **2.15** across 16 random 6-month windows.

The sweep result doc
[`scripts/BTC_DEEP_SWEEP_RESULTS.md`](./scripts/BTC_DEEP_SWEEP_RESULTS.md),
lines 175–180, shows what actually happens when you take that same exact
baseline config and re-run the evaluation under three alternative random
seeds:

| Config | seed 42 | seed 7 | seed 100 | seed 999 | **avg** |
|---|---:|---:|---:|---:|---:|
| **baseline (H-Vol @ q=0.70)** | **2.15** | **−0.27** | 1.38 | 0.74 | **1.00** |

Let me spell out exactly what this table says, because the author does not
spell it out in the README:

1. The "best result in project history" appears at **only one of the four
   seeds tested**.
2. At **seed 7**, the same config produces a **negative** Sharpe on BTC —
   i.e., the strategy is *worse than cash*.
3. The mean across the four seeds is **1.00** — approximately tied with the
   16-window buy-and-hold mean Sharpe of ~1.03 that appears elsewhere in the
   repo.
4. The spread is **±1.21 Sharpe** from seed to seed — which is astronomically
   larger than any parameter-tuning effect the author reports.

What the author *does* say in that same document is accurate and correct: the
sweep-winning configs (e.g. `sell_bps=-20` at a seed-42 Sharpe of **2.40**)
are data-mined artifacts, and the correct conclusion is that they don't beat
baseline. Good. But the logical chain doesn't stop at the sweep winners — it
applies just as forcefully to the **baseline itself**. The baseline was also
chosen (over the "legacy" composite-3×3 default) because of its seed-42
performance. The same evidence that disqualified `sell_bps=-20` disqualifies
`q=0.70` as the "best" quantile — at least until someone runs the full sweep
at 10+ seeds and picks the multi-seed winner.

The author's own disciplinary rule — explicitly memorialized in
[`scripts/BTC_DEEP_SWEEP_RESULTS.md`](./scripts/BTC_DEEP_SWEEP_RESULTS.md) — is:

> Every sweep that produces a "winner" MUST be re-run at 3+ alternative seeds
> before promotion.

The baseline is the output of a 16-window sweep. It has never been re-run at
3+ seeds with the same multi-seed criteria used to kill its competitors. The
rule is applied to challengers and waived for incumbents.

**Headline-worthy number corrected for seed variance**: `H-Vol @ q=0.70,
4-seed average Sharpe ≈ 1.00 ± 0.6`. That's the number that should be in the
README. In fairness, this would still put the hybrid roughly at parity with
buy-and-hold on Sharpe — which is consistent with the well-known difficulty
of beating a levered-long benchmark in a multi-year BTC bull. But
"approximately as good as buy-and-hold" is a very different marketing pitch
than "2.15, best in project history."

---

## 2. "16 random non-overlapping 6-month windows" — except they overlap a lot

This one is a documentation-vs-code discrepancy, and it matters because it
inflates the apparent statistical power of *every* result in the project.

The pinned evaluation doc
[`scripts/RANDOM_WINDOW_EVAL.md`](./scripts/RANDOM_WINDOW_EVAL.md), line 7,
states verbatim:

> **Windows**: 16 random **non-overlapping** 6-month (126 trading-bar) windows

The actual implementation at
[`scripts/random_window_eval.py`](./scripts/random_window_eval.py), lines
208–209:

```python
rng = random.Random(seed)
starts = sorted(rng.sample(range(min_start, max_start), n_windows))
```

`random.sample(...)` draws N distinct start indices. It does not enforce any
spacing between them. With a 126-bar window and ~1,800 eligible starts in
BTC's 2015–2024 history, you are *guaranteed* overlap.

You don't need to trust me on this. The per-window results table in the same
doc is self-indicting:

| # | Window | Start | End |
|---|---|---|---|
| 1 | window 1 | 2019-01-05 | 2019-05-10 |
| 2 | window 2 | 2019-01-11 | 2019-05-16 |
| 3 | window 3 | 2019-01-25 | 2019-05-30 |
| 4 | window 4 | 2019-02-02 | 2019-06-07 |

The first four "random" windows are four starts within **28 calendar days**,
covering nearly the same 4-month slice of price history. Windows 1 and 2
share **~95%** of their bars. Windows 1 and 4 share **~80%**. They are not
four observations of the strategy's performance; they are approximately one
observation, replicated four times with small initial-condition perturbations.

Looking at the full 16-window list, you can visually cluster them into
roughly **6 distinct market episodes**:

| Cluster | Approx period | # "windows" it contains |
|---|---|---:|
| Early 2019 rally | Jan–Jun 2019 | 4 |
| Late 2019 → COVID crash | Sep 2019 – Apr 2020 | 4 |
| Post-COVID bull | Apr–Aug 2020 | 1 |
| May 2021 crash | Mar–Sep 2021 | 3 |
| Mid-2021 rally → crypto winter onset | Jun 2021 – Feb 2022 | 2 |
| 2023–2024 recovery | Jun 2023 – Sep 2024 | 2 |

The effective independent sample count is ~6, not 16. This is a ~2.6×
overestimate of the degrees of freedom in every "median across 16 windows"
statistic in the project — which means every reported standard error, every
Sharpe p-value, every "robustness count" like *"beats B&H in 9/16 windows"*
needs to be deflated accordingly.

The fix is one line: enforce a minimum spacing of 126 bars between sampled
starts. But the author would then need to document that a 10-year price
history can only support ~16 genuinely non-overlapping 6-month windows if
each one is placed as far from its neighbors as possible — and the
evaluation would have noticeably fewer bars to work with.

Worth noting: even if you enforced non-overlap, your 16 samples are all drawn
from the same 10-year span, so they're not IID in the statistical sense
either. The right tool here is a **moving-block bootstrap** with block length
≥ 126, resampling from the same underlying bar series. Nothing in this repo
does that.

---

## 3. DSR Theater: the deflated Sharpe reports zero; the project proceeds anyway

The author ships a correct implementation of Bailey & López de Prado's
Deflated Sharpe Ratio at
[`signals/backtest/metrics.py`](./signals/backtest/metrics.py), lines 83–108.
It is mechanically correct.

It is also ignored.

From the project's own memory and the result docs:

- [`scripts/HOMC_ORDER7_RESULTS.md`](./scripts/HOMC_ORDER7_RESULTS.md):
  "All 25 sweep configs had DSR = 0.00."
- [`scripts/HOMC_ORDER5_W1000_RESULTS.md`](./scripts/HOMC_ORDER5_W1000_RESULTS.md):
  "DSR (in-sample) = 0.00 for both HOMC and Composite."
- [`scripts/HOMC_TIER0C_HYBRID_RESULTS.md`](./scripts/HOMC_TIER0C_HYBRID_RESULTS.md):
  "H-Vol's in-sample DSR is still 0.00 on every test."
- [`IMPROVEMENTS.md`](./IMPROVEMENTS.md), Tier-0d notes: "DSR correctly
  identified the in-sample as indistinguishable from noise, but it cannot
  tell you when an in-sample-noisy model has out-of-sample structure."

That last sentence is the rationalization that the whole project's promotion
logic now rests on. Let me state it plainly: Bailey & Prado's DSR is
specifically a test of whether an observed Sharpe from a multi-trial search
is distinguishable from the max of N IID noise draws under
H₀: true SR = 0. The author encounters a situation where DSR = 0 for every
config in a 25-config sweep, and concludes that DSR is "necessary but not
sufficient" and overrides it with an out-of-sample validation.

This would be defensible if the out-of-sample window were pristine. It isn't.
The "out-of-sample validation" is the trailing 20% of the same series, and
the "tightened defaults" that produced the headline numbers — `vol_window=10`,
`alpha=0.01`, `buy_bps=25`, `sell_bps=-35`, `target_scale_bps=20`,
`retrain_freq=21` — were tuned in iterations that had full visibility of that
same trailing 20%. The project's own internal notes say:

> Caveat: the tightened defaults were originally tuned over a window that
> overlaps the holdout, so it's not a pristine OOS test. A truly clean test
> would re-tune on 2018–2022 only and then evaluate on 2023–2024.

So the decision chain is:

1. DSR = 0 → "DSR is misleading."
2. Holdout Sharpe = 1.26 → "This is our validation."
3. (Acknowledged but never corrected: the holdout was used in tuning.)
4. Random-window eval (seed 42) Sharpe = 2.15 → "This is the headline."
5. (Acknowledged but never surfaced to README: at seed ≠ 42, the same config
   averages 1.00.)

Any *one* of steps 1/3/5 would be forgivable in isolation. Stacked, they
produce a number that has walked around every single statistical guardrail
Bailey & Prado wrote the DSR to enforce in the first place. The DSR is the
control, and the control fires, and the author overrides the control, and
the result is then promoted. This is classic DSR theater: include the test
because it's in the literature, but route around it when it disagrees.

---

## 4. The "walk-forward engine is lookahead-clean" — and the lookahead is in the metadata instead

I want to give the author credit on one point here:
[`tests/test_lookahead.py`](./tests/test_lookahead.py) is a genuinely good
test. It asserts bit-identical equity curves up to bar N regardless of how
much future data is in the input. Covers composite, HOMC, HMM, and hybrid
routing. This is the cheapest and most effective data-leakage canary you can
write, and most AI-generated trading repos don't have it at all. If you are
building a backtest engine, steal this file.

What the lookahead test **cannot** catch is **selection-time leakage** —
i.e., when the developer picks hyperparameters based on performance over a
span of data that the backtest then evaluates against. A strict lookahead
test only catches leakage inside a single `engine.run()` call. It doesn't
catch the situation where the human-in-the-loop iterated on `buy_bps` /
`sell_bps` / `vol_window` while staring at Sharpe numbers from 2023–2024.
Which is exactly what happened here.

The "tightened defaults" in `BacktestConfig`
([`signals/backtest/engine.py`](./signals/backtest/engine.py), lines 43–124)
were chosen across multiple sweep iterations in which the full 2015–2024
price history was visible. The author then reports holdout metrics against
the same 2023–2024 window and treats them as out-of-sample evidence. This is
a subtler form of lookahead than the one the test file protects against, and
the lookahead test's presence on the README gives a false sense of security
about the class of leakage that the project *is* actually suffering from.

The correct procedure:

1. Freeze a universe of candidate hyperparameters **before looking at
   holdout data**.
2. Run walk-forward backtests on train-only data (2018–2022).
3. Choose the best config.
4. Run it *once* on 2023–2024 and report the result.
5. Never iterate on 2023–2024 again.

Step 5 is the one that gets skipped in every AI-assisted research loop,
because the whole point of fast iteration is that you can re-check the
"holdout" after every change. The moment you do, it isn't a holdout.

---

## 5. The hybrid is a volatility regime filter wearing a Markov-chain costume

The author's central claim is that a regime-routed ensemble of the composite
Markov chain and the higher-order Markov chain beats both individually —
because composite has bear defense and HOMC has bull participation, and a
good router uses each where it's strongest.

Take a look at how the router actually decides
([`signals/model/hybrid.py`](./signals/model/hybrid.py), lines 303–310):

```python
# "vol" or "blend": get the latest non-NaN 20d vol and compare
# to threshold.
assert self._vol_threshold_value is not None
vol_series = observations["volatility_20d"].dropna()
if vol_series.empty:
    return "neutral"
current_vol = float(vol_series.iloc[-1])
return "bear" if current_vol >= self._vol_threshold_value else "bull"
```

That's it. That is the entire "regime detector" for the production default:
is trailing realized volatility above its 70th percentile in the training
window or not? If yes, route to composite (which the author has tuned to be
risk-off); if no, route to HOMC (which the author has tuned to be risk-on).

There's nothing wrong with a vol-regime filter. Vol-switched trend-following
is a respected technique in the equity literature going back decades. What
*is* a problem is the framing: the README describes this as a "regime-routed
ensemble" of two Markov chain models, but structurally the only thing the
Markov chains are doing in the high-vol branch is a *flatten-and-wait*
behavior that a simple vol-threshold rule would accomplish with one line of
pandas. The composite chain produces an expected return that the engine
compares to `sell_threshold_bps = -35`; when you route to composite in
high-vol periods, the composite almost always produces a near-zero or
negative expected return, hits the `SELL` threshold, and flattens. The Markov
machinery is the vehicle, not the source of the edge.

You can confirm this by looking at where the routing doc itself admits the
HMM variant was abandoned because "HMM latent-state routing whipsaws on
ambiguous regimes"
([`scripts/HOMC_TIER0C_HYBRID_RESULTS.md`](./scripts/HOMC_TIER0C_HYBRID_RESULTS.md)).
The HMM was the one doing "regime detection in the strict sense." It lost to
a single quantile-of-realized-vol threshold. That is your signal that the
regime-detection sophistication isn't doing the work — the directional vol
filter is.

**What the author should test**: drop both Markov components and replace
them with `target = +1.0 if vol_20d < q70 else 0.0`. If that posts a median
Sharpe within ±0.2 of 2.15 at seed 42, the Markov chain apparatus is
decorative. I strongly suspect it will.

---

## 6. The asset scope is post-hoc survivorship

The README, prominently:

> **Production scope**: BTC-USD and ^GSPC (S&P 500).

The internal project notes say this was a deliberate "deprioritization" of
ETH and SOL. What actually happened is preserved in
[`scripts/HOMC_TIER0C_HYBRID_RESULTS.md`](./scripts/HOMC_TIER0C_HYBRID_RESULTS.md):

| Asset | H-Vol (Hybrid) holdout Sharpe |
|---|---:|
| BTC | **2.21** |
| SOL | **2.11** |
| ETH | **-0.40** ❌ |

The hybrid model — the production default — produces a **negative Sharpe**
on ETH holdout. The author's response is to take ETH out of the production
scope and note that "composite remains the default for ETH."

This is survivorship bias at the asset-universe level. A strategy that wins
on 2 of 3 major cryptos and loses on the third is not a crypto-regime-aware
model; it's a model that fit BTC-specific and SOL-specific features and
failed to generalize. Every subsequent result in the project that reports
"BTC performance" is then conditioned on having pre-selected BTC because BTC
worked — you cannot run a pure forward experiment on BTC in isolation once
you've filtered it out of a larger universe.

The S&P 500 scoping is the same pattern inverted. The author tested 4
strategy classes on S&P — Markov, vol quantile tuning, trend filters, and
HOMC memory depth sweep — all failed to beat buy-and-hold, and the
recommendation became "use B&H for S&P." Fine. But then a 40/60 BTC/SP
portfolio is shipped as "the first genuine alpha in the project that isn't
from parameter tuning"
([`scripts/BTC_SP500_PORTFOLIO_RESULTS.md`](./scripts/BTC_SP500_PORTFOLIO_RESULTS.md)),
which is unfortunate phrasing: *picking the 40/60 weight* is parameter
tuning, the only difference is the parameter lives at the
portfolio-construction layer instead of the strategy layer. The multi-seed
robustness check notes it works on 3/4 seeds and fails on seed 100. Again,
this isn't being validated with the same discipline that was used to kill
the strategy-level parameter sweeps.

The right way to frame the scope: **"This project produced a BTC-specific,
vol-regime-filtered trading rule that survives a 4-seed robustness check at
an average Sharpe of ~1.00 — roughly tied with buy-and-hold on Sharpe but
with noticeably lower drawdowns."** That's still a legitimate result. It's
just a very different-sounding result than the one on the README.

---

## 7. Transaction costs are a point estimate masquerading as a surface

At [`signals/backtest/engine.py`](./signals/backtest/engine.py), lines 90–91:

```python
commission_bps: float = 5.0
slippage_bps: float = 5.0
```

So: 10 bps round-trip cost. These are not crazy assumptions for a
sophisticated crypto execution setup (Kraken post-VIP, Coinbase Advanced,
some DMA routes). They are very aggressive assumptions for a retail user
running this via `signals paper-trade` on a Coinbase brokerage account, where
you can easily see 50–150 bps of spread + fees per round trip on a $10k
portfolio in BTC-USD.

Nothing in the repo runs the backtest at multiple cost levels. There's no
chart of "Sharpe vs commission_bps" or "CAGR vs slippage_bps." For a strategy
whose edge is largely expressed through *position sizing and rebalancing
frequency*, that is the single most important robustness test you can do. A
defensive overlay that rebalances weekly will shred on 15 bps round-trip
costs in a way that's invisible at 10 bps.

Specifically dangerous: the `min_trade_fraction = 0.20` deadband
(`BacktestConfig`, line 113) is a 20% hysteresis band on the rebalance
trigger. That was tuned into the defaults to *reduce commission churn*, which
means cost-sensitivity has already been baked into the choice of the
parameter itself — you can't cleanly vary cost assumptions without re-tuning
`min_trade_fraction` to match. The two are entangled and neither has been
swept independently.

**The test that would settle this**: a 2-D grid over `commission_bps ∈ {2,
5, 10, 15, 25}` × `min_trade_fraction ∈ {0.05, 0.10, 0.15, 0.20}` at the
production H-Vol config. Look for the Sharpe ridge — if it's narrow, the
strategy is cost-brittle.

---

## 8. Miscellaneous — the kind of thing that signals an LLM wrote it without review

These are individually minor. Collectively, they tell you how much of the
code was written with a human in the loop for each decision vs. how much was
generated and accepted.

### 8a. BTC is annualized at 252, not 365

At [`signals/backtest/metrics.py`](./signals/backtest/metrics.py), lines
36–44:

```python
def _annualization_factor(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 252.0
    deltas = (equity.index[1:] - equity.index[:-1]).total_seconds()
    median = float(np.median(deltas))
    if median <= 0:
        return 252.0
    bars_per_day = max(1.0, 86400.0 / median)
    return 252.0 * bars_per_day if bars_per_day < 24 else 365.0 * bars_per_day
```

For daily bars, `bars_per_day == 1.0`, and `1.0 < 24`, so this returns `252`.
That's the equities convention. BTC trades 365 days a year. Meanwhile, at
[`signals/backtest/vol_target.py`](./signals/backtest/vol_target.py),
line 71 — the *volatility targeting* overlay — the same author writes:

```python
periods_per_year: int = 365
```

...and the docstring explicitly says "365 for crypto (BTC trades daily), 252
for equities."

So the Sharpe metric and the vol-target overlay disagree about the number of
periods in a year for the same asset. The direction of this error
*understates* the reported BTC Sharpe by a factor of
`sqrt(252/365) ≈ 0.83`. At the 2.15 headline, the "correct" annualization
would give 2.59. This is in the strategy's favor, which is how errors like
this survive a long time: nobody's incentivized to fix a bug that makes the
number smaller. But the fact that the same file agrees with itself only some
of the time is the telltale sign of code generated in chunks without a pass
for global consistency.

### 8b. `volatility_20d` is a 10-day volatility

At [`signals/backtest/engine.py`](./signals/backtest/engine.py), line 52:

```python
vol_window: int = 10              # tightened: short window reacts faster to vol regimes
```

At [`signals/backtest/engine.py`](./signals/backtest/engine.py), line 145:

```python
feats["volatility_20d"] = rolling_volatility(feats["return_1d"], window=vol_window)
```

The column is named `volatility_20d` but computed with a 10-day window.
Forever. The feature name was presumably correct when `vol_window=20` was the
default; during one of the "tightening" iterations the window was changed
and the column name was never updated. Every consumer of the feature —
composite encoder, HMM, HOMC, hybrid router — now reads `volatility_20d` and
operates on 10d realized vol. The user-facing names in every logging
dashboard, every test fixture, every saved model manifest are wrong.

This is a small bug. It is also the kind of bug that never happens in code
you read carefully before merging, and always happens in code that was
generated, tested for correctness of behavior, and then shipped.
Naming-without-semantics is an LLM failure mode; humans notice when a column
named `_20d` contains a 10d number.

### 8c. Risk-free rate = 0 through a 4-year hiking cycle

`BacktestConfig.risk_free_rate: float = 0.0`
([`signals/backtest/engine.py`](./signals/backtest/engine.py)). The reporting
period is 2018–2024. The US 3-month Treasury averaged ~2.3% over that window
and peaked over 5% in 2023–2024. Buy-and-hold on short treasuries would have
earned ~10% cumulative over the backtest window at zero risk. The reported
Sharpe does not subtract this. Applied consistently, this isn't as severe as
it first sounds — it shifts both the strategy and the benchmark by the same
amount — but it inflates *absolute* Sharpe readings by a small amount that
compounds with the other inflating choices on this list.

### 8d. The "92 → 140+ tests passing" badge

Appears on the README. It is a code-quality signal, not a methodology signal.
The tests cover: "does save/load roundtrip work," "is the engine bit-identical
under truncation," "does the boost model fit without errors." None of them
cover: "does the headline result survive a different seed," "is the claim of
non-overlapping windows actually true," "does the DSR bound hold when you
correct for how many sweeps the project has run across tiers." Test count is
not validation count.

---

## What the author got right (credit where due)

I've been hard on this repo. Let me be explicit about the things in it that
are better than the median AI-generated trading project:

- **Real walk-forward engine with strict lookahead regression tests.**
  [`tests/test_lookahead.py`](./tests/test_lookahead.py) is the first thing
  I'd copy from this repo into a new project. It catches the single most
  common data-leakage bug class in four parameterized lines.
- **The author publishes negative results.**
  [`scripts/HOMC_ORDER7_RESULTS.md`](./scripts/HOMC_ORDER7_RESULTS.md),
  [`scripts/SP500_TREND_AND_HOMC_MEMORY.md`](./scripts/SP500_TREND_AND_HOMC_MEMORY.md),
  [`scripts/TIER3_COMPREHENSIVE_RESULTS.md`](./scripts/TIER3_COMPREHENSIVE_RESULTS.md)
  — each of these is the author saying, in writing, "I tried this and it
  didn't work and here's why." Most AI-assisted quant repos bury their
  failures. This one narrates them.
- **Multi-seed robustness was eventually adopted as a project-level rule.**
  The rule came late, wasn't applied retroactively to the baseline, and
  doesn't appear in the README, but it *does* appear as an internal
  discipline. That alone puts this repo in the top quartile of the genre.
- **The "`buy & hold` on S&P 500" verdict.** It's rare to see an
  AI-assisted quant project openly recommend buy-and-hold against its own
  strategy on any asset. That verdict is correct, was reached via a
  reasonable chain of experiments, and should be applied back to BTC on the
  same grounds.
- **Reasonable execution model.** Next-bar-open fills are correct, the
  stop-loss + cooldown is standard, the target-fraction portfolio
  reconciliation is correct, the long/short accounting handles covers +
  opens in the same bar properly. No gross execution bugs.
- **DSR implementation is mechanically correct.** Even if it's then
  overridden, the math at
  [`signals/backtest/metrics.py`](./signals/backtest/metrics.py),
  lines 65–108, is right and cites the Bailey & López de Prado paper. This
  is not universally true.
- **`SECURITY.md` and a clear "this is experimental research" disclaimer.**
  A distressing number of AI-generated trading repos are posted with no
  disclaimer and a copy-pastable trading CLI. This one is explicit that it's
  a research artifact.

---

## The improvements list (extremely detailed)

If I were reviewing this for promotion to a paper-trading pilot, here is the
punch list I'd want closed before anyone routes a single dollar through it.
Tiered by how much they change the headline claims.

### Tier A — must-fix before the README can keep its current numbers

**A1. Replace every reported number with a multi-seed summary.**
Every "median Sharpe" and "mean CAGR" in the README should be
`mean ± stderr across ≥ 10 random seeds`. For the current H-Vol config the
4-seed data from
[`scripts/BTC_DEEP_SWEEP_RESULTS.md`](./scripts/BTC_DEEP_SWEEP_RESULTS.md)
already puts the true number at ~1.00 ± 0.6. Run 10+ seeds and publish the
distribution.

**A2. Enforce non-overlap in `random_window_eval.py`.**
Replace `rng.sample(range(min_start, max_start), n_windows)` with a
rejection-sampling loop that requires at least 126 bars between any two
starts. If a 10-year price history can't support 16 genuinely non-overlapping
6-month windows, the evaluator should raise and tell you how many it *can*
support (it's roughly 14 for BTC 2015–2024 if you pack them tightly). Then
re-run every headline number.

**A3. Moving-block bootstrap for confidence intervals.**
Non-overlap fixes sampling. It doesn't fix the fact that your ~16 windows
are all drawn from the same ~2,500 bars. The right tool is a moving-block
bootstrap with block length ≈ window length.
Stochastic-dominance-tests like Hansen's SPA or Romano-Wolf would give you
honest CIs on "beats B&H by X" claims.

**A4. A pristine holdout.**
Freeze a 24-month "never seen by any parameter choice" slice — I'd pick
2023-01-01 → 2024-12-31 — wipe it from the sweep-eligible range, re-tune
everything on 2015–2022 only, and then run the holdout **exactly once**.
Whatever Sharpe comes out of that is the real number. Every other reported
Sharpe in the project should be labeled "in-sample."

**A5. Apply multi-seed robustness to the baseline retroactively.**
The `q=0.70` choice needs to be re-justified against
`q=0.50, 0.60, 0.70, 0.75, 0.80, 0.90` at 10+ seeds each, picking the
multi-seed winner rather than the seed-42 winner. Then report the whole
curve, not just the peak.

### Tier B — methodological gaps

**B1. Monte-Carlo permutation test.**
Shuffle the returns within each eval window (preserving the price structure
but destroying the directional information) and re-run the strategy. If the
shuffled-series Sharpe is within 1σ of the real-series Sharpe, the model is
fitting noise. This test is cheap and catches model classes that look like
they're predicting direction but are actually just vol-sizing.

**B2. Transaction cost sensitivity surface.**
A 2-D grid over `commission_bps × slippage_bps` at
`{2.5, 5, 10, 15, 25}` × `{2.5, 5, 10, 15, 25}`. Plot the Sharpe surface. If
the headline Sharpe collapses by >0.5 when commission doubles, the strategy
is cost-brittle and the README should say so.

**B3. Decouple the deadband from cost assumptions.**
Sweep `min_trade_fraction ∈ {0.02, 0.05, 0.10, 0.15, 0.20, 0.30}`
independently. The current 0.20 is a very aggressive hysteresis that papers
over cost sensitivity. Show the full curve.

**B4. Trivial baselines on BTC.**
The S&P notebook correctly benchmarks against buy-and-hold, 200-day MA
trend, and 50/200 golden cross. BTC is never compared to any of these. At
minimum, run:

- Always long (buy & hold)
- 200-day MA trend filter
- A pure `target = +1 if vol < q70 else 0` rule (the "is the Markov
  machinery doing anything" test from section 5)
- Vol-targeted buy-and-hold (cap at the same vol as the hybrid's average
  realized)

If any of these come within 0.2 Sharpe of the hybrid on multi-seed
evaluation, the Markov chain machinery is not what's producing the edge.

**B5. Project-level DSR.**
The DSR's `n_trials` argument should count *all* strategy evaluations the
researcher has run while picking this config, not just the number of configs
inside a single sweep. The project has run, by my count, ~5,000+ distinct
parameter combinations across Tiers 0 through 3. A DSR deflation at
`n_trials = 5000` is much harsher than the per-sweep DSRs in the result
docs. Report it.

**B6. Fix the annualization and the column name.**
At [`signals/backtest/metrics.py`](./signals/backtest/metrics.py),
lines 36–44, annualization should be asset-specific or at minimum
configurable via `BacktestConfig.periods_per_year` — as it already is in the
vol-target module. Rename `volatility_20d` to `volatility_{window}d` or
simply `volatility` with a docstring. Cosmetic but shows care.

**B7. Non-zero risk-free rate in headlines.**
Swap in a 3-month T-bill series for the reporting window and recompute
Sharpe against it. The strategy's actual excess return is what matters, not
its absolute return.

### Tier C — structural (bigger work, bigger payoff)

**C1. Hourly / intraday data.**
Daily bars + a once-per-day signal gives you ~2,500 training observations
for a 5-state, order-5 HOMC. The state space is 5⁵ = 3,125. You are fitting
a model whose parameter count exceeds your sample size. *Of course* most
k-tuples never repeat. Hourly bars would give you 24× more samples against
the same parameter budget, and would resolve some of the intra-day
execution slippage questions that daily bars can't address.

**C2. Forward-paper-trade log, published.**
The repo already has
[`signals/broker/paper_trade_log.py`](./signals/broker/paper_trade_log.py)
and a `signals paper-trade record/reconcile/report` CLI. Commit the
paper-trade log to the repo. 90 days of real forward signals with
timestamps, rationales, and realized PnL — updated daily — is the only
piece of evidence that is robust to every class of retroactive bias in the
backtest. No amount of holdout discipline beats a committed, timestamped
live log.

**C3. Publish per-window daily equity curves, not just aggregate metrics.**
Every result doc in the project reports `median Sharpe`, `mean CAGR`,
`max DD` aggregates. The per-window daily equity series is almost never
shown. A reader should be able to see the 16 equity curves overlaid and
visually check for the cases where the strategy "tied" by holding cash
through a flat period — tied Sharpes mask a lot of behavior.

**C4. A "regime filter" ablation.**
The project's own working theory is that the hybrid wins because composite
contributes bear defense and HOMC contributes bull participation. Test this
directly: strip each component, keep the router, substitute a constant
signal (`target = 0` for the composite branch, `target = 1` for the HOMC
branch), and re-run the evaluation. If the resulting Sharpe is close to the
full hybrid, the components aren't doing what the author thinks they're
doing.

**C5. Drop the `hybrid_routing_strategy == "hmm"` code path if it's been
shown to whipsaw.**
The HMM routing is preserved in the code and tested. Zero production
guidance recommends it. Dead code produces confusion. Either delete or mark
explicitly as "known-bad."

**C6. Actually compute the confidence interval on "beats B&H in 9/16
windows."**
Under the null hypothesis that the strategy is random (50/50 vs B&H on each
truly-independent window), the 95% CI for a binomial at 16 trials is 3–13.
So "9/16" is literally consistent with the null at standard significance.
Worse, at my effective sample count of ~6 (section 2), "beats B&H in 9/16"
is a meaningless statistic because the 16 trials aren't trials. Either
publish the proper binomial test with the corrected N, or stop quoting it.

### Tier D — things the author should not do, in spite of obvious temptation

**D1. Don't sweep more hyperparameters.**
The project has already run ≥ 5,000 parameter combinations. Each additional
sweep strictly worsens the multi-trial correction. If Tier A and Tier B
above produce a smaller headline number, the response should not be to
re-sweep until the number comes back up.

**D2. Don't add more model classes.**
The paper trail shows HOMC, HMM, composite, hybrid, ensemble, boost, trend,
golden_cross. Eight model classes in a repo that can't reliably beat
buy-and-hold on its best-performing asset under a 4-seed robustness check.
More model classes is not the answer; fewer, more rigorously evaluated
models is.

**D3. Don't add alternative data before the core claim is solid.**
Onchain data, funding rates, options skew, macro features — all tempting,
all appear in the roadmap, none will help until the headline methodology is
solid. You cannot bolt macro features onto a strategy whose random-seed
variance is ±0.6 Sharpe and expect the macro features to be the thing that
sticks.

**D4. Don't route real capital through this until Tier A + C2 are done.**
The repo has an Alpaca SDK wired up with explicit dry-run defaults and a
safety-gated `live=True`. Good. Keep it that way. A 30-day forward paper
trade log, committed to the repo, is the single fastest way to produce
evidence that would override most of the critique above.

---

## The summary slide

The README for `jlgreen11/signals` opens with "median Sharpe 2.15, the
highest result in the project's history." The project's own internal
documents — written by the same author, linked from the same repo — quietly
show that the same configuration averages a Sharpe of ~**1.00** across four
random seeds, with one seed producing a **negative** result. The 16-window
evaluation that produced the 2.15 headline draws from a sampler that was
documented as non-overlapping but is not enforced as such, inflating the
effective sample count by roughly 2.6×. The holdout window used to validate
the model was shown to the tuner during the "tightening" phase, a fact
preserved in the author's notes and never corrected in the README. The
deflated Sharpe — included in the code, mechanically correct, cited
accurately — returns 0.00 for every reported configuration and is overridden
on out-of-sample grounds that are themselves compromised.

None of that means the project is worthless. It means the project has
produced a vol-regime-filtered BTC strategy whose true out-of-sample Sharpe
is probably somewhere in the band [0.5, 1.5], is roughly at parity with
buy-and-hold, and has a meaningfully lower drawdown profile — which is a
perfectly respectable, publishable research result. It's just not the result
the README is selling.

If the author reads this: **the defensive-overlay interpretation is the real
story**, and it's a good one. Edit the README to say that. Run Tier A and
you'll have a repo that stands up to review. Skip Tier A and sooner or later
someone on Hacker News is going to do this teardown under their own byline,
and the comment section will be less charitable than mine.

*— the skeptic*
