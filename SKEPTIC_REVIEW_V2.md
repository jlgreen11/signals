# I Built a Momentum Strategy That Returned 28% a Year. It's Probably Fake.

*A self-critique of the signals project, April 2026*

---

I spent three weeks building an equity momentum backtest that shows a
Sharpe of 1.075 and 28.4% CAGR over 26 years. The README on GitHub
says the result "survives deflated Sharpe correction" and "passes
survivorship bias validation." Both of those claims are misleading.
Here is what is actually wrong.

## The Setup

The strategy is simple: rank S&P 500 stocks by momentum acceleration
(3-month return minus the annualized 12-month pace), buy the top 15
with a 2-per-sector cap, hold for 105 trading days, repeat. On the
S&P 500 alone it gets Sharpe 0.659 and CAGR 13.6%. Not bad, but not
statistically significant under deflated Sharpe correction.

Then I expanded the universe. Instead of restricting to current S&P 500
members, I let the strategy buy from ALL stocks with available price
data, including the S&P 400 MidCap. Sharpe jumped from 0.659 to 1.075.
CAGR doubled from 13.6% to 28.4%. I ran survivorship bias tests. Three
out of four passed. I wrote it up, pushed to GitHub, and felt good
about it for about six hours.

Then I looked harder.

## Problem 1: The Universe Expansion Is Survivorship Bias

The S&P 400 MidCap stocks I added are today's S&P 400. I downloaded
the current constituent list from Wikipedia and fetched 26 years of
historical prices for each ticker. This means every stock in my
expanded universe is a company that:

- Still exists in April 2026
- Grew to mid-cap status ($5B-$15B market cap)
- Maintained enough financial health to stay in a major index

When I apply this list retroactively to 2000, I am giving the strategy
foreknowledge of which mid-caps survived. The strategy then buys
accelerating mid-caps, and because these are guaranteed to still be
alive in 2026, it avoids the worst failures. Enron, WorldCom, Lehman,
Washington Mutual, Countrywide, Bear Stearns — none of them are in my
expanded universe. These are exactly the stocks that momentum would
have bought in the months before they collapsed.

My own validation suite detected this. Test 1 (time-decay) shows the
full-universe advantage over S&P-only:

| Period | Delta Sharpe (full-universe vs S&P-only) |
|--------|------------------------------------------|
| 2000-2005 | **+0.46** |
| 2005-2010 | +0.35 |
| 2010-2015 | +0.22 |
| 2015-2020 | +0.10 |
| 2020-2026 | **-0.05** |

The improvement is concentrated in the years with the most dead
stocks and vanishes in recent data where survivorship bias is minimal.
I labeled this "CONCERN" in the validation report. It should have
been labeled "the strategy's edge is an artifact."

## Problem 2: DSR Trial Counting Is Dishonest

The README says "DSR@2 = 1.000, PASS" and calls n_trials=2 the
"honest trial count" for the universe choice. This is false. I tested
at least 8 universe variants:

1. S&P 500 only
2. Full universe (all tickers)
3. Full universe excluding dead stocks
4. Full universe with delisting penalty
5. Former-S&P-only
6. S&P 500+400 (the headline result)
7. S&P 500+400+600
8. Broad US (4,000+ tickers)

I saw results from all 8 before selecting #6 as the "sweet spot." The
honest trial count for the universe dimension alone is 8. DSR fails at
n_trials >= 4 (DSR = 0.833). At n_trials=8, it's well below 0.05.

And this compounds with the 108-config parameter sweep that produced
the base configuration. The canonical short=63, long=252, hold=105,
n_long=15, max_per_sector=2 came from a grid search. The total trial
count is at minimum 108 * 8 = 864. The strategy does not survive any
honest statistical test.

## Problem 3: The Delisting Simulation Is Too Gentle

My Monte Carlo test randomly kills 3% of stocks per year with -50%
terminal returns and reports Sharpe 0.888 (vs 1.075 baseline). Three
problems:

**Real delisting losses are -100%, not -50%.** Enron went from $90 to
$0.26. Lehman went to zero. The "average delisting return of -50%"
includes orderly acquisitions at premiums. The stocks missing from my
dataset are disproportionately the catastrophic ones.

**3%/year is uniform. Real delistings cluster in crises.** The 375
missing tickers are concentrated in 2001-2002 and 2008-2009. A
uniform 3%/year dramatically understates crisis-period losses, which is
exactly when momentum blows up.

**Random killing doesn't correlate with the signal.** The simulation
kills stocks at random. Real delistings correlate with momentum — the
stocks momentum buys (high recent acceleration) are exactly the ones
that blow up when the cycle turns. Enron had outstanding momentum in
2000. Lehman was accelerating into early 2008. A random simulation
cannot capture this correlation.

## Problem 4: The Backtest Has a Hidden Return Inflator

In `signals/backtest/bias_free.py`, Step 2 runs every single trading
day:

```python
if holdings and cash > 100:
    per = cash / len(holdings)
    for col in holdings:
        holdings[col]["sh"] += per / p
        cash -= per
```

When a position exits at a profit, the cash is immediately
redistributed into the remaining holdings on the same bar, before new
entries are considered. This happens at **zero transaction cost** —
no spread, no commission, no market impact.

This creates a compounding amplification effect on momentum winners.
Winning positions generate cash when they exit. That cash is
immediately deployed into other positions that are still running (and
are, by construction, momentum winners that haven't hit their hold
period). Those positions grow on a larger base. When they exit, the
cycle repeats.

In a real portfolio, cash from exits would sit until the next rebalance.
The daily cash sweep is a form of synthetic leverage on momentum
winners that doesn't exist in practice.

## Problem 5: Transaction Costs Are Understated

The backtest assumes 10 bps round-trip. For S&P 400 mid-cap stocks
with high recent acceleration (i.e., stocks that are moving fast),
realistic costs include:

- Wider bid-ask spreads: 10-30 bps for mid-caps, especially in
  2000-2010
- Adverse selection: the market maker knows you're buying momentum,
  and fills at the expensive side
- The idle cash redeployment (Step 2) executes ~3,750 small trades per
  year at zero cost

There has never been a transaction cost sensitivity analysis. This was
flagged in the original SKEPTIC_REVIEW.md and remains unfixed.

## Problem 6: Sharpe 1.075 Is Not Credible

For calibration:

| Strategy | Sharpe | Period |
|----------|--------|--------|
| AQR Large Cap Momentum (AMOMX) | ~0.5-0.6 | 2009-2026 |
| DFA US Large Cap Momentum | ~0.5-0.7 | — |
| Fama-French UMD factor (1927-2024) | ~0.6 | 97 years |
| **This backtest** | **1.075** | 26 years |

No long-only momentum fund in the world has achieved a sustained
Sharpe above 1.0 over multi-decade periods. A Sharpe of 1.075 would
place a 15-stock, single-developer, yfinance-backtested portfolio
above every professional momentum manager. The CAGR of 28.4% would
turn $100K into ~$60 million, beating Warren Buffett's lifetime record
by 8 percentage points annually.

Either I have discovered something that AQR, DFA, and every quant
fund missed, or my numbers are wrong. Occam's razor is clear.

## Problem 7: The Sector Cap Uses Future Information

The sector diversification constraint (max 2 per GICS sector) uses
today's GICS sector assignments. But GICS was reclassified multiple
times:

- 2018: Facebook and Google moved from Information Technology to
  Communication Services
- 2016: Real Estate was split from Financials

In the pre-2018 backtest, the code classifies Google as Communication
Services, allowing more IT positions than reality would have permitted.
This is lookahead bias in the trading constraint.

## Problem 8: No Live Trading Record Exists

The paper trading infrastructure is built. The Alpaca accounts are
configured. The CLI commands are ready. Zero days of forward evidence
have been collected. The strategy has never traded a single real dollar.

## What's Actually True

Strip out the survivorship-biased universe expansion and you get:

| Metric | S&P-only (honest) | Full-universe (inflated) |
|--------|-------------------|--------------------------|
| Sharpe | 0.659 | 1.075 |
| CAGR | 13.6% | 28.4% |
| Max DD | -62.9% | -55.2% |
| DSR@6 | FAIL | FAIL |

The S&P-only result — Sharpe 0.659, CAGR 13.6% — is consistent with
the known US equity momentum premium. It's a real effect, well-
documented in the academic literature, worth ~5 percentage points of
CAGR over buy-and-hold with significantly worse drawdowns. It is not
statistically distinguishable from chance after correcting for the
108-config sweep.

The full-universe result is the same strategy applied to a survivorship-
biased mid-cap universe where every stock is guaranteed to still be
alive in 2026. The 28.4% CAGR headline is the sound of looking up the
answers in the back of the book and being surprised that you got an A.

## What I Should Do Next

1. **Buy Norgate Data ($50/month).** Get survivorship-bias-free
   historical constituents for the S&P 400. Re-run the full-universe
   backtest with point-in-time mid-cap membership. If the improvement
   survives, it's real. If it doesn't, the S&P-only Sharpe 0.659 is
   the ceiling.

2. **Fix the idle cash mechanism.** Cash from exits should sit until
   the next rebalance, not compound daily into remaining winners.
   Apply transaction costs to all trades including redeployments.

3. **Run a proper cost sensitivity.** Sweep 5-50 bps in 5 bps
   increments. Show where the strategy breaks even with SPY.

4. **Start paper trading.** The infrastructure exists. Use it. One day
   of real forward evidence is worth more than another year of
   backtesting.

5. **Stop reporting the full-universe Sharpe as the headline.** The
   honest number is 0.659. The full-universe number is an upper bound
   on what you'd get with better data. Frame it that way.

---

*Every backtest is a hypothesis. This one got ahead of its evidence.*
