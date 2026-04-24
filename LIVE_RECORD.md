# Live Paper Trading Record

**Last updated:** 2026-04-24
**Venue:** Alpaca paper (paper-api.alpaca.markets), three sub-accounts
**Raw snapshot:** [`scripts/data/paper_trading_20260424.json`](scripts/data/paper_trading_20260424.json)
**How to refresh:** `python scripts/pull_alpaca_state.py -o scripts/data/paper_trading_$(date +%Y%m%d).json`

This file is the running forward-evidence log. [SKEPTIC_REVIEW_V2.md](SKEPTIC_REVIEW_V2.md)
Problem 8 flagged zero days of live trading evidence. As of 2026-04-24 there are
**10 trading days of paper-trading returns** — enough to prove the plumbing
works, **not enough to conclude anything about the strategy's edge**.

## Headline (2026-04-10 → 2026-04-23, close-to-close)

| Account        | Equity       | Return  | vs SPY   | Cash       | Long MV     | Gross exposure |
|----------------|-------------:|--------:|---------:|-----------:|------------:|---------------:|
| **momentum**   | $104,794.69  | +4.79%  | +0.52%   | $14.93     | $107,244.18 | 1.02×          |
| **multifactor**| $100,013.14  | +0.01%  | -4.26%   | -$19,773.93| $118,810.04 | 1.19× (margin) |
| **baseline**   | $105,618.24  | +5.62%  | +1.35%   | -$23,192.14| $129,699.48 | 1.23× (margin) |
| SPY (bench)    | —            | +4.27%  | 0        | —          | —           | 1.00×          |

SPY return is total-return close-to-close over the same window. The momentum
account's intraday equity at the time of this snapshot was $107,259.11 — an
additional +2.35% on 2026-04-24 — but that single day is excluded from the
headline since the portfolio-history series ends at the 2026-04-23 close.

## What this proves

1. **The plumbing works.** Orders are being generated, submitted, filled
   (mostly), reconciled, and reflected in account equity.
2. **No catastrophic bug.** No -50% day, no runaway position, no account
   frozen, no margin call.
3. **The momentum portfolio behaves like a ~beta-1 long-only US-equity
   portfolio.** Over 10 days of a strong SPY rally (+4.27%), it returned
   +4.79%. That is what you would expect from a 15-position long-only equity
   portfolio in a rising market. It does not yet show alpha.

## What this does NOT prove

1. **Nothing about Sharpe.** Backtest Sharpe claims (0.659 honest / 1.075
   inflated per SKEPTIC_REVIEW_V2) need 60+ months of daily returns to
   distinguish from chance. 10 days is an anecdote.
2. **Nothing about the backtested edge.** The backtest's headline number used
   a survivorship-biased S&P 500+400 universe. The live momentum account
   trades current-list S&P 500. A live test of the inflated universe is
   not possible without point-in-time constituents data (Norgate, $50/mo).
3. **Nothing about transaction-cost adequacy.** The backtest assumes 10 bps
   round-trip. No realized-vs-projected-pnl reconciliation has been run.
   That is the core purpose of `signals/broker/paper_trade_log.py` and it
   has not been wired up yet for the momentum account.

## What each account actually did

### momentum (the strategy under test)
- 16 filled buys across 16 unique tickers, 48 canceled orders.
- Fully invested: cash $14.93, long MV $107,244.
- Biggest winners: **AMD +41.35%**, **INTC +27.93%**, **MPWR +21.58%**,
  **CVNA +20.31%**.
- No sells yet — all 16 positions still open within the 105-day hold window.
- **75% order cancellation rate** is worth diagnosing. If these are limit
  orders expiring, the strategy is missing three-quarters of its intended
  entries. Backtest fills at the open; live fills depend on order type and
  routing. Treat this as the first concrete finding.

### multifactor (underperformed)
- 10 filled buys, 10 live positions, 0 sells.
- **Account is on margin.** Cash -$19,774, long MV $118,810 → ~1.19× gross
  exposure on a $100K base. This is an unintended deviation from the
  strategy design; the paper_runner is sizing positions as if each gets a
  fresh $12K slice instead of splitting the $100K 10 ways.
- Picks are recognizably multifactor (JNJ, ULTA, DG, SPG, DLTR, EIX, MPC,
  DVN, LDOS, INCY) — value/quality tilt.
- Net return +0.01% vs SPY +4.27% over the window.
- **Result is not comparable to the backtest** until sizing is fixed.

### baseline (was meant to be passive)
- 1 filled buy: **181.82 shares of SPY** on 2026-04-14, at ~$676.
- Also on margin: cash -$23,192, long MV $129,699 → ~1.23× gross exposure.
- So the "baseline" is actually 1.23× leveraged SPY. Not a useful benchmark.
- Return +5.62% is 1.23× SPY's +4.27% (as expected for levered long).
- **Fix the sizing before using this account as a benchmark again.**

## Open items (in priority order)

1. **Diagnose the 75% cancellation rate on momentum.** If this is limit-order
   expiry, switch to marketable limits with wider tolerance or accept market
   orders at the open (matches backtest model). Either way, understand it.
2. **Fix position sizing on multifactor and baseline.** Neither account is
   supposed to be using margin. This is a paper_runner bug.
3. **Wire the paper_trade_log reconciliation loop for momentum.** This is
   the cheapest falsification test available: compare each trade's realized
   return to the backtest's projected return. If they diverge by >20%, the
   backtest's execution model is wrong.
4. **Do nothing else for 3-6 months.** 10 days is noise. 60 days gets
   meaningful, 250 days (one year) starts to support honest inference.
   Resist the urge to iterate on the strategy during the forward test —
   that would reintroduce the 108-config selection bias from the backtest.

## Running tally

| As of       | Days | Momentum | Multifactor | Baseline   | SPY    |
|-------------|-----:|---------:|------------:|-----------:|-------:|
| 2026-04-23  |   10 |  +4.79%  |     +0.01%  |   +5.62%   | +4.27% |

(Add one row per snapshot; never edit old rows.)
