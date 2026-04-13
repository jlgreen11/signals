# Future Improvements

Prioritized list of ideas to test and evaluate. Items are grouped by
category and marked with their current status.

## Validated findings (do not re-test)

These have been tested on the 26-year survivorship-bias-free backtest
and conclusively resolved.

- **Exit rules (profit targets, stop losses, trailing stops)**: 26 rules
  tested, all performed worse than doing nothing. Momentum's edge is in
  the fat right tail. Don't cut winners.
- **Regime filters (golden cross, SPY >200ma, drawdown limits)**: reduce
  drawdowns but cost 3-4% CAGR. Not worth it for a max-returns objective.
- **Portfolio optimizers (risk parity, mean-variance, max diversification)**:
  no improvement over equal weight on momentum portfolios.
- **Cash reserve**: 5% cash reserve creates drag that kills alpha with DCA
  contributions. Fully invested (0% reserve) is optimal.
- **Classic 12-month momentum**: survivorship bias inflates results ~80%.
  Early-breakout acceleration signal is strictly better.
- **Acceleration window**: 21d/126d (1-month vs 6-month) beats all other
  short/long window combinations (12 tested).
- **Hold period**: 105 trading days (~5 months) is optimal across 5
  hold periods tested (42/63/84/105/126).
- **Sector cap**: 2/sector maximizes CAGR; 1/sector maximizes Sharpe but
  structurally limits to 11 positions.

## Feature ideas to test

### From volume data (tested — none helped)

- **Volume confirmation** (21d avg > 63d avg): no improvement.
- **Relative volume spike** (5d avg > 1.5-2x 63d avg): hurt performance.
- **Volume as tiebreaker**: zero effect (no ties in acceleration scores).

Status: REJECTED. Volume does not predict which breakouts persist.

### From price position (tested — mixed)

- **52-week range boost**: improved both CAGR and Sharpe in one sweep
  implementation, but when tested in the validated framework with correct
  acceleration formula and fixed-hold logic, it HURT performance. The
  improvement was an artifact of implementation bugs in the sweep script.
  Needs careful re-test with the exact production backtest to confirm.
- **Recovery filter** (buy dips, price <80% of 126d high): catastrophic.
  Negative CAGR. The model explicitly needs stocks at highs.
- **Anti-recovery** (only stocks >90% of 126d high): over-filters, hurts CAGR.

Status: LIKELY REJECTED but 52-week boost deserves one more clean test.

### From volatility (tested — tradeoff)

- **Exclude top-25% volatility**: best MaxDD (-50%) but costs ~1% CAGR.
- **Exclude top-10% volatility**: costs CAGR without enough DD improvement.
- **Low-vol preference** (penalize vol in score): Sharpe improvement but
  CAGR loss.

Status: REJECTED for max-CAGR objective. Revisit if objective changes to
risk-adjusted returns.

**Open question**: every volatility filter we tested HURT returns. This
is counterintuitive — most academic literature says low-vol outperforms.
Possible explanations:
1. Our model already selects for quality via the acceleration signal
   (steady grind-ups score higher than spiky moves)
2. High-vol stocks that pass the 10% minimum 1-month return filter are
   the ones making real moves, not just noise
3. The sector cap already provides diversification that a vol filter
   would duplicate
4. Survivorship-bias-free universe includes more volatile dead companies
   that a vol filter would correctly exclude — but we already filter
   corrupted data

This deserves deeper investigation: run the vol filter test separately
on pre-2010 vs post-2010 data to see if the effect is regime-dependent.

### From TensorTrade (pending test)

- **Drawdown-from-6m-high filter**: reject stocks >10-15% below their
  126-day high. Tests clean-uptrend hypothesis. Priority: HIGH.
- **Sortino ranking**: replace acceleration with 21-day Sortino ratio,
  or blend 50/50 with acceleration. Rewards consistent momentum over
  gap-driven spikes. Priority: MEDIUM.
- **Inverse-vol position sizing**: weight = (1/vol_i) / sum(1/vol_j).
  Different from vol FILTER — this doesn't exclude stocks, just sizes
  them smaller. Priority: MEDIUM.
- **Continuity bonus**: +0.10 added to acceleration for stocks already
  held. Reduces turnover at rebalance boundaries. Priority: LOW.

Status: TESTED. All four hurt CAGR vs baseline. Drawdown filters
screen out the emerging breakouts the strategy targets. Sortino loses
the acceleration signal's information. Inverse-vol sizing is the least
harmful (-3pp CAGR) but still worse. Continuity bonus creates
concentration risk. Combining features compounds the damage.

REJECTED for max-CAGR objective.

### Not yet testable (need data)

- **Earnings surprise filter**: only buy breakouts that also had a
  recent positive earnings surprise. Requires reliable historical
  earnings data for 1,081 tickers across 26 years. Could use the
  existing PEAD infrastructure but coverage is limited.
- **Short interest**: high short interest + breakout = potential squeeze.
  Requires paid data (not available in yfinance historically).
- **Insider buying**: insider purchases preceding breakout = informed
  signal. Requires SEC Form 4 data at scale.
- **Analyst revision momentum**: upward EPS estimate revisions. Requires
  paid consensus data.

### Structural ideas (not yet tested)

- **Staggered entry**: instead of entering all 15 positions at once on
  rebalance day, enter 3-5 per week over 3 weeks. Reduces timing risk.
- **Adaptive hold period**: hold longer when the position is still
  accelerating, exit earlier when acceleration reverses. Would need
  careful backtesting to avoid curve-fitting.
- **Universe expansion**: add mid-cap stocks (SP400) or international
  (developed markets). More opportunities but need historical constituent
  data for bias-free testing.
- **Multi-timeframe confirmation**: require both 1-week and 1-month
  acceleration to be positive before entering. Reduces false breakouts.
