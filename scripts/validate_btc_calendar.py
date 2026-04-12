"""Validate that the 4-asset portfolio captures ALL BTC trading days.

The 4-asset portfolio lives on the equity shared calendar (inner join
of SP/TLT/GLD), which is ~252 days/year. BTC trades 365 days/year, so
113 BTC bars/year are "weekend BTC bars" that don't appear as their
own portfolio rows.

**Question**: does the portfolio lose the BTC price action that
happened on those dropped weekend bars?

**Answer**: no. The weekend price action is absorbed into the next
equity trading day's bar (typically Monday). When pandas computes
`pct_change()` on the inner-joined leg_df, the BTC column's Monday
value is compared against the previous row in leg_df — which is
Friday, not Sunday. So Monday's BTC "return" is a 3-day return
covering Fri-close → Mon-close, capturing every weekend tick.

This script demonstrates that explicitly on a real 30-day BTC window:

  1. Count BTC bars in the raw data (should be ~30, one per day)
  2. Count BTC bars surviving the inner-join with SP (should be ~22, weekdays only)
  3. Sum the per-bar BTC returns on the raw 365-day series (total BTC move)
  4. Sum the per-bar BTC returns on the inner-joined 252-day series (also total BTC move)
  5. Assert they're equal to within floating-point noise

If step 5 passes, no BTC price action is being lost.
"""

from __future__ import annotations

import pandas as pd

from signals.backtest.engine import BTC_HYBRID_PRODUCTION, BacktestConfig, BacktestEngine
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

START = pd.Timestamp("2024-03-01", tz="UTC")
END = pd.Timestamp("2024-03-31", tz="UTC")


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    btc = store.load("BTC-USD", "1d").sort_index()
    sp = store.load("^GSPC", "1d").sort_index()

    btc_win = btc.loc[(btc.index >= START) & (btc.index <= END)]
    sp_win = sp.loc[(sp.index >= START) & (sp.index <= END)]

    # For apples-to-apples comparison, both calendars must START and END
    # on the same dates. Trim the native BTC series to match SP's first
    # and last bar — otherwise the "native" total includes the extra
    # tail weekend that the equity calendar hasn't reached yet.
    first_sp = sp_win.index.min()
    last_sp = sp_win.index.max()
    btc_aligned = btc_win.loc[(btc_win.index >= first_sp) & (btc_win.index <= last_sp)]

    print("=" * 72)
    print(f"BTC calendar validation — window {START.date()} → {END.date()}")
    print("=" * 72)
    print(f"  BTC bars (raw 365-day in window)   : {len(btc_win):3d}")
    print(f"  SP bars  (equity calendar)         : {len(sp_win):3d}")
    print(f"  BTC bars (trimmed to SP endpoints) : {len(btc_aligned):3d}")
    print(f"  SP first bar : {first_sp.date()}  |  SP last bar : {last_sp.date()}")

    # Which BTC dates are NOT in SP's calendar? (weekend BTC bars inside
    # the aligned window)
    btc_dates = {d.date() for d in btc_aligned.index}
    sp_dates = {d.date() for d in sp_win.index}
    dropped = sorted(btc_dates - sp_dates)
    print(f"  Weekend BTC bars inside window     : {len(dropped)}")
    print(f"    First 5: {[str(d) for d in dropped[:5]]}")

    # --- Method 1: full BTC compounding on the TRIMMED-to-same-endpoints 365-day calendar
    btc_returns_native = btc_aligned["close"].pct_change().dropna()
    total_btc_native = (1 + btc_returns_native).prod() - 1

    # --- Method 2: inner-join with SP, then pct_change (Fri→Mon becomes
    # a single 3-day bar)
    combined = pd.concat(
        {"BTC": btc_aligned["close"], "SP": sp_win["close"]},
        axis=1,
    ).dropna(how="any")
    combined_returns = combined.pct_change().dropna()
    total_btc_on_sp_cal = (1 + combined_returns["BTC"]).prod() - 1

    print()
    print("BTC total return over the aligned window, computed two ways:")
    print(f"  (1) Native 365-day (31 bars)   : {total_btc_native:+.6%}")
    print(f"  (2) Equity calendar (22 bars)  : {total_btc_on_sp_cal:+.6%}")
    diff = abs(total_btc_native - total_btc_on_sp_cal)
    print(f"  Absolute difference: {diff:.2e}")
    assert diff < 1e-10, f"MISMATCH: {diff}"
    print("  ✓ IDENTICAL to floating-point precision")
    print()
    print("  The compounding property does the job: even though the equity")
    print("  calendar has fewer bars, each Monday bar carries the full 3-day")
    print("  return since Friday, so the product of all bar returns equals")
    print("  the product over the native 365-day calendar.")
    print()
    print("Interpretation: every BTC tick between Friday close and Monday")
    print("close is captured as Monday's portfolio-bar return. No BTC")
    print("price action is lost; the weekend's cumulative move appears")
    print("on Monday's bar as a 3-day return, which is then multiplied")
    print("by 0.25 to get the BTC-leg contribution to Monday's portfolio.")
    print()

    # Demonstrate with a specific weekend
    # Pick a weekend visible in the output
    if len(dropped) >= 2:
        weekend_start = pd.Timestamp(dropped[0], tz="UTC")
        weekend_end = pd.Timestamp(dropped[1], tz="UTC")
        print(f"Example weekend: {weekend_start.date()} and {weekend_end.date()}")
        # Find the Friday before and the Monday after
        fri = btc_win.index[btc_win.index < weekend_start].max()
        mon = btc_win.index[btc_win.index > weekend_end].min()
        if fri is not pd.NaT and mon is not pd.NaT:
            p_fri = float(btc_win.loc[fri, "close"])
            p_sat = float(btc_win.loc[weekend_start, "close"])
            p_sun = float(btc_win.loc[weekend_end, "close"])
            p_mon = float(btc_win.loc[mon, "close"])
            print(f"  Fri {fri.date()}: ${p_fri:,.2f}")
            print(f"  Sat {weekend_start.date()}: ${p_sat:,.2f} "
                  f"({(p_sat/p_fri - 1):+.2%} from Fri)")
            print(f"  Sun {weekend_end.date()}: ${p_sun:,.2f} "
                  f"({(p_sun/p_sat - 1):+.2%} from Sat)")
            print(f"  Mon {mon.date()}: ${p_mon:,.2f} "
                  f"({(p_mon/p_sun - 1):+.2%} from Sun)")
            fri_mon = p_mon / p_fri - 1
            print(f"  Fri→Mon 3-day return (what Monday's portfolio bar shows): "
                  f"{fri_mon:+.4%}")
            fri_mon_compound = ((p_sat / p_fri) * (p_sun / p_sat)
                                * (p_mon / p_sun)) - 1
            print(f"  Compound of Sat+Sun+Mon native returns: {fri_mon_compound:+.4%}")
            print(f"  Match: {abs(fri_mon - fri_mon_compound) < 1e-10}")

    # Also validate that the BacktestEngine returns 365-day BTC equity
    print()
    print("=" * 72)
    print("BTC hybrid engine validation — uses the full 365-day calendar")
    print("=" * 72)
    base = dict(BTC_HYBRID_PRODUCTION)
    base["risk_free_rate"] = historical_usd_rate("2018-2024")
    cfg = BacktestConfig(**base)
    # Give it enough warmup
    warmup_start = pd.Timestamp("2019-01-01", tz="UTC")
    engine_input = btc.loc[(btc.index >= warmup_start) & (btc.index <= END)]
    result = BacktestEngine(cfg).run(engine_input, symbol="BTC-USD")
    eq_in_window = result.equity_curve.loc[
        (result.equity_curve.index >= START) & (result.equity_curve.index <= END)
    ]
    print(f"  Engine equity curve in window: {len(eq_in_window)} bars")
    print("  Expected (BTC calendar): ~31 bars")
    # BTC has calendar days Mar 1-31 = 31 days
    weekend_count = sum(
        1 for d in eq_in_window.index
        if d.dayofweek in (5, 6)  # Saturday=5, Sunday=6
    )
    print(f"  Weekend bars in engine curve: {weekend_count} "
          f"(if > 0, BTC weekend bars ARE preserved)")
    assert weekend_count > 0, (
        "BacktestEngine appears to be dropping BTC weekend bars! "
        "That would be a bug."
    )
    print("  ✓ BacktestEngine preserves BTC weekend bars")


if __name__ == "__main__":
    main()
