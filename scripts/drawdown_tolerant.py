"""Drawdown-tolerant allocation analysis.

For a hypothetical investor with unlimited tolerance for path pain,
what actually maximizes terminal wealth? This question is NOT answered
by Sharpe — Sharpe penalizes volatility even when volatility is fine.

Runs the same 7-year window as trailing_7y_view (2019-04-01 → 2026-04-01)
on a menu of BTC-heavy allocations + the project's hybrid variants +
pure single assets, and ranks purely by terminal wealth and CAGR. The
drawdown column is shown for calibration, not for ranking.

All mixes are daily-rebalanced equal-weight or specified-weight blends
on the equity shared calendar (the one the 4-asset script already uses,
validated by scripts/validate_btc_calendar.py to preserve BTC weekend
moves via pandas compounding).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from signals.backtest.engine import (
    BTC_HYBRID_PRODUCTION,
    BacktestConfig,
    BacktestEngine,
)
from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

WINDOW_START = pd.Timestamp("2019-04-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-04-01", tz="UTC")
INITIAL = 10_000.0


def _load(store: DataStore, sym: str) -> pd.DataFrame:
    df = store.load(sym, "1d").sort_index()
    return df.loc[df.index <= WINDOW_END]


def _anchor(sp: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    eligible = sp.loc[(sp.index >= WINDOW_START) & (sp.index <= WINDOW_END)]
    return eligible.index[0], eligible.index[-1]


def _btc_hybrid_equity(
    btc: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
    max_long: float = 1.0,
) -> pd.Series:
    base = dict(BTC_HYBRID_PRODUCTION)
    base["risk_free_rate"] = historical_usd_rate("2018-2024")
    base["max_long"] = max_long
    cfg = BacktestConfig(**base)
    warmup = pd.Timestamp("2015-01-01", tz="UTC")
    engine_input = btc.loc[(btc.index >= warmup) & (btc.index <= end_ts)]
    result = BacktestEngine(cfg).run(engine_input, symbol="BTC-USD")
    eq = result.equity_curve.loc[
        (result.equity_curve.index >= start_ts)
        & (result.equity_curve.index <= end_ts)
    ]
    if eq.empty or eq.iloc[0] <= 0:
        return pd.Series(dtype=float)
    return eq / eq.iloc[0]


def _bh_equity(
    prices: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.Series:
    sl = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)]
    if sl.empty:
        return pd.Series(dtype=float)
    return sl["close"] / sl["close"].iloc[0]


def _blend(
    weights: dict[str, float],
    legs: dict[str, pd.Series],
) -> pd.Series:
    """Daily-rebalanced weighted blend. `weights` maps leg name → weight;
    the legs dict must have matching keys. Returns rebased to 1.0 at
    first shared trading day."""
    assert abs(sum(weights.values()) - 1.0) < 1e-9, f"weights must sum to 1: {weights}"
    leg_df = pd.concat(
        {k: legs[k] for k in weights}, axis=1
    ).dropna(how="any")
    if leg_df.empty:
        return pd.Series(dtype=float)
    leg_returns = leg_df.pct_change().fillna(0.0)
    port_returns = sum(
        weights[k] * leg_returns[k] for k in weights
    )
    port_equity = (1 + port_returns).cumprod()
    port_equity.iloc[0] = 1.0
    return port_equity


@dataclass
class Result:
    label: str
    end_value: float
    total_return: float
    cagr: float
    sharpe: float
    max_dd: float
    calmar: float


def _stats(equity: pd.Series, label: str) -> Result:
    if equity.empty:
        return Result(label, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0)
    values = equity * INITIAL
    m = compute_metrics(
        values, [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=252.0,
    )
    total = float(values.iloc[-1] / values.iloc[0] - 1)
    return Result(
        label=label,
        end_value=float(values.iloc[-1]),
        total_return=total,
        cagr=m.cagr,
        sharpe=m.sharpe,
        max_dd=m.max_drawdown,
        calmar=m.calmar,
    )


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    btc = _load(store, "BTC-USD")
    sp = _load(store, "^GSPC")
    tlt = _load(store, "TLT")
    gld = _load(store, "GLD")

    start_ts, end_ts = _anchor(sp)
    print(f"Window: {start_ts.date()} → {end_ts.date()}")

    # Legs — each a rebased-to-1.0 daily equity series on its native calendar
    btc_bh = _bh_equity(btc, start_ts, end_ts)
    sp_bh = _bh_equity(sp, start_ts, end_ts)
    tlt_bh = _bh_equity(tlt, start_ts, end_ts)
    gld_bh = _bh_equity(gld, start_ts, end_ts)
    btc_hybrid = _btc_hybrid_equity(btc, start_ts, end_ts)

    # Align BTC series to the equity calendar for portfolio blends
    btc_bh_on_sp = btc_bh.reindex(sp_bh.index, method="ffill")
    btc_bh_on_sp = btc_bh_on_sp / btc_bh_on_sp.iloc[0]
    btc_hybrid_on_sp = btc_hybrid.reindex(sp_bh.index, method="ffill")
    btc_hybrid_on_sp = btc_hybrid_on_sp / btc_hybrid_on_sp.iloc[0]

    legs_for_blend = {
        "BTC_BH": btc_bh_on_sp,
        "BTC_HYBRID": btc_hybrid_on_sp,
        "SP": sp_bh,
        "TLT": tlt_bh,
        "GLD": gld_bh,
    }

    results: list[Result] = []

    # Single-asset controls
    results.append(_stats(btc_bh_on_sp, "100% BTC buy-and-hold"))
    results.append(_stats(btc_hybrid_on_sp, "100% BTC_HYBRID_PRODUCTION"))
    results.append(_stats(sp_bh, "100% SP500 B&H"))
    results.append(_stats(gld_bh, "100% GLD"))

    # BTC-heavy blends (raw BTC for the BTC leg)
    for btc_w, gld_w in [
        (0.90, 0.10),
        (0.80, 0.20),
        (0.70, 0.30),
        (0.60, 0.40),
        (0.50, 0.50),
    ]:
        label = f"{int(btc_w*100)}/{int(gld_w*100)} BTC/GLD"
        eq = _blend({"BTC_BH": btc_w, "GLD": gld_w}, legs_for_blend)
        results.append(_stats(eq, label))

    # 3-asset BTC-heavy mixes
    for btc_w, gld_w, sp_w in [
        (0.70, 0.15, 0.15),
        (0.60, 0.20, 0.20),
        (0.50, 0.25, 0.25),
    ]:
        label = f"{int(btc_w*100)}/{int(gld_w*100)}/{int(sp_w*100)} BTC/GLD/SP"
        eq = _blend(
            {"BTC_BH": btc_w, "GLD": gld_w, "SP": sp_w},
            legs_for_blend,
        )
        results.append(_stats(eq, label))

    # 4-asset equal-weight (the existing production recommendation, for reference)
    eq = _blend(
        {"BTC_BH": 0.25, "SP": 0.25, "TLT": 0.25, "GLD": 0.25},
        legs_for_blend,
    )
    results.append(_stats(eq, "25/25/25/25 BTC/SP/TLT/GLD (reference)"))

    # Hybrid-leg variants (use the vol-routed hybrid for BTC instead of raw)
    for btc_w, gld_w in [(0.80, 0.20), (0.70, 0.30), (0.60, 0.40)]:
        label = f"{int(btc_w*100)}/{int(gld_w*100)} BTC_HYBRID/GLD"
        eq = _blend(
            {"BTC_HYBRID": btc_w, "GLD": gld_w},
            legs_for_blend,
        )
        results.append(_stats(eq, label))

    # Higher-leverage hybrid (max_long=1.5) — the Round-3 Tier-0f sweep
    # showed Sharpe is flat across max_long ∈ [1.0, 2.0] but CAGR scales
    # approximately linearly. For a drawdown-tolerant investor this is
    # the cheapest way to get more CAGR out of the hybrid.
    btc_hybrid_15x = _btc_hybrid_equity(btc, start_ts, end_ts, max_long=1.5)
    btc_hybrid_15x_on_sp = btc_hybrid_15x.reindex(sp_bh.index, method="ffill")
    btc_hybrid_15x_on_sp = btc_hybrid_15x_on_sp / btc_hybrid_15x_on_sp.iloc[0]
    results.append(_stats(btc_hybrid_15x_on_sp, "100% BTC_HYBRID max_long=1.5"))

    # Rank by end value (what a drawdown-tolerant investor actually cares about)
    results.sort(key=lambda r: r.end_value, reverse=True)

    print()
    print("=" * 98)
    print("Drawdown-tolerant ranking — sorted by END VALUE on $10,000 initial")
    print("=" * 98)
    print(
        f"  {'strategy':<42}"
        f"{'end $':>14}{'total':>12}"
        f"{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'Calmar':>9}"
    )
    print("  " + "-" * 96)
    for r in results:
        marker = " " if r.label != "100% SP500 B&H" else "←"
        print(
            f"  {r.label:<42}"
            f"${r.end_value:>12,.0f}"
            f"{r.total_return:>+11.1%}"
            f"{r.cagr:>+8.1%}"
            f"{r.sharpe:>+9.3f}"
            f"{r.max_dd:>+8.1%}"
            f"{r.calmar:>+8.2f}"
            f" {marker}"
        )

    # Persist
    rows = [
        {
            "label": r.label, "end_value": r.end_value,
            "total_return": r.total_return, "cagr": r.cagr,
            "sharpe": r.sharpe, "max_dd": r.max_dd, "calmar": r.calmar,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    out_parquet = Path(__file__).parent / "data" / "drawdown_tolerant.parquet"
    df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}")

    # Markdown
    out_md = Path(__file__).parent / "DRAWDOWN_TOLERANT.md"
    md_lines = [
        "# Drawdown-tolerant allocation — trailing 7 years",
        "",
        f"Window: **{start_ts.date()} → {end_ts.date()}**. Starting capital "
        "$10,000. Ranked by terminal wealth because the hypothesis is "
        "'unlimited stomach for drawdowns, maximize compounded return'.",
        "",
        "## Sorted by end value",
        "",
        "| strategy | end value | total | CAGR | Sharpe | MDD | Calmar |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        md_lines.append(
            f"| `{r.label}` | ${r.end_value:,.0f} | "
            f"{r.total_return:+.1%} | "
            f"{r.cagr:+.1%} | "
            f"{r.sharpe:+.3f} | "
            f"{r.max_dd:+.1%} | "
            f"{r.calmar:+.2f} |"
        )
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
