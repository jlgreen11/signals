"""$50k SP-complement allocation — 5-year forward horizon.

Context: investor already holds SP500 as their core, has $50k extra
capital, and wants to maximize 5-year return. Since the existing core
is SP-heavy, MORE SP adds no diversification. The $50k should go into
assets that are (a) lowly correlated with SP, (b) have positive
expected forward return, and (c) not redundant with what's already
held.

This script ranks candidate SP-complement allocations on a trailing
5-year window (2021-04-01 → 2026-04-01) that includes the 2022 bear,
the 2023 recovery, the 2024 BTC halving rally, the 2025 GLD run,
and Q1 2026. All allocations start at $50,000.

Candidates:
  - Single assets (BTC, GLD, TLT, SP) as calibration anchors
  - BTC-heavy mixes (BTC+GLD, BTC+GLD+SP)
  - Gold-heavy mixes (GLD+something defensive)
  - Classic diversifiers (60/40 SP/TLT, 60/40 SP/GLD)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from signals.backtest.metrics import compute_metrics
from signals.backtest.risk_free import historical_usd_rate
from signals.config import SETTINGS
from signals.data.storage import DataStore

WINDOW_START = pd.Timestamp("2021-04-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-04-01", tz="UTC")
INITIAL = 50_000.0


def _load(store: DataStore, sym: str) -> pd.DataFrame:
    df = store.load(sym, "1d").sort_index()
    return df.loc[df.index <= WINDOW_END]


def _anchor(sp: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    eligible = sp.loc[(sp.index >= WINDOW_START) & (sp.index <= WINDOW_END)]
    return eligible.index[0], eligible.index[-1]


def _bh(prices: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Series:
    sl = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)]
    if sl.empty:
        return pd.Series(dtype=float)
    return sl["close"] / sl["close"].iloc[0]


def _blend(weights: dict[str, float], legs: dict[str, pd.Series]) -> pd.Series:
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    leg_df = pd.concat({k: legs[k] for k in weights}, axis=1).dropna(how="any")
    if leg_df.empty:
        return pd.Series(dtype=float)
    leg_returns = leg_df.pct_change().fillna(0.0)
    port_returns = sum(weights[k] * leg_returns[k] for k in weights)
    port_equity = (1 + port_returns).cumprod()
    port_equity.iloc[0] = 1.0
    return port_equity


@dataclass
class Result:
    label: str
    weights: str
    end_value: float
    total_return: float
    cagr: float
    sharpe: float
    max_dd: float
    calmar: float
    corr_vs_sp: float


def _stats(equity: pd.Series, label: str, weights: str, sp_eq: pd.Series) -> Result:
    if equity.empty:
        return Result(label, weights, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = equity * INITIAL
    m = compute_metrics(
        values, [],
        risk_free_rate=historical_usd_rate("2018-2024"),
        periods_per_year=252.0,
    )
    total = float(values.iloc[-1] / values.iloc[0] - 1)
    # Correlation of daily returns vs SP B&H
    common = equity.index.intersection(sp_eq.index)
    corr = float(
        equity.loc[common].pct_change().corr(sp_eq.loc[common].pct_change())
    )
    return Result(
        label=label,
        weights=weights,
        end_value=float(values.iloc[-1]),
        total_return=total,
        cagr=m.cagr,
        sharpe=m.sharpe,
        max_dd=m.max_drawdown,
        calmar=m.calmar,
        corr_vs_sp=corr,
    )


def main() -> None:
    store = DataStore(SETTINGS.data.dir)
    btc = _load(store, "BTC-USD")
    sp = _load(store, "^GSPC")
    tlt = _load(store, "TLT")
    gld = _load(store, "GLD")

    start_ts, end_ts = _anchor(sp)
    print(f"Window: {start_ts.date()} → {end_ts.date()}  (5-year trailing)")
    print(f"Initial capital: ${INITIAL:,.0f}")
    print()

    # Single-asset legs
    btc_bh = _bh(btc, start_ts, end_ts)
    sp_bh = _bh(sp, start_ts, end_ts)
    tlt_bh = _bh(tlt, start_ts, end_ts)
    gld_bh = _bh(gld, start_ts, end_ts)

    btc_on_sp = btc_bh.reindex(sp_bh.index, method="ffill")
    btc_on_sp = btc_on_sp / btc_on_sp.iloc[0]

    legs = {"BTC": btc_on_sp, "SP": sp_bh, "TLT": tlt_bh, "GLD": gld_bh}

    results: list[Result] = []

    # Single-asset references
    results.append(_stats(sp_bh, "100% SP500 B&H (redundant w/ core)", "100/0", sp_bh))
    results.append(_stats(btc_on_sp, "100% BTC buy-and-hold", "100/0", sp_bh))
    results.append(_stats(gld_bh, "100% GLD buy-and-hold", "100/0", sp_bh))
    results.append(_stats(tlt_bh, "100% TLT buy-and-hold", "100/0", sp_bh))

    # BTC-heavy SP-complements
    for btc_w, gld_w in [
        (1.00, 0.00),
        (0.80, 0.20),
        (0.70, 0.30),
        (0.60, 0.40),
        (0.50, 0.50),
    ]:
        if btc_w == 1.00:
            continue  # covered above
        eq = _blend({"BTC": btc_w, "GLD": gld_w}, legs)
        results.append(_stats(
            eq, f"{int(btc_w*100)}/{int(gld_w*100)} BTC/GLD",
            f"{int(btc_w*100)}/{int(gld_w*100)}", sp_bh,
        ))

    # BTC + GLD + TLT (real diversification — each leg uncorrelated with SP)
    for btc_w, gld_w, tlt_w in [
        (0.60, 0.25, 0.15),
        (0.50, 0.30, 0.20),
        (0.50, 0.25, 0.25),
        (0.40, 0.30, 0.30),
    ]:
        label = f"{int(btc_w*100)}/{int(gld_w*100)}/{int(tlt_w*100)} BTC/GLD/TLT"
        eq = _blend({"BTC": btc_w, "GLD": gld_w, "TLT": tlt_w}, legs)
        results.append(_stats(eq, label, label.split()[0], sp_bh))

    # Gold-heavy defensives
    results.append(_stats(
        _blend({"GLD": 0.70, "TLT": 0.30}, legs),
        "70/30 GLD/TLT (defensive)", "70/30", sp_bh,
    ))
    results.append(_stats(
        _blend({"GLD": 0.60, "BTC": 0.40}, legs),
        "60/40 GLD/BTC (inverted)", "60/40", sp_bh,
    ))

    # Classic mixes for comparison
    results.append(_stats(
        _blend({"SP": 0.60, "TLT": 0.40}, legs),
        "60/40 SP/TLT (classic pension)", "60/40", sp_bh,
    ))
    results.append(_stats(
        _blend({"SP": 0.60, "GLD": 0.40}, legs),
        "60/40 SP/GLD (gold-bond)", "60/40", sp_bh,
    ))

    # Max-diversification: 4-asset equal weight
    results.append(_stats(
        _blend({"BTC": 0.25, "SP": 0.25, "TLT": 0.25, "GLD": 0.25}, legs),
        "25/25/25/25 BTC/SP/TLT/GLD (ref)", "25×4", sp_bh,
    ))

    # Sort by end value
    results.sort(key=lambda r: r.end_value, reverse=True)

    print("=" * 108)
    print(
        f"$50k SP-complement ranking — 5-year trailing "
        f"({start_ts.date()} → {end_ts.date()})"
    )
    print("=" * 108)
    print(
        f"  {'strategy':<42}"
        f"{'end $':>12}{'total':>10}"
        f"{'CAGR':>8}{'Sharpe':>9}{'MDD':>8}{'Calmar':>8}{'ρ(SP)':>9}"
    )
    print("  " + "-" * 106)
    for r in results:
        print(
            f"  {r.label:<42}"
            f"${r.end_value:>10,.0f}"
            f"{r.total_return:>+9.1%}"
            f"{r.cagr:>+7.1%}"
            f"{r.sharpe:>+9.3f}"
            f"{r.max_dd:>+7.1%}"
            f"{r.calmar:>+7.2f}"
            f"{r.corr_vs_sp:>+9.3f}"
        )

    # Persist
    df = pd.DataFrame([
        {
            "label": r.label, "end_value": r.end_value,
            "total_return": r.total_return, "cagr": r.cagr,
            "sharpe": r.sharpe, "max_dd": r.max_dd,
            "calmar": r.calmar, "corr_vs_sp": r.corr_vs_sp,
        }
        for r in results
    ])
    out_parquet = Path(__file__).parent / "data" / "sp_complement_50k.parquet"
    df.to_parquet(out_parquet)
    print(f"\n[wrote] {out_parquet}")

    # Markdown
    out_md = Path(__file__).parent / "SP_COMPLEMENT_50K.md"
    md_lines = [
        "# $50k SP-complement allocation — 5-year trailing backtest",
        "",
        f"**Window**: {start_ts.date()} → {end_ts.date()}  "
        f"(~5 years, {(end_ts - start_ts).days} calendar days)",
        "**Initial**: $50,000",
        "",
        "Context: investor holds SP500 as their core and has $50k "
        "additional capital for a 5-year horizon. The $50k should "
        "go into SP-complementary assets — i.e. low correlation with "
        "the existing core, positive expected forward return.",
        "",
        "The `ρ(SP)` column shows daily-return correlation vs the "
        "existing SP core. **Lower correlation = better "
        "diversification** for an SP-heavy portfolio.",
        "",
        "## Ranking by end value on $50,000",
        "",
        "| strategy | end $ | total | CAGR | Sharpe | MDD | Calmar | ρ(SP) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        md_lines.append(
            f"| `{r.label}` | ${r.end_value:,.0f} | "
            f"{r.total_return:+.1%} | "
            f"{r.cagr:+.1%} | "
            f"{r.sharpe:+.3f} | "
            f"{r.max_dd:+.1%} | "
            f"{r.calmar:+.2f} | "
            f"{r.corr_vs_sp:+.3f} |"
        )
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    main()
