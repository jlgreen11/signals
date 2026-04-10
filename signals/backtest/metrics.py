"""Performance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    sharpe: float
    cagr: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar: float
    final_equity: float
    n_trades: int

    def to_dict(self) -> dict:
        return {
            "sharpe": self.sharpe,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "calmar": self.calmar,
            "final_equity": self.final_equity,
            "n_trades": self.n_trades,
        }


def _annualization_factor(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 252.0
    deltas = (equity.index[1:] - equity.index[:-1]).total_seconds()
    median = float(np.median(deltas))
    if median <= 0:
        return 252.0
    bars_per_day = max(1.0, 86400.0 / median)
    return 252.0 * bars_per_day if bars_per_day < 24 else 365.0 * bars_per_day


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total_seconds = (equity.index[-1] - equity.index[0]).total_seconds()
    years = total_seconds / (365.25 * 86400)
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def compute_metrics(equity: pd.Series, trades: list) -> Metrics:
    if equity.empty:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    returns = equity.pct_change().dropna()
    periods = _annualization_factor(equity)
    sr = sharpe_ratio(returns, periods)
    mdd = max_drawdown(equity)
    cg = cagr(equity)

    # Any trade that *closes* a position carries realized pnl. SELL closes longs,
    # COVER closes shorts. Treat any trade with non-zero pnl as a closed event.
    closing_sides = {"SELL", "COVER"}
    closed_pnls = [
        t.pnl for t in trades
        if getattr(t, "side", None) in closing_sides and getattr(t, "pnl", 0.0) != 0.0
    ]
    wins = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p < 0]
    win_rate = len(wins) / len(closed_pnls) if closed_pnls else 0.0
    gross_win = sum(wins) if wins else 0.0
    gross_loss = -sum(losses) if losses else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0.0
    calmar = cg / abs(mdd) if mdd != 0 else 0.0

    return Metrics(
        sharpe=sr,
        cagr=cg,
        max_drawdown=mdd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        calmar=calmar,
        final_equity=float(equity.iloc[-1]),
        n_trades=len(closed_pnls),
    )
