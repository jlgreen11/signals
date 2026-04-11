"""Performance metrics."""

from __future__ import annotations

import math
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


def _annualization_factor(
    equity: pd.Series,
    periods_per_year: float | None = None,
) -> float:
    """Bars-per-year used to annualize the Sharpe ratio.

    If `periods_per_year` is provided (e.g. 365 for crypto, 252 for equities)
    we use it directly. Otherwise we infer from the index spacing — but the
    legacy inference rule returns 252 for any daily-cadence series, which is
    wrong for BTC (trades 365 days/year). SKEPTIC_REVIEW.md § 8a flags this.
    Callers that know the asset should pass `periods_per_year` explicitly;
    the engine now does so via `BacktestConfig.periods_per_year`.
    """
    if periods_per_year is not None:
        return float(periods_per_year)
    if len(equity) < 2:
        return 252.0
    deltas = (equity.index[1:] - equity.index[:-1]).total_seconds()
    median = float(np.median(deltas))
    if median <= 0:
        return 252.0
    bars_per_day = max(1.0, 86400.0 / median)
    return 252.0 * bars_per_day if bars_per_day < 24 else 365.0 * bars_per_day


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio.

    `risk_free_rate` is the *annualized* risk-free rate (e.g. 0.02 for 2%);
    it gets converted to a per-period rate via division by `periods_per_year`
    before being subtracted from the return series.
    """
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    excess = r - (risk_free_rate / periods_per_year)
    return float(excess.mean() / r.std() * np.sqrt(periods_per_year))


def expected_max_sharpe(n_trials: int) -> float:
    """Expected max of N IID standard-normal Sharpe estimates under H0 (true SR=0).

    Used as the baseline for the Deflated Sharpe Ratio. Bailey & López de Prado
    (2014), eq. 8. Assumes the variance of the Sharpe estimates across trials
    is 1 — a conservative default that makes the deflation easier to interpret.
    """
    if n_trials <= 1:
        return 0.0
    from scipy.stats import norm

    emc = 0.5772156649015329  # Euler-Mascheroni
    return float(
        (1 - emc) * norm.ppf(1 - 1.0 / n_trials)
        + emc * norm.ppf(1 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio: Pr(true SR > 0 | observed SR, N trials).

    Bailey & López de Prado (2014). Corrects an observed Sharpe for selection
    bias when the strategy was picked from N trials, and for non-normality of
    returns (skew and excess kurtosis).

    Returns a probability in [0, 1]. A DSR < 0.95 means the observed Sharpe
    is consistent with chance under N trials and should not be celebrated.
    """
    if n_observations < 2 or n_trials < 1:
        return 0.0
    from scipy.stats import norm

    e_max = expected_max_sharpe(n_trials)
    var_term = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe * sharpe
    if var_term <= 0:
        return 0.0
    z = (sharpe - e_max) * math.sqrt(max(1, n_observations - 1)) / math.sqrt(var_term)
    return float(norm.cdf(z))


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


def compute_metrics(
    equity: pd.Series,
    trades: list,
    risk_free_rate: float = 0.0,
    periods_per_year: float | None = None,
) -> Metrics:
    """Compute performance metrics for an equity curve.

    `periods_per_year`, when provided, overrides the index-inferred
    annualization factor. Pass 365 for crypto (BTC trades 365 days/year) or
    252 for equities. If omitted, falls back to legacy index-inference which
    returns 252 for any daily cadence — see SKEPTIC_REVIEW.md § 8a.
    """
    if equity.empty:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    returns = equity.pct_change().dropna()
    periods = _annualization_factor(equity, periods_per_year=periods_per_year)
    sr = sharpe_ratio(returns, periods, risk_free_rate=risk_free_rate)
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
