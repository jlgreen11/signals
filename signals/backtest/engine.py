"""Walk-forward backtest engine — model-agnostic, with sized longs/shorts and stops.

Strict no-lookahead:
  - At bar `t`, the model is fit on data through `t-1` only.
  - The signal generated at `t` is acted upon at the open of `t+1`.

Three model backends are supported via `BacktestConfig.model_type`:
  - "composite" → CompositeMarkovChain (1st-order discrete, 3×3 by default)
  - "hmm"       → HiddenMarkovModel (Gaussian HMM, fit via Baum-Welch / EM)
  - "homc"      → HigherOrderMarkovChain (paper-inspired k-th order discrete)

Strategy layer (applies to all model types):
  - SignalGenerator emits a sized target_position in [-max_short, +max_long]
  - Portfolio is reconciled via set_target() at next bar's open
  - Per-bar stop-loss check (close-to-close) flattens the position if hit
  - Optional cooldown after a stop fires before re-entering
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from signals.backtest.metrics import Metrics, compute_metrics
from signals.backtest.portfolio import Portfolio
from signals.backtest.vol_target import VolTargetConfig, apply_vol_target
from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.boost import GradientBoostingModel
from signals.model.composite import CompositeMarkovChain
from signals.model.ensemble import EnsembleModel
from signals.model.hmm import HiddenMarkovModel
from signals.model.homc import HigherOrderMarkovChain
from signals.model.hybrid import DEFAULT_ROUTING, HybridRegimeModel
from signals.model.signals import SignalGenerator
from signals.model.trend import DualMovingAverage, TrendFilter
from signals.model.vol_filter import NaiveVolFilter
from signals.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestConfig:
    model_type: str = "composite"     # "composite" | "hmm" | "homc" | "hybrid" | "trend" | "golden_cross" | "boost" | "vol_filter"
    train_window: int = 252
    retrain_freq: int = 21
    n_states: int = 9                 # composite: ignored (uses return_bins×vol_bins); homc/hmm: used
    return_bins: int = 3              # composite only
    volatility_bins: int = 3          # composite only
    order: int = 3                    # homc only
    vol_window: int = 10              # tightened: short window reacts faster to vol regimes
    n_iter: int = 200                 # hmm only
    n_init: int = 1                   # hmm only — multi-start, keeps best LL
    tol: float = 1e-3                 # hmm only — EM convergence tolerance
    strict_convergence: bool = False  # hmm only — raise if not converged
    random_state: int = 42
    laplace_alpha: float = 0.01       # tightened: low smoothing for composite (1.0 for hmm/homc)
    # Hybrid-specific: lets a CLI caller override the default routing
    # without knowing the internals of HybridRegimeModel.
    # Default is "vol" because the 16-window random eval showed HMM-based
    # routing whipsaws on ambiguous regimes (median Sharpe 0.37, far below
    # both composite and HOMC), while vol-based routing produces the best
    # median Sharpe of any model tested (1.92, +5% over HOMC, +33% over
    # composite). See scripts/HOMC_TIER0C_HYBRID_RESULTS.md for the
    # comparison.
    hybrid_routing: dict[str, str] | None = None
    hybrid_routing_strategy: str = "vol"   # "hmm", "vol", "blend", or "adaptive_vol"
    # ------------------------------------------------------------------
    # Historical q-value timeline (SKEPTIC_REVIEW.md / Tier A5 + Round 3)
    # ------------------------------------------------------------------
    # q=0.75 — original ad-hoc default (Tier 0c hybrid launch, 2026-04-10)
    # q=0.70 — seed-42 sweep winner (Tier 0e, 2026-04-11). Also the
    #          multi-seed winner when evaluated in isolation with
    #          retrain_freq=21 and train_window=1000 held fixed
    #          (confirm_winners.py, 1.175 ± 0.083).
    # q=0.50 — Round-3 large-grid search winner (see
    #          explore_improvements.py). When combined with
    #          retrain_freq=14 and train_window=750, the hybrid
    #          produces multi-seed avg Sharpe **1.551 ± 0.099**
    #          across 10 seeds, min seed 1.010, max 1.949.
    #          +0.659 Sharpe vs q=0.70 baseline. Dominates on
    #          min-seed (1.010 vs 0.345). See BTC_HYBRID_PRODUCTION
    #          constant below and scripts/data/explore_improvements.md.
    #          NOTE: q=0.50 alone (with default retrain_freq=21 and
    #          train_window=1000) only achieves 1.081 Sharpe. The
    #          +0.659 edge requires all three parameters tuned together.
    # ------------------------------------------------------------------
    # Every historical result doc in scripts/*.md reports numbers
    # measured at the q-value prevailing at the time of that doc's run.
    # Check each doc's "Test parameters (historical)" header.
    hybrid_vol_quantile: float = 0.50
    hybrid_blend_low: float = 0.50         # blend ramp lower quantile
    hybrid_blend_high: float = 0.85        # blend ramp upper quantile
    hybrid_adaptive_low: float = 0.60      # adaptive-vol low-regime threshold quantile
    hybrid_adaptive_high: float = 0.80     # adaptive-vol high-regime threshold quantile
    hybrid_adaptive_lookback: int = 30     # bars of recent vol to average for regime detection
    # Naive vol filter (null hypothesis baseline — see signals/model/vol_filter.py):
    vol_filter_quantile: float = 0.50  # match hybrid's default for fair comparison
    # Trend-filter models (for equity indices — see signals/model/trend.py):
    trend_window: int = 200                # single-MA trend filter window
    trend_fast_window: int = 50            # dual-MA fast window
    trend_slow_window: int = 200           # dual-MA slow window
    # Gradient boosting model (Tier-3 Phase E — signals/model/boost.py):
    boost_n_estimators: int = 100
    boost_max_depth: int = 3
    boost_learning_rate: float = 0.1
    initial_cash: float = 10_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    buy_threshold_bps: float = 25.0
    sell_threshold_bps: float = -35.0
    target_scale_bps: float = 20.0
    allow_short: bool = False         # tightened: BTC's secular uptrend punishes shorts
    # max_long default is conservative (1.0 = no leverage). The Tier-0f sizing
    # sweep (scripts/HOMC_TIER0F_SIZING_BLEND.md) showed Sharpe is flat across
    # max_long ∈ [1.0, 2.0] while CAGR scales nearly linearly with leverage:
    #
    #   max_long=1.00  → median Sharpe 2.15, CAGR +156%, MDD -21% (default)
    #   max_long=1.25  → median Sharpe 2.15, CAGR +216%, MDD -26%
    #   max_long=1.50  → median Sharpe 2.16, CAGR +288%, MDD -30%
    #   max_long=2.00  → median Sharpe 2.13, CAGR +480%, MDD -38%
    #
    # If you can tolerate larger drawdowns, passing --max-long 1.5 gives
    # ~1.85× the CAGR of the default at the same Sharpe. This is a risk-
    # tolerance decision, not a methodology one — the default is left at
    # 1.0 deliberately.
    max_long: float = 1.0
    max_short: float = 1.0
    stop_loss_pct: float = 0.0        # tightened: empirically unhelpful for composite (sell signal exits faster than any reasonable stop)
    stop_cooldown_bars: int = 5
    min_trade_fraction: float = 0.20  # don't rebalance for changes smaller than this
    hold_preserves_position: bool = True  # HOLD signal keeps current position (trend-follow)
    # Annualized risk-free rate subtracted from returns in Sharpe. Default
    # is 0.0 for backwards compatibility with the project's historical
    # result docs, but SKEPTIC_REVIEW.md § 8c flags this as a small upward
    # bias on reported Sharpes — the US 3-month T-bill averaged ~2.3% over
    # 2018-2024 and peaked > 5% in 2023-2024. New result runs should pass
    # `risk_free_rate=historical_usd_rate(window)` from
    # signals.backtest.risk_free, which returns 0.023 for the 2018-2024
    # reporting window by default.
    risk_free_rate: float = 0.0
    # Annualization factor for Sharpe. None = legacy index-inference, which
    # returns 252 for any daily-cadence series — wrong for BTC (trades 365
    # days/year). Set to 365 for crypto, 252 for equities. See
    # SKEPTIC_REVIEW.md § 8a / Tier B6. The CLI auto-detects crypto vs
    # equity symbols and overrides this; passing it explicitly via
    # BacktestConfig(periods_per_year=...) always wins.
    periods_per_year: float | None = None
    # Volatility-targeting overlay (Tier-4 item — see signals/backtest/vol_target.py).
    # Disabled by default to preserve baseline behavior; enable via
    # `vol_target_enabled=True` plus an annual target (e.g., 0.20 for 20%).
    vol_target_enabled: bool = False
    vol_target_annual: float = 0.20
    vol_target_periods_per_year: int = 365  # BTC trades daily; use 252 for equities
    vol_target_max_scale: float = 2.0
    vol_target_min_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.train_window < 10:
            raise ValueError(f"train_window must be >= 10, got {self.train_window}")
        if self.retrain_freq < 1:
            raise ValueError(f"retrain_freq must be >= 1, got {self.retrain_freq}")
        if self.retrain_freq > self.train_window:
            raise ValueError(
                f"retrain_freq ({self.retrain_freq}) must be <= "
                f"train_window ({self.train_window})"
            )
        if not 0.0 <= self.hybrid_vol_quantile <= 1.0:
            raise ValueError(
                f"hybrid_vol_quantile must be in [0, 1], got {self.hybrid_vol_quantile}"
            )
        if self.max_long < 0:
            raise ValueError(f"max_long must be >= 0, got {self.max_long}")
        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be > 0, got {self.initial_cash}")


#: Round-3 large-grid search winner for BTC hybrid — the current "best
#: known" configuration. Pass as `BacktestConfig(**BTC_HYBRID_PRODUCTION)`
#: to get the 1.551 ± 0.099 Sharpe result from
#: `scripts/data/explore_improvements.md` Tier 4. This dict intentionally
#: does NOT become the field-level default because some params
#: (retrain_freq, train_window) are shared across model types and the
#: composite/HOMC models have different preferred values. The hybrid
#: production config is a bundle; pass all three together.
BTC_HYBRID_PRODUCTION: dict = {
    "model_type": "hybrid",
    "hybrid_routing_strategy": "vol",
    "hybrid_vol_quantile": 0.50,
    "retrain_freq": 14,
    "train_window": 750,
    "n_states": 5,
    "order": 5,
    "return_bins": 3,
    "volatility_bins": 3,
    "vol_window": 10,
    "laplace_alpha": 0.01,
    "buy_threshold_bps": 25.0,
    "sell_threshold_bps": -35.0,
    "target_scale_bps": 20.0,
    "max_long": 1.0,
    "allow_short": False,
    "periods_per_year": 365.0,
    # risk_free_rate should be passed explicitly from
    # signals.backtest.risk_free.historical_usd_rate(window)
    # based on the reporting window the caller cares about.
}


@dataclass
class BacktestResult:
    config: BacktestConfig
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    signals: pd.DataFrame
    metrics: Metrics
    benchmark_metrics: Metrics
    trades: list = field(default_factory=list)


#: Canonical feature column name for trailing realized volatility. The
#: period is always `BacktestConfig.vol_window` bars — historically this
#: column was called `volatility_20d` back when the window defaulted to
#: 20, but the "tightened defaults" pass cut it to 10 without renaming.
#: SKEPTIC_REVIEW.md § 8b flagged the name-vs-value mismatch. Tier B6 /
#: § 19 landed the rename to a neutral `volatility` — every consumer
#: (composite encoder, HMM, HOMC, hybrid router, vol-target overlay)
#: reads this name. Do not re-introduce a hard-coded window in the name.
VOLATILITY_COLUMN = "volatility"


def _prepare_features(prices: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    feats = pd.DataFrame(index=prices.index)
    feats["close"] = prices["close"]
    feats["open"] = prices["open"]
    feats["return_1d"] = log_returns(prices["close"])
    feats[VOLATILITY_COLUMN] = rolling_volatility(feats["return_1d"], window=vol_window)
    return feats


class BacktestEngine:
    """Walk-forward backtest with periodic retraining + sized long/short execution."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    # ----- Model factory -----
    def _make_model(self):
        cfg = self.config
        if cfg.model_type == "composite":
            return CompositeMarkovChain(
                return_bins=cfg.return_bins,
                volatility_bins=cfg.volatility_bins,
                alpha=cfg.laplace_alpha,
            )
        if cfg.model_type == "hmm":
            return HiddenMarkovModel(
                n_states=cfg.n_states,
                n_iter=cfg.n_iter,
                n_init=cfg.n_init,
                tol=cfg.tol,
                strict_convergence=cfg.strict_convergence,
                random_state=cfg.random_state,
            )
        if cfg.model_type == "homc":
            return HigherOrderMarkovChain(
                n_states=cfg.n_states,
                order=cfg.order,
                alpha=cfg.laplace_alpha,
            )
        if cfg.model_type == "boost":
            return GradientBoostingModel(
                n_estimators=cfg.boost_n_estimators,
                max_depth=cfg.boost_max_depth,
                learning_rate=cfg.boost_learning_rate,
                random_state=cfg.random_state,
            )
        if cfg.model_type == "ensemble":
            # Default ensemble: composite-3×3 + HOMC@5 + boost@100
            return EnsembleModel()
        if cfg.model_type == "trend":
            return TrendFilter(window=cfg.trend_window)
        if cfg.model_type == "golden_cross":
            return DualMovingAverage(
                fast_window=cfg.trend_fast_window,
                slow_window=cfg.trend_slow_window,
            )
        if cfg.model_type == "vol_filter":
            return NaiveVolFilter(
                vol_window=cfg.vol_window,
                quantile=cfg.vol_filter_quantile,
            )
        if cfg.model_type == "hybrid":
            return HybridRegimeModel(
                regime_n_states=3,
                regime_n_iter=cfg.n_iter,
                regime_random_state=cfg.random_state,
                composite_return_bins=cfg.return_bins,
                composite_volatility_bins=cfg.volatility_bins,
                composite_alpha=cfg.laplace_alpha,
                homc_n_states=cfg.n_states if cfg.n_states >= 2 else 5,
                homc_order=cfg.order,
                homc_alpha=max(cfg.laplace_alpha, 1.0),  # HOMC prefers alpha>=1
                routing=cfg.hybrid_routing or dict(DEFAULT_ROUTING),
                routing_strategy=cfg.hybrid_routing_strategy,
                vol_quantile_threshold=cfg.hybrid_vol_quantile,
                blend_low_quantile=cfg.hybrid_blend_low,
                blend_high_quantile=cfg.hybrid_blend_high,
                adaptive_low_quantile=cfg.hybrid_adaptive_low,
                adaptive_high_quantile=cfg.hybrid_adaptive_high,
                adaptive_lookback=cfg.hybrid_adaptive_lookback,
            )
        raise ValueError(f"unknown model_type: {cfg.model_type!r}")

    def _fit_kwargs(self) -> dict:
        if self.config.model_type == "hmm":
            return {
                "feature_cols": ["return_1d", VOLATILITY_COLUMN],
                "return_col": "return_1d",
            }
        if self.config.model_type == "composite":
            return {
                "return_feature": "return_1d",
                "volatility_feature": VOLATILITY_COLUMN,
                "return_col": "return_1d",
            }
        if self.config.model_type == "hybrid":
            # HybridRegimeModel.fit accepts any kwargs and handles component
            # dispatch internally. Pass a neutral default that matches its
            # signature; unused kwargs are ignored by the hybrid.
            return {"feature_col": "return_1d", "return_col": "return_1d"}
        if self.config.model_type in ("trend", "golden_cross"):
            # Trend models fit() is a no-op that only reads the "close"
            # column. They accept and ignore any additional kwargs.
            return {}
        if self.config.model_type == "boost":
            return {"feature_col": "return_1d", "return_col": "return_1d"}
        if self.config.model_type == "ensemble":
            return {"feature_col": "return_1d", "return_col": "return_1d"}
        return {"feature_col": "return_1d", "return_col": "return_1d"}

    # ----- Run -----
    def run(
        self,
        prices: pd.DataFrame,
        symbol: str = "",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> BacktestResult:
        if not {"open", "close"}.issubset(prices.columns):
            raise ValueError("prices must contain 'open' and 'close' columns")

        df = prices.sort_index()
        if start is not None:
            df = df.loc[df.index >= pd.Timestamp(start, tz=df.index.tz or "UTC")]
        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end, tz=df.index.tz or "UTC")]

        feats = _prepare_features(df, self.config.vol_window)
        feats = feats.dropna(subset=["return_1d", VOLATILITY_COLUMN])
        if len(feats) <= self.config.train_window + 1:
            raise ValueError(
                f"Not enough data ({len(feats)}) for train_window={self.config.train_window}"
            )

        portfolio = Portfolio(
            initial_cash=self.config.initial_cash,
            commission_bps=self.config.commission_bps,
            slippage_bps=self.config.slippage_bps,
        )

        vol_target_cfg = VolTargetConfig(
            enabled=self.config.vol_target_enabled,
            annual_target=self.config.vol_target_annual,
            periods_per_year=self.config.vol_target_periods_per_year,
            max_scale=self.config.vol_target_max_scale,
            min_scale=self.config.vol_target_min_scale,
        )

        model = None
        generator: SignalGenerator | None = None
        bars_since_retrain = self.config.retrain_freq  # force initial fit
        cooldown = 0

        records: list[dict] = []
        idx = feats.index
        start_i = self.config.train_window
        end_i = len(feats) - 1

        for i in range(start_i, end_i):
            ts = idx[i]
            next_ts = idx[i + 1]
            row = feats.iloc[i]
            close_price = float(row["close"])
            next_open = float(feats.iloc[i + 1]["open"])

            # Stop-loss is checked first, against current bar's close.
            if self.config.stop_loss_pct > 0 and not portfolio.is_flat:
                fired = portfolio.check_stop(ts, close_price, self.config.stop_loss_pct)
                if fired:
                    cooldown = self.config.stop_cooldown_bars

            # Retrain on a strict-past window
            if bars_since_retrain >= self.config.retrain_freq:
                window = feats.iloc[i - self.config.train_window : i]
                model = self._make_model()
                try:
                    model.fit(window, **self._fit_kwargs())
                except Exception as e:
                    log.warning("model fit failed at bar %d: %s", i, e)
                    portfolio.mark(ts, close_price)
                    bars_since_retrain += 1
                    if cooldown > 0:
                        cooldown -= 1
                    continue
                generator = SignalGenerator(
                    model=model,
                    buy_threshold_bps=self.config.buy_threshold_bps,
                    sell_threshold_bps=self.config.sell_threshold_bps,
                    target_scale_bps=self.config.target_scale_bps,
                    max_long=self.config.max_long,
                    max_short=self.config.max_short,
                    allow_short=self.config.allow_short,
                )
                bars_since_retrain = 0

            if model is None or generator is None:
                raise RuntimeError(
                    f"Model/generator not initialized by bar {i}. "
                    f"Check that train_window ({self.config.train_window}) "
                    f"<= data length ({len(feats)}) and initial fit succeeded."
                )

            inference_window = feats.iloc[i - self.config.train_window + 1 : i + 1]
            try:
                current_state = model.predict_state(inference_window)
            except Exception as e:
                log.warning("decode failed at bar %d: %s", i, e)
                portfolio.mark(ts, close_price)
                bars_since_retrain += 1
                if cooldown > 0:
                    cooldown -= 1
                continue

            decision = generator.generate(current_state)

            # If we're in cooldown after a stop, force flat regardless of signal.
            if cooldown > 0:
                target = 0.0
                cooldown -= 1
            elif decision.signal.value == "HOLD" and self.config.hold_preserves_position:
                # HOLD = no actionable edge. Don't flatten an existing position;
                # let the trend run until an opposite signal or stop fires.
                target = portfolio.position_fraction(close_price)
            else:
                target = decision.target_position

            # Volatility-targeting overlay. Scales the raw target by
            # target_annual_vol / realized_annualized_vol so that
            # expected portfolio vol over the next bar approximates the
            # target. No-op when disabled. Uses bar-t's realized-vol
            # column (trailing-window realized vol, no lookahead).
            if vol_target_cfg.enabled and target != 0.0:
                realized_vol = float(row[VOLATILITY_COLUMN])
                target = apply_vol_target(target, realized_vol, vol_target_cfg)

            portfolio.set_target(
                next_ts, next_open, target,
                min_trade_fraction=self.config.min_trade_fraction,
            )
            portfolio.mark(ts, close_price)
            records.append(
                {
                    "ts": ts,
                    "state": str(decision.state),
                    "state_label": decision.state_label,
                    "signal": decision.signal.value,
                    "confidence": decision.confidence,
                    "expected_return": decision.expected_return,
                    "target_position": target,
                }
            )
            bars_since_retrain += 1

        last_close = float(feats.iloc[end_i]["close"])
        last_ts = idx[end_i]
        portfolio.flatten(last_ts, last_close)
        portfolio.mark(last_ts, last_close)

        equity_curve = portfolio.equity_series()
        signals_df = pd.DataFrame(records).set_index("ts") if records else pd.DataFrame()

        bench_window = feats.iloc[start_i : end_i + 1]
        if len(bench_window) > 0:
            initial_price = float(bench_window.iloc[0]["close"])
            bench = (bench_window["close"] / initial_price) * self.config.initial_cash
            bench.name = "benchmark"
        else:
            bench = pd.Series(dtype=float, name="benchmark")

        metrics = compute_metrics(
            equity_curve,
            portfolio.trades,
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=self.config.periods_per_year,
        )
        bench_metrics = compute_metrics(
            bench,
            [],
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=self.config.periods_per_year,
        )

        return BacktestResult(
            config=self.config,
            symbol=symbol,
            start=feats.index[start_i],
            end=feats.index[end_i],
            equity_curve=equity_curve,
            benchmark_curve=bench,
            signals=signals_df,
            metrics=metrics,
            benchmark_metrics=bench_metrics,
            trades=portfolio.trades,
        )
