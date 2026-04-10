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
from signals.features.returns import log_returns
from signals.features.volatility import rolling_volatility
from signals.model.composite import CompositeMarkovChain
from signals.model.hmm import HiddenMarkovModel
from signals.model.homc import HigherOrderMarkovChain
from signals.model.signals import SignalGenerator
from signals.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestConfig:
    model_type: str = "composite"     # "composite" | "hmm" | "homc"
    train_window: int = 252
    retrain_freq: int = 21
    n_states: int = 9                 # composite: ignored (uses return_bins×vol_bins); homc/hmm: used
    return_bins: int = 3              # composite only
    volatility_bins: int = 3          # composite only
    order: int = 3                    # homc only
    vol_window: int = 10              # tightened: short window reacts faster to vol regimes
    n_iter: int = 200                 # hmm only
    random_state: int = 42
    laplace_alpha: float = 0.01       # tightened: low smoothing for composite (1.0 for hmm/homc)
    initial_cash: float = 10_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    buy_threshold_bps: float = 25.0
    sell_threshold_bps: float = -35.0
    target_scale_bps: float = 20.0
    allow_short: bool = False         # tightened: BTC's secular uptrend punishes shorts
    max_long: float = 1.0
    max_short: float = 1.0
    stop_loss_pct: float = 0.0        # tightened: empirically unhelpful for composite (sell signal exits faster than any reasonable stop)
    stop_cooldown_bars: int = 5
    min_trade_fraction: float = 0.20  # don't rebalance for changes smaller than this
    hold_preserves_position: bool = True  # HOLD signal keeps current position (trend-follow)


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


def _prepare_features(prices: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    feats = pd.DataFrame(index=prices.index)
    feats["close"] = prices["close"]
    feats["open"] = prices["open"]
    feats["return_1d"] = log_returns(prices["close"])
    feats["volatility_20d"] = rolling_volatility(feats["return_1d"], window=vol_window)
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
                random_state=cfg.random_state,
            )
        if cfg.model_type == "homc":
            return HigherOrderMarkovChain(
                n_states=cfg.n_states,
                order=cfg.order,
                alpha=cfg.laplace_alpha,
            )
        raise ValueError(f"unknown model_type: {cfg.model_type!r}")

    def _fit_kwargs(self) -> dict:
        if self.config.model_type == "hmm":
            return {
                "feature_cols": ["return_1d", "volatility_20d"],
                "return_col": "return_1d",
            }
        if self.config.model_type == "composite":
            return {
                "return_feature": "return_1d",
                "volatility_feature": "volatility_20d",
                "return_col": "return_1d",
            }
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
        feats = feats.dropna(subset=["return_1d", "volatility_20d"])
        if len(feats) <= self.config.train_window + 1:
            raise ValueError(
                f"Not enough data ({len(feats)}) for train_window={self.config.train_window}"
            )

        portfolio = Portfolio(
            initial_cash=self.config.initial_cash,
            commission_bps=self.config.commission_bps,
            slippage_bps=self.config.slippage_bps,
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

            assert model is not None and generator is not None

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

        metrics = compute_metrics(equity_curve, portfolio.trades)
        bench_metrics = compute_metrics(bench, [])

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
