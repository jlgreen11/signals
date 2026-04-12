"""Portfolio optimizer package.

Provides four weighting schemes beyond equal-weight:
- equal_volatility: inverse-volatility weights
- risk_parity: equal risk contribution (Spinu 2013-style)
- mean_variance: max Sharpe via scipy SLSQP
- max_diversification: maximize diversification ratio (Choueifaty & Coignard)

Each module exposes an ``optimize_weights()`` convenience function that takes
the signals project's native data format: ``prices_dict`` (dict of ticker ->
price DataFrame) and ``selected_tickers`` (list of tickers to allocate across).
"""

from signals.backtest.optimizers.equal_volatility import EqualVolatilityOptimizer
from signals.backtest.optimizers.equal_volatility import (
    optimize_weights as equal_volatility_weights,
)
from signals.backtest.optimizers.max_diversification import MaxDiversificationOptimizer
from signals.backtest.optimizers.max_diversification import (
    optimize_weights as max_diversification_weights,
)
from signals.backtest.optimizers.mean_variance import MeanVarianceOptimizer
from signals.backtest.optimizers.mean_variance import (
    optimize_weights as mean_variance_weights,
)
from signals.backtest.optimizers.risk_parity import RiskParityOptimizer
from signals.backtest.optimizers.risk_parity import (
    optimize_weights as risk_parity_weights,
)

__all__ = [
    "EqualVolatilityOptimizer",
    "MaxDiversificationOptimizer",
    "MeanVarianceOptimizer",
    "RiskParityOptimizer",
    "equal_volatility_weights",
    "max_diversification_weights",
    "mean_variance_weights",
    "risk_parity_weights",
]
