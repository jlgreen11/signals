"""Automation layer: signal storage, cash overlay, insights engine, and paper trading."""

from signals.automation.cash_overlay import CashOverlay
from signals.automation.insights_engine import InsightsEngine
from signals.automation.paper_runner import PaperTradeRunner
from signals.automation.signal_store import SignalStore

__all__ = ["SignalStore", "CashOverlay", "InsightsEngine", "PaperTradeRunner"]
