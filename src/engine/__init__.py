"""Shared computation interface for CLI and localhost dashboard."""
from .backtest import Snapshot, BacktestResult, run_backtest
from .config import normalize_config

__all__ = ['Snapshot', 'BacktestResult', 'run_backtest', 'normalize_config']
