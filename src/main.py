import warnings

from environment import load_config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = load_config()
RF_DAILY = CFG["rf_daily"]
COST_BPS = CFG["cost_bps"]


def main():
    """Pipeline orchestrating data loading, backtesting, and visualization."""
    pass
