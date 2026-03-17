import yaml
from utils.rate_helper import _fetch_risk_free_rate


def load_config():
    """
    Load configuration from config.yaml, fetch risk-free rate, and compute derived parameters.
    
    Returns:
        dict: Configuration dictionary with injected risk-free rate and derived parameters.
    """
    with open("src/environment/config.yaml") as f:
        CFG = yaml.safe_load(f)

    CFG["rf_annual"] = _fetch_risk_free_rate()  # injected at runtime

    RF_DAILY = (1 + CFG["rf_annual"]) ** (1 / 252) - 1
    COST_BPS = CFG["transaction_cost_bps"] / 10000  # convert bps to decimal
    CFG["rf_daily"] = RF_DAILY
    CFG["cost_bps"] = COST_BPS
    return CFG
