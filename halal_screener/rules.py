"""AAOIFI Standard 21 screen for one stock. Pure: no I/O."""
import math
import re

# Yahoo Finance industry labels. Edit here if your scholar draws the line elsewhere.
PROHIBITED_INDUSTRIES = {
    "Banks - Diversified", "Banks - Regional",
    "Insurance - Life", "Insurance - Property & Casualty", "Insurance - Diversified",
    "Insurance - Specialty", "Insurance - Reinsurance", "Insurance Brokers",
    "Capital Markets", "Credit Services", "Mortgage Finance",
    "Beverages - Brewers", "Beverages - Wineries & Distilleries",
    "Tobacco", "Gambling", "Resorts & Casinos",
}
GREY_INDUSTRIES = {"Aerospace & Defense", "Lodging", "Entertainment", "Broadcasting", "Restaurants"}
# Whole sector fails: catches financial industries missing from the list above
# (asset managers, exchanges, ...). Use the allowlist for any you've reviewed.
PROHIBITED_SECTORS = {"Financial Services"}


def _norm(label: str) -> str:
    """Yahoo has used '-', '–' and '—' in the same labels; compare loosely."""
    return re.sub(r"\s*[-–—]\s*", " - ", label).strip().casefold()


_PROHIBITED = {_norm(x): x for x in PROHIBITED_INDUSTRIES}
_GREY = {_norm(x): x for x in GREY_INDUSTRIES}
_PROHIBITED_SECTORS = {_norm(x): x for x in PROHIBITED_SECTORS}

FIELDS = ["sector", "industry", "total_debt", "cash_and_st", "interest_income",
          "total_revenue", "market_cap", "avg_market_cap_12m", "total_assets", "receivables"]
STANDARDS = ("aaoifi", "wahed")

# AAOIFI Standard 21: vs market cap, all strictly below.
MAX_DEBT = 0.30
MAX_CASH = 0.30
MAX_INCOME = 0.05
# Wahed / FTSE Yasaar ground rules v4.6 s.4.3.2: vs total assets.
WAHED_MAX_DEBT = 0.33333
WAHED_MAX_CASH = 0.33333
WAHED_MAX_RECEIVABLES_CASH = 0.50
WAHED_MAX_INCOME = 0.05  # "should not exceed": exactly 5% passes

_NEEDS = {
    "aaoifi": ["total_debt", "cash_and_st", "interest_income", "total_revenue",
               "market_cap", "avg_market_cap_12m"],
    "wahed": ["total_debt", "cash_and_st", "interest_income", "total_revenue",
              "total_assets", "receivables"],
}
_DENOMINATORS = {"market_cap", "avg_market_cap_12m", "total_revenue", "total_assets"}


def _num(v):
    """Float or None for None/NaN/empty/non-numeric."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def evaluate(row: dict, standard: str = "aaoifi") -> dict:
    if standard not in STANDARDS:
        raise ValueError(f"standard must be one of {STANDARDS}, got {standard!r}")
    reasons = []
    for f in ("sector", "industry"):
        v = row.get(f)
        if not isinstance(v, str) or not v.strip():
            reasons.append(f"missing:{f}")
    vals = {}
    for f in _NEEDS[standard]:
        v = _num(row.get(f))
        if v is None or (f in _DENOMINATORS and v <= 0):
            reasons.append(f"missing:{f}")
            v = None
        vals[f] = v

    sector = _norm(row.get("sector") or "")
    industry = _norm(row.get("industry") or "")
    if sector in _PROHIBITED_SECTORS:
        reasons.append(f"prohibited:{_PROHIBITED_SECTORS[sector]}")
    if industry in _PROHIBITED:
        reasons.append(f"prohibited:{_PROHIBITED[industry]}")
    elif industry in _GREY:
        reasons.append(f"grey:{_GREY[industry]}")

    def ratio(num, den):
        n = sum(vals[x] for x in num) if all(vals[x] is not None for x in num) else None
        return None if n is None or vals[den] is None else n / vals[den]

    out = dict.fromkeys(["debt_ratio", "debt_ratio_avg", "cash_ratio", "cash_ratio_avg",
                         "receivables_cash_ratio"])
    out["income_ratio"] = ratio(["interest_income"], "total_revenue")
    if standard == "aaoifi":
        out.update(debt_ratio=ratio(["total_debt"], "market_cap"),
                   debt_ratio_avg=ratio(["total_debt"], "avg_market_cap_12m"),
                   cash_ratio=ratio(["cash_and_st"], "market_cap"),
                   cash_ratio_avg=ratio(["cash_and_st"], "avg_market_cap_12m"))
        # (key, code, limit, exactly-at-limit passes?)
        limits = [("debt_ratio", "debt", MAX_DEBT, False), ("debt_ratio_avg", "debt_avg", MAX_DEBT, False),
                  ("cash_ratio", "cash", MAX_CASH, False), ("cash_ratio_avg", "cash_avg", MAX_CASH, False),
                  ("income_ratio", "income", MAX_INCOME, False)]
    else:
        out.update(debt_ratio=ratio(["total_debt"], "total_assets"),
                   cash_ratio=ratio(["cash_and_st"], "total_assets"),
                   receivables_cash_ratio=ratio(["receivables", "cash_and_st"], "total_assets"))
        limits = [("debt_ratio", "debt", WAHED_MAX_DEBT, False), ("cash_ratio", "cash", WAHED_MAX_CASH, False),
                  ("receivables_cash_ratio", "receivables_cash", WAHED_MAX_RECEIVABLES_CASH, False),
                  ("income_ratio", "income", WAHED_MAX_INCOME, True)]
    for key, code, limit, at_limit_ok in limits:
        r = out[key]
        if r is not None and (r > limit if at_limit_ok else r >= limit):
            reasons.append(code)

    out["passed"] = not reasons
    out["reasons"] = ";".join(reasons)
    out["purification_pct"] = out["income_ratio"] if out["passed"] else None
    return out
