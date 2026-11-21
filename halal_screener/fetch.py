"""S&P 500 constituents (Wikipedia) and per-ticker fundamentals (Yahoo via yfinance)."""
import io

import pandas as pd
import requests
import yfinance as yf

from .rules import FIELDS

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "halal-screener/0.1"


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().replace(".", "-")


def pick_label(frame: pd.DataFrame, *labels: str) -> str:
    """First of `labels` present in the statement, else the first one (which then reads as missing)."""
    present = [] if frame is None else frame.index
    return next((lb for lb in labels if lb in present), labels[0])


def latest_set(frame: pd.DataFrame, labels: list[str],
               optional: list[str] = ()) -> dict[str, float | None]:
    """Values of `labels` from ONE period: the newest column where every present label is non-NaN.

    Never mixes periods (an older, lower debt next to a newer market cap). Absent labels -> None.
    `optional` labels are read from that same column (None if NaN there); they never pick the column.
    """
    out = dict.fromkeys([*labels, *optional])
    if frame is None or frame.empty:
        return out
    present = [lb for lb in labels if lb in frame.index]
    for col in sorted(frame.columns, reverse=True):
        vals = frame.loc[present, col]
        if present and not vals.isna().any():
            out.update({lb: float(vals[lb]) for lb in present})
            for lb in optional:
                if lb in frame.index and not pd.isna(frame.loc[lb, col]):
                    out[lb] = float(frame.loc[lb, col])
            return out
    return out


def income_fields(inc: pd.DataFrame) -> tuple[float | None, float | None]:
    """(interest_income, total_revenue) from one annual period.

    Falls back to Net Interest Income + Interest Expense (= interest income) when Yahoo
    has no Interest Income row; an inconsistent (negative) result counts as missing.
    """
    label = pick_label(inc, "Interest Income", "Interest Income Non Operating")
    v = latest_set(inc, [label, "Total Revenue"])
    if v[label] is not None or label in (inc.index if inc is not None else []):
        return v[label], v["Total Revenue"]
    d = latest_set(inc, ["Net Interest Income", "Interest Expense", "Total Revenue"])
    if d["Net Interest Income"] is None or d["Interest Expense"] is None:
        return None, d["Total Revenue"]
    derived = d["Net Interest Income"] + d["Interest Expense"]
    return (derived if derived >= 0 else None), d["Total Revenue"]


def sp500_tickers() -> list[str]:
    resp = requests.get(SP500_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    return [normalize_symbol(s) for s in table["Symbol"]]


def fetch_fundamentals(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info or {}
    bs = t.quarterly_balance_sheet
    inc = t.income_stmt
    shares = info.get("sharesOutstanding")
    closes = t.history(period="1y")["Close"]
    avg_cap = shares * float(closes.mean()) if shares and len(closes) else None
    debt, cash = "Total Debt", "Cash Cash Equivalents And Short Term Investments"
    receivables = pick_label(bs, "Accounts Receivable", "Receivables")
    # debt + cash pick the period (as AAOIFI needs); Wahed-only rows read from that same period
    b = latest_set(bs, [debt, cash], optional=["Total Assets", receivables])
    interest_income, revenue = income_fields(inc)
    row = {
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "total_debt": b[debt],
        "cash_and_st": b[cash],
        "interest_income": interest_income,
        "total_revenue": revenue,
        "avg_market_cap_12m": avg_cap,
        "total_assets": b["Total Assets"],
        "receivables": b[receivables],
    }
    return {f: row[f] for f in FIELDS}
