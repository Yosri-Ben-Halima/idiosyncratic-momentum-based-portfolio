import pandas as pd
from halal_screener.fetch import normalize_symbol, latest_set, pick_label


def test_normalize_symbol():
    assert normalize_symbol(" BRK.B ") == "BRK-B"
    assert normalize_symbol("AAPL") == "AAPL"


def frame():
    cols = pd.to_datetime(["2026-09-30", "2026-06-30"])
    return pd.DataFrame({cols[0]: [float("nan"), 5.0], cols[1]: [7.0, 4.0]},
                        index=["Total Debt", "Total Revenue"])


def test_latest_set_empty_frame():
    assert latest_set(pd.DataFrame(), ["Total Debt"]) == {"Total Debt": None}


def test_latest_set_takes_one_period_for_all_rows():
    # newest column has NaN debt -> both values come from the older column, never mixed
    assert latest_set(frame(), ["Total Debt", "Total Revenue"]) == {"Total Debt": 7.0, "Total Revenue": 4.0}


def test_latest_set_absent_label_is_none_others_still_read():
    assert latest_set(frame(), ["Interest Income", "Total Revenue"]) == {"Interest Income": None, "Total Revenue": 5.0}


def test_pick_label_falls_back():
    f = pd.DataFrame({"x": [1.0]}, index=["Interest Income Non Operating"])
    assert pick_label(f, "Interest Income", "Interest Income Non Operating") == "Interest Income Non Operating"


from halal_screener.fetch import income_fields


def stmt(rows):
    col = pd.to_datetime(["2025-12-31"])[0]
    return pd.DataFrame({col: list(rows.values())}, index=list(rows))


def test_income_fields_direct_label():
    assert income_fields(stmt({"Interest Income": 3.0, "Total Revenue": 100.0})) == (3.0, 100.0)


def test_income_fields_derived_from_net_interest():
    # AOS-style: no Interest Income row; net interest = income - expense
    f = stmt({"Net Interest Income": -13.5, "Interest Expense": 13.5, "Total Revenue": 100.0})
    assert income_fields(f) == (0.0, 100.0)


def test_income_fields_inconsistent_derivation_is_missing():
    f = stmt({"Net Interest Income": -20.0, "Interest Expense": 13.5, "Total Revenue": 100.0})
    assert income_fields(f) == (None, 100.0)


def test_latest_set_optional_same_column():
    cols = pd.to_datetime(["2026-09-30", "2026-06-30"])
    f = pd.DataFrame({cols[0]: [float("nan"), 50.0, 1.0], cols[1]: [7.0, 40.0, 2.0]},
                     index=["Total Debt", "Total Assets", "Accounts Receivable"])
    # newest column lacks debt -> older column for everything, including the optional row
    assert latest_set(f, ["Total Debt", "Total Assets"], optional=["Accounts Receivable"]) == \
        {"Total Debt": 7.0, "Total Assets": 40.0, "Accounts Receivable": 2.0}
    f.loc["Accounts Receivable", cols[1]] = float("nan")
    assert latest_set(f, ["Total Debt"], optional=["Accounts Receivable"])["Accounts Receivable"] is None


def test_balance_sheet_period_chosen_by_debt_and_cash_only(monkeypatch):
    # Newest quarter lacks Total Assets: AAOIFI must still get the NEWEST debt/cash
    import halal_screener.fetch as fm
    cols = pd.to_datetime(["2026-09-30", "2026-06-30"])
    bs = pd.DataFrame({cols[0]: [50.0, 10.0, float("nan"), 5.0], cols[1]: [20.0, 10.0, 300.0, 5.0]},
                      index=["Total Debt", "Cash Cash Equivalents And Short Term Investments",
                             "Total Assets", "Accounts Receivable"])

    class FakeTicker:
        info = {"sector": "Technology", "industry": "Software", "marketCap": 100.0, "sharesOutstanding": 1.0}
        quarterly_balance_sheet = bs
        income_stmt = pd.DataFrame()

        def history(self, period):
            return pd.DataFrame({"Close": [100.0]})

    monkeypatch.setattr(fm.yf, "Ticker", lambda t: FakeTicker())
    row = fm.fetch_fundamentals("X")
    assert row["total_debt"] == 50.0 and row["total_assets"] is None
