# Halal S&P 500 Screener Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python package whose `halal_tickers()` returns the S&P 500 tickers passing AAOIFI Standard 21, read from a snapshot refreshed monthly by `python -m halal_screener refresh`.

**Architecture:** Three files with one job each: `rules.py` (pure AAOIFI decision on one row), `fetch.py` (S&P 500 list + Yahoo fundamentals), `api.py` (refresh → dated CSV snapshot; read snapshot + allowlist → tickers). `__main__.py` is the CLI.

**Tech Stack:** Python 3.10+, pandas, yfinance, requests, lxml, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-halal-screener-design.md`

## Global Constraints

- Standard: AAOIFI Standard 21 only. Thresholds are **strictly below**: debt < 0.30, cash < 0.30, non-permissible income < 0.05 (a ratio equal to the limit fails).
- Ratio tests use market cap **and** 12-month average market cap (current shares × mean daily close over 1 year); both must pass.
- Grey-area industries **fail**. Missing or unusable data **fails** with reason `missing:<field>`.
- Free data only (Yahoo via `yfinance`, S&P 500 list from Wikipedia). No paid APIs, no scheduler, no UI.
- No network calls in tests.
- HTTP User-Agent is the generic `halal-screener/0.1`; no personal details in requests.
- Commits carry no AI attribution trailers.

## Review Focus

1. **Ratio exactly at a limit** (e.g. debt = 30% of market cap) → fails. Pinned in Task 1 `test_ratio_at_limit_fails`.
2. **Yahoo returns NaN / omits a field** → that stock fails with `missing:<field>`, no crash. Pinned in Task 1 `test_missing_field_fails` and Task 2 `test_latest_missing_label`.
3. **Zero or negative market cap or revenue** → fails as missing, no `ZeroDivisionError`. Pinned in Task 1 `test_zero_denominator_fails`.
4. **One ticker's fetch raises mid-refresh** (delisted, timeout) → refresh finishes, that row fails with reason `error:<ExceptionName>`. Pinned in Task 3 `test_refresh_survives_fetch_error`.
5. **Allowlisted ticker no longer in the S&P 500 snapshot** → not returned. Pinned in Task 3 `test_allowlist_ignores_tickers_outside_snapshot`.

Known consequence (by spec, not a bug): Yahoo often omits `Total Debt` for debt-free firms or `Interest Income` for some firms; those fail and need the allowlist after manual review.

---

### Task 1: Project setup + AAOIFI rules

**Files:**
- Create: `requirements.txt` (`pandas`, `yfinance`, `requests`, `lxml`, `pytest`), `.gitignore` (`__pycache__/`, `.venv/`)
- Create: `halal_screener/__init__.py` (empty for now), `halal_screener/rules.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Produces:
  - `PROHIBITED_INDUSTRIES: set[str]` — Yahoo industry labels: `Banks - Diversified`, `Banks - Regional`, `Insurance - Life`, `Insurance - Property & Casualty`, `Insurance - Diversified`, `Insurance - Specialty`, `Insurance - Reinsurance`, `Insurance Brokers`, `Capital Markets`, `Credit Services`, `Mortgage Finance`, `Beverages - Brewers`, `Beverages - Wineries & Distilleries`, `Tobacco`, `Gambling`, `Resorts & Casinos`.
  - `GREY_INDUSTRIES: set[str]` — `Aerospace & Defense`, `Lodging`, `Entertainment`, `Broadcasting`, `Restaurants`.
  - `FIELDS: list[str]` = `["sector", "industry", "total_debt", "cash_and_st", "interest_income", "total_revenue", "market_cap", "avg_market_cap_12m"]`
  - `evaluate(row: dict) -> dict` — returns `{"debt_ratio", "cash_ratio", "debt_ratio_avg", "cash_ratio_avg", "income_ratio": float | None, "passed": bool, "reasons": str, "purification_pct": float | None}`. `reasons` is `;`-joined codes, empty string when passed. Codes: `missing:<field>`, `prohibited:<industry>`, `grey:<industry>`, `debt`, `debt_avg`, `cash`, `cash_avg`, `income`. `purification_pct` = `income_ratio` when passed, else `None`.

- [ ] **Step 1: `git init`, write `requirements.txt` and `.gitignore`, `pip install -r requirements.txt`**

- [ ] **Step 2: Write the failing tests in `tests/test_rules.py`**

```python
from halal_screener.rules import evaluate

def ok(**kw):
    row = dict(sector="Technology", industry="Software - Infrastructure",
               total_debt=10.0, cash_and_st=10.0, interest_income=1.0,
               total_revenue=100.0, market_cap=100.0, avg_market_cap_12m=100.0)
    row.update(kw)
    return row

def test_clean_stock_passes():
    r = evaluate(ok())
    assert r["passed"] and r["reasons"] == ""
    assert r["purification_pct"] == 0.01

def test_ratio_at_limit_fails():
    assert "debt" in evaluate(ok(total_debt=30.0))["reasons"].split(";")
    assert "income" in evaluate(ok(interest_income=5.0))["reasons"].split(";")

def test_avg_market_cap_must_also_pass():
    r = evaluate(ok(total_debt=25.0, avg_market_cap_12m=80.0))  # 25/80 = 31%
    assert not r["passed"] and r["reasons"] == "debt_avg"

def test_cash_fails():
    assert evaluate(ok(cash_and_st=40.0))["reasons"] == "cash;cash_avg"

def test_prohibited_and_grey_industries_fail():
    assert evaluate(ok(industry="Banks - Regional"))["reasons"] == "prohibited:Banks - Regional"
    assert evaluate(ok(industry="Restaurants"))["reasons"] == "grey:Restaurants"

def test_missing_field_fails():
    r = evaluate(ok(interest_income=float("nan")))
    assert not r["passed"] and "missing:interest_income" in r["reasons"]
    assert r["purification_pct"] is None
    assert "missing:industry" in evaluate(ok(industry=None))["reasons"]

def test_zero_denominator_fails():
    assert "missing:market_cap" in evaluate(ok(market_cap=0.0))["reasons"]
    assert "missing:total_revenue" in evaluate(ok(total_revenue=-5.0))["reasons"]
```

- [ ] **Step 3: Run `pytest tests/test_rules.py -v`** — Expected: FAIL (`ModuleNotFoundError: halal_screener.rules`)

- [ ] **Step 4: Implement `rules.py` per the Interfaces block.** Treat `None`, NaN, empty string, and non-positive denominators (`market_cap`, `avg_market_cap_12m`, `total_revenue`) as missing. Numerators (`total_debt`, `cash_and_st`, `interest_income`) may be 0 but not missing. Collect all applicable reason codes (order: missing, industry, debt, debt_avg, cash, cash_avg, income); a ratio is computed only when its inputs are present.

- [ ] **Step 5: Run `pytest tests/test_rules.py -v`** — Expected: 7 passed

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: AAOIFI rule evaluation"`

---

### Task 2: Data fetching

**Files:**
- Create: `halal_screener/fetch.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: `FIELDS` from `rules.py` (keys of the returned dict).
- Produces:
  - `normalize_symbol(symbol: str) -> str` — `"BRK.B"` → `"BRK-B"`, strips whitespace.
  - `latest(frame: pandas.DataFrame, label: str) -> float | None` — value of row `label` in the most recent column (yfinance statements have dates as columns, newest first); first non-NaN going back in time; `None` if the row is absent or all NaN.
  - `sp500_tickers() -> list[str]` — `requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "halal-screener/0.1"}, timeout=30)`, `pandas.read_html(io.StringIO(text))[0]["Symbol"]`, normalized.
  - `fetch_fundamentals(ticker: str) -> dict` — keys exactly `FIELDS`, missing values `None`. Sources (`t = yfinance.Ticker(ticker)`): `sector`, `industry`, `market_cap` from `t.info` (`sector`, `industry`, `marketCap`); `total_debt` = `latest(t.quarterly_balance_sheet, "Total Debt")`; `cash_and_st` = `latest(t.quarterly_balance_sheet, "Cash Cash Equivalents And Short Term Investments")`; `interest_income` and `total_revenue` = `latest(t.income_stmt, "Interest Income" / "Total Revenue")` (annual); `avg_market_cap_12m` = `t.info["sharesOutstanding"] * t.history(period="1y")["Close"].mean()`, `None` if either is missing.

- [ ] **Step 1: Write failing tests in `tests/test_fetch.py`**

```python
import pandas as pd
from halal_screener.fetch import normalize_symbol, latest

def test_normalize_symbol():
    assert normalize_symbol(" BRK.B ") == "BRK-B"
    assert normalize_symbol("AAPL") == "AAPL"

def frame():
    cols = pd.to_datetime(["2026-09-30", "2026-06-30"])
    return pd.DataFrame({cols[0]: [float("nan"), 5.0], cols[1]: [7.0, 4.0]},
                        index=["Total Debt", "Total Revenue"])

def test_latest_takes_newest_non_nan():
    assert latest(frame(), "Total Revenue") == 5.0
    assert latest(frame(), "Total Debt") == 7.0

def test_latest_missing_label():
    assert latest(frame(), "Interest Income") is None
    assert latest(pd.DataFrame(), "Total Debt") is None
```

- [ ] **Step 2: Run `pytest tests/test_fetch.py -v`** — Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `fetch.py` per the Interfaces block.** Sort columns descending before picking, don't trust yfinance order.

- [ ] **Step 4: Run `pytest tests/test_fetch.py -v`** — Expected: 3 passed

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: S&P 500 list and Yahoo fundamentals fetch"`

---

### Task 3: Snapshot, public API, CLI

**Files:**
- Create: `halal_screener/api.py`, `halal_screener/__main__.py`, `data/allowlist.txt` (header comment only: `# One ticker per line, manually reviewed and approved. Use # for the reason.`)
- Modify: `halal_screener/__init__.py` — `from .api import halal_tickers, screen, refresh`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `evaluate`, `FIELDS` (Task 1); `sp500_tickers`, `fetch_fundamentals` (Task 2).
- Produces:
  - `DATA_DIR: Path` = `Path(__file__).resolve().parent.parent / "data"`
  - `refresh(data_dir: Path = DATA_DIR, limit: int | None = None, tickers: list[str] | None = None, fetch: Callable[[str], dict] = fetch_fundamentals, pause: float = 0.5) -> Path` — tickers default to `sp500_tickers()[:limit]`; per ticker, up to 3 attempts with `pause` seconds sleep between; on final failure the row is all-`None` fields and reasons `error:<ExceptionName>` (passed False). Writes `data_dir/snapshot_<YYYY-MM-DD>.csv` (today, overwrite same day) with columns `ticker` + `FIELDS` + evaluate keys; prints progress `i/N ticker` per row; returns the path.
  - `screen(snapshot: str | None = None, data_dir: Path = DATA_DIR) -> pandas.DataFrame` — `snapshot` is `"YYYY-MM-DD"`; `None` = newest `snapshot_*.csv` by name. No snapshot → `FileNotFoundError("No snapshot in <dir>; run: python -m halal_screener refresh")`.
  - `halal_tickers(snapshot: str | None = None, data_dir: Path = DATA_DIR) -> list[str]` — sorted union of `passed` tickers and allowlist tickers **that appear in the snapshot**. Allowlist: strip text after `#`, strip whitespace, skip blanks, upper-case + `normalize_symbol`. Missing allowlist file = empty.
  - CLI: `python -m halal_screener refresh [--limit N]` → calls `refresh`, prints the path and `<passed>/<total> passed`.

- [ ] **Step 1: Write failing tests in `tests/test_api.py`**

```python
from halal_screener.api import refresh, screen, halal_tickers

GOOD = dict(sector="Technology", industry="Software - Infrastructure", total_debt=1.0,
            cash_and_st=1.0, interest_income=0.0, total_revenue=100.0,
            market_cap=100.0, avg_market_cap_12m=100.0)

def fake_fetch(t):
    if t == "BOOM":
        raise TimeoutError("down")
    return dict(GOOD, industry="Banks - Regional") if t == "BANK" else dict(GOOD)

def snap(tmp_path):
    return refresh(tmp_path, tickers=["AAA", "BANK", "BOOM"], fetch=fake_fetch, pause=0)

def test_refresh_and_halal_tickers(tmp_path):
    path = snap(tmp_path)
    assert path.name.startswith("snapshot_") and path.exists()
    assert halal_tickers(data_dir=tmp_path) == ["AAA"]

def test_refresh_survives_fetch_error(tmp_path):
    snap(tmp_path)
    row = screen(data_dir=tmp_path).set_index("ticker").loc["BOOM"]
    assert not row["passed"] and row["reasons"] == "error:TimeoutError"

def test_allowlist_adds_reviewed_ticker(tmp_path):
    snap(tmp_path)
    (tmp_path / "allowlist.txt").write_text("# reviewed\nbank  # ok per scholar\n")
    assert halal_tickers(data_dir=tmp_path) == ["AAA", "BANK"]

def test_allowlist_ignores_tickers_outside_snapshot(tmp_path):
    snap(tmp_path)
    (tmp_path / "allowlist.txt").write_text("ZZZ\n")
    assert halal_tickers(data_dir=tmp_path) == ["AAA"]

def test_no_snapshot_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError, match="refresh"):
        halal_tickers(data_dir=tmp_path)
```

- [ ] **Step 2: Run `pytest tests/test_api.py -v`** — Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `api.py`, `__main__.py`, update `__init__.py`, create `data/allowlist.txt` per the Interfaces block.** Read `passed` back from CSV as bool (`df["passed"].astype(bool)` after `read_csv` handles `True`/`False` strings).

- [ ] **Step 4: Run `pytest -v`** — Expected: all 15 tests pass

- [ ] **Step 5: Live smoke test (network):** `python -m halal_screener refresh --limit 10` — Expected: a `data/snapshot_<today>.csv` with 10 rows, prints `k/10 passed`; then `python -c "import halal_screener as h; print(h.halal_tickers())"` prints a list. Open the CSV and check a few rows' ratios look sane (e.g. a bank failing as `prohibited:...`). If a Yahoo label in Task 2 doesn't match, fix the label, not the rules.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: snapshot refresh, halal_tickers API and CLI"`

---

### Task 4: Wahed / FTSE standard + compare (spec addendum)

**Files:** Modify `halal_screener/rules.py`, `fetch.py`, `api.py`, `__main__.py`, `__init__.py`; tests in `tests/test_rules.py`, `tests/test_fetch.py`, `tests/test_api.py`.

**Interfaces:**
- `FIELDS` gains `"total_assets"`, `"receivables"`. `STANDARDS = ("aaoifi", "wahed")`.
- `evaluate(row: dict, standard: str = "aaoifi") -> dict` — same keys as before plus `"receivables_cash_ratio"`; unknown standard → `ValueError`. Missing-data check covers only that standard's fields (AAOIFI: debt, cash, interest, revenue, market_cap, avg_market_cap_12m; Wahed: debt, cash, interest, revenue, total_assets, receivables). Wahed codes: `debt`, `cash`, `receivables_cash`, `income`. Limits: `WAHED_MAX_DEBT = WAHED_MAX_CASH = 0.33333` (strict), `WAHED_MAX_RECEIVABLES_CASH = 0.50` (strict), `WAHED_MAX_INCOME = 0.05` (ratio ≤ passes). Wahed ratios use total assets; `debt_ratio_avg`/`cash_ratio_avg` are None for Wahed.
- `latest_set(frame, labels, optional=()) -> dict` — `optional` labels are read from the column chosen for `labels`, None if NaN there.
- Snapshot columns: `ticker`, `FIELDS`, then every evaluate key suffixed `_aaoifi` and `_wahed`.
- `halal_tickers(snapshot=None, data_dir=DATA_DIR, standard="aaoifi")`; `compare(snapshot=None, data_dir=DATA_DIR) -> dict[str, list[str]]` built from `halal_tickers` of each standard (allowlist applies to both).
- CLI: `python -m halal_screener compare` prints the three lists with counts; `refresh` prints pass counts per standard.

**Tests (RED first):** `test_wahed_clean_passes`, `test_wahed_uses_total_assets_not_market_cap`, `test_wahed_limits` (debt 33.333% fails, receivables+cash 50% fails, income exactly 5% passes), `test_each_standard_requires_only_its_fields`, `test_unknown_standard_raises`, `test_latest_set_optional_same_column`, `test_snapshot_has_both_standards`, `test_compare_splits_lists`.

**Review Focus additions:** (6) a field only one standard uses is missing → only that standard fails; (7) Wahed income exactly 5% → passes (≤), other Wahed limits at the boundary → fail.
