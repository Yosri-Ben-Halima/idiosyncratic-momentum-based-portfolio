# Halal S&P 500 Screener — Design

## Intent
A Python module that returns the S&P 500 tickers that are Shariah-compliant under **AAOIFI Standard 21** (the standard the user's scholar follows), for use in the user's own portfolio code. A wrong "pass" is the costly error: when in doubt, a stock is excluded.

## Goals
- `halal_tickers()` returns a list of compliant tickers, fast and reproducible.
- Monthly manual refresh using free data (Yahoo Finance via `yfinance`).
- Every stock's ratios and fail reasons saved, so manual review is possible.
- Purification % per passing stock (user's call to keep — proposed by Claude).

## Non-goals
- No UI, dashboard, or scheduler. No other standards (S&P, MSCI). No paid data.
- No investment advice; the screen applies rules, the user and their scholar own the ruling.

## Anti-goals
- A doubtful stock silently passing (missing data, grey sector, unknown revenue mix).

## Rules (AAOIFI)
1. **Business activity** — fail if Yahoo `sector`/`industry` is in a prohibited list: conventional banks, insurance, capital markets/credit services, alcohol (brewers, wineries & distillers), tobacco, gambling/casinos, adult entertainment. **Grey areas fail** (user decision): weapons/aerospace & defense, hotels/resorts, entertainment/media, restaurants. Lists live in one editable config dict.
2. **Interest-bearing debt / market cap < 30%** — debt = Yahoo `Total Debt`.
3. **Cash + interest-bearing securities / market cap < 30%** — `Cash, Cash Equivalents & Short Term Investments`.
4. **Non-permissible income / revenue < 5%** — proxy: `Interest Income / Total Revenue`. Hidden revenue (e.g. alcohol sales inside a retailer) is not visible in free data; that is why rule 1 is strict and the allowlist exists.
- **Market cap = strictest of both** (user decision): rules 2–3 must pass with current market cap AND with 12-month average market cap (approximated as current shares × 12-month average price).
- **Missing data = fail**, with reason `missing:<field>`.

## Purification
For passing stocks: `purification_pct = interest_income / total_revenue` — share of each dividend to give to charity.

## Approaches considered
- **A. Fetch on every call** — simplest, but minutes per call, fragile, not reproducible. Rejected.
- **B. Refresh-to-snapshot, read from snapshot** — **chosen** by user, refresh monthly.
- **C. Start from S&P 500 Shariah index list** — uses S&P's rules, wrongly drops AAOIFI-compliant stocks. Dropped.

## How it works
- `python -m halal_screener refresh` — gets S&P 500 constituents (Wikipedia table), fetches each ticker from Yahoo with retry/pause, applies rules, writes `data/snapshot_YYYY-MM-DD.csv` (one row per ticker: ratios, pass/fail, reasons, purification %). Old snapshots are kept.
- `halal_tickers(snapshot=None)` — reads the latest snapshot (or a given date), returns passing tickers plus tickers in `allowlist.txt` (stocks the user manually reviewed and approved; one ticker per line, `#` comments for the reason).
- `screen(snapshot=None)` — returns the full snapshot DataFrame for inspection.

## Left to the builder
- Exact Yahoo field names and fallbacks; ticker format (`BRK.B` → `BRK-B`).
- Throttle/retry values; refresh progress output.
- Tests: the rule function tested on hand-made rows (pass, each fail case, missing data, grey sector, allowlist override). No network in tests.

## Addendum (2026-11-20): second standard — Wahed / FTSE Shariah
User decision: keep AAOIFI and add the standard Wahed's HLAL ETF tracks (FTSE Shariah USA Index, screened by Yasaar), to compare the two lists of passing tickers.

**Source:** FTSE Yasaar Global Equity Shariah Index Series Ground Rules v4.6, February 2026, §4.3.2 (verified from the PDF). Assumption: the FTSE Shariah USA Index follows these rules (Wahed's page says Yasaar screens it).

**Wahed rules** (all against total assets; market cap not used):
1. Debt < 33.333% of total assets.
2. Cash and interest-bearing items < 33.333% of total assets.
3. Accounts receivable + cash < 50% of total assets.
4. Interest + non-compliant income **≤** 5% of revenue ("should not exceed").

**Shared with AAOIFI:** business-activity screen (incl. whole Financial Services sector), grey areas fail, missing data fails (only for the fields a standard uses), allowlist, purification %, monthly snapshot.

**Not implemented (free-data / scope limits):** FTSE's 5%-of-revenue tolerance for prohibited business lines (we fail on industry label, stricter); FTSE's two-consecutive-quarter ±5% hysteresis band.

**Proxies (free data):** "cash and interest-bearing items" = Yahoo `Cash Cash Equivalents And Short Term Investments` (long-term interest-bearing securities not counted, so the cash ratio can be understated); "interest + non-compliant income" = interest income only.

**Interface:** `evaluate(row, standard="aaoifi" | "wahed")`; snapshot stores `<key>_aaoifi` and `<key>_wahed` for every evaluate key; `halal_tickers(standard="aaoifi")`; `compare(snapshot=None) -> {"both", "aaoifi_only", "wahed_only"}` (sorted lists); CLI `python -m halal_screener compare`. New fetched fields: `total_assets` ("Total Assets"), `receivables` ("Accounts Receivable", else "Receivables"), read from the same balance-sheet period as debt and cash.
