"""Refresh a dated snapshot of the screen; read halal tickers back from it."""
import time
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from .fetch import fetch_fundamentals, normalize_symbol, sp500_tickers
from .rules import FIELDS, STANDARDS, evaluate

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _fetch_with_retry(ticker, fetch, pause, attempts=3):
    for i in range(attempts):
        try:
            return fetch(ticker)
        except Exception as e:  # any failure on one ticker must not stop the whole refresh
            if i == attempts - 1:
                return e
            time.sleep(pause)


def refresh(data_dir: Path = DATA_DIR, limit: int | None = None,
            tickers: list[str] | None = None,
            fetch: Callable[[str], dict] = fetch_fundamentals, pause: float = 0.5,
            max_error_share: float = 0.05) -> Path:
    """Write today's snapshot. A --limit run, or one where more than `max_error_share`
    of tickers errored (e.g. Yahoo rate-limiting), is saved as *_partial.csv and is never
    served as the latest snapshot."""
    data_dir = Path(data_dir)
    tickers = tickers if tickers is not None else sp500_tickers()[:limit]
    rows = []
    for i, t in enumerate(tickers, 1):
        print(f"{i}/{len(tickers)} {t}", flush=True)
        data = _fetch_with_retry(t, fetch, pause)
        failed = isinstance(data, Exception)
        row = {"ticker": t, **dict.fromkeys(FIELDS)}
        if not failed:
            row.update({f: data.get(f) for f in FIELDS})
        for std in STANDARDS:
            result = evaluate(data if not failed else {}, std)
            if failed:
                result = {**dict.fromkeys(result), "passed": False, "reasons": f"error:{type(data).__name__}"}
            row.update({f"{k}_{std}": v for k, v in result.items()})
        rows.append(row)
        time.sleep(pause)
    data_dir.mkdir(parents=True, exist_ok=True)
    errors = sum(str(r["reasons_aaoifi"]).startswith("error:") for r in rows)
    partial = limit is not None or errors > max_error_share * len(rows)
    path = data_dir / f"snapshot_{date.today().isoformat()}{'_partial' if partial else ''}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def screen(snapshot: str | None = None, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if snapshot:
        path = data_dir / f"snapshot_{snapshot}.csv"
    else:
        found = sorted(data_dir.glob("snapshot_????-??-??.csv"))  # full runs only, never *_partial
        path = found[-1] if found else None
    if path is None or not path.exists():
        raise FileNotFoundError(f"No snapshot in {data_dir}; run: python -m halal_screener refresh")
    df = pd.read_csv(path, dtype={"ticker": str}, keep_default_na=False, na_values=[""])
    for std in STANDARDS:
        df[f"passed_{std}"] = df[f"passed_{std}"].astype(str).str.lower().eq("true")
        df[f"reasons_{std}"] = df[f"reasons_{std}"].fillna("")
    return df


def _allowlist(data_dir: Path) -> set[str]:
    path = Path(data_dir) / "allowlist.txt"
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        sym = line.split("#", 1)[0].strip()
        if sym:
            out.add(normalize_symbol(sym.upper()))
    return out


def halal_tickers(snapshot: str | None = None, data_dir: Path = DATA_DIR,
                  standard: str = "aaoifi") -> list[str]:
    if standard not in STANDARDS:
        raise ValueError(f"standard must be one of {STANDARDS}, got {standard!r}")
    df = screen(snapshot, data_dir)
    passed = set(df.loc[df[f"passed_{standard}"], "ticker"])
    in_universe = _allowlist(data_dir) & set(df["ticker"])  # allowlist can't add stocks outside the S&P 500
    return sorted(passed | in_universe)


def compare(snapshot: str | None = None, data_dir: Path = DATA_DIR) -> dict[str, list[str]]:
    a = set(halal_tickers(snapshot, data_dir, "aaoifi"))
    w = set(halal_tickers(snapshot, data_dir, "wahed"))
    return {"both": sorted(a & w), "aaoifi_only": sorted(a - w), "wahed_only": sorted(w - a)}
