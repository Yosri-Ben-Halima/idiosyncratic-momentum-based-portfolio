import argparse

from .api import compare, refresh, screen
from .rules import STANDARDS

parser = argparse.ArgumentParser(prog="python -m halal_screener")
sub = parser.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("refresh", help="fetch S&P 500 data and write a new snapshot")
p.add_argument("--limit", type=int, help="only the first N tickers (for a quick test)")
c = sub.add_parser("compare", help="AAOIFI vs Wahed/FTSE passing tickers")
c.add_argument("--snapshot", help="YYYY-MM-DD (default: latest full snapshot)")
args = parser.parse_args()

if args.cmd == "refresh":
    path = refresh(limit=args.limit)
    df = screen(path.stem.removeprefix("snapshot_"), path.parent)
    errors = int(df["reasons_aaoifi"].str.startswith("error:").sum())
    print(path)
    for std in STANDARDS:
        print(f"{std}: {int(df[f'passed_{std}'].sum())}/{len(df)} passed")
    print(f"{errors} fetch errors")
    if path.stem.endswith("_partial"):
        print("PARTIAL snapshot (--limit or too many errors): halal_tickers() will not use it. Rerun refresh.")
else:
    for name, tickers in compare(args.snapshot).items():
        print(f"{name} ({len(tickers)}): {' '.join(tickers)}")
