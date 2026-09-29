#!/usr/bin/env python3
"""
fetch_DFF.py (v2) — daily Federal Funds Rate, target range, and any other
FRED series straight from FRED's no-API-key CSV endpoint.

  python src/fetch_DFF.py                              # DFF+DFEDTAR+DFEDTARU+DFEDTARL since FTMA inception -> data/
  python src/fetch_DFF.py --series DFF DGS10 DGS30 --start 2025-11-10
  python src/fetch_DFF.py --series DFF --start 2020-01-01 --outdir .
  python src/fetch_DFF.py --align data/FTMA.csv         # inner-join onto trading days

Run from the munidata repo root; output defaults to ./data regardless of cwd
(anchored to this script's location), override with --outdir.

Needs only pandas + stdlib. No API key: FRED serves
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>&cosd=<start>&coed=<end>

Series worth knowing
--------------------
  DFF        Federal Funds Effective Rate, daily 7-DAY  (weekend rows repeat Friday)
  EFFR       Effective Federal Funds Rate, BUSINESS DAYS only
  DFEDTARU   Fed funds target range, upper limit (daily)
  DFEDTARL   Fed funds target range, lower limit (daily)
  FEDFUNDS   Fed funds effective, MONTHLY average
  SOFR       Secured Overnight Financing Rate, business days
  DGS2/5/10/30   Treasury constant maturity yields, business days

Missing values arrive as "." (holidays in business-day series). Those are
dropped by default; --ffill carries the prior value forward instead.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

NOTES = {
    "DFF": "Fed funds effective, daily 7-day (weekends carry Friday's value)",
    "EFFR": "Fed funds effective, business days only",
    "DFEDTARU": "Fed funds target range, upper limit",
    "DFEDTARL": "Fed funds target range, lower limit",
    "FEDFUNDS": "Fed funds effective, MONTHLY average — not daily",
    "SOFR": "Secured Overnight Financing Rate",
    "DGS2": "2Y Treasury constant maturity",
    "DGS5": "5Y Treasury constant maturity",
    "DGS10": "10Y Treasury constant maturity",
    "DGS30": "30Y Treasury constant maturity",
}

# Why a long pull comes back short. These are regime facts, not data errors:
# the instrument or the reporting convention did not exist before these dates.
SERIES_BEGINS = {
    "DFEDTARU": "target RANGE begins 2008-12-16; before that the FOMC set a "
                "single point target — use DFEDTAR for the pre-2008 era",
    "DFEDTARL": "target RANGE begins 2008-12-16; see DFEDTAR for pre-2008",
    "DFEDTAR":  "single-point target, DISCONTINUED 2008-12-15 when the range began",
    "IORB":     "interest on reserve balances begins 2021-07-29; IOER covers "
                "2008-10 to 2021-07; no interest was paid on reserves before 2008",
    "IOER":     "superseded by IORB on 2021-07-29",
    "SOFR":     "published from 2018-04-03; nothing comparable exists earlier",
    "EFFR":     "NY Fed transaction-based series; DFF reaches much further back",
}


def fetch_series(series: str, start: str, end: str | None = None,
                 timeout: int = 30) -> pd.DataFrame:
    """One FRED series as a two-column frame: date, <series>."""
    end = end or date.today().isoformat()
    url = f"{FRED_CSV}?id={series}&cosd={start}&coed={end}"
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_fred/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{series}: FRED returned HTTP {e.code} "
                           f"(check the series ID at fred.stlouisfed.org/series/{series})") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"{series}: could not reach FRED: {e}") from e

    df = pd.read_csv(io.StringIO(body), dtype=str)
    if df.shape[1] < 2:
        raise RuntimeError(f"{series}: unexpected response shape {df.shape}; "
                           f"first bytes: {body[:120]!r}")

    # FRED's date column has been 'DATE' and 'observation_date' at different times
    dcol = df.columns[0]
    vcol = df.columns[1]

    out = pd.DataFrame({
        "date": pd.to_datetime(df[dcol], errors="coerce"),
        series: pd.to_numeric(df[vcol].replace(".", pd.NA), errors="coerce"),
    })
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def describe(df: pd.DataFrame, series: str, requested_start: str | None = None) -> None:
    s = df[series]
    have = s.notna()
    first = df.loc[have, "date"].min()
    print(f"{series:9s} {have.sum():5d} obs  "
          f"{first.date()} → {df.loc[have, 'date'].max().date()}"
          f"   last {s[have].iloc[-1]:.4f}")
    if NOTES.get(series):
        print(f"          {NOTES[series]}")

    # A series that simply did not exist yet is the #1 surprise on long pulls.
    if requested_start:
        req = pd.Timestamp(requested_start)
        if first > req + pd.Timedelta(days=7):
            short = (first - req).days
            print(f"          ** starts {first.date()}, {short} days after your "
                  f"--start {req.date()} — series did not exist earlier")
            if series in SERIES_BEGINS:
                print(f"          ** {SERIES_BEGINS[series]}")
    if (~have).any():
        print(f"          {(~have).sum()} missing rows (FRED '.', typically holidays)")

    # 7-day vs business-day is checkable, not a matter of belief
    weekend = df.loc[have, "date"].dt.weekday >= 5
    if weekend.any():
        print(f"          {weekend.sum()} weekend observations present → 7-day series")
    else:
        print(f"          no weekend observations → business-day series")

    # A policy rate barely moves day to day; say so with the actual number.
    chg = s[have].diff().abs()
    moved = (chg > 1e-9).sum()
    if len(chg) > 1:
        print(f"          changed on {moved}/{len(chg)-1} day-pairs "
              f"({moved/(len(chg)-1)*100:.1f}%)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download daily FRED series (default: DFF, DFEDTAR, DFEDTARU, DFEDTARL).")
    p.add_argument("--series", nargs="+", default=["DFF", "DFEDTAR", "DFEDTARU", "DFEDTARL"])
    p.add_argument("--start", default="2025-11-10", help="default: FTMA inception")
    p.add_argument("--years", type=float, default=None,
                   help="shorthand: N years back from today, overrides --start")
    p.add_argument("--end", default=None, help="inclusive; default today")
    p.add_argument("--outdir", type=Path, default=DATA_DIR)
    p.add_argument("--ffill", action="store_true",
                   help="carry the prior value over FRED '.' gaps instead of dropping")
    p.add_argument("--align", metavar="CSV",
                   help="inner-join onto the trading dates in this file's date column")
    p.add_argument("--out", default=None, help="output filename (default fred_<series>.csv)")
    args = p.parse_args(argv)

    if args.years:
        args.start = (pd.Timestamp.today()
                      - pd.Timedelta(days=round(args.years * 365.25))).date().isoformat()
        print(f"--years {args.years:g} -> start {args.start}\n")

    merged = None
    for s in args.series:
        s = s.upper()
        try:
            df = fetch_series(s, args.start, args.end)
        except RuntimeError as e:
            print(f"  ! {e}", file=sys.stderr)
            continue
        describe(df, s, args.start)
        if args.ffill:
            df[s] = df[s].ffill()
        else:
            df = df.dropna(subset=[s])
        merged = df if merged is None else merged.merge(df, on="date", how="outer")
        print()

    if merged is None or merged.empty:
        print("nothing downloaded", file=sys.stderr)
        return 1
    merged = merged.sort_values("date").reset_index(drop=True)

    if args.align:
        ref = pd.read_csv(args.align, dtype=str)
        dcol = next((c for c in ref.columns if c.strip().lower() == "date"), ref.columns[0])
        days = pd.DataFrame({"date": pd.to_datetime(ref[dcol], errors="coerce")}).dropna()
        days["date"] = days["date"].dt.normalize()
        before = len(merged)
        merged = days.drop_duplicates().merge(merged, on="date", how="left")
        print(f"aligned to {len(merged)} trading days from {args.align} "
              f"(was {before} calendar rows)")
        gaps = merged[args.series[0].upper()].isna().sum() if args.series else 0
        if gaps:
            print(f"  {gaps} trading days with no rate observation — "
                  f"rerun with --ffill if you need them filled")

    args.outdir.mkdir(parents=True, exist_ok=True)
    name = args.out or f"fred_{'_'.join(s.upper() for s in args.series)}.csv"
    path = args.outdir / name
    merged.to_csv(path, index=False)
    print(f"\n{len(merged)} rows -> {path}")
    print(merged.tail(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
