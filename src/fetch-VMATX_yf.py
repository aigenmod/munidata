#!/usr/bin/env python3
"""
fetch_yf.py (v1) — pull FTMA / VMATX daily EOD from yfinance and top off
an existing CSV table without re-downloading or disturbing history.

  pip install yfinance pandas

  python src/fetch-VMATX_yf.py                # full history -> data/FTMA.csv, data/VMATX.csv, data/jointVMATX-FTMA_daily.csv
  python src/fetch-VMATX_yf.py --topoff FTMA=FTMA_HistoricalData.csv \
                     --topoff VMATX=VMATX_2025-2026_-_Sheet1.csv
  python src/fetch-VMATX_yf.py --start 2026-09-01 --end 2026-09-26 --print

Run from the munidata repo root; output defaults to ./data regardless of cwd
(anchored to this script's location), override with --outdir.

Notes that actually bite:
  * auto_adjust defaults to True in modern yfinance and back-adjusts closes for
    distributions. A muni fund distributes monthly, so an adjusted series will
    NOT line up with the NAV table you collected by hand. This uses
    auto_adjust=False everywhere.
  * VMATX is a mutual fund: Yahoo reports one NAV strike per day and echoes it
    into Open/High/Low, with Volume 0. That is expected, not a bad pull.
  * yf.download() with >1 ticker returns MultiIndex columns. This fetches one
    ticker at a time to keep the frame shape boring.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from zoneinfo import ZoneInfo
except ImportError:                                    # py<3.9
    ZoneInfo = None

yf.config.debug.hide_exceptions = False  # let fetch failures raise, don't just log

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

ET = "America/New_York"
# US equity close is 16:00 ET; give the consolidated tape time to settle.
SESSION_SETTLED_HOUR_ET = 16
SESSION_SETTLED_MINUTE_ET = 15

TICKERS = ["FTMA", "VMATX"]
INCEPTION = {"FTMA": "2025-11-10", "VMATX": "1998-12-09"}

COLS = ["date", "open", "high", "low", "close", "volume"]


# ------------------------------------------------- partial-bar protection
def drop_unsettled_bar(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Remove today's bar when the session has not finished.

    Why this exists: yfinance happily returns an in-progress bar for the
    current session. Its close is simply the last trade so far, and its
    volume is a partial count. Committing that to the repo writes a number
    that is NOT the closing price into a column every downstream model
    reads as one. It self-heals on the next full re-download, but any
    analysis running in the same job consumes the bad value first.

    Detected on time, not on the data, because a thin ETF can legitimately
    post a tiny real volume. The round-lot check below is advisory only.
    """
    if df.empty or ZoneInfo is None:
        return df

    now_et = datetime.now(ZoneInfo(ET))
    cutoff = now_et.replace(hour=SESSION_SETTLED_HOUR_ET,
                            minute=SESSION_SETTLED_MINUTE_ET,
                            second=0, microsecond=0)
    today = now_et.date()

    last = df["date"].iloc[-1].date()
    if last == today and now_et < cutoff:
        vol = df["volume"].iloc[-1]
        px = df["close"].iloc[-1]
        print(f"  ! {ticker}: dropping {last} — session still open "
              f"({now_et:%H:%M} ET, settles {SESSION_SETTLED_HOUR_ET}:"
              f"{SESSION_SETTLED_MINUTE_ET:02d}). "
              f"Partial bar was close={px:.4f} volume={vol:.0f}",
              file=sys.stderr)
        return df.iloc[:-1].reset_index(drop=True)

    # Advisory: settled Yahoo EOD volumes are reported in round lots. A
    # non-round figure on the newest bar is a hint it is still live.
    if last == today and len(df) and df["volume"].iloc[-1] % 100 != 0:
        print(f"  ~ {ticker}: {last} volume {df['volume'].iloc[-1]:.0f} is not a "
              f"round lot; bar may still be settling", file=sys.stderr)
    return df


# ---------------------------------------------------------------- fetch
def load(ticker: str, start: str | date, end: str | date | None = None,
         retries: int = 3) -> pd.DataFrame:
    """Daily EOD bars for one ticker as date/open/high/low/close/volume.

    `end` is inclusive here (yfinance's own end is exclusive; we add a day).
    """
    start = str(start)
    end = str(end or date.today())
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()

    last_err = None
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(
                start=start,
                end=end_exclusive,
                interval="1d",
                auto_adjust=False,   # keep raw closes / NAV
                actions=False,
            )
            break
        except Exception as e:                      # network, rate limit, curl
            last_err = e
            if attempt == retries - 1:
                raise RuntimeError(
                    f"{ticker}: yfinance failed after {retries} tries: {e}") from e
            import time
            time.sleep(2 ** attempt)
    else:                                            # pragma: no cover
        raise RuntimeError(str(last_err))

    if hist is None or hist.empty:
        return pd.DataFrame(columns=COLS)

    df = hist.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # index column is 'date' or 'datetime' depending on version/interval
    dcol = "date" if "date" in df.columns else "datetime"
    out = pd.DataFrame({
        "date": pd.to_datetime(df[dcol]).dt.tz_localize(None).dt.normalize(),
        "open": pd.to_numeric(df.get("open"), errors="coerce"),
        "high": pd.to_numeric(df.get("high"), errors="coerce"),
        "low": pd.to_numeric(df.get("low"), errors="coerce"),
        "close": pd.to_numeric(df.get("close"), errors="coerce"),
        "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
    })
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    out = out.drop_duplicates(subset="date", keep="last").sort_values("date")
    out = out.reset_index(drop=True)
    return drop_unsettled_bar(out, ticker)


# ------------------------------------------------------- existing tables
_DATE_KEYS = ["date", "trade date", "nav date"]
_CLOSE_KEYS = ["close/last", "close", "nav", "adj close", "price", "last"]


def _pick(cols, candidates):
    low = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for cand in candidates:
        for lc, orig in low.items():
            if cand in lc:
                return orig
    return None


def read_existing(path: Path) -> pd.DataFrame:
    """Read a vendor CSV in whatever shape it was exported.

    Handles Nasdaq ('Close/Last', '$9.06', newest-first), Yahoo/Stooq OHLCV,
    and plain Date,NAV sheets. Dates may be mm/dd/yyyy or ISO.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    dcol = _pick(raw.columns, _DATE_KEYS)
    ccol = _pick(raw.columns, _CLOSE_KEYS)
    if not dcol or not ccol:
        raise ValueError(f"{path.name}: need a date and a close/NAV column; "
                         f"saw {list(raw.columns)}")

    def num(s):
        s = str(s).strip().replace("$", "").replace(",", "")
        return pd.to_numeric(s, errors="coerce")

    out = pd.DataFrame({
        "date": pd.to_datetime(raw[dcol], errors="coerce", format="mixed"),
        "open": raw[_pick(raw.columns, ["open"])].map(num) if _pick(raw.columns, ["open"]) else pd.NA,
        "high": raw[_pick(raw.columns, ["high"])].map(num) if _pick(raw.columns, ["high"]) else pd.NA,
        "low": raw[_pick(raw.columns, ["low"])].map(num) if _pick(raw.columns, ["low"]) else pd.NA,
        "close": raw[ccol].map(num),
        "volume": raw[_pick(raw.columns, ["volume"])].map(num) if _pick(raw.columns, ["volume"]) else pd.NA,
    })
    out = out.dropna(subset=["date", "close"])
    return out.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)


def topoff(existing: pd.DataFrame, ticker: str, through: str | date | None = None,
           max_move: float = 0.05) -> tuple[pd.DataFrame, int]:
    """Fetch only sessions after the last stored date and append them.

    Existing rows are never modified. Returns (combined, n_added).
    """
    if existing.empty:
        fetched = load(ticker, INCEPTION.get(ticker, "2000-01-01"), through)
        return fetched, len(fetched)

    last = pd.Timestamp(existing["date"].max())
    start = (last + pd.Timedelta(days=1)).date()
    end = pd.Timestamp(through or date.today()).date()
    if start > end:
        return existing, 0

    new = load(ticker, start, end)
    new = new[new["date"] > last]
    if new.empty:
        return existing, 0

    # reject implausible ticks against the last stored close
    prev = float(existing.loc[existing["date"].idxmax(), "close"])
    keep = []
    for _, r in new.iterrows():
        move = abs(r["close"] / prev - 1.0)
        if move > max_move:
            print(f"  ! {ticker} {r['date'].date()}: {move*100:.1f}% move vs "
                  f"{prev:.4f} — skipped as a suspect tick", file=sys.stderr)
            continue
        keep.append(r)
        prev = float(r["close"])
    new = pd.DataFrame(keep, columns=COLS) if keep else pd.DataFrame(columns=COLS)

    combined = (pd.concat([existing, new], ignore_index=True)
                  .drop_duplicates(subset="date", keep="first")   # existing wins
                  .sort_values("date").reset_index(drop=True))
    return combined, len(new)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Load FTMA/VMATX EOD from yfinance.")
    p.add_argument("--tickers", nargs="*", default=TICKERS)
    p.add_argument("--start", default=None, help="default: ticker inception")
    p.add_argument("--end", default=None, help="inclusive; default today")
    p.add_argument("--topoff", action="append", default=[], metavar="TICKER=CSV",
                   help="append only new sessions onto this existing table")
    p.add_argument("--outdir", type=Path, default=DATA_DIR)
    p.add_argument("--print", dest="show", action="store_true")
    args = p.parse_args(argv)

    seeds = {}
    for spec in args.topoff:
        t, path = spec.split("=", 1)
        seeds[t.upper()] = Path(path).expanduser()

    args.outdir.mkdir(parents=True, exist_ok=True)
    frames = {}

    for t in args.tickers:
        t = t.upper()
        if t in seeds:
            existing = read_existing(seeds[t])
            df, added = topoff(existing, t, args.end)
            print(f"{t:6s} {len(existing)} existing + {added} new = {len(df)} rows")
        else:
            df = load(t, args.start or INCEPTION.get(t, "2000-01-01"), args.end)
            print(f"{t:6s} {len(df)} rows")

        if df.empty:
            continue
        span = f"{df['date'].min().date()} → {df['date'].max().date()}"
        print(f"       {span}   last close {df['close'].iloc[-1]:.4f}")
        df.to_csv(args.outdir / f"{t}.csv", index=False)
        frames[t] = df[["date", "close"]].rename(columns={"close": f"{t.lower()}_close"})
        if args.show:
            print(df.tail(5).to_string(index=False), "\n")

    if len(frames) >= 2:
        merged = None
        for f in frames.values():
            merged = f if merged is None else merged.merge(f, on="date", how="inner")
        merged.to_csv(args.outdir / "jointVMATX-FTMA_daily.csv", index=False)
        print(f"\nmerged {len(merged)} matched sessions -> {args.outdir/'jointVMATX-FTMA_daily.csv'}")
        if {"ftma_close", "vmatx_close"} <= set(merged.columns) and len(merged) > 3:
            r = merged["ftma_close"].pct_change().corr(merged["vmatx_close"].pct_change())
            print(f"daily-return correlation: {r:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
