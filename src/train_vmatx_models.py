#!/usr/bin/env python3
"""
train_vmatx_models.py — fit the four VMATX-vs-FTMA models and save them.

  python src/train_vmatx_models.py                 # fetch distributions with yfinance, fit, save
  python src/train_vmatx_models.py --no-fetch      # reuse data/vmatx_ftma_distributions.csv

Reads   data/jointVMATX-FTMA_daily.csv           (written by fetch-VMATX_yf.py)
Writes  data/vmatx_ftma_distributions.csv        (date, ticker, amount; refreshed from Yahoo)
        data/models/vmatx_models.json            (coefficients + fit diagnostics)

See vmatx_models.py for the model definitions.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vmatx_models as vm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
JOINT = DATA_DIR / "jointVMATX-FTMA_daily.csv"
DIVS = DATA_DIR / "vmatx_ftma_distributions.csv"
MODELS = DATA_DIR / "models" / "vmatx_models.json"


def fetch_distributions(start: str) -> pd.DataFrame:
    """Distribution history for FTMA and VMATX from Yahoo via yfinance.
    FTMA dates are ex-dates; Yahoo records VMATX's on its month-end pay date."""
    import yfinance as yf
    rows = []
    for t in ("FTMA", "VMATX"):
        s = yf.Ticker(t).dividends
        if s is None or s.empty:
            raise RuntimeError(f"yfinance returned no distributions for {t}")
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        s.index = idx.normalize()
        s = s[s.index >= pd.Timestamp(start) - pd.Timedelta(days=40)]
        rows += [(d.date().isoformat(), t, float(a)) for d, a in s.items()]
    return pd.DataFrame(rows, columns=["date", "ticker", "amount"]).sort_values(["ticker", "date"])


def load_inputs(no_fetch: bool):
    joint = pd.read_csv(JOINT, parse_dates=["date"]).set_index("date").sort_index()
    if no_fetch:
        divs = pd.read_csv(DIVS)
    else:
        divs = fetch_distributions(str(joint.index[0].date()))
        DIVS.parent.mkdir(parents=True, exist_ok=True)
        divs.to_csv(DIVS, index=False)
    divs["date"] = pd.to_datetime(divs["date"])
    fdiv = divs[divs.ticker == "FTMA"].set_index("date")["amount"].groupby(level=0).sum()
    vdiv = divs[divs.ticker == "VMATX"].set_index("date")["amount"].groupby(level=0).sum()
    return joint, fdiv, vdiv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fetch", action="store_true", help="use the saved distributions CSV")
    ap.add_argument("--assume-vmatx-div", type=float, default=None,
                    help="amount for a VMATX month-end distribution not yet posted (default: mean of last 3)")
    ap.add_argument("--out", default=str(MODELS))
    a = ap.parse_args(argv)

    joint, fdiv, vdiv = load_inputs(a.no_fetch)
    models = vm.train(joint, fdiv, vdiv, a.assume_vmatx_div)
    models["trained_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(models, indent=1))

    print(f"Trained on {models['sample_start']} .. {models['sample_end']} -> {out}")
    for n in models["notes"]:
        print("NOTE:", n)
    hdr = f"{'basis':10s}{'a':>8s}{'b':>7s}{'EG p':>7s}{'h':>7s}{'g':>8s}{'g p':>6s}{'R2 ecm':>8s}{'se%':>7s}{'n':>5s}"
    print(hdr)
    for k in ("raw", "tr"):
        m = models[k]
        print(f"{k:10s}{m['a']:8.4f}{m['b']:7.3f}{m['eg_p']:7.2f}{m['h']:7.3f}{m['g']:8.4f}"
              f"{m['g_p']:6.2f}{m['R2_ecm']:8.3f}{m['se']*100:7.3f}{m['n']:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
