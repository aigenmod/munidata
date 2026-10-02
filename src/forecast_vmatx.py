#!/usr/bin/env python3
"""
forecast_vmatx.py — forecast today's VMATX NAV from FTMA's intraday move.

  python src/forecast_vmatx.py                     # FTMA 5-min bars from yfinance, 09:30 ET to now
  python src/forecast_vmatx.py --morning-end 11:00 # cap the averaging window
  python src/forecast_vmatx.py --ftma-now 8.59     # supply the FTMA input yourself
  python src/forecast_vmatx.py --no-chart

Reads   data/jointVMATX-FTMA_daily.csv, data/vmatx_ftma_distributions.csv,
        data/models/vmatx_models.json            (run train_vmatx_models.py first)
Writes  data/forecasts/vmatx_forecasts.csv       one row per target date (re-runs replace that
                                                 date's row); `actual` and errors are filled in
                                                 for past rows once the NAV is in the joint CSV
        data/forecasts/latest.json
        data/forecasts/vmatx_forecast_<date>.png  (one file per day; never overwritten
                                                  by a later day)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vmatx_models as vm  # noqa: E402
from train_vmatx_models import DATA_DIR, JOINT, MODELS, load_inputs  # noqa: E402

ET = "America/New_York"
FC_DIR = DATA_DIR / "forecasts"
LOG = FC_DIR / "vmatx_forecasts.csv"
VMATX_CSV = DATA_DIR / "VMATX.csv"


def ftma_intraday_mean(start="09:30", end=None):
    """Mean of today's FTMA 5-minute closes (regular session) from yfinance."""
    import yfinance as yf
    bars = yf.Ticker("FTMA").history(period="1d", interval="5m", prepost=False, auto_adjust=False)
    if bars.empty:
        raise RuntimeError("no FTMA intraday bars from yfinance (market closed?)")
    idx = bars.index.tz_convert(ET) if bars.index.tz is not None else bars.index.tz_localize(ET)
    bars.index = idx
    s = bars["Close"].dropna()
    s = s[s.index.normalize() == s.index[-1].normalize()].between_time(start, end or "23:59")
    vol = int(bars.loc[s.index, "Volume"].sum())
    return float(s.mean()), s.index[0], s.index[-1], len(s), vol


def update_log(row: dict, joint: pd.DataFrame) -> pd.DataFrame:
    """Append/replace today's row, then score any past rows whose actual NAV is now known."""
    log = pd.read_csv(LOG, dtype={"target_date": str}) if LOG.exists() else pd.DataFrame()
    if not log.empty:
        log = log[log.target_date != row["target_date"]]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True).sort_values("target_date")
    actual = joint.vmatx_close.copy()
    actual.index = actual.index.strftime("%Y-%m-%d")
    log["actual"] = log.target_date.map(actual)
    for k in vm.MODEL_KEYS:
        log[f"err_{k}"] = log["actual"] - log[k]
    return log


def plot(fc: dict, path: Path, split: str, start: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    inp = fc["inputs"]
    s = pd.read_csv(VMATX_CSV, parse_dates=["date"]).set_index("date")["close"][start:inp["last_close_date"]]
    sp, t0, t1 = pd.Timestamp(split), s.index[-1], pd.Timestamp(inp["target_date"])
    B, R, O, G, bg = "#1f5fbf", "#d62728", "#e08a00", "#555", "#fbf9f4"
    vals = [fc[k]["forecast"] for k in vm.MODEL_KEYS]

    def draw(ax, big):
        ax.set_facecolor(bg)
        ax.plot(s[s.index <= sp].index, s[s.index <= sp], color=B, lw=2, label=f"VMATX NAV (through {sp:%b %d})")
        ax.plot(s[s.index >= sp].index, s[s.index >= sp], color=R, lw=2, label=f"After {sp:%b %d}")
        for k, col, lab in [("tr_ecm", R, "returns model, total return"), ("raw_ecm", O, "returns model, raw")]:
            y = fc[k]["forecast"]
            ax.plot([t0, t1], [s.iloc[-1], y], color=col, ls=(0, (3, 2)), lw=1.6, label=f"{t1:%b %d} {lab}: {y:.3f}")
            ax.scatter([t1], [y], s=30 if big else 20, facecolors="none", edgecolors=col,
                       linewidths=1.6, linestyles="--", zorder=6)
        for k, mk, lab in [("tr_level", "x", "level, total return (option)"), ("raw_level", "+", "level, raw (option)")]:
            y = fc[k]["forecast"]
            ax.scatter([t1], [y], marker=mk, s=30 if big else 20, color=G, zorder=6, label=f"{lab}: {y:.3f}")
        ax.axvline(sp, color="#888", ls=":", lw=1)
        ax.grid(alpha=.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(bg)
    draw(ax, False)
    ax.set_title(f"VMATX daily NAV, {s.index[0]:%b %d} – {t0:%b %d, %Y}, with {t1:%b %d} forecasts", loc="left")
    ax.set_ylabel("NAV ($)")
    ax.xaxis.set_major_formatter(md.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(md.MonthLocator())
    ax.set_xlim(right=t1 + pd.Timedelta(days=4))
    lo = min(s.min(), *vals)
    ax.set_ylim(lo - (s.max() - lo) * 0.55, s.max() + (s.max() - lo) * 0.05)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ins = ax.inset_axes([0.52, 0.06, 0.30, 0.40])
    draw(ins, True)
    z0 = t1 - pd.Timedelta(days=22)
    zv = list(s[z0:]) + vals
    pad = (max(zv) - min(zv)) * 0.12
    ins.set_xlim(z0, t1 + pd.Timedelta(days=1.5))
    ins.set_ylim(min(zv) - pad, max(zv) + pad)
    ins.xaxis.set_major_formatter(md.DateFormatter("%b %d"))
    ins.xaxis.set_major_locator(md.AutoDateLocator(maxticks=4))
    ins.tick_params(labelsize=7.5)
    ins.set_title("Zoom: last 3 weeks", fontsize=8, loc="left")
    fig.text(0.01, 0.01, f"FTMA input {inp['ftma_input']:.3f} ({inp['ftma_input_note']}). "
             "Data: aigenmod/munidata; Yahoo Finance.", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ftma-now", type=float, help="FTMA input price (skips the intraday fetch)")
    ap.add_argument("--morning-end", default=None, help="last ET time to average, e.g. 11:00")
    ap.add_argument("--target-date", default=None, help="YYYY-MM-DD (default: next business day after last close)")
    ap.add_argument("--assume-vmatx-div", type=float, default=None)
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--split", default="2026-09-16", help="regime date for chart colors (last FOMC decision)")
    ap.add_argument("--chart-start", default="2026-02-01")
    a = ap.parse_args(argv)

    models = json.loads(MODELS.read_text())
    joint, fdiv, vdiv = load_inputs(no_fetch=True)  # distributions CSV saved by the training run
    last = joint.index[-1]
    target = pd.Timestamp(a.target_date) if a.target_date else last + pd.offsets.BDay(1)
    today_et = pd.Timestamp.now(tz=ET).normalize().tz_localize(None)
    if a.target_date is None and last >= today_et:
        print(f"Joint CSV already has {last.date()} (today's NAV is out); nothing to forecast.")
        return 0
    if models["sample_end"] != str(last.date()):
        print(f"WARNING: models trained through {models['sample_end']}, data through {last.date()}; "
              "re-run train_vmatx_models.py")

    if a.ftma_now is not None:
        Fin, note, nbars, vol = a.ftma_now, "supplied", None, None
    else:
        Fin, t0, t1, nbars, vol = ftma_intraday_mean(end=a.morning_end)
        note = f"mean of {nbars} 5-min closes {t0:%H:%M}-{t1:%H:%M} ET, volume {vol:,}"
        if t1.normalize().tz_localize(None) != target:
            print(f"WARNING: intraday bars are dated {t1.date()}, target is {target.date()}")

    fc = vm.forecast(models, joint, fdiv, vdiv, Fin, target, a.assume_vmatx_div)
    fc["inputs"]["ftma_input_note"] = note
    fc["inputs"]["models_trained_through"] = models["sample_end"]
    fc["run_at_et"] = pd.Timestamp.now(tz=ET).strftime("%Y-%m-%d %H:%M")

    FC_DIR.mkdir(parents=True, exist_ok=True)
    (FC_DIR / "latest.json").write_text(json.dumps(fc, indent=1, default=float))
    row = dict(target_date=str(target.date()), run_at_et=fc["run_at_et"],
               last_close=fc["inputs"]["V0"], ftma_prev_close=fc["inputs"]["F0"], ftma_input=Fin,
               ftma_note=note, **{k: round(fc[k]["forecast"], 4) for k in vm.MODEL_KEYS},
               tr_ecm_lo95=round(fc["tr_ecm"]["lo95"], 4), tr_ecm_hi95=round(fc["tr_ecm"]["hi95"], 4),
               raw_ecm_lo95=round(fc["raw_ecm"]["lo95"], 4), raw_ecm_hi95=round(fc["raw_ecm"]["hi95"], 4),
               notes=" | ".join(fc["inputs"]["notes"]))
    log = update_log(row, joint)
    log.to_csv(LOG, index=False)

    print(f"Last close {fc['inputs']['last_close_date']}: VMATX {fc['inputs']['V0']:.3f}, "
          f"FTMA {fc['inputs']['F0']:.3f}. Target {target.date()}")
    print(f"FTMA input {Fin:.4f} ({note}); move {fc['inputs']['ftma_move_pct']:+.3f}%")
    for n in fc["inputs"]["notes"]:
        print("NOTE:", n)
    for k in vm.MODEL_KEYS:
        r = fc[k]
        rng = f"  95% {r['lo95']:.3f}-{r['hi95']:.3f}" if "lo95" in r else ""
        extra = (f"  [FTMA {r['terms_pct']['ftma']:+.3f}%, gap {r['terms_pct']['gap']:+.3f}%, "
                 f"const {r['terms_pct']['const']:+.3f}%, u {r['u_last_pct']:+.2f}%]") if "terms_pct" in r else ""
        print(f"  {vm.MODEL_LABELS[k]:40s}{r['forecast']:8.3f}{rng}{extra}")
    scored = log.dropna(subset=["actual"])
    if len(scored):
        print(f"Track record over {len(scored)} scored days (MAE, $):",
              ", ".join(f"{k} {scored[f'err_{k}'].abs().mean():.4f}" for k in vm.MODEL_KEYS))

    if not a.no_chart:
        png = FC_DIR / f"vmatx_forecast_{target.date()}.png"
        plot(fc, png, a.split, a.chart_start)
        print("chart:", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
