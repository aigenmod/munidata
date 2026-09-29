#!/usr/bin/env python3
"""
analyze.py (v1) — daily analysis pass over the committed munidata tables.

  python src/analyze.py            # read data/, write data/analysis/
  python src/analyze.py --quiet    # no stdout, just write files

What this does that a one-off notebook cannot
---------------------------------------------
The headline output is a LOGGED OUT-OF-SAMPLE NOWCAST. Each run:

  1. reads the newest FTMA close whose VMATX NAV has not published yet,
  2. predicts that day's VMATX NAV and appends the prediction to
     data/analysis/nowcast_log.csv with the model that produced it,
  3. goes back and scores every earlier prediction whose actual NAV has
     since arrived.

That ordering is the whole point. A prediction written down before the
answer exists is evidence; the same number computed afterwards from a
model fitted on the full sample is not. A daily scheduled job is the only
practical way to accumulate the former, and it is the reason to run this
on a schedule rather than re-running a backtest by hand.

Everything else here is monitoring, meant to catch the relationship
breaking rather than to re-litigate it:

  * level regression WITH a linear trend (a constant-mean specification
    understates the fit when the equilibrium itself is trending)
  * returns regression, which is the defensible read on predictive power
  * rolling 60-day correlation, to see drift rather than one number
  * residual z-score against the trailing distribution, as a drift alarm
  * Engle-Granger cointegration, refreshed, with and without the trend
  * data-integrity flags: stale feeds, calendar gaps, volume spikes,
    suspected partial bars

Outputs (all committed, so history accumulates in git):
  data/analysis/metrics.json       latest run, machine-readable
  data/analysis/metrics_log.csv    one row per run, appended
  data/analysis/nowcast_log.csv    predictions + realised errors
  data/analysis/REPORT.md          human-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller, coint
    HAVE_SM = True
except ImportError:                                     # keep the job alive
    HAVE_SM = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = DATA_DIR / "analysis"

ROLL_WINDOW = 60          # trading days for rolling stats
RESID_WINDOW = 120        # trailing window for the residual z-score
STALE_DAYS = 5            # feed considered stale after this many calendar days


# --------------------------------------------------------------- loading
def load_tables() -> dict[str, pd.DataFrame]:
    def rd(name: str, **kw) -> pd.DataFrame:
        p = DATA_DIR / name
        if not p.exists():
            raise SystemExit(f"missing {p} — run the fetch steps first")
        return pd.read_csv(p, parse_dates=["date"], **kw)

    ftma = rd("FTMA.csv").sort_values("date").reset_index(drop=True)
    vmatx = rd("VMATX.csv").sort_values("date").reset_index(drop=True)

    fred_files = sorted(DATA_DIR.glob("fred_*.csv"))
    fred = (pd.read_csv(fred_files[0], parse_dates=["date"])
              .sort_values("date").reset_index(drop=True)) if fred_files else pd.DataFrame()

    return {"ftma": ftma, "vmatx": vmatx, "fred": fred}


# ------------------------------------------------------------ regression
def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """Plain OLS with HAC-free standard errors. X should include a constant."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Durbin-Watson: <1.5 means the residuals are serially correlated, which
    # is the classic tell that a level regression is picking up shared trend
    dw = float(((np.diff(resid)) ** 2).sum() / ss_res) if ss_res > 0 else np.nan

    return {"beta": beta.tolist(), "se": se.tolist(), "r2": float(r2),
            "rmse": float(np.sqrt(ss_res / n)), "dw": dw, "n": int(n),
            "resid": resid}


def fit_models(joint: pd.DataFrame) -> dict:
    """Level-with-trend and returns specifications."""
    out = {}
    x = joint["ftma_close"].to_numpy(float)
    y = joint["vmatx_close"].to_numpy(float)
    t = np.arange(len(joint), dtype=float)

    # Level + trend. The trend term matters: without it the equilibrium is
    # forced to a constant mean it demonstrably does not have.
    X = np.column_stack([np.ones_like(x), x, t])
    lvl = ols(y, X)
    out["level_trend"] = {k: v for k, v in lvl.items() if k != "resid"}
    out["_level_resid"] = lvl["resid"]

    # Level without trend, for comparison
    lvl0 = ols(y, np.column_stack([np.ones_like(x), x]))
    out["level_plain"] = {k: v for k, v in lvl0.items() if k != "resid"}

    # Returns
    r = joint["ftma_close"].pct_change().to_numpy(float)
    s = joint["vmatx_close"].pct_change().to_numpy(float)
    m = ~(np.isnan(r) | np.isnan(s))
    if m.sum() > 5:
        Xr = np.column_stack([np.ones(m.sum()), r[m]])
        ret = ols(s[m], Xr)
        out["returns"] = {k: v for k, v in ret.items() if k != "resid"}
        out["returns_corr"] = float(np.corrcoef(r[m], s[m])[0, 1])
    return out


def rolling_stats(joint: pd.DataFrame, window: int = ROLL_WINDOW) -> dict:
    r = joint["ftma_close"].pct_change()
    s = joint["vmatx_close"].pct_change()
    roll = r.rolling(window).corr(s)
    valid = roll.dropna()
    if valid.empty:
        return {}
    return {
        "window": window,
        "latest": float(valid.iloc[-1]),
        "mean": float(valid.mean()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "latest_date": str(joint["date"].iloc[valid.index[-1]].date()),
    }


def cointegration(joint: pd.DataFrame) -> dict:
    """Engle-Granger, with and without a trend in the cointegrating relation.

    Reported as a pair on purpose: the constant-mean version failing while
    the trend version passes is a specification result, not evidence the
    two funds are unrelated.
    """
    if not HAVE_SM or len(joint) < 30:
        return {"available": False}
    x = joint["ftma_close"].to_numpy(float)
    y = joint["vmatx_close"].to_numpy(float)
    out = {"available": True}
    try:
        out["p_constant"] = float(coint(y, x, trend="c")[1])
        out["p_trend"] = float(coint(y, x, trend="ct")[1])
    except Exception as e:
        out["error"] = str(e)[:120]
    try:
        out["adf_vmatx_p"] = float(adfuller(y, autolag="AIC", result_object=False)[1])
        out["adf_ftma_p"] = float(adfuller(x, autolag="AIC", result_object=False)[1])
    except TypeError:                       # statsmodels < 0.15 has no result_object
        out["adf_vmatx_p"] = float(adfuller(y, autolag="AIC")[1])
        out["adf_ftma_p"] = float(adfuller(x, autolag="AIC")[1])
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- nowcast
def nowcast(ftma: pd.DataFrame, vmatx: pd.DataFrame, joint: pd.DataFrame,
            models: dict) -> dict | None:
    """Predict the VMATX NAV for the newest FTMA session that has no NAV yet.

    The timing is what makes this legitimate rather than a leak: FTMA's
    close is observable at 16:00 ET, VMATX's NAV strikes at 16:00 but does
    not publish until roughly 18:00. So same-day FTMA is genuinely
    available before the target is known.
    """
    if joint.empty or ftma.empty:
        return None

    last_nav = vmatx["date"].max()
    ahead = ftma[ftma["date"] > last_nav]
    if ahead.empty:
        return None
    row = ahead.iloc[-1]

    # Refuse to log a prediction built on an unsettled bar. A mid-session
    # FTMA price is not the close the model was fitted on, so scoring it
    # against the real NAV later would measure the wrong thing — and the
    # error would be silently baked into the out-of-sample record, which is
    # the one number here that has to stay clean.
    vol = row.get("volume")
    if pd.notna(vol) and float(vol) % 100 != 0:
        return {"skipped": True, "target_date": str(row["date"].date()),
                "reason": f"FTMA bar looks unsettled (volume {float(vol):,.0f} "
                          f"is not a round lot); no prediction logged"}

    # The model is fitted only on data through the last known NAV, so the
    # prediction uses nothing from the day being predicted except FTMA.
    b = models["level_trend"]["beta"]
    t_next = float(len(joint))          # next index position on the trend
    pred_level = b[0] + b[1] * float(row["close"]) + b[2] * t_next

    prev_ftma = joint["ftma_close"].iloc[-1]
    prev_nav = joint["vmatx_close"].iloc[-1]
    pred_ret = None
    if "returns" in models and prev_ftma:
        rb = models["returns"]["beta"]
        ftma_ret = float(row["close"]) / float(prev_ftma) - 1.0
        pred_ret = float(prev_nav) * (1.0 + rb[0] + rb[1] * ftma_ret)

    return {
        "target_date": str(row["date"].date()),
        "ftma_close": float(row["close"]),
        "prev_nav": float(prev_nav),
        "pred_level_trend": float(pred_level),
        "pred_returns": float(pred_ret) if pred_ret is not None else None,
        "fitted_through": str(joint["date"].iloc[-1].date()),
        "n_train": int(len(joint)),
    }


def update_nowcast_log(pred: dict | None, vmatx: pd.DataFrame) -> pd.DataFrame:
    """Append today's prediction, then score any past ones the NAV caught up to."""
    path = OUT_DIR / "nowcast_log.csv"
    cols = ["target_date", "made_at", "ftma_close", "prev_nav",
            "pred_level_trend", "pred_returns", "n_train",
            "actual_nav", "err_level", "err_returns", "err_naive"]
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)

    if pred is not None and pred.get("skipped"):
        pred = None                      # scoring below still runs

    if pred is not None and pred["target_date"] not in set(log.get("target_date", [])):
        log = pd.concat([log, pd.DataFrame([{
            "target_date": pred["target_date"],
            "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ftma_close": pred["ftma_close"],
            "prev_nav": pred["prev_nav"],
            "pred_level_trend": pred["pred_level_trend"],
            "pred_returns": pred["pred_returns"],
            "n_train": pred["n_train"],
        }])], ignore_index=True)

    # score whatever has resolved since the last run
    nav = vmatx.set_index(vmatx["date"].dt.strftime("%Y-%m-%d"))["close"]
    for i, r in log.iterrows():
        if pd.notna(r.get("actual_nav")):
            continue
        if r["target_date"] in nav.index:
            a = float(nav.loc[r["target_date"]])
            log.at[i, "actual_nav"] = a
            log.at[i, "err_level"] = a - float(r["pred_level_trend"])
            if pd.notna(r.get("pred_returns")):
                log.at[i, "err_returns"] = a - float(r["pred_returns"])
            log.at[i, "err_naive"] = a - float(r["prev_nav"])   # random-walk benchmark

    log = log.drop_duplicates(subset="target_date", keep="last").sort_values("target_date")
    log.to_csv(path, index=False)
    return log


def score_log(log: pd.DataFrame) -> dict:
    """Out-of-sample accuracy so far, against a random-walk benchmark."""
    done = log.dropna(subset=["actual_nav"]) if "actual_nav" in log else pd.DataFrame()
    if done.empty:
        return {"scored": 0,
                "note": "no predictions have resolved yet — needs one more run "
                        "after the next NAV publishes"}
    out = {"scored": int(len(done))}
    for name, col in [("level_trend", "err_level"), ("returns", "err_returns"),
                      ("naive", "err_naive")]:
        e = pd.to_numeric(done.get(col), errors="coerce").dropna()
        if len(e):
            out[f"mae_{name}"] = float(e.abs().mean())
            out[f"rmse_{name}"] = float(np.sqrt((e ** 2).mean()))
    if "rmse_level_trend" in out and out.get("rmse_naive"):
        out["skill_vs_naive"] = float(1 - out["rmse_level_trend"] / out["rmse_naive"])
    return out


# ----------------------------------------------------------- data health
def integrity(ftma: pd.DataFrame, vmatx: pd.DataFrame, fred: pd.DataFrame) -> dict:
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    flags: list[str] = []
    out: dict = {}

    for name, df in [("FTMA", ftma), ("VMATX", vmatx)]:
        last = df["date"].max()
        lag = int((today - last).days)
        out[f"{name.lower()}_last"] = str(last.date())
        out[f"{name.lower()}_lag_days"] = lag
        if lag > STALE_DAYS:
            flags.append(f"{name} stale: last observation {last.date()} ({lag}d ago)")

    if not fred.empty and "DFF" in fred:
        d = fred.dropna(subset=["DFF"])
        if not d.empty:
            last = d["date"].max()
            out["dff_last"] = str(last.date())
            out["dff_lag_days"] = int((today - last).days)

    # partial-bar suspicion: settled Yahoo EOD volume is reported in round lots
    if not ftma.empty:
        v = ftma["volume"].iloc[-1]
        if pd.notna(v) and float(v) % 100 != 0:
            flags.append(f"FTMA {ftma['date'].max().date()} volume {float(v):,.0f} "
                         f"is not a round lot — possible partial/in-session bar")

    # volume spike, which usually means a creation/redemption rather than flow
    if len(ftma) > 20:
        med = float(ftma["volume"].tail(60).median())
        v = float(ftma["volume"].iloc[-1])
        out["volume_vs_median"] = round(v / med, 2) if med else None
        if med and v > 8 * med:
            flags.append(f"FTMA volume {v:,.0f} is {v/med:.1f}x its 60-day median")

    # calendar gaps against business days
    for name, df in [("FTMA", ftma)]:
        idx = pd.DatetimeIndex(df["date"])
        expected = pd.bdate_range(idx.min(), idx.max())
        missing = expected.difference(idx)
        out[f"{name.lower()}_missing_bdays"] = int(len(missing))
        if len(missing) > 12:      # ~holidays only; more than that is suspicious
            flags.append(f"{name} missing {len(missing)} business days in span")

    out["flags"] = flags
    return out


def drift(models: dict, joint: pd.DataFrame) -> dict:
    """Is today's residual unusual against its own trailing distribution?"""
    resid = models.get("_level_resid")
    if resid is None or len(resid) < RESID_WINDOW + 5:
        return {}
    trail = resid[-RESID_WINDOW:-1]
    latest = float(resid[-1])
    mu, sd = float(trail.mean()), float(trail.std(ddof=1))
    z = (latest - mu) / sd if sd > 0 else np.nan
    return {"latest_residual": latest, "z_score": float(z),
            "window": RESID_WINDOW,
            "alert": bool(abs(z) > 3) if np.isfinite(z) else False}


# ---------------------------------------------------------------- report
def write_report(m: dict) -> None:
    L = m["models"]["level_trend"]
    R = m["models"].get("returns", {})
    ro = m.get("rolling", {})
    ci = m.get("cointegration", {})
    sc = m.get("nowcast_score", {})
    nc = m.get("nowcast")
    dr = m.get("drift", {})
    ig = m["integrity"]

    lines = [
        f"# munidata daily analysis",
        "",
        f"Run {m['run_utc']} · joint sample {m['sample']['start']} → "
        f"{m['sample']['end']} ({m['sample']['n']} matched sessions)",
        "",
    ]

    if ig["flags"]:
        lines += ["## ⚠ Flags", ""] + [f"- {f}" for f in ig["flags"]] + [""]
    else:
        lines += ["## Flags", "", "None.", ""]

    lines += [
        "## Fit",
        "",
        "| Spec | R² | RMSE | Durbin–Watson | n |",
        "|---|---|---|---|---|",
        f"| Level + trend | {L['r2']:.4f} | {L['rmse']:.4f} | {L['dw']:.3f} | {L['n']} |",
        f"| Level, no trend | {m['models']['level_plain']['r2']:.4f} | "
        f"{m['models']['level_plain']['rmse']:.4f} | "
        f"{m['models']['level_plain']['dw']:.3f} | {m['models']['level_plain']['n']} |",
    ]
    if R:
        lines.append(f"| Returns | {R['r2']:.4f} | {R['rmse']:.6f} | {R['dw']:.3f} | {R['n']} |")
    lines += [
        "",
        "Level R² is inflated by shared trend; the returns row is the honest "
        "read on day-to-day predictive power. Durbin–Watson well below 2 on "
        "the level rows is that inflation showing itself.",
        "",
    ]

    if ro:
        lines += [
            f"## Rolling {ro['window']}-day return correlation",
            "",
            f"Latest **{ro['latest']:.4f}** (as of {ro['latest_date']}) · "
            f"mean {ro['mean']:.4f} · range {ro['min']:.4f} to {ro['max']:.4f}",
            "",
        ]

    if ci.get("available") and "p_trend" in ci:
        lines += [
            "## Cointegration (Engle–Granger)",
            "",
            f"- constant-mean specification: p = {ci['p_constant']:.4f}",
            f"- with trend in the cointegrating relation: p = {ci['p_trend']:.4f}",
            "",
            "Reported as a pair deliberately. The constant-mean test failing "
            "while the trend version passes is a statement about "
            "specification, not evidence the two funds are unrelated.",
            "",
        ]

    if dr:
        state = "**ALERT**" if dr.get("alert") else "normal"
        lines += [
            "## Relationship drift",
            "",
            f"Latest level residual {dr['latest_residual']:+.4f}, "
            f"z = {dr['z_score']:+.2f} against the trailing {dr['window']} days — {state}",
            "",
        ]

    lines += ["## Nowcast", ""]
    if nc and nc.get("skipped"):
        lines += [f"No prediction logged for {nc['target_date']}: {nc['reason']}.", ""]
    elif nc:
        lines += [
            f"Predicting VMATX NAV for **{nc['target_date']}** from that day's "
            f"FTMA close of {nc['ftma_close']:.4f} "
            f"(model fitted through {nc['fitted_through']}, n={nc['n_train']}):",
            "",
            f"- level+trend: **{nc['pred_level_trend']:.4f}**",
        ]
        if nc.get("pred_returns"):
            lines.append(f"- returns: **{nc['pred_returns']:.4f}**")
        lines += [f"- random-walk benchmark (prior NAV): {nc['prev_nav']:.4f}", ""]
    else:
        lines += ["No open prediction — VMATX NAV is current with FTMA.", ""]

    if sc.get("scored"):
        lines += [
            f"### Realised out-of-sample accuracy ({sc['scored']} resolved)",
            "",
            "| Model | MAE | RMSE |",
            "|---|---|---|",
        ]
        for label, key in [("Level + trend", "level_trend"), ("Returns", "returns"),
                           ("Random walk", "naive")]:
            if f"mae_{key}" in sc:
                lines.append(f"| {label} | {sc[f'mae_{key}']:.4f} | {sc[f'rmse_{key}']:.4f} |")
        if "skill_vs_naive" in sc:
            lines += ["", f"Skill vs random walk: **{sc['skill_vs_naive']:+.1%}** "
                          f"(positive means the model beats simply carrying the "
                          f"prior NAV forward)."]
        lines.append("")
    else:
        lines += [sc.get("note", ""), ""]

    lines += [
        "---",
        "",
        f"FTMA last {ig['ftma_last']} ({ig['ftma_lag_days']}d) · "
        f"VMATX last {ig['vmatx_last']} ({ig['vmatx_lag_days']}d)"
        + (f" · DFF last {ig['dff_last']} ({ig['dff_lag_days']}d)" if "dff_last" in ig else ""),
        "",
        "Generated by `src/analyze.py`. Not investment advice.",
    ]

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Daily analysis over data/.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = load_tables()
    ftma, vmatx, fred = t["ftma"], t["vmatx"], t["fred"]

    joint = (ftma[["date", "close"]].rename(columns={"close": "ftma_close"})
             .merge(vmatx[["date", "close"]].rename(columns={"close": "vmatx_close"}),
                    on="date", how="inner")
             .sort_values("date").reset_index(drop=True))
    if len(joint) < 30:
        print(f"only {len(joint)} matched sessions — too few to analyse", file=sys.stderr)
        return 1

    models = fit_models(joint)
    pred = nowcast(ftma, vmatx, joint, models)
    log = update_nowcast_log(pred, vmatx)

    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample": {"start": str(joint["date"].min().date()),
                   "end": str(joint["date"].max().date()),
                   "n": int(len(joint))},
        "models": {k: v for k, v in models.items() if not k.startswith("_")},
        "rolling": rolling_stats(joint),
        "cointegration": cointegration(joint),
        "drift": drift(models, joint),
        "nowcast": pred,
        "nowcast_score": score_log(log),
        "integrity": integrity(ftma, vmatx, fred),
    }

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # flat one-row-per-run history, easy to plot straight from git
    flat = {
        "run_utc": metrics["run_utc"],
        "sample_end": metrics["sample"]["end"],
        "n": metrics["sample"]["n"],
        "level_r2": models["level_trend"]["r2"],
        "returns_r2": models.get("returns", {}).get("r2"),
        "returns_corr": models.get("returns_corr"),
        "roll_corr": metrics["rolling"].get("latest"),
        "coint_p_trend": metrics["cointegration"].get("p_trend"),
        "resid_z": metrics["drift"].get("z_score"),
        "n_flags": len(metrics["integrity"]["flags"]),
    }
    lp = OUT_DIR / "metrics_log.csv"
    hist = pd.read_csv(lp) if lp.exists() else pd.DataFrame()
    pd.concat([hist, pd.DataFrame([flat])], ignore_index=True).to_csv(lp, index=False)

    write_report(metrics)

    if not args.quiet:
        print(f"joint sample: {metrics['sample']['n']} sessions "
              f"({metrics['sample']['start']} → {metrics['sample']['end']})")
        print(f"level+trend R2 {models['level_trend']['r2']:.4f}  "
              f"DW {models['level_trend']['dw']:.3f}")
        if "returns" in models:
            print(f"returns R2     {models['returns']['r2']:.4f}  "
                  f"corr {models['returns_corr']:.4f}")
        if metrics["rolling"]:
            print(f"rolling {metrics['rolling']['window']}d corr "
                  f"{metrics['rolling']['latest']:.4f}")
        if pred and pred.get("skipped"):
            print(f"nowcast {pred['target_date']}: SKIPPED — {pred['reason']}")
        elif pred:
            print(f"nowcast {pred['target_date']}: "
                  f"level {pred['pred_level_trend']:.4f}"
                  + (f", returns {pred['pred_returns']:.4f}" if pred['pred_returns'] else ""))
        sc = metrics["nowcast_score"]
        if sc.get("scored"):
            print(f"scored {sc['scored']} past predictions; "
                  f"skill vs naive {sc.get('skill_vs_naive', float('nan')):+.1%}")
        else:
            print(sc.get("note", ""))
        for f in metrics["integrity"]["flags"]:
            print(f"  ! {f}")
        print(f"\nwrote {OUT_DIR}/REPORT.md, metrics.json, metrics_log.csv, nowcast_log.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
