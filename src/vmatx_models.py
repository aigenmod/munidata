"""
vmatx_models.py — VMATX-vs-FTMA models used by train_vmatx_models.py and
forecast_vmatx.py.

Four models, each fit on daily closes (data/jointVMATX-FTMA_daily.csv):

  tr_ecm     FINAL (primary)  returns / error-correction model, total-return basis
  raw_ecm    FINAL            returns / error-correction model, raw closes
  tr_level   option           level regression, total-return basis
  raw_level  option           level regression, raw closes

  level:    log V_t = a + b log F_t            u_t = log V_t - (a + b log F_t)
  returns:  dlog V_t = c + h dlog F_t + g u_{t-1}

Total-return basis:
  * FTMA (ETF): each distribution is added back on its ex-date,
        r_F,t = log((F_t + D_t) / F_{t-1})
  * VMATX (daily-accrual mutual fund, NAV does not drop on payment): each
    monthly distribution is spread evenly over that month's business days,
        r_V,t = log((V_t + A_t) / V_{t-1}),  A_t = monthly amount / business days in month
  * Indices are cumulated from these returns and anchored to the actual close on
    the first sample date, so fitted coefficients do not depend on the run date.
  * A total-return forecast is converted back to published NAV with
        NAV_t = NAV_{t-1} * exp(dlog V_t) - A_t
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

MODEL_KEYS = ["tr_ecm", "raw_ecm", "tr_level", "raw_level"]
MODEL_LABELS = {
    "tr_ecm": "FINAL  returns model, total return",
    "raw_ecm": "FINAL  returns model, raw closes",
    "tr_level": "option level regression, total return",
    "raw_level": "option level regression, raw closes",
}


# ------------------------------------------------------------- distributions
def fill_missing_vmatx(vdiv: pd.Series, first: pd.Timestamp, last: pd.Timestamp,
                       assume: float | None = None) -> tuple[pd.Series, list[str]]:
    """VMATX pays at month end. If a completed month inside [first, last] has no
    distribution on record yet (Yahoo posts it late), fill it with `assume` or the
    mean of the last three. Returns (filled series, notes)."""
    vdiv = vdiv.copy().sort_index()
    fill = float(assume) if assume is not None else float(vdiv.tail(3).mean())
    notes = []
    for m in pd.period_range(first, last, freq="M"):
        month_end_bd = m.to_timestamp(how="end").normalize() - pd.offsets.BDay(0)
        have = ((vdiv.index.year == m.year) & (vdiv.index.month == m.month)).any()
        if month_end_bd <= last and not have:
            vdiv.loc[month_end_bd] = fill
            notes.append(f"VMATX {m} distribution not on record; assumed ${fill:.4f}")
    return vdiv.sort_index(), notes


def vmatx_accrual(day: pd.Timestamp, vdiv: pd.Series, fallback: float) -> float:
    """One business day's VMATX income accrual for the month containing `day`."""
    m = vdiv[(vdiv.index.year == day.year) & (vdiv.index.month == day.month)]
    amt = float(m.sum()) if len(m) else fallback
    nbd = len(pd.bdate_range(day.replace(day=1), day + pd.offsets.MonthEnd(0)))
    return amt / nbd


def total_return_logs(V: pd.Series, F: pd.Series, fdiv: pd.Series, vdiv: pd.Series,
                      fallback: float) -> tuple[pd.Series, pd.Series]:
    """Log total-return indices for VMATX and FTMA, anchored to actual closes on the
    first date. `fdiv`/`vdiv` are distribution amounts indexed by (naive) date."""
    D = pd.Series(0.0, index=F.index)
    for d, amt in fdiv.items():
        if d in D.index:
            D[d] += amt
    rF = np.log((F + D) / F.shift(1))
    A = pd.Series([vmatx_accrual(d, vdiv, fallback) for d in V.index], index=V.index)
    rV = np.log((V + A) / V.shift(1))
    lV = np.log(V.iloc[0]) + rV.fillna(0).cumsum()
    lF = np.log(F.iloc[0]) + rF.fillna(0).cumsum()
    return lV, lF


# ------------------------------------------------------------- fitting
def fit_pair(lv: pd.Series, lf: pd.Series) -> dict:
    """Level regression + returns (ECM) regression on log series."""
    eg_t, eg_p, _ = coint(lv, lf)
    lev = sm.OLS(lv, sm.add_constant(lf)).fit()
    a, b = (float(x) for x in lev.params)
    u = lev.resid
    X = pd.DataFrame({"dv": lv.diff(), "df": lf.diff(), "u1": u.shift(1)}).dropna()
    ecm = sm.OLS(X.dv, sm.add_constant(X[["df", "u1"]])).fit()
    c, h, g = (float(x) for x in ecm.params)
    return dict(
        a=a, b=b, R2_level=float(lev.rsquared), u_sd=float(u.std()),
        c=c, h=h, g=g,
        c_p=float(ecm.pvalues["const"]), h_p=float(ecm.pvalues["df"]), g_p=float(ecm.pvalues["u1"]),
        R2_ecm=float(ecm.rsquared), se=float(np.sqrt(ecm.scale)),
        eg_t=float(eg_t), eg_p=float(eg_p), n=int(len(lv)),
    )


def train(joint: pd.DataFrame, fdiv: pd.Series, vdiv: pd.Series,
          assume_vmatx_div: float | None = None) -> dict:
    """Fit raw and total-return model pairs. `joint` has a DatetimeIndex and
    columns vmatx_close, ftma_close."""
    V, F = joint.vmatx_close.astype(float), joint.ftma_close.astype(float)
    vdiv_f, notes = fill_missing_vmatx(vdiv, V.index[0], V.index[-1], assume_vmatx_div)
    fallback = float(vdiv_f.tail(3).mean())
    lV, lF = total_return_logs(V, F, fdiv, vdiv_f, fallback)
    return dict(
        sample_start=str(V.index[0].date()), sample_end=str(V.index[-1].date()),
        raw=fit_pair(np.log(V), np.log(F)),
        tr=fit_pair(lV, lF),
        notes=notes,
    )


# ------------------------------------------------------------- forecasting
def forecast(models: dict, joint: pd.DataFrame, fdiv: pd.Series, vdiv: pd.Series,
             ftma_input: float, target: pd.Timestamp,
             assume_vmatx_div: float | None = None) -> dict:
    """Forecast VMATX NAV on `target` given an FTMA price for that day.
    Uses the coefficients in `models`; state (last close, gap u) comes from `joint`."""
    V, F = joint.vmatx_close.astype(float), joint.ftma_close.astype(float)
    V0, F0 = float(V.iloc[-1]), float(F.iloc[-1])
    vdiv_f, notes = fill_missing_vmatx(vdiv, V.index[0], V.index[-1], assume_vmatx_div)
    fallback = float(vdiv_f.tail(3).mean())
    lV, lF = total_return_logs(V, F, fdiv, vdiv_f, fallback)

    d_target = float(fdiv.get(target, 0.0))
    if d_target:
        notes.append(f"FTMA ex-date on {target.date()} (${d_target:.4f}): added to FTMA total return; "
                     "raw models are biased low today")
    acc = vmatx_accrual(target, vdiv_f, fallback)

    cases = {
        # name: (fit, log V last, log F last, FTMA log move, accrual)
        "raw": (models["raw"], np.log(V0), np.log(F0), np.log(ftma_input / F0), 0.0),
        "tr": (models["tr"], float(lV.iloc[-1]), float(lF.iloc[-1]),
               np.log((ftma_input + d_target) / F0), acc),
    }
    out = {}
    for name, (m, lv0, lf0, dF, a_t) in cases.items():
        u0 = lv0 - (m["a"] + m["b"] * lf0)
        dv = m["c"] + m["h"] * dF + m["g"] * u0
        z = 1.96 * m["se"]
        out[f"{name}_ecm"] = dict(
            forecast=V0 * np.exp(dv) - a_t,
            lo95=V0 * np.exp(dv - z) - a_t, hi95=V0 * np.exp(dv + z) - a_t,
            ret_pct=dv * 100, u_last_pct=u0 * 100,
            terms_pct=dict(ftma=m["h"] * dF * 100, gap=m["g"] * u0 * 100, const=m["c"] * 100),
            accrual=a_t)
        lv_level = m["a"] + m["b"] * (lf0 + dF)
        out[f"{name}_level"] = dict(forecast=V0 * np.exp(lv_level - lv0) - a_t, accrual=a_t)
    out["inputs"] = dict(last_close_date=str(V.index[-1].date()), target_date=str(target.date()),
                         V0=V0, F0=F0, ftma_input=ftma_input,
                         ftma_move_pct=np.log(ftma_input / F0) * 100, notes=notes)
    return out
