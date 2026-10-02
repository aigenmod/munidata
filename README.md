# munidata

Scripts for tracking **VMATX** (Vanguard Massachusetts Tax-Exempt Fund) —
its daily NAV history plus the rate context (FRED) needed to explain its
moves. Because VMATX only prints one NAV per day, **FTMA** — an ETF holding
similar Massachusetts muni paper — is pulled in too: its hourly bars stand
in for the moves happening *within* a day that VMATX's single daily strike
can't show. Everything writes into `data/` and
is safe to re-run (fetchers top off rather than re-download where it
matters).

## Scripts

### `src/fetch-VMATX_yf.py` — VMATX (and FTMA) daily EOD (yfinance)

Pulls daily NAV/OHLCV bars for VMATX and keeps a running table without
disturbing already-stored history; FTMA is pulled alongside it so the two
can be joined and compared.

| Function | Purpose |
|---|---|
| `load(ticker, start, end, retries)` | Fetches raw (unadjusted) daily history for one ticker from yfinance, with retry/backoff on transient failures. |
| `_pick(cols, candidates)` | Finds the right column in a vendor CSV whose headers vary (`Close/Last` vs `NAV` vs `Adj Close`, etc.). |
| `read_existing(path)` | Reads a hand-collected or vendor-exported CSV (Nasdaq, Yahoo, Stooq, plain date/NAV sheets) into the script's standard OHLCV shape. |
| `topoff(existing, ticker, through, max_move)` | Fetches only the sessions after the last stored date and appends them; rejects any new tick that moves more than `max_move` (default 5%) from the prior close as a suspect print. Never modifies existing rows. |
| `main(argv)` | CLI entry point. Loads or tops off each requested ticker, writes a per-ticker CSV, and — when both VMATX and FTMA are present — an inner-joined daily table plus their return correlation. |

### `src/fetch_DFF.py` — Fed funds rate & target range (FRED)

Rate context for VMATX's moves. Pulls daily rate series straight from
FRED's no-API-key CSV endpoint.

| Function | Purpose |
|---|---|
| `fetch_series(series, start, end, timeout)` | Downloads one FRED series and returns it as a `date, <series>` frame. Converts FRED's `.` missing-value marker to `NaN`. |
| `describe(df, series, requested_start)` | Prints diagnostics for a fetched series: observation count, date range, whether it's a 7-day or business-day series, how often the rate actually changes, and — the main gotcha — a warning when the series starts later than the `--start` you asked for, because the instrument or reporting convention didn't exist yet (e.g. `DFEDTARU` before 2008‑12‑16). |
| `main(argv)` | CLI entry point. Fetches each `--series`, merges them into one wide table by date, optionally aligns to another file's trading days (`--align`, typically VMATX's) or forward-fills gaps (`--ffill`), and writes the merged CSV to `data/`. |

### `src/fetch_ftma_hourly.py` — FTMA hourly bars, as an intraday stand-in for VMATX

Pulls hourly FTMA bars and reduces them to daily features — because VMATX
gives exactly one NAV strike per day, this is how intraday muni-market
movement gets represented at all.

| Function | Purpose |
|---|---|
| `fetch_hourly(ticker, start, end)` | Fetches 1‑hour OHLCV bars from yfinance, localizes timestamps to `America/New_York`, and flags whether each bar had a real trade (`had_trade`) vs a carried-forward price. |
| `daily_features(bars)` | Collapses one session's hourly bars into one row: OHLC, a `vwap_proxy` (volume-weighted typical price), and staleness of the closing print (how many hours since the last bar that actually traded). |
| `main(argv)` | CLI entry point. Fetches hourly bars, derives the daily feature table, writes both CSVs, and prints liquidity/staleness diagnostics (FTMA trades thinly enough that zero-volume hourly bars are common). |

## Reference: rate series (FRED)

| Series | Name | What it measures | Frequency | Official reference |
|---|---|---|---|---|
| **DFF** | Federal Funds Effective Rate | The actual, volume-weighted overnight rate banks charge each other for reserves — the real-world market outcome, not a policy setting. 7-day calendar (weekend rows repeat Friday's value). | Daily | [fred.stlouisfed.org/series/DFF](https://fred.stlouisfed.org/series/DFF) |
| **DFEDTARU** | Fed Funds Target Range – Upper Limit | The top of the FOMC's current target range — the policy ceiling DFF is meant to stay under. Only exists from 2008‑12‑16 onward (before that the Fed set a single-point target, `DFEDTAR`, not a range). | Daily | [fred.stlouisfed.org/series/DFEDTARU](https://fred.stlouisfed.org/series/DFEDTARU) |
| **DFEDTARL** | Fed Funds Target Range – Lower Limit | The bottom of that same target range — the policy floor. Same 2008‑12‑16 start date as DFEDTARU. | Daily | [fred.stlouisfed.org/series/DFEDTARL](https://fred.stlouisfed.org/series/DFEDTARL) |

## Reference: VMATX and FTMA

| Ticker | Name | Type | Inception | What it is | Official reference |
|---|---|---|---|---|---|
| **VMATX** | Vanguard Massachusetts Tax-Exempt Fund | Open-end mutual fund | 1998‑12‑09 | Seeks high current income exempt from federal and Massachusetts personal income tax; one NAV strike per day, priced at the 16:00 close. Intended for Massachusetts residents. | [investor.vanguard.com/investment-products/mutual-funds/profile/vmatx](https://investor.vanguard.com/investment-products/mutual-funds/profile/vmatx) |
| **FTMA** | Franklin Massachusetts Municipal Income ETF | Exchange-traded fund | 2025‑11‑10 | Franklin Templeton ETF investing in investment-grade municipal bonds exempt from federal and Massachusetts personal income tax, maturities of at least three years. Pulled in here because it trades intraday like a stock, giving real hourly bars that VMATX's single daily NAV can't provide. | [franklintempleton.com/.../franklin-massachusetts-municipal-income-etf/FTMA](https://www.franklintempleton.com/investments/options/exchange-traded-funds/products/48338/SINGLCLASS/franklin-massachusetts-municipal-income-etf/FTMA) |
