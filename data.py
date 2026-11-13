"""Market data: Binance (spot + perp + funding) and Yahoo Finance (stocks, dollar, gold...).

All functions return DataFrames indexed by the UTC time the value became KNOWN
(bar close), so features never peek at the future.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from config import DATA_DIR

SPOT_BASES = ["https://api.binance.com", "https://data-api.binance.vision"]
FAPI_BASE = "https://fapi.binance.com"

_session = requests.Session()
_session.headers["User-Agent"] = "btc-signal-bot/1.0"

INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}

# Cross-asset tickers on Yahoo. Futures trade ~23h/day, so they move while
# crypto is moving (cash indices only update 6.5h/day).
CROSS_TICKERS = {
    "es": "ES=F",        # S&P 500 futures
    "nq": "NQ=F",        # Nasdaq 100 futures
    "dxy": "DX-Y.NYB",   # US dollar index
    "gold": "GC=F",      # gold futures
    "vix": "^VIX",       # equity volatility
    "us10y": "^TNX",     # 10y Treasury yield (x10)
    "oil": "CL=F",       # WTI crude futures
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _get(url: str, params: dict, tries: int = 4):
    last: Exception | None = None
    for i in range(tries):
        try:
            r = _session.get(url, params=params, timeout=20)
            if r.status_code in (418, 429):
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = e
            time.sleep(1 + i)
    raise RuntimeError(f"request failed: {url} {params} ({last})")


_spot_base_ok: str | None = None


def _spot_get(path: str, params: dict):
    """Try api.binance.com, fall back to the public market-data mirror."""
    global _spot_base_ok
    bases = [_spot_base_ok] if _spot_base_ok else SPOT_BASES
    err = None
    for b in bases:
        try:
            out = _get(b + path, params)
            _spot_base_ok = b
            return out
        except RuntimeError as e:
            err = e
    raise err  # type: ignore[misc]


KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def klines(symbol: str, interval: str = "1h", start_ms: int | None = None,
           end_ms: int | None = None, market: str = "spot") -> pd.DataFrame:
    """Paginated Binance klines. Index = bar CLOSE time (when the bar is known).
    Only fully closed bars are returned."""
    step = INTERVAL_MS[interval]
    limit = 1000 if market == "spot" else 1500
    path = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
    if start_ms is None:
        start_ms = _now_ms() - 500 * step
    rows: list = []
    cur = start_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": cur}
        if end_ms:
            params["endTime"] = end_ms
        data = _spot_get(path, params) if market == "spot" else _get(FAPI_BASE + path, params)
        if not data:
            break
        rows.extend(data)
        nxt = data[-1][0] + step
        if len(data) < limit or nxt >= (end_ms or _now_ms()):
            break
        cur = nxt
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "taker_buy_base"])
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df = df[df["close_time"] < _now_ms()]  # drop the still-forming bar
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df["open_time"] + step, unit="ms", utc=True)
    df.index.name = "t"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]]


def funding_rates(symbol: str = "BTCUSDT", start_ms: int | None = None) -> pd.Series:
    """Perp funding rate history (every 8h). Index = funding time."""
    if start_ms is None:
        start_ms = _now_ms() - 30 * 86_400_000
    out: list = []
    cur = start_ms
    while True:
        data = _get(FAPI_BASE + "/fapi/v1/fundingRate",
                    {"symbol": symbol, "startTime": cur, "limit": 1000})
        if not data:
            break
        out.extend(data)
        if len(data) < 1000:
            break
        cur = data[-1]["fundingTime"] + 1
    if not out:
        return pd.Series(dtype=float, name="funding")
    s = pd.Series([float(d["fundingRate"]) for d in out],
                  index=pd.to_datetime([d["fundingTime"] for d in out], unit="ms", utc=True),
                  name="funding")
    return s[~s.index.duplicated()].sort_index()


# --------------------------------------------------------------------------
# Cached history (used for training; incremental updates)
# --------------------------------------------------------------------------

def _cache_path(name: str):
    return DATA_DIR / f"{name}.parquet"


def cached_klines(symbol: str, interval: str, days: int, market: str = "spot") -> pd.DataFrame:
    path = _cache_path(f"{symbol}_{market}_{interval}")
    start = _now_ms() - days * 86_400_000
    old = pd.read_parquet(path) if path.exists() else None
    if old is not None and len(old) and old.index[0].value // 1_000_000 <= start + 86_400_000:
        fetch_from = int(old.index[-1].value // 1_000_000) - 3 * INTERVAL_MS[interval]
        new = klines(symbol, interval, fetch_from, market=market)
        df = pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = klines(symbol, interval, start, market=market)
    df.to_parquet(path)
    return df[df.index >= pd.Timestamp(start, unit="ms", tz="UTC")]


def cached_funding(symbol: str, days: int) -> pd.Series:
    path = _cache_path(f"{symbol}_funding")
    start = _now_ms() - days * 86_400_000
    s = funding_rates(symbol, start)
    s.to_frame().to_parquet(path)
    return s


def yahoo_hourly(days: int = 729) -> pd.DataFrame:
    """Hourly closes for CROSS_TICKERS. Index = bar close time (UTC).
    Missing tickers are skipped (column absent), never fatal."""
    import yfinance as yf

    period = f"{min(days, 729)}d"
    cols = {}
    for name, tic in CROSS_TICKERS.items():
        try:
            df = yf.download(tic, period=period, interval="1h", progress=False,
                             auto_adjust=False, threads=False)
            if df is None or df.empty:
                print(f"  [yahoo] no data for {tic}")
                continue
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            idx = close.index
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            # Yahoo labels bars by their START; the value is known one hour later.
            close.index = idx + pd.Timedelta(hours=1)
            cols[name] = close.astype(float)
        except Exception as e:  # yfinance raises many kinds of things
            print(f"  [yahoo] {tic} failed: {e}")
    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    # Drop anything not yet complete.
    return out[out.index <= pd.Timestamp.now(tz="UTC")]


def cached_yahoo(days: int, max_age_min: int = 0) -> pd.DataFrame:
    path = _cache_path("yahoo_cross_1h")
    if max_age_min and path.exists():
        age = time.time() - path.stat().st_mtime
        if age < max_age_min * 60:
            return pd.read_parquet(path)
    df = yahoo_hourly(days)
    if df.empty and path.exists():
        print("  [yahoo] download failed, using cached copy")
        return pd.read_parquet(path)
    if not df.empty:
        df.to_parquet(path)
    return df


# --------------------------------------------------------------------------
# Bundles
# --------------------------------------------------------------------------

def load_bundle(days: int, live: bool = False) -> dict:
    """Everything the feature builder needs.
    live=True: small, fresh download for a single prediction.
    live=False: long cached history for training."""
    bundle: dict = {}

    def attempt(key, fn):
        try:
            bundle[key] = fn()
        except Exception as e:
            print(f"  [data] {key} unavailable: {e}")
            bundle[key] = None

    if live:
        start = _now_ms() - days * 86_400_000
        attempt("btc", lambda: klines("BTCUSDT", "1h", start))
        attempt("eth", lambda: klines("ETHUSDT", "1h", start))
        attempt("sol", lambda: klines("SOLUSDT", "1h", start))
        attempt("perp", lambda: klines("BTCUSDT", "1h", start, market="futures"))
        attempt("funding", lambda: funding_rates("BTCUSDT", start))
        attempt("cross", lambda: cached_yahoo(60, max_age_min=20))
    else:
        attempt("btc", lambda: cached_klines("BTCUSDT", "1h", days))
        attempt("eth", lambda: cached_klines("ETHUSDT", "1h", days))
        attempt("sol", lambda: cached_klines("SOLUSDT", "1h", days))
        attempt("perp", lambda: cached_klines("BTCUSDT", "1h", days, market="futures"))
        attempt("funding", lambda: cached_funding("BTCUSDT", days))
        attempt("cross", lambda: cached_yahoo(days))
    if bundle.get("btc") is None or len(bundle["btc"]) == 0:
        raise RuntimeError("Could not download BTC prices from Binance - check your internet "
                           "connection (or whether Binance is reachable from your network).")
    return bundle


def open_interest(symbol: str = "BTCUSDT", period: str = "15m", limit: int = 500) -> pd.Series:
    """Perp open interest (BTC units). Binance keeps the last 30 days."""
    data = _get(FAPI_BASE + "/futures/data/openInterestHist",
                {"symbol": symbol, "period": period, "limit": limit})
    s = pd.Series([float(d["sumOpenInterest"]) for d in data],
                  index=pd.to_datetime([int(d["timestamp"]) for d in data], unit="ms", utc=True),
                  name="oi")
    return s[~s.index.duplicated()].sort_index()


def btc_5m_since(start: datetime) -> pd.DataFrame:
    """5-minute BTC bars from `start` until now (for checking paper-trade exits)."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return klines("BTCUSDT", "5m", int(start.timestamp() * 1000) - 300_000)


def funding_since(start: datetime) -> pd.Series:
    """Perp funding rates from `start` until now (for leveraged paper-trade costs)."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return funding_rates("BTCUSDT", int(start.timestamp() * 1000))


def last_price() -> float:
    return float(_spot_get("/api/v3/ticker/price", {"symbol": "BTCUSDT"})["price"])
