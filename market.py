"""Market snapshot (/market, and the daily briefing): how futures traders are positioned, what
options price in, who is buying and getting liquidated, the price levels people watch, the slow
backdrop, and what is scheduled next.

All of it is description. Nothing here is backtested and nothing feeds the model, the call or any
wallet (CLAUDE.md rules 3-4). One compact row an hour goes to `market_snap` so that an idea about
any of these numbers can be tested later - once, with the hypothesis written down first.

Every source is free and needs no key. Each is fetched once with a short timeout; one that
doesn't answer leaves its line out and is named at the bottom of the message.
"""
from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import news
import paper_log as pl
import whales
from data import FAPI_BASE, _cache_path, _session, _spot_get, klines
from vol_model import RANGE_TO_SIGMA

log = logging.getLogger("market")

DERIBIT = "https://www.deribit.com/api/v2/public/"
TIMEOUT = 8               # seconds per request, one try
COLLECT_TIMEOUT = 25      # the whole snapshot gives up on slow sources after this
SKEW_DAYS = 30            # skew is read off the expiry nearest this many days out...
SKEW_MIN_DAYS = 7         # ...but never one closer than this (short-dated skew is jumpy)
SKEW_DELTA = 0.25         # the standard "25-delta" put and call
SKEW_TOL = 0.08           # no option within this of 25 delta = no skew shown
SKEW_EVEN = 1.0           # under this many vol points apart, puts and calls count as priced alike
BIG_EXPIRY_DAYS = 35      # "biggest expiry ahead" looks this far
FUTURE_MIN_DAYS = 14      # the dated future used for the premium must have at least this long left
OKX_CT_BTC = 0.01         # BTC per OKX BTC-USDT-SWAP contract
CORR_MIN_N = 40           # hourly pairs needed before a correlation is shown
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}

SCHEMA = "CREATE TABLE IF NOT EXISTS market_snap (t TEXT PRIMARY KEY, px REAL, data TEXT)"

_cache: dict = {}   # key -> (unix time, value), for the sources that barely move


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _get(url: str, params: dict | None = None):
    r = _session.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _cached(key: str, minutes: float, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < minutes * 60:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def _ms(x) -> datetime:
    return datetime.fromtimestamp(int(x) / 1000, timezone.utc)


# --------------------------------------------------------------------------
# 1) Positioning: open interest, long/short split, funding, dated-future premium
# --------------------------------------------------------------------------

def f_bn() -> dict:
    p = _get(FAPI_BASE + "/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"})
    return {"index": float(p["indexPrice"]), "funding": float(p["lastFundingRate"]), "next": _ms(p["nextFundingTime"])}


def f_oi() -> dict:
    """Binance perp open interest and its 24h change in BTC terms (in $ it would just follow the price)."""
    d = _get(FAPI_BASE + "/futures/data/openInterestHist", {"symbol": "BTCUSDT", "period": "1h", "limit": 25})
    return {"usd": float(d[-1]["sumOpenInterestValue"]),
            "chg": float(d[-1]["sumOpenInterest"]) / float(d[0]["sumOpenInterest"]) - 1}


def f_ls() -> dict:
    """Share of Binance futures ACCOUNTS that are net long, and the long share of the top traders' POSITIONS."""
    q = {"symbol": "BTCUSDT", "period": "1h", "limit": 1}
    acc = _get(FAPI_BASE + "/futures/data/globalLongShortAccountRatio", q)[-1]
    top = _get(FAPI_BASE + "/futures/data/topLongShortPositionRatio", q)[-1]
    return {"acc_long": float(acc["longAccount"]), "top_long": float(top["longAccount"])}


def f_bybit() -> dict:
    t = _get("https://api.bybit.com/v5/market/tickers", {"category": "linear", "symbol": "BTCUSDT"})["result"]["list"][0]
    return {"funding": float(t["fundingRate"]) * 8 / float(t.get("fundingIntervalHour") or 8),
            "oi_usd": float(t["openInterestValue"])}


def f_okx() -> dict:
    d = _get("https://www.okx.com/api/v5/public/funding-rate", {"instId": "BTC-USDT-SWAP"})["data"][0]
    hours = (int(d["nextFundingTime"]) - int(d["fundingTime"])) / 3_600_000 or 8
    return {"funding": float(d["fundingRate"]) * 8 / hours}


def f_hl() -> dict:
    r = _session.post("https://api.hyperliquid.xyz/info", json={"type": "metaAndAssetCtxs"}, timeout=TIMEOUT)
    r.raise_for_status()
    meta, ctxs = r.json()
    c = ctxs[[u["name"] for u in meta["universe"]].index("BTC")]
    return {"funding": float(c["funding"]) * 8,    # Hyperliquid pays hourly; shown per 8h like the others
            "oi_usd": float(c["openInterest"]) * float(c["markPx"])}


def _expiry(code: str) -> datetime:
    """'4DEC26' -> 4 Dec 2026 08:00 UTC, when Deribit contracts settle."""
    return datetime(2000 + int(code[-2:]), MONTHS[code[-5:-2]], int(code[:-5]), 8, tzinfo=timezone.utc)


def future_premium(rows: list[dict], now: datetime) -> dict | None:
    """The most-held dated future with FUTURE_MIN_DAYS+ left: how far above the index it trades, and
    that as a yearly rate. Unlike funding it doesn't flip every few hours."""
    best = None
    for x in rows:
        parts = x["instrument_name"].split("-")
        if len(parts) != 2 or parts[1] == "PERPETUAL":
            continue
        days = (_expiry(parts[1]) - now).total_seconds() / 86400
        if days >= FUTURE_MIN_DAYS and (best is None or x["open_interest"] > best[0]["open_interest"]):
            best = (x, days)
    if not best:
        return None
    x, days = best
    basis = x["mark_price"] / x["estimated_delivery_price"] - 1
    return {"t": _expiry(x["instrument_name"].split("-")[1]), "days": days, "basis": basis, "annual": basis * 365 / days}


def f_futs() -> dict | None:
    return future_premium(_get(DERIBIT + "get_book_summary_by_currency", {"currency": "BTC", "kind": "future"})["result"], _now())


# --------------------------------------------------------------------------
# 2) Options (Deribit, where most BTC options trade)
# --------------------------------------------------------------------------

def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def options_view(rows: list[dict], now: datetime) -> dict | None:
    """rows: Deribit's book summary for every BTC option. Pure, so it can be tested on made-up rows.
    - pc: open puts / open calls.
    - big: the expiry within BIG_EXPIRY_DAYS with the most open contracts, its most-held call and put
      strikes, and "max pain" (the settlement price at which option holders collect the least).
    - skew: implied vol of the 25-delta put minus the 25-delta call on the expiry nearest SKEW_DAYS.
      Deltas are Black-Scholes on the future (rates 0), from Deribit's own mark vol."""
    opts = []
    for x in rows:
        try:
            _, code, strike, cp = x["instrument_name"].split("-")
            t = _expiry(code)
        except (ValueError, KeyError):
            continue
        if t > now:
            opts.append({"t": t, "k": float(strike), "cp": cp, "oi": float(x.get("open_interest") or 0),
                         "iv": x.get("mark_iv"), "f": x.get("underlying_price"), "px": x.get("estimated_delivery_price")})
    if not opts:
        return None
    calls, puts = sum(o["oi"] for o in opts if o["cp"] == "C"), sum(o["oi"] for o in opts if o["cp"] == "P")
    index = next((o["px"] for o in opts if o["px"]), None)
    out = {"pc": puts / calls if calls else None, "oi_btc": calls + puts, "index": index, "big": None, "skew": None}

    by_exp: dict = {}
    for o in opts:
        by_exp.setdefault(o["t"], []).append(o)
    near = {t: v for t, v in by_exp.items() if t - now <= timedelta(days=BIG_EXPIRY_DAYS)}
    if near:
        t = max(near, key=lambda t: sum(o["oi"] for o in near[t]))
        exp = near[t]
        top = lambda cp: max((o for o in exp if o["cp"] == cp), key=lambda o: o["oi"], default=None)
        strikes = sorted({o["k"] for o in exp})
        paid = lambda s: sum(o["oi"] * max(s - o["k"], 0) if o["cp"] == "C" else o["oi"] * max(o["k"] - s, 0) for o in exp)
        c, p = top("C"), top("P")
        oi = sum(o["oi"] for o in exp)
        out["big"] = {"t": t, "oi_btc": oi, "usd": oi * index if index else None, "call": c and c["k"], "put": p and p["k"],
                      "pain": min(strikes, key=paid)}

    far = [t for t in by_exp if t - now >= timedelta(days=SKEW_MIN_DAYS)]
    if far:
        t = min(far, key=lambda t: abs((t - now).total_seconds() / 86400 - SKEW_DAYS))
        yrs = (t - now).total_seconds() / (365 * 86400)
        best = {"C": None, "P": None}
        for o in by_exp[t]:
            if not o["iv"] or not o["f"]:
                continue
            sd = o["iv"] / 100 * math.sqrt(yrs)
            delta = _ncdf((math.log(o["f"] / o["k"]) + 0.5 * sd * sd) / sd) - (o["cp"] == "P")
            miss = abs(abs(delta) - SKEW_DELTA)
            if miss <= SKEW_TOL and (best[o["cp"]] is None or miss < best[o["cp"]][0]):
                best[o["cp"]] = (miss, o["iv"])
        if best["C"] and best["P"]:
            out["skew"] = {"t": t, "put_iv": best["P"][1], "call_iv": best["C"][1], "skew": best["P"][1] - best["C"][1]}
    return out


def f_opts() -> dict | None:
    return options_view(_get(DERIBIT + "get_book_summary_by_currency", {"currency": "BTC", "kind": "option"})["result"], _now())


def f_dvol() -> dict:
    """Deribit's implied-volatility index: the yearly volatility options price in for the next 30 days."""
    now = int(time.time() * 1000)
    q = lambda res, days: _get(DERIBIT + "get_volatility_index_data", {
        "currency": "BTC", "resolution": res, "start_timestamp": now - days * 86_400_000, "end_timestamp": now})["result"]["data"]
    h = q(3600, 7)
    cur = float(h[-1][4])
    year = _cached("dvol_year", 360, lambda: [float(r[4]) for r in q("1D", 366)])
    at = lambda hours: float(h[-1 - hours][4]) if len(h) > hours else None
    return {"now": cur, "d1": at(24), "d7": float(h[0][4]) if len(h) >= 150 else None,
            "pct_year": sum(v < cur for v in year) / len(year) if len(year) >= 200 else None}


# --------------------------------------------------------------------------
# 3) Flows: liquidations, spot vs futures, Coinbase premium
# --------------------------------------------------------------------------

def okx_liq_view(details: list[dict], now: datetime, minutes: int = 60) -> dict:
    """OKX BTC-USDT perp liquidations in the last `minutes`. posSide is the position that was closed.
    The feed returns its newest 100; in a cascade those may not reach back the full hour (`partial`)."""
    cutoff = now - timedelta(minutes=minutes)
    out = {"long": 0.0, "short": 0.0}
    for x in details:
        if _ms(x["ts"]) >= cutoff and x.get("posSide") in out:
            out[x["posSide"]] += float(x["sz"]) * OKX_CT_BTC * float(x["bkPx"])
    out["partial"] = len(details) >= 100 and min(_ms(x["ts"]) for x in details) > cutoff
    return out


def f_okx_liq() -> dict:
    d = _get("https://www.okx.com/api/v5/public/liquidation-orders", {"instType": "SWAP", "uly": "BTC-USDT", "state": "filled"})["data"]
    return okx_liq_view(d[0]["details"] if d else [], _now())


def f_flows() -> dict:
    """Last 24 closed hours on Binance: $ traded and net market-order buying, spot vs the perp."""
    start = int((time.time() - 25 * 3600) * 1000)
    spot, perp = klines("BTCUSDT", "1h", start).tail(24), klines("BTCUSDT", "1h", start, market="futures").tail(24)
    net = lambda df: float(((2 * df["taker_buy_base"] - df["volume"]) * df["close"]).sum())
    return {"spot_usd": float(spot["quote_volume"].sum()), "perp_usd": float(perp["quote_volume"].sum()),
            "spot_net": net(spot), "perp_net": net(perp), "last": float(spot["close"].iloc[-1]),
            "chg": float(spot["close"].iloc[-1] / spot["open"].iloc[0] - 1),
            "high": float(spot["high"].max()), "low": float(spot["low"].min())}


def f_prem() -> dict:
    """Coinbase (dollars, mostly US) vs Binance (USDT, mostly offshore). Part of any gap is just USDT
    not being worth exactly $1, so that part is taken out (`adj`)."""
    cb = float(_get("https://api.exchange.coinbase.com/products/BTC-USD/ticker")["price"])
    usdt = float(_get("https://api.exchange.coinbase.com/products/USDT-USD/ticker")["price"])
    bn = float(_spot_get("/api/v3/ticker/price", {"symbol": "BTCUSDT"})["price"])
    return {"bn": bn, "usdt": usdt, "raw": cb / bn - 1, "adj": cb / (bn * usdt) - 1}


def liq_local(now: datetime) -> dict:
    """From the live Binance feed (whales.py): BTC and all-coin liquidations in the last hour."""
    ok, why = whales.feed_text(now)
    started = whales.feed.get("started")
    mins = min(60, int((now - started).total_seconds() // 60)) if started else 0   # she may have just started
    return {"ok": ok, "why": why, "mins": mins, "btc": whales.liq_totals(60, now), "all": whales.liq_market(60, now)}


# --------------------------------------------------------------------------
# 4) Levels (from daily bars; days and weeks are UTC, weeks start on Monday)
# --------------------------------------------------------------------------

def levels_view(d: pd.DataFrame) -> dict:
    """d: daily bars indexed by the day's start (UTC), the last row being today's unfinished bar."""
    today = d.index[-1]
    done = d.iloc[:-1]
    monday = today - pd.Timedelta(days=today.dayofweek)
    last_wk = done[(done.index >= monday - pd.Timedelta(days=7)) & (done.index < monday)]
    lv = []
    if len(done):
        lv += [("yesterday's high", done["high"].iloc[-1]), ("yesterday's low", done["low"].iloc[-1])]
    if len(last_wk) == 7:
        lv += [("last week's high", last_wk["high"].max()), ("last week's low", last_wk["low"].min())]
    if monday in d.index:
        lv.append(("this week's open", d.loc[monday, "open"]))
    first = today.replace(day=1)
    if first in d.index:
        lv.append(("this month's open", d.loc[first, "open"]))
    top = d["high"].idxmax()
    return {"levels": [(n, float(p)) for n, p in lv],
            "ma200": float(done["close"].tail(200).mean()) if len(done) >= 200 else None,
            "top": float(d["high"].max()), "top_day": top.to_pydatetime(), "since": d.index[0].to_pydatetime()}


def f_levels() -> dict:
    def get():
        rows = _spot_get("/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000})
        d = pd.DataFrame([r[:5] for r in rows], columns=["t", "open", "high", "low", "close"]).astype(float)
        d.index = pd.to_datetime(d.pop("t"), unit="ms", utc=True)
        return levels_view(d)
    return _cached("levels", 10, get)


# --------------------------------------------------------------------------
# 5) Backdrop: correlations, stablecoins, dominance, sentiment
# --------------------------------------------------------------------------

def corr_view(btc: pd.Series, cross: pd.DataFrame, now: datetime) -> dict:
    """Correlation of hourly returns, BTC vs S&P futures and gold, over 7 and 30 days. Only hours in
    which the other market actually traded count (it is shut at weekends)."""
    r = np.log(btc).diff()
    out = {}
    for name in ("es", "gold"):
        if name not in cross:
            continue
        x = cross[name].dropna()
        xr = np.log(x).diff()
        both = pd.concat([r, xr.where(xr != 0)], axis=1, keys=["b", "x"]).dropna()
        for days in (7, 30):
            w = both[both.index >= now - timedelta(days=days)]
            out[f"{name}{days}"] = float(w["b"].corr(w["x"])) if len(w) >= CORR_MIN_N else None
    return out


def f_corr() -> dict:
    """Reads the cross-asset cache the hourly scan keeps fresh (no download of its own, so it can't
    collide with the scan writing that file)."""
    path = _cache_path("yahoo_cross_1h")
    if not path.exists() or time.time() - path.stat().st_mtime > 6 * 3600:
        raise RuntimeError("cross-asset cache is over 6h old (the hourly scan refreshes it)")
    btc = klines("BTCUSDT", "1h", int((time.time() - 31 * 86400) * 1000))["close"]
    return corr_view(btc, pd.read_parquet(path), _now())


def f_stables() -> dict:
    def get():
        j = _get("https://stablecoins.llama.fi/stablecoincharts/all")
        v = [float(x["totalCirculatingUSD"]["peggedUSD"]) for x in j[-31:]]
        return {"usd": v[-1], "d7": v[-1] - v[-8], "d30": v[-1] - v[0]}
    return _cached("stables", 360, get)


def f_fng() -> dict:
    def get():
        d = _get("https://api.alternative.me/fng/", {"limit": 2})["data"]
        return {"v": int(d[0]["value"]), "label": d[0]["value_classification"], "prev": int(d[1]["value"]) if len(d) > 1 else None}
    return _cached("fng", 180, get)


def f_dom() -> dict:
    return _cached("dom", 30, lambda: {"btc": float(_get("https://api.coingecko.com/api/v3/global")["data"]["market_cap_percentage"]["btc"])})


# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------

FETCHERS = {"bn": f_bn, "oi": f_oi, "ls": f_ls, "bybit": f_bybit, "okx": f_okx, "hl": f_hl, "futs": f_futs,
            "opts": f_opts, "dvol": f_dvol, "okx_liq": f_okx_liq, "flows": f_flows, "prem": f_prem,
            "levels": f_levels, "corr": f_corr, "stables": f_stables, "fng": f_fng, "dom": f_dom}
NAMES = {"bn": "Binance funding", "oi": "Binance open interest", "ls": "Binance long/short", "bybit": "Bybit",
         "okx": "OKX funding", "hl": "Hyperliquid", "futs": "Deribit futures", "opts": "Deribit options",
         "dvol": "Deribit DVOL", "okx_liq": "OKX liquidations", "flows": "Binance volume", "prem": "Coinbase premium",
         "levels": "daily bars", "corr": "correlations", "stables": "stablecoins", "fng": "Fear & Greed",
         "dom": "BTC dominance"}


def last_forecast(now: datetime) -> dict | None:
    """Amadeus's own newest 4h range forecast, if the hourly scan made one in the last 90 minutes."""
    with pl._db() as con:
        r = con.execute("SELECT created_utc, range_med, percentile FROM vol_forecasts ORDER BY created_utc DESC LIMIT 1").fetchone()
    if not r or now - datetime.fromisoformat(r[0]) > timedelta(minutes=90):
        return None
    return {"range": r[1], "pct": r[2]}


def collect(now: datetime | None = None) -> dict:
    """Everything, fetched in parallel. A source that fails or is slow is None and listed in `errors`."""
    now = now or _now()
    ex = ThreadPoolExecutor(len(FETCHERS))
    futs = {k: ex.submit(fn) for k, fn in FETCHERS.items()}
    wait(futs.values(), timeout=COLLECT_TIMEOUT)
    snap: dict = {"t": now, "errors": {}}
    for k, f in futs.items():
        try:
            snap[k] = f.result(timeout=0)
        except Exception as e:
            snap[k] = None
            snap["errors"][k] = f"{type(e).__name__}: {e}"[:120] if str(e) else "too slow"
    ex.shutdown(wait=False, cancel_futures=True)
    snap["liq"] = liq_local(now)
    for k, fn in (("forecast", lambda: last_forecast(now)), ("events", lambda: news.events_between(now, now + timedelta(hours=48)))):
        try:
            snap[k] = fn()
        except Exception as e:
            log.warning("market %s: %s", k, e)
            snap[k] = None
    snap["px"] = next((v for v in ((snap["bn"] or {}).get("index"), (snap["prem"] or {}).get("bn"),
                                   (snap["flows"] or {}).get("last")) if v), None)
    return snap


def log_snap(snap: dict) -> None:
    """One row an hour of the plain numbers, for a later one-shot test."""
    g = lambda k, f: (snap.get(k) or {}).get(f)
    opts = snap.get("opts") or {}
    liq = snap["liq"]
    d = {"oi_usd": g("oi", "usd"), "oi_chg": g("oi", "chg"), "acc_long": g("ls", "acc_long"), "top_long": g("ls", "top_long"),
         "f_bn": g("bn", "funding"), "f_bybit": g("bybit", "funding"), "f_okx": g("okx", "funding"), "f_hl": g("hl", "funding"),
         "basis_ann": g("futs", "annual"), "dvol": g("dvol", "now"), "skew": (opts.get("skew") or {}).get("skew"),
         "pc": opts.get("pc"), "pain": (opts.get("big") or {}).get("pain"), "prem_raw": g("prem", "raw"),
         "prem_adj": g("prem", "adj"), "usdt": g("prem", "usdt"), "spot_usd": g("flows", "spot_usd"),
         "perp_usd": g("flows", "perp_usd"), "spot_net": g("flows", "spot_net"), "perp_net": g("flows", "perp_net"),
         "liq_feed_ok": liq["ok"], "liq_btc_long": liq["btc"]["long"], "liq_btc_short": liq["btc"]["short"],
         "liq_all_long": liq["all"]["long"], "liq_all_short": liq["all"]["short"],
         "okx_liq_long": g("okx_liq", "long"), "okx_liq_short": g("okx_liq", "short"),
         "stables": g("stables", "usd"), "fng": g("fng", "v"), "dom": g("dom", "btc")}
    with pl._db() as con:
        con.execute(SCHEMA)
        con.execute("INSERT OR REPLACE INTO market_snap VALUES (?,?,?)",
                    (snap["t"].replace(minute=0, second=0, microsecond=0).isoformat(), snap["px"],
                     json.dumps({k: (round(v, 8) if isinstance(v, float) else v) for k, v in d.items() if v is not None})))


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

def _usd(x: float, sign: bool = False) -> str:
    a = abs(x)
    if a < 500:
        return "$0"
    s = f"${a / 1e9:.2f}B" if a >= 1e9 else f"${a / 1e6:.0f}M" if a >= 1e7 else f"${a / 1e6:.1f}M" if a >= 1e6 else f"${a / 1e3:.0f}k"
    return ("-" if x < 0 else "+" if sign else "") + s


def _in(td: timedelta) -> str:
    m = max(0, int(td.total_seconds() // 60))
    return f"{m // 1440}d {m % 1440 // 60}h" if m >= 1440 else f"{m // 60}h{m % 60:02d}"


def _positioning(s: dict) -> list[str]:
    out = []
    oi = [f"Binance {_usd(s['oi']['usd'])} ({s['oi']['chg']:+.1%} in 24h, counted in BTC)"] if s["oi"] else []
    oi += [f"{n} {_usd(s[k]['oi_usd'])}" for k, n in (("bybit", "Bybit"), ("hl", "Hyperliquid")) if s[k]]
    if oi:
        out.append("• Open contracts: " + ", ".join(oi) + ".")
    if s["ls"]:
        out.append(f"• Binance accounts: {s['ls']['acc_long']:.0%} long, {1 - s['ls']['acc_long']:.0%} short. "
                   f"Top traders, by position size: {s['ls']['top_long']:.0%} long. "
                   "(That counts people, not money: every contract has one long and one short.)")
    fund = [(n, s[k]["funding"]) for k, n in (("bn", "Binance"), ("bybit", "Bybit"), ("okx", "OKX"), ("hl", "Hyperliquid")) if s[k]]
    if fund:
        yearly = sum(f for _, f in fund) / len(fund) * 3 * 365
        nxt = f" Next payment in {_in(s['bn']['next'] - s['t'])}." if s["bn"] else ""
        out.append("• Funding per 8h: " + ", ".join(f"{n} {f:+.4%}" for n, f in fund)
                   + f". On average {yearly:+.1%} a year ({'longs pay shorts' if yearly > 0 else 'shorts pay longs'}).{nxt}")
    if s["futs"]:
        f = s["futs"]
        out.append(f"• The {f['t']:%d %b} future trades {f['basis']:+.2%} from spot ({f['annual']:+.1%} a year): "
                   "what traders pay to hold leverage until then.")
    return out


def _options(s: dict, tz) -> list[str]:
    out = []
    v = s["dvol"]
    if v:
        was = ", ".join(f"{x:.1f} {w} ago" for x, w in ((v["d1"], "a day"), (v["d7"], "a week")) if x)
        rank = f", higher than {v['pct_year']:.0%} of the past year" if v["pct_year"] is not None else ""
        out.append(f"• Implied volatility (DVOL) {v['now']:.1f}" + (f" ({was})" if was else "") + rank
                   + f". That prices moves of about ±{v['now'] / math.sqrt(365):.1f}% a day over the next month, "
                     f"or a ~{v['now'] / math.sqrt(365 * 6) * RANGE_TO_SIGMA:.1f}% range per 4h.")
    fc = s.get("forecast")
    if fc and v:
        out.append(f"• My own forecast for the next 4h: {fc['range']:.1f}% range"
                   + (f", busier than {fc['pct']:.0%} of my past year" if fc["pct"] is not None else "")
                   + ". (Options usually price in a bit more than actually happens.)")
    o = s["opts"]
    if o and o["skew"]:
        k = o["skew"]
        read = ("priced about the same" if abs(k["skew"]) < SKEW_EVEN else
                "protection against a fall costs more" if k["skew"] > 0 else "bets on a rise cost more")
        out.append(f"• {k['t']:%d %b} options: 25-delta puts {k['put_iv']:.1f} vs calls {k['call_iv']:.1f} implied vol - {read}.")
    if o and o["big"]:
        b = o["big"]
        pc = f"Open puts per call: {o['pc']:.2f}. " if o["pc"] else ""
        size = f"{b['oi_btc'] / 1e3:.0f}k BTC" + (f" ({_usd(b['usd'])})" if b["usd"] else "")
        held = ", ".join(f"{k:,.0f} {n}" for k, n in ((b["put"], "puts"), (b["call"], "calls")) if k)
        out.append(f"• {pc}Biggest expiry ahead: {b['t'].astimezone(tz):%a %d %b %H:%M} - {size}. "
                   f"Most held there: {held}. Max pain {b['pain']:,.0f}.")
    return out


def _flows(s: dict) -> list[str]:
    out = []
    lq = s["liq"]
    pair = lambda d: f"longs {_usd(d['long'])} / shorts {_usd(d['short'])}"
    venues = []
    span = "" if lq["mins"] >= 60 else f" in the {lq['mins']} min I've been listening"
    if lq["ok"]:
        venues.append(f"Binance {pair(lq['btc'])}{span} (at least: its feed is capped at one a second)")
    if s["okx_liq"]:
        venues.append(f"OKX {pair(s['okx_liq'])}" + (" (at least)" if s["okx_liq"]["partial"] else ""))
    if venues:
        out.append("• BTC liquidations, last hour: " + "; ".join(venues) + ".")
    if not lq["ok"]:
        out.append(f"• ⚠ Binance liquidation feed: {lq['why']}.")
    elif lq["all"]["long"] + lq["all"]["short"] > 0:
        a = lq["all"]
        tot = a["long"] + a["short"]
        top = ", ".join(f"{c} {_usd(u)}" for c, u in a["top"])
        out.append(f"• All coins on Binance, last {'hour' if lq['mins'] >= 60 else str(lq['mins']) + ' min'}: "
                   f"{_usd(tot)} liquidated, {a['long'] / tot:.0%} of it longs. Most: {top}.")
    f = s["flows"]
    if f:
        out.append(f"• 24h volume on Binance: futures {_usd(f['perp_usd'])}, spot {_usd(f['spot_usd'])} "
                   f"({f['perp_usd'] / (f['perp_usd'] + f['spot_usd']):.0%} futures). Net market buying: "
                   f"futures {_usd(f['perp_net'], True)}, spot {_usd(f['spot_net'], True)}.")
    p = s["prem"]
    if p:
        out.append(f"• Coinbase vs Binance: {p['raw']:+.2%}. USDT is at ${p['usdt']:.4f}, so the real gap is {p['adj']:+.2%} "
                   "(above 0 = US buyers paying more).")
    return out


def _levels(s: dict) -> list[str]:
    lv, px = s["levels"], s["px"]
    if not lv or not px:
        return []
    rows = sorted(lv["levels"] + [("now", px)], key=lambda r: -r[1])
    out = [f"▸ {p:,.0f}  now" if n == "now" else f"  {p:,.0f}  {n} ({p / px - 1:+.1%})" for n, p in rows]
    avg = f"200-day average {lv['ma200']:,.0f} (price is {px / lv['ma200'] - 1:+.1%} from it). " if lv["ma200"] else ""
    return out + [f"• {avg}Highest since {lv['since']:%b %Y}: {lv['top']:,.0f} on {lv['top_day']:%d %b %Y} "
                  f"({px / lv['top'] - 1:+.1%})."]


def _backdrop(s: dict) -> list[str]:
    out, bits = [], []
    if s["stables"]:
        st = s["stables"]
        bits.append(f"Stablecoins {_usd(st['usd'])} ({_usd(st['d7'], True)} in 7 days, {_usd(st['d30'], True)} in 30)")
    if s["dom"]:
        bits.append(f"BTC is {s['dom']['btc']:.1f}% of the crypto market")
    if s["fng"]:
        g = s["fng"]
        bits.append(f"Fear & Greed {g['v']} ({g['label']}" + (f", {g['prev']} yesterday" if g["prev"] is not None else "") + ")")
    if bits:
        out.append("• " + ". ".join(bits) + ".")
    c = s["corr"] or {}
    pairs = [f"{n} {c[k + '7']:+.2f} over 7 days, {c[k + '30']:+.2f} over 30"
             for k, n in (("es", "S&P futures"), ("gold", "gold")) if c.get(k + "7") is not None and c.get(k + "30") is not None]
    if pairs:
        out.append("• BTC's hourly moves vs " + "; ".join(pairs) + ". (1 = together, 0 = unrelated, -1 = opposite)")
    return out


def _coming(s: dict, tz) -> list[str]:
    now, items = s["t"], []
    if s["bn"]:
        items.append((s["bn"]["next"], f"funding payment (Binance {s['bn']['funding']:+.4%})"))
    big = (s["opts"] or {}).get("big")
    if big:
        items.append((big["t"], f"options expiry, {big['oi_btc'] / 1e3:.0f}k BTC"))
    for e in s.get("events") or []:
        items.append((e["time"], "US data: " + e["title"]))
    when = lambda t: f"{t.astimezone(tz):%a %d %b %H:%M}" if t - now > timedelta(hours=24) else f"{t.astimezone(tz):%a %H:%M}"
    return [f"• {when(t)} (in {_in(t - now)}): {what}" for t, what in sorted(items, key=lambda i: i[0]) if t > now]


def text(s: dict, tz, title: str = "Market snapshot") -> str:
    head = [f"📊 {title} · {s['t'].astimezone(tz):%a %d %b %H:%M %Z}"]
    if s["px"]:
        f = s["flows"]
        head.append(f"BTC {s['px']:,.0f}" + (f" · {f['chg']:+.1%} in 24h (low {f['low']:,.0f}, high {f['high']:,.0f})" if f else ""))
    else:
        head.append("⚠ I couldn't get a price: the exchanges aren't answering.")
    lines = head
    for name, body in (("POSITIONING (futures)", _positioning(s)), ("OPTIONS (Deribit)", _options(s, tz)),
                       ("FLOWS", _flows(s)), ("LEVELS (UTC days, vs now)", _levels(s)), ("BACKDROP", _backdrop(s)),
                       ("COMING UP", _coming(s, tz))):
        if body:
            lines += ["", name] + body
    lines += ["", "Description only: none of this is tested, and it never changes my call."]
    if s["errors"]:
        lines.append("⚠ Not answering: " + ", ".join(NAMES[k] for k in s["errors"]) + ".")
    return "\n".join(lines)
