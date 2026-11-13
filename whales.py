"""Whale watch: big market buying/selling and liquidation cascades on Binance BTCUSDT, and
order-book walls on Binance, Coinbase, OKX (spot + perp), Kraken, Bitfinex and Bitstamp.

None of this is a trade signal - nothing here is backtested. Bursts and liquidations are
things that already happened (often they ARE the move); walls are often fake (placed to
scare people, then pulled). Every event is logged in `whale_events`, so whether price
tended to continue or reverse afterwards can be checked later - once, with a stated
hypothesis, not by tuning thresholds until it looks good.

Who placed an order is never visible: Binance shows sizes, not names.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd

import paper_log as pl
from config import WHALE_BURST_USD, WHALE_LIQ_USD, WHALE_WALL_PERP_USD, WHALE_WALL_USD
from data import _session, _spot_get, klines

log = logging.getLogger("whales")

WINDOW_MIN = 5            # bursts and liquidations are summed over this many minutes
REPEAT_MIN = 30           # same kind + direction is held back this long, unless twice as big
WALL_BAND = 50.0          # $ price band a wall is measured over (big orders get split over nearby ticks)
WALL_NEAR = 0.0015        # new walls must be 0.15%+ from price: normal liquidity sits closer and churns
WALL_MIN_STANDING = 10    # minutes a wall must stand before it's reported (most fakes are pulled sooner)
WALL_MAX_PER_HOUR = 3     # new-wall pings per hour; more wait their turn (the thresholds are lightly calibrated)
CLUSTER_BAND = 100.0      # $ price band liquidity clusters are added up over, across the spot books
CLUSTER_N = 3             # clusters shown per side
DEPTH_STALE_MIN = 5       # a book reading older than this is not shown
LIQ_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"

SCHEMA = """CREATE TABLE IF NOT EXISTS whale_events (
    t TEXT, kind TEXT, side TEXT, usd REAL, price REAL, detail TEXT)"""
DEPTH_SCHEMA = """CREATE TABLE IF NOT EXISTS whale_depth (
    t TEXT, price REAL, spot_n INTEGER, spot_bid REAL, spot_ask REAL, perp_bid REAL, perp_ask REAL, clusters TEXT)"""

state: dict = {"net_15m": None, "price": None, "venues": {}, "depth": None}   # venues: name -> last error, or None if OK
_walls: dict = {}                       # (venue, side, band) -> {"first", "usd", "max", "alerted"}
_liqs: deque = deque(maxlen=20000)      # (time, side, usd) - side "long" = a long was force-sold
_wall_pings: deque = deque(maxlen=50)   # times of new-wall pings, for WALL_MAX_PER_HOUR
feed: dict = {"connected": False, "since": None, "error": None}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _held_back(kind: str, side: str, usd: float) -> bool:
    """One ping per kind + direction per REPEAT_MIN, unless the new one is at least twice as big."""
    k = f"whale_last_{kind}_{side}"
    last = pl.get_kv(k)
    now = _now()
    if last:
        t, prev = last.split("|")
        if now - datetime.fromisoformat(t) < timedelta(minutes=REPEAT_MIN) and usd < 2 * float(prev):
            return True
    pl.set_kv(k, f"{now.isoformat()}|{usd}")
    return False


# --------------------------------------------------------------------------
# 1) Market buying/selling bursts
# --------------------------------------------------------------------------

def _net_flow(df: pd.DataFrame) -> pd.Series:
    """Net market-order dollars per minute: aggressive buys minus aggressive sells."""
    return (2 * df["taker_buy_base"] - df["volume"]) * df["close"]


def check_burst(net: pd.Series, px: pd.DataFrame) -> dict | None:
    """net: perp + spot net flow per closed minute (index = bar close). px: perp 1m bars."""
    w = net.iloc[-WINDOW_MIN:]
    if len(w) < WINDOW_MIN:
        return None
    total = float(w.sum())
    side = "buy" if total > 0 else "sell"
    if abs(total) < WHALE_BURST_USD or _held_back("burst", side, abs(total)):
        return None
    bars = px.reindex(w.index)
    big_t = w.abs().idxmax()
    return {"kind": "burst", "side": side, "usd": abs(total),
            "t0": w.index[0] - timedelta(minutes=1), "t1": w.index[-1],
            "p0": float(bars["open"].iloc[0]), "p1": float(bars["close"].iloc[-1]),
            "big_t": big_t, "big_usd": float(w[big_t])}


# --------------------------------------------------------------------------
# 2) Liquidations (live websocket; Binance keeps no history to download)
# --------------------------------------------------------------------------

def on_liquidation(o: dict) -> None:
    if o.get("s") != "BTCUSDT":
        return
    side = "long" if o["S"] == "SELL" else "short"   # a liquidated long is force-SOLD
    _liqs.append((datetime.fromtimestamp(o["T"] / 1000, timezone.utc), side, float(o["ap"]) * float(o["z"])))


async def liquidation_listener() -> None:
    """Runs for the bot's lifetime. Binance sends at most one liquidation per second per
    symbol, so totals are a lower bound in a real cascade."""
    import websockets
    backoff = 5
    while True:
        try:
            async with websockets.connect(LIQ_URL, open_timeout=15, ping_interval=60) as ws:
                feed.update(connected=True, since=_now(), error=None)
                backoff = 5
                async for raw in ws:
                    on_liquidation(json.loads(raw)["o"])
            feed["error"] = "closed by Binance"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            feed["error"] = f"{type(e).__name__}: {e}"[:150]
            log.warning("liquidation feed: %s (retry in %ss)", feed["error"], backoff)
        feed["connected"] = False
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 300)


def liq_totals(minutes: float, now: datetime | None = None) -> dict:
    cutoff = (now or _now()) - timedelta(minutes=minutes)
    out = {"long": 0.0, "short": 0.0}
    for t, side, usd in list(_liqs):     # copy: the listener appends from the event loop
        if t >= cutoff:
            out[side] += usd
    return out


def check_liqs(now: datetime | None = None) -> list[dict]:
    return [{"kind": "liq", "side": side, "usd": usd}
            for side, usd in liq_totals(WINDOW_MIN, now).items()
            if usd >= WHALE_LIQ_USD and not _held_back("liq", side, usd)]


# --------------------------------------------------------------------------
# 3) Order-book walls, on every big exchange whose free book feed reaches ~1% from price.
#    Left out: Bybit (its feed only reaches ~0.25%) and the Binance perp (~0.16%).
#    Perp books are 10-30x deeper than spot books, so they get their own threshold.
# --------------------------------------------------------------------------

def _fetch(url: str, params: dict):
    """One quick try: a slow exchange is skipped this minute rather than holding up the rest."""
    r = _session.get(url, params=params, timeout=8)
    r.raise_for_status()
    return r.json()


def _levels(rows, k: float = 1.0) -> list[tuple[float, float]]:
    return [(float(r[0]), float(r[1]) * k) for r in rows]


def _okx(inst: str, k: float):
    d = _fetch("https://www.okx.com/api/v5/market/books-full", {"instId": inst, "sz": 5000})["data"][0]
    return _levels(d["bids"], k), _levels(d["asks"], k)


def _kraken():
    d = _fetch("https://api.kraken.com/0/public/Depth", {"pair": "XBTUSD", "count": 500})
    book = next(iter(d["result"].values()))
    return _levels(book["bids"]), _levels(book["asks"])


def _bitfinex():
    rows = _fetch("https://api-pub.bitfinex.com/v2/book/tBTCUSD/P0", {"len": 250})   # [price, count, amount]
    return [(p, a) for p, n, a in rows if a > 0], [(p, -a) for p, n, a in rows if a < 0]


VENUES = {   # name: (fetch -> (bids, asks) as [(price, BTC)], is_perp)
    "Binance": (lambda: (lambda d: (_levels(d["bids"]), _levels(d["asks"])))(
        _spot_get("/api/v3/depth", {"symbol": "BTCUSDT", "limit": 5000})), False),
    "Coinbase": (lambda: (lambda d: (_levels(d["bids"]), _levels(d["asks"])))(
        _fetch("https://api.exchange.coinbase.com/products/BTC-USD/book", {"level": 2})), False),
    "OKX": (lambda: _okx("BTC-USDT", 1.0), False),
    "OKX perp": (lambda: _okx("BTC-USDT-SWAP", 0.01), True),     # perp sizes are contracts of 0.01 BTC
    "Kraken": (_kraken, False),
    "Bitfinex": (_bitfinex, False),
    "Bitstamp": (lambda: (lambda d: (_levels(d["bids"]), _levels(d["asks"])))(
        _fetch("https://www.bitstamp.net/api/v2/order_book/btcusd/", {})), False),
}
WALL_REACH = 0.01   # look within 1% of price: every venue above shows at least that much


def fetch_books() -> dict:
    """All venues in parallel. A venue that fails is left out this minute (see state["venues"])."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(VENUES)) as ex:
        futs = {name: ex.submit(fn) for name, (fn, _) in VENUES.items()}
    books = {}
    for name, f in futs.items():
        try:
            bids, asks = f.result()
            if not bids or not asks:
                raise ValueError("empty book")
            books[name] = (sorted(bids, key=lambda x: -x[0]), sorted(asks, key=lambda x: x[0]))
            state["venues"][name] = None
        except Exception as e:
            state["venues"][name] = f"{type(e).__name__}: {e}"[:120]
    return books


def _bands(levels: list, mid: float) -> pd.Series:
    """$ resting per $50 price band, within WALL_REACH of price (Coinbase and Bitstamp send
    their whole book, including joke orders at $1M a coin)."""
    near = [(p, q) for p, q in levels if abs(p - mid) / mid <= WALL_REACH]
    if not near:
        return pd.Series(dtype=float)
    df = pd.DataFrame(near, columns=["p", "q"])
    return (df["p"] * df["q"]).groupby((df["p"] // WALL_BAND) * WALL_BAND).sum()


def check_walls(books: dict, lo: float, hi: float, now: datetime | None = None) -> list[dict]:
    """books: venue -> (bids, asks). lo/hi: BTC's price range since the last check, to tell
    whether a vanished wall was reached (hit) or pulled before price got there."""
    now = now or _now()
    events = []
    for venue, (bids, asks) in books.items():
        thr = WHALE_WALL_PERP_USD if VENUES[venue][1] else WHALE_WALL_USD
        mid = (bids[0][0] + asks[0][0]) / 2
        seen, view = {}, {}
        for side, levels in (("bid", bids), ("ask", asks)):
            bands = _bands(levels, mid)
            view[side] = (bands.index.min(), bands.index.max()) if len(bands) else (mid, mid)
            for band, usd in bands.items():
                key = (venue, side, float(band))
                if key in _walls:      # tracked walls survive small dips (hysteresis)
                    if usd >= 0.7 * thr:
                        seen[key] = usd
                elif usd >= thr and abs(band + WALL_BAND / 2 - mid) / mid >= WALL_NEAR:
                    seen[key] = usd
        for key, usd in seen.items():
            w = _walls.setdefault(key, {"first": now, "max": 0.0, "alerted": False})
            w["usd"], w["max"] = usd, max(w["max"], usd)
            if not w["alerted"] and now - w["first"] >= timedelta(minutes=WALL_MIN_STANDING):
                if sum(now - t < timedelta(hours=1) for t in _wall_pings) >= WALL_MAX_PER_HOUR:
                    continue   # over the hourly cap: try again next minute
                _wall_pings.append(now)
                w["alerted"] = True
                events.append({"kind": "wall", "venue": venue, "side": key[1], "band": key[2], "usd": usd,
                               "mid": mid, "mins": int((now - w["first"]).total_seconds() // 60)})
        # Only venues that answered this minute: a failed fetch says nothing about its walls.
        for key in [k for k in _walls if k[0] == venue and k not in seen]:
            w = _walls.pop(key)
            _, side, band = key
            lo_v, hi_v = view[side]
            if not w["alerted"] or band < lo_v or band > hi_v:
                continue   # never reported, or price moved away so it's out of view - not pulled
            reached = lo <= band + WALL_BAND if side == "bid" else hi >= band
            events.append({"kind": "wall_gone", "venue": venue, "side": side, "band": band, "usd": w["max"],
                           "mid": mid, "reached": reached, "mins": int((now - w["first"]).total_seconds() // 60)})
    return events


# --------------------------------------------------------------------------
# 4) Depth: how much is resting on each side, and where it's bunched up (liquidity clusters).
#    A description of the books right now, shown in /whales and under a call. Never pinged,
#    never fed to the model: resting orders can be cancelled in a millisecond.
# --------------------------------------------------------------------------

def depth_stats(books: dict, now: datetime | None = None) -> dict | None:
    """books: venue -> (bids, asks), both sorted from the best price outwards.
    Totals are everything within WALL_REACH of each venue's own price. Clusters add up the spot
    books per $CLUSTER_BAND of price, leaving out the first WALL_NEAR (ordinary churn); the
    futures book is kept out of them because it is 10-30x deeper and would drown the rest."""
    if not books:
        return None
    tot = {"spot": {"bid": 0.0, "ask": 0.0}, "perp": {"bid": 0.0, "ask": 0.0}}
    bands = {"bid": {}, "ask": {}}   # band -> {venue: $}
    mids = []
    for venue, (bids, asks) in books.items():
        kind = "perp" if VENUES[venue][1] else "spot"
        mid = (bids[0][0] + asks[0][0]) / 2
        if kind == "spot":
            mids.append(mid)
        for side, levels in (("bid", bids), ("ask", asks)):
            for p, q in levels:
                d = abs(p - mid) / mid
                if d > WALL_REACH:
                    break
                tot[kind][side] += p * q
                if kind == "spot" and d >= WALL_NEAR:
                    b = bands[side].setdefault((p // CLUSTER_BAND) * CLUSTER_BAND, {})
                    b[venue] = b.get(venue, 0.0) + p * q
    if not mids:
        return None

    def top(side):   # "venues" = exchanges holding 10%+ of the cluster, so one stray order doesn't count
        rows = [{"band": b, "usd": sum(v.values()), "venues": sum(x >= 0.1 * sum(v.values()) for x in v.values())}
                for b, v in bands[side].items()]
        return sorted(rows, key=lambda c: -c["usd"])[:CLUSTER_N]
    return {"t": now or _now(), "mid": float(pd.Series(mids).median()), "spot_n": len(mids),
            "spot": tot["spot"], "perp": tot["perp"] if len(mids) < len(books) else None,
            "clusters": {"bid": top("bid"), "ask": top("ask")}}


def _log_depth(d: dict) -> None:
    c = d["clusters"]
    with pl._db() as con:
        con.execute(DEPTH_SCHEMA)
        con.execute("INSERT INTO whale_depth VALUES (?,?,?,?,?,?,?,?)",
                    (d["t"].isoformat(), d["mid"], d["spot_n"], d["spot"]["bid"], d["spot"]["ask"],
                     d["perp"]["bid"] if d["perp"] else None, d["perp"]["ask"] if d["perp"] else None,
                     json.dumps({s: [[x["band"], round(x["usd"])] for x in c[s]] for s in c})))


def wall_record(days: int = 7) -> dict | None:
    """What became of the walls I reported: hit (price got there) or pulled first."""
    since = (_now() - timedelta(days=days)).isoformat()
    try:
        with pl._db() as con:
            rows = con.execute("SELECT kind, detail FROM whale_events WHERE kind IN ('wall','wall_gone') AND t >= ?",
                               (since,)).fetchall()
    except Exception:
        return None   # no table yet: nothing has been logged
    gone = [json.loads(d).get("reached") for k, d in rows if k == "wall_gone"]
    return {"days": days, "reported": sum(k == "wall" for k, _ in rows), "hit": sum(bool(r) for r in gone),
            "pulled": sum(not r for r in gone)}


def fresh_depth(now: datetime | None = None) -> dict | None:
    d = state.get("depth")
    return d if d and (now or _now()) - d["t"] <= timedelta(minutes=DEPTH_STALE_MIN) else None


def _share(a: float, b: float) -> str:
    return f"{a / (a + b) * 100:.0f}% / {b / (a + b) * 100:.0f}%" if a + b else "-"


def _cluster(c: dict, mid: float) -> str:
    dist = (c["band"] + CLUSTER_BAND / 2) / mid - 1
    return (f"{c['band']:,.0f}-{c['band'] + CLUSTER_BAND:,.0f} {_m(c['usd'])} "
            f"({abs(dist) * 100:.1f}% {'below' if dist < 0 else 'above'}, {c['venues']} exch)")


def depth_lines(d: dict) -> list[str]:
    s, c = d["spot"], d["clusters"]
    lines = [f"Order books now (BTC {d['mid']:,.0f}, everything within {WALL_REACH * 100:.0f}% of price, "
             f"{d['spot_n']} spot exchanges):",
             f"• Buy orders {_m(s['bid'])} vs sell orders {_m(s['ask'])} ({_share(s['bid'], s['ask'])})."
             + (f" OKX futures: {_m(d['perp']['bid'])} vs {_m(d['perp']['ask'])} "
                f"({_share(d['perp']['bid'], d['perp']['ask'])})." if d["perp"] else "")]
    for side, word in (("bid", "buy"), ("ask", "sell")):
        if c[side]:
            lines.append(f"• Biggest {word} clusters: " + "; ".join(_cluster(x, d["mid"]) for x in c[side]) + ".")
    return lines


def call_section() -> str:
    """A short note under a call. Context only: it never changes the call."""
    d = fresh_depth()
    if not d:
        return ""
    s, c = d["spot"], d["clusters"]
    near = lambda side: _cluster(c[side][0], d["mid"]) if c[side] else "none"
    return (f"🧱 Order books ({d['spot_n']} spot exchanges, within {WALL_REACH * 100:.0f}%): buy {_m(s['bid'])} vs "
            f"sell {_m(s['ask'])} ({_share(s['bid'], s['ask'])}).\n"
            f"Biggest buy cluster {near('bid')}; biggest sell cluster {near('ask')}.\n"
            f"Resting orders can be pulled at any moment - context, not part of the call. /whales for more.")


# --------------------------------------------------------------------------
# Run all three (every minute, from a thread: it makes blocking network calls)
# --------------------------------------------------------------------------

def check_all() -> list[dict]:
    start = int((time.time() - 20 * 60) * 1000)
    perp = klines("BTCUSDT", "1m", start, market="futures")
    spot = klines("BTCUSDT", "1m", start)
    net = _net_flow(perp).add(_net_flow(spot), fill_value=0).sort_index()
    price = float(perp["close"].iloc[-1])
    state.update(net_15m=float(net.iloc[-15:].sum()), price=price)

    events = []
    ev = check_burst(net, perp)
    if ev:
        events.append(ev)
    for ev in check_liqs():
        ev["price"] = price
        ev["move_10m"] = price / float(perp["close"].iloc[-11]) - 1 if len(perp) > 10 else None
        events.append(ev)
    recent = spot.iloc[-2:]
    books = fetch_books()
    events += check_walls(books, float(recent["low"].min()), float(recent["high"].max()))
    state["depth"] = depth_stats(books)
    if state["depth"]:
        _log_depth(state["depth"])   # one row a minute, so "did the heavier side matter?" can be tested later
    _log_events(events, price)
    return events


def _log_events(events: list[dict], price: float) -> None:
    if not events:
        return
    with pl._db() as con:
        con.execute(SCHEMA)
        con.executemany("INSERT INTO whale_events VALUES (?,?,?,?,?,?)",
                        [(_now().isoformat(), e["kind"], e["side"], e["usd"], price,
                          json.dumps({k: v for k, v in e.items() if k not in ("kind", "side", "usd")},
                                     default=str)) for e in events])


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------

def _m(usd: float) -> str:
    return f"${abs(usd) / 1e6:,.1f}M"


def format_event(ev: dict, name: str, tz) -> str:
    k = ev["kind"]
    if k == "burst":
        word = "BUYING" if ev["side"] == "buy" else "SELLING"
        verb = "market-bought than market-sold" if ev["side"] == "buy" else "market-sold than market-bought"
        t = lambda x: x.astimezone(tz).strftime("%H:%M")
        return (f"🐋 {name}: whale {word} on Binance\n"
                f"{_m(ev['usd'])} more was {verb} in {WINDOW_MIN} min ({t(ev['t0'])}-{t(ev['t1'])}, perp + spot)\n"
                f"BTC {ev['p0']:,.0f} → {ev['p1']:,.0f} ({(ev['p1'] / ev['p0'] - 1) * 100:+.2f}%) · "
                f"biggest minute {'+' if ev['big_usd'] > 0 else '-'}{_m(ev['big_usd'])} at {t(ev['big_t'])}\n\n"
                f"This already happened - often it's the move itself, not a warning before it. Not a trade signal.")
    if k == "liq":
        who, forced = ("longs", "selling") if ev["side"] == "long" else ("shorts", "buying")
        move = f" ({ev['move_10m'] * 100:+.1f}% in 10 min)" if ev.get("move_10m") is not None else ""
        return (f"💥 {name}: {who} liquidated\n"
                f"At least {_m(ev['usd'])} of leveraged {who} force-closed on Binance in {WINDOW_MIN} min. "
                f"BTC {ev['price']:,.0f}{move}.\n\n"
                f"Forced {forced} can push price further for a few minutes, and it often snaps back after. "
                f"(Binance reports at most one liquidation per second, so the real total is higher.)")
    side = "buy" if ev["side"] == "bid" else "sell"
    band = f"{ev['band']:,.0f}-{ev['band'] + WALL_BAND:,.0f}"
    venue = ev["venue"] + (" (futures)" if VENUES[ev["venue"]][1] else "")
    if k == "wall":
        dist = (ev["band"] + WALL_BAND / 2) / ev["mid"] - 1
        return (f"🧱 {name}: big {side} wall on {venue}\n"
                f"{_m(ev['usd'])} of {side} orders at {band} "
                f"({abs(dist) * 100:.2f}% {'below' if dist < 0 else 'above'} price), standing for {ev['mins']} min.\n\n"
                f"Walls are often pulled before price gets there (spoofing) - treat it as a maybe. "
                f"I'll tell you whether it gets hit or pulled.")
    if ev["reached"]:
        return (f"🧱 {name}: the {band} {side} wall on {venue} ({_m(ev['usd'])}) is gone - price reached it, "
                f"so it was hit (or pulled at the last second). Stood {ev['mins']} min.")
    return (f"🧱 {name}: the {band} {side} wall on {venue} ({_m(ev['usd'])}) was pulled before price got there "
            f"(BTC {ev['mid']:,.0f}). Stood {ev['mins']} min. That's what a spoof looks like.")


def status_text(on: bool, tz) -> str:
    now = _now()
    lines = [f"🐋 Whale watch is {'ON' if on else 'OFF'} (/whales {'off' if on else 'on'}). I ping:", ""]
    net = state["net_15m"]
    lines.append(f"• Market buy/sell bursts on Binance: {_m(WHALE_BURST_USD)}+ net in {WINDOW_MIN} min (perp + spot)."
                 + (f" Last 15 min: {'+' if net >= 0 else '-'}{_m(net)}." if net is not None else ""))
    h = liq_totals(60, now)
    if feed["connected"]:
        conn = f"live feed connected {int((now - feed['since']).total_seconds() // 60)} min"
    else:
        conn = f"⚠ live feed down ({feed['error'] or 'not started'})"
    lines.append(f"• Liquidations on Binance: {_m(WHALE_LIQ_USD)}+ of longs or shorts in {WINDOW_MIN} min. "
                 f"Last hour: longs {_m(h['long'])}, shorts {_m(h['short'])}. {conn}.")
    watching = sorted(_walls.items(), key=lambda kv: -kv[1]["usd"])[:4]
    w = "; ".join(f"{ven} {'buy' if s == 'bid' else 'sell'} {b:,.0f} ({_m(v['usd'])}, "
                  f"{int((now - v['first']).total_seconds() // 60)} min)" for (ven, s, b), v in watching)
    spot = ", ".join(n for n, (_, perp) in VENUES.items() if not perp)
    perps = ", ".join(n for n, (_, perp) in VENUES.items() if perp)
    lines.append(f"• Walls: {_m(WHALE_WALL_USD)}+ in one ${WALL_BAND:.0f} band on {spot}; {_m(WHALE_WALL_PERP_USD)}+ on "
                 f"{perps} (futures books are far deeper). Within 1% of price, standing {WALL_MIN_STANDING}+ min. "
                 + (f"Watching: {w}." if w else "None right now."))
    down = {n: e for n, e in state["venues"].items() if e}
    if down:
        lines.append("  ⚠ Not answering last check: " + "; ".join(f"{n} ({e[:60]})" for n, e in down.items()))
    d = fresh_depth(now)
    lines += [""] + (depth_lines(d) if d else ["Order books: no fresh reading yet (I read them every minute)."])
    r = wall_record()
    if r and r["reported"]:
        lines.append(f"• Walls I reported in the last {r['days']} days: {r['reported']} - {r['hit']} were hit, "
                     f"{r['pulled']} were pulled before price got there.")
    lines.append("  More buy orders than sell orders isn't a reason to expect a rise: I haven't tested it, and "
                 "the orders can vanish. Clusters are where price may slow down or bounce, if they're still there.")
    lines += ["", "None of these are trade signals: bursts and liquidations already happened, and walls are often "
                  "fake. I log them all so we can check later whether price tended to follow."]
    return "\n".join(lines)
