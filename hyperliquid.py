"""Hyperliquid top traders: who's winning on Hyperliquid lately, what they hold, and when they
open or close something big.

Hyperliquid is an exchange that runs on its own public blockchain, so every account's positions
are public (Binance only shows sizes, never who). Each day the 20 accounts with the best 30-day
profit (after filters) become the list; every minute their positions are polled.

Nothing here is backtested: the leaderboard only shows today's rankings, so "do last month's
winners keep winning?" has no history to test against. Three paper wallets test it forward,
each rule fixed up front (don't tune them on their results):
- mirror (here, kv `hl_mirror`): holds the top 20's average BTC exposure right now (each
  trader's BTC position / their account, flat = 0), rebalancing when it moves 25%+;
- copy (paper_log source 'copy'): every hour, the top 20's BTC majority as a 4h trade;
- follow (source 'follow'): a 4h trade with a top trader's fresh big BTC open/add.
Copy and follow use the same entry, stops, leverage and fees as Amadeus's own lean, so only
the choice of side differs. None of it ever changes the model's call (CLAUDE.md rules 3 and 4).
Daily leaderboard snapshots (hl_snapshots) and hourly BTC positions (hl_positions) build the
history for a proper test later.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import paper_log as pl
from config import FEE_ROUND_TRIP, HL_DEXES, HL_PING_ALT_USD, HL_PING_USD, HL_REFRESH_H, HL_TOP_N, START_EQUITY

log = logging.getLogger("hyperliquid")

INFO_URL = "https://api.hyperliquid.xyz/info"
LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"   # ~40MB, all accounts
EXPLORER = "https://app.hyperliquid.xyz/explorer/address/"
MAX_TURNOVER = 100        # 30-day volume / account value above this = market maker, kept off the list
MIN_POS_USD = 100_000    # a BTC position smaller than this counts as flat for the consensus
CONSENSUS_SHARE = 0.6     # the copy wallet trades only if 60%+ of the traders holding BTC agree...
CONSENSUS_MIN = 3         # ...and at least this many hold BTC at all
STALE_MIN = 10            # positions older than this aren't used for a call or a copy trade
WORKERS = 5               # parallel requests (Hyperliquid allows ~600 of these a minute per IP)

SNAPSHOT_N = 1000         # leaderboard rows saved per day (best by 30-day pnl, before any filter)
MAP_ACCOUNTS = 1000       # biggest accounts by value kept for the liquidation map (liqmap.py)
MIRROR_BAND = 0.25        # the mirror wallet rebalances when the top 20's exposure moves 25%+ from its own...
MIRROR_MIN_STEP = 0.05    # ...and by at least 0.05x of the wallet (so tiny exposures don't churn fees)

SCHEMA = """CREATE TABLE IF NOT EXISTS hl_events (
    t TEXT, addr TEXT, rank INTEGER, coin TEXT, kind TEXT, side TEXT, usd REAL, price REAL, detail TEXT)"""
SNAP_SCHEMA = """CREATE TABLE IF NOT EXISTS hl_snapshots (
    day TEXT, rank INTEGER, addr TEXT, name TEXT, acct REAL, pnl_day REAL, pnl_week REAL, pnl_month REAL,
    pnl_all REAL, roi_month REAL, vlm_month REAL, vlm_all REAL)"""
POS_SCHEMA = """CREATE TABLE IF NOT EXISTS hl_positions (
    t TEXT, addr TEXT, acct REAL, btc_usd REAL, entry REAL, upnl REAL, lev REAL)"""   # hourly, the tracked 20
MIRROR_SCHEMA = """CREATE TABLE IF NOT EXISTS hl_mirror (
    t TEXT, price REAL, x_from REAL, x_to REAL, equity REAL, fee REAL, target REAL)"""   # every rebalance

state: dict = {"positions": {}, "accts": {}, "polled": None, "errors": {}}   # addr -> {coin: position} / account $
_pinged: dict = {}     # (addr, coin) -> signed $ size at the last ping (or when first seen: seeded silently)
_pinged_t: dict = {}   # (addr, coin) -> when that was: exit prices are averaged over the fills since then
_seeded: set = set()   # addresses whose positions have been seen at least once since startup
LIQ_SHOWN = 4                # nearest liquidation prices listed per side in /hl
LIQ_NEAR = 0.10              # "close" = within 10% of price
FILL_PAGES = 3         # up to 6,000 fills since the last ping (busy traders fill thousands an hour, mostly other coins)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _post(body: dict, timeout: float = 10):
    r = requests.post(INFO_URL, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------
# The list: top 20 by 30-day profit, refreshed daily
# --------------------------------------------------------------------------

def _windows(row: dict) -> dict:
    return {name: {k: float(v) for k, v in perf.items()} for name, perf in row["windowPerformances"]}


def pick_top(rows: list[dict], n: int = HL_TOP_N) -> list[dict]:
    """Best 30-day profit, skipping accounts that look like they didn't earn it trading:
    - losing all-time (one lucky month on top of a losing record),
    - under $10M traded in 30 days, or more profit than volume (a deposit or transfer, not trades),
    - under $1M in the account,
    - traded over MAX_TURNOVER x the account in 30 days: market makers (e.g. Auros; one at 424x on
      2026-10-29). Their position comes from quoting both sides, not a view on direction."""
    out = []
    for r in sorted(rows, key=lambda r: -float(_windows(r)["month"]["pnl"])):
        w = _windows(r)
        m, acct = w["month"], float(r["accountValue"])
        if w["allTime"]["pnl"] <= 0 or m["vlm"] < 10e6 or m["pnl"] > m["vlm"] or acct < 1e6 \
                or m["vlm"] > MAX_TURNOVER * acct:
            continue
        out.append({"addr": r["ethAddress"], "name": r.get("displayName"), "acct": acct,
                    "pnl30": m["pnl"], "roi30": m["roi"], "pnl7": w["week"]["pnl"], "pnl_all": w["allTime"]["pnl"]})
        if len(out) == n:
            break
    return out


def snapshot(rows: list[dict], n: int = SNAPSHOT_N) -> int:
    """Save today's leaderboard (best n by 30-day pnl, unfiltered, every window) in hl_snapshots.
    The leaderboard only ever shows the present; a month of these lets "do last month's winners
    keep winning next month?" be tested with only what was known at the time. One row set per day."""
    day = _now().date().isoformat()
    best = sorted(rows, key=lambda r: -float(_windows(r)["month"]["pnl"]))[:n]
    with pl._db() as con:
        con.execute(SNAP_SCHEMA)
        if con.execute("SELECT 1 FROM hl_snapshots WHERE day=? LIMIT 1", (day,)).fetchone():
            return 0
        recs = []
        for i, r in enumerate(best, 1):
            w = _windows(r)
            recs.append((day, i, r["ethAddress"], r.get("displayName"), float(r["accountValue"]),
                         w["day"]["pnl"], w["week"]["pnl"], w["month"]["pnl"], w["allTime"]["pnl"],
                         w["month"]["roi"], w["month"]["vlm"], w["allTime"]["vlm"]))
        con.executemany("INSERT INTO hl_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", recs)
    return len(recs)


def top() -> list[dict]:
    raw = pl.get_kv("hl_top")
    return json.loads(raw)["top"] if raw else []


def top_refreshed() -> datetime | None:
    raw = pl.get_kv("hl_top")
    return datetime.fromisoformat(json.loads(raw)["t"]) if raw else None


def refresh_top(force: bool = False) -> dict | None:
    """Re-pick the list if it's over HL_REFRESH_H old. Returns who joined and who left, or None
    if nothing was due. A trader who starts losing drops down the 30-day ranking and falls off."""
    t = top_refreshed()
    if not force and t and _now() - t < timedelta(hours=HL_REFRESH_H) and pl.get_kv("hl_map"):
        return None
    r = requests.get(LEADERBOARD_URL, timeout=120)
    r.raise_for_status()
    rows = r.json()["leaderboardRows"]
    snapshot(rows)
    new = pick_top(rows)
    if len(new) < HL_TOP_N // 2:
        raise RuntimeError(f"leaderboard looked wrong ({len(new)} accounts passed the filters)")
    old = top()
    pl.set_kv("hl_top", json.dumps({"t": _now().isoformat(), "top": new}))
    # The biggest accounts of all, whatever their profit: the liquidation map (liqmap.py) reads these.
    big = sorted(rows, key=lambda r: -float(r["accountValue"]))[:MAP_ACCOUNTS]
    pl.set_kv("hl_map", json.dumps({"t": _now().isoformat(), "addrs": [r["ethAddress"] for r in big]}))
    old_a, new_a = {x["addr"] for x in old}, {x["addr"] for x in new}
    for a in old_a - new_a:     # stop tracking: forget their positions
        state["positions"].pop(a, None)
        state["accts"].pop(a, None)
        state["errors"].pop(a, None)
    return {"first": not old, "joined": [x for x in new if x["addr"] not in old_a],
            "left": [x for x in old if x["addr"] not in new_a]}


# --------------------------------------------------------------------------
# Positions, polled every minute
# --------------------------------------------------------------------------

def _positions(addr: str) -> tuple[dict, float | None]:
    """({coin: position}, account value) for one trader: the main exchange plus each HL_DEXES
    exchange (stock/commodity futures, coins named like "xyz:GOLD"). Any failed request fails the
    whole trader for this minute - half a reading would look like their other positions closed.
    The account value is the main exchange's (each extra exchange has its own margin)."""
    out, acct = {}, None
    for dex in ("", *HL_DEXES):
        st = _post({"type": "clearinghouseState", "user": addr, **({"dex": dex} if dex else {})})
        for p in st.get("assetPositions", []):
            p = p["position"]
            szi = float(p["szi"])
            if szi == 0:
                continue
            value = abs(float(p["positionValue"]))
            out[p["coin"]] = {"usd": value * (1 if szi > 0 else -1), "mark": value / abs(szi),
                              "entry": float(p["entryPx"] or 0), "lev": (p.get("leverage") or {}).get("value"),
                              "liq": float(p["liquidationPx"]) if p.get("liquidationPx") else None,
                              "upnl": float(p.get("unrealizedPnl") or 0)}
        if not dex:
            acct = (st.get("marginSummary") or {}).get("accountValue")
    return out, float(acct) if acct is not None else None


def _label(coin: str) -> str:
    """'xyz:GOLD' -> 'GOLD (xyz)'; crypto stays as is."""
    dex, _, name = coin.rpartition(":")
    return f"{name} ({dex})" if dex else coin


def fetch_positions(addrs: list[str]) -> dict:
    """addr -> ({coin: position}, account value) or the exception. One try each; a trader who fails
    is skipped this minute."""
    def one(a):
        try:
            return a, _positions(a)
        except Exception as e:
            return a, e
    with ThreadPoolExecutor(WORKERS) as ex:
        return dict(ex.map(one, addrs))


def poll() -> dict:
    """Fetch everyone on the list and update `state`. Returns the fresh positions."""
    tr = top()
    got = fetch_positions([x["addr"] for x in tr])
    fresh = {}
    for a, res in got.items():
        if isinstance(res, Exception):
            state["errors"][a] = f"{type(res).__name__}: {res}"[:100]
        else:
            state["errors"].pop(a, None)
            state["positions"][a] = fresh[a] = res[0]
            if res[1]:
                state["accts"][a] = res[1]
    if fresh:
        state["polled"] = _now()
    return fresh


def classify(v: float, last: float, T: float) -> str | None:
    """What changed between the position at the last ping (`last`, signed $) and now (`v`).
    Steps scale with the position, so a $200M holder needs a $50M add to count, not $5M."""
    if abs(v) >= T and (abs(last) < T or v * last < 0):
        return "open"                       # new, or flipped sides
    if abs(last) >= T and abs(v) < 0.1 * abs(last):
        return "close"                      # (nearly) all of it gone
    if v * last > 0 and abs(v) - abs(last) >= max(T, 0.25 * abs(last)):
        return "add"
    if v * last > 0 and abs(last) - abs(v) >= max(T, 0.5 * abs(last)):
        return "cut"                        # at least half closed
    return None


def check_moves(fresh: dict, tr: list[dict]) -> list[dict]:
    """Big opens, flips, adds, cuts and closes since the last ping. A trader's positions the first
    time they're seen (startup, or newly on the list) are only recorded, never pinged."""
    rank = {x["addr"]: i + 1 for i, x in enumerate(tr)}
    info = {x["addr"]: x for x in tr}
    events = []
    for a, pos in fresh.items():
        if a not in rank:
            continue
        coins = set(pos) | {c for (aa, c) in _pinged if aa == a}
        for coin in coins:
            p = pos.get(coin)
            v = p["usd"] if p else 0.0
            key = (a, coin)
            if a not in _seeded:
                _pinged[key], _pinged_t[key] = v, _now()
                continue
            last = _pinged.get(key, 0.0)
            kind = classify(v, last, HL_PING_USD if coin == "BTC" else HL_PING_ALT_USD)
            if kind:
                prev = state.get("_prev", {}).get(a, {}).get(coin)
                events.append({"addr": a, "rank": rank[a], "trader": info[a], "coin": coin, "kind": kind,
                               "usd": v, "last": last, "pos": p, "prev": prev, "since": _pinged_t.get(key)})
                _pinged[key], _pinged_t[key] = v, _now()
        _seeded.add(a)
    return events


def consensus(positions: dict | None = None, tr: list[dict] | None = None, accts: dict | None = None) -> dict:
    """The top 20's BTC positions, one vote per trader (by dollars, two or three huge accounts
    would decide everything). side = LONG/SHORT if 60%+ of those holding BTC agree, else None.
    avg_x = their average BTC exposure as a multiple of each one's own account (flat counts as 0):
    what the mirror wallet holds. The account is their Hyperliquid perp account (marginSummary),
    which can be well below the leaderboard's figure (that includes spot), so -1.5x, not -0.9x."""
    positions = state["positions"] if positions is None else positions
    accts = state["accts"] if accts is None else accts
    tr = top() if tr is None else tr
    xs = []
    longs = shorts = 0
    long_usd = short_usd = 0.0
    entry_usd = {"long": 0.0, "short": 0.0}   # sum of entry x $ size, for dollar-weighted average entries
    entry_w = {"long": 0.0, "short": 0.0}     # the $ those entries cover
    seen, mark = 0, None
    for x in tr:
        pos = positions.get(x["addr"])
        if pos is None:
            continue
        seen += 1
        b = pos.get("BTC") or {}
        v = b.get("usd", 0.0)
        acct = accts.get(x["addr"]) or x["acct"]
        if acct and acct > 0:
            xs.append(v / acct)
        mark = b.get("mark") or mark
        if v >= MIN_POS_USD:
            longs, long_usd = longs + 1, long_usd + v
        elif v <= -MIN_POS_USD:
            shorts, short_usd = shorts + 1, short_usd - v
        if abs(v) >= MIN_POS_USD and b.get("entry"):
            k = "long" if v > 0 else "short"
            entry_usd[k] += b["entry"] * abs(v)
            entry_w[k] += abs(v)
    holders = longs + shorts
    side, share = None, None
    if holders:
        share = max(longs, shorts) / holders
        if holders >= CONSENSUS_MIN and share >= CONSENSUS_SHARE:
            side = "LONG" if longs > shorts else "SHORT"
    biggest = max((abs((positions.get(x["addr"], {}).get("BTC") or {}).get("usd", 0.0)) for x in tr), default=0.0)
    return {"n": len(tr), "seen": seen, "long": longs, "short": shorts, "flat": seen - holders,
            "long_usd": long_usd, "short_usd": short_usd, "net_usd": long_usd - short_usd,
            "side": side, "share": share, "biggest": biggest, "polled": state["polled"], "mark": mark,
            "avg_x": sum(xs) / len(xs) if xs else None,
            "long_entry": entry_usd["long"] / entry_w["long"] if entry_w["long"] else None,
            "short_entry": entry_usd["short"] / entry_w["short"] if entry_w["short"] else None}


def fresh_consensus() -> dict:
    """Consensus from positions at most STALE_MIN old, polling now if needed. side is None if it can't."""
    if state["polled"] is None or _now() - state["polled"] > timedelta(minutes=STALE_MIN):
        if top():
            poll()
    c = consensus()
    if c["polled"] is None or _now() - c["polled"] > timedelta(minutes=STALE_MIN) or c["seen"] < c["n"] // 2:
        c["side"] = None
        c["stale"] = True
    return c


# --------------------------------------------------------------------------
# Mirror wallet: hold what the top 20 hold right now, in proportion to the account
# --------------------------------------------------------------------------

def btc_price() -> float:
    return float(_post({"type": "allMids"})["BTC"])


def _next_funding(t: datetime) -> datetime:
    """First 00/08/16 UTC funding time strictly after t."""
    day = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return day + timedelta(hours=(t.hour // 8 + 1) * 8)


def mirror_state() -> dict | None:
    raw = pl.get_kv("hl_mirror")
    return json.loads(raw) if raw else None


def mirror_reset() -> None:
    pl.set_kv("hl_mirror", "")     # starts again with START_EQUITY at the next minute


def mirror_step(target: float | None, price: float, now: datetime, fetch_funding) -> dict | None:
    """One minute of the mirror wallet. Marks the position to market, pays or receives perp funding
    at 00/08/16 UTC (Binance's rates, like the other wallets), then rebalances to `target` - the
    top 20's average BTC exposure as a multiple of the account (None = no fresh reading: hold) -
    once it's MIRROR_BAND away from what the wallet holds. Fees 0.05% a side on what's traded.
    Returns the rebalance, or None."""
    st = mirror_state()
    if st is None:
        st = {"start_t": now.isoformat(), "start_px": price, "start_eq": START_EQUITY, "eq": START_EQUITY,
              "q": 0.0, "px": price, "fund_t": now.isoformat(), "fees": 0.0, "funding": 0.0, "n": 0, "broke": False}
    st["eq"] += st["q"] * (price - st["px"])      # q = BTC held (negative = short)
    st["px"] = price
    fund_t = datetime.fromisoformat(st["fund_t"])
    if now >= _next_funding(fund_t):
        if st["q"]:
            rates = fetch_funding(fund_t)
            due = rates[(rates.index > pd.Timestamp(fund_t)) & (rates.index.floor("min") <= pd.Timestamp(now))] \
                if len(rates) else rates
            if len(due):     # not published yet: try again next minute
                pay = float(st["q"] * price * due.sum())   # longs pay a positive rate, shorts receive it
                st["eq"] -= pay
                st["funding"] += pay
                st["fund_t"] = due.index[-1].isoformat()
        else:
            st["fund_t"] = now.isoformat()
    if st["eq"] <= 0 and not st["broke"]:
        st.update(eq=0.0, q=0.0, broke=True)
    x = st["q"] * price / st["eq"] if st["eq"] > 0 else 0.0
    ev = None
    if target is not None and not st["broke"]:
        tgt = 0.0 if abs(target) < MIRROR_MIN_STEP else target
        if abs(tgt - x) >= MIRROR_MIN_STEP and (abs(tgt - x) >= MIRROR_BAND * abs(x) or tgt * x <= 0):
            new_q = tgt * st["eq"] / price
            fee = abs(new_q - st["q"]) * price * FEE_ROUND_TRIP / 2
            st["eq"] -= fee
            st["fees"] += fee
            st["n"] += 1
            sign = lambda u: (u > 0) - (u < 0)
            ev = {"t": now, "price": price, "x_from": x, "x_to": tgt, "eq": st["eq"], "fee": fee, "target": target,
                  "side_change": sign(x) != sign(tgt)}
            st["q"] = new_q
            with pl._db() as con:
                con.execute(MIRROR_SCHEMA)
                con.execute("INSERT INTO hl_mirror VALUES (?,?,?,?,?,?,?)",
                            (now.isoformat(), price, x, tgt, st["eq"], fee, target))
    st["t"] = now.isoformat()
    pl.set_kv("hl_mirror", json.dumps(st))
    return ev


def mirror_check(fetch_funding) -> dict | None:
    """The every-minute mirror step, from the positions just polled. A stale or patchy reading
    (under half the traders answered) holds the current position."""
    c = consensus()
    fresh = c["polled"] is not None and _now() - c["polled"] < timedelta(minutes=STALE_MIN) and c["seen"] >= c["n"] // 2
    return mirror_step(c["avg_x"] if fresh else None, btc_price(), _now(), fetch_funding)


def _x_text(x: float) -> str:
    return "FLAT" if abs(x) < 1e-9 else f"{'LONG' if x > 0 else 'SHORT'} {abs(x):.2f}x"


def mirror_text() -> str:
    st = mirror_state()
    if not st:
        return "🪞 Mirror wallet (HL top 20, proportional): starts at the next check."
    eq, start = st["eq"], st["start_eq"]
    x = st["q"] * st["px"] / eq if eq > 0 else 0.0
    btc = st["px"] / st["start_px"] - 1
    since = datetime.fromisoformat(st["start_t"])
    lines = [f"🪞 Mirror wallet (HL top 20, proportional): ${eq:,.2f}  (started with ${start:,.0f}, "
             f"{(eq / start - 1) * 100:+.1f}%)",
             f"Holding {_x_text(x)}" + (f" (${abs(st['q'] * st['px']):,.0f} of BTC)" if st["q"] else "")
             + f". {st['n']} rebalance{'' if st['n'] == 1 else 's'}, fees ${st['fees']:,.2f}, funding {'paid' if st['funding'] >= 0 else 'received'} "
               f"${abs(st['funding']):,.2f}.",
             f"BTC since it started ({since:%d %b}): {btc * 100:+.1f}%. It holds the top 20's average BTC exposure "
             f"(each one's position ÷ their account, flat = 0), rebalancing when that moves {MIRROR_BAND:.0%}+."]
    if st["broke"]:
        lines.append("💸 Broke. /wallet reset starts again.")
    return "\n".join(lines)


def mirror_event_text(ev: dict, name: str) -> str:
    return (f"🪞 {name}: mirror wallet now {_x_text(ev['x_to'])} (was {_x_text(ev['x_from'])}) at "
            f"{_px(ev['price'])} - the top 20's average BTC exposure is {ev['target']:+.2f}x of their accounts. "
            f"Wallet ${ev['eq']:,.2f}.")


def _vwap(fills: list[dict]) -> float | None:
    sz = sum(float(f["sz"]) for f in fills)
    return sum(float(f["px"]) * float(f["sz"]) for f in fills) / sz if sz else None


def fill_summary(addr: str, coin: str, since: datetime) -> dict:
    """The trader's actual fills in `coin` since `since`: average price of the closing part (the
    exit) and of the opening part, and the profit Hyperliquid booked on the closing part
    (closedPnl, before fees). A flip's fills count as both. partial = more fills than we fetched."""
    t0, fills, partial = int((since - timedelta(minutes=1)).timestamp() * 1000), [], False
    for page in range(FILL_PAGES):
        f = _post({"type": "userFillsByTime", "user": addr, "startTime": t0, "aggregateByTime": True}, timeout=15)
        fills += f
        if len(f) < 2000:
            break
        t0 = max(x["time"] for x in f) + 1
        partial = page == FILL_PAGES - 1
    mine = [f for f in fills if f["coin"] == coin]
    closing = [f for f in mine if f["dir"].startswith("Close") or ">" in f["dir"]]   # "Long > Short" = flip
    opening = [f for f in mine if f["dir"].startswith("Open") or ">" in f["dir"]]
    times = [f["time"] for f in (closing or mine)]   # how long the exit took
    return {"exit_px": _vwap(closing), "open_px": _vwap(opening), "n": len(mine),
            "pnl": sum(float(f["closedPnl"]) for f in closing), "fees": sum(float(f.get("fee") or 0) for f in mine),
            "first": datetime.fromtimestamp(min(times) / 1000, timezone.utc) if times else None,
            "last": datetime.fromtimestamp(max(times) / 1000, timezone.utc) if times else None, "partial": partial}


def check_all() -> list[dict]:
    """The every-minute job: poll, find big moves, look up their fills (exit prices), log them."""
    tr = top()
    if not tr:
        return []
    state["_prev"] = {a: dict(p) for a, p in state["positions"].items()}
    fresh = poll()
    _snapshot_positions(fresh)
    events = check_moves(fresh, tr)
    for ev in events:
        try:   # without the fills the message falls back to the last unrealised pnl
            ev["fills"] = fill_summary(ev["addr"], ev["coin"], ev["since"] or _now() - timedelta(minutes=5))
        except Exception as e:
            log.warning("fills for %s %s: %s", ev["addr"][:10], ev["coin"], e)
            ev["fills"] = None
    _log_events(events)
    return events


def _snapshot_positions(fresh: dict) -> None:
    """Once an hour: each tracked trader's BTC position and account, for testing later which
    traders were actually good at BTC (the leaderboard only ranks total profit)."""
    hour = _now().strftime("%Y-%m-%dT%H")
    if not fresh or pl.get_kv("hl_pos_hour") == hour:
        return
    with pl._db() as con:
        con.execute(POS_SCHEMA)
        con.executemany("INSERT INTO hl_positions VALUES (?,?,?,?,?,?,?)",
                        [(_now().isoformat(), a, state["accts"].get(a), (pos.get("BTC") or {}).get("usd", 0.0),
                          (pos.get("BTC") or {}).get("entry"), (pos.get("BTC") or {}).get("upnl"),
                          (pos.get("BTC") or {}).get("lev")) for a, pos in fresh.items()])
    pl.set_kv("hl_pos_hour", hour)


def _log_events(events: list[dict]) -> None:
    if not events:
        return
    with pl._db() as con:
        con.execute(SCHEMA)
        con.executemany("INSERT INTO hl_events VALUES (?,?,?,?,?,?,?,?,?)",
                        [(_now().isoformat(), e["addr"], e["rank"], e["coin"], e["kind"],
                          "long" if (e["usd"] or e["last"]) > 0 else "short", e["usd"],
                          (e["pos"] or {}).get("entry"),
                          json.dumps({"last": e["last"], "prev_entry": (e.get("prev") or {}).get("entry"),
                                      **({k: e["fills"][k] for k in ("exit_px", "open_px", "pnl", "n")}
                                         if e.get("fills") else {})})) for e in events])


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------

def _m(usd: float) -> str:
    usd = abs(usd)
    return f"${usd / 1e9:,.2f}B" if usd >= 1e9 else f"${usd / 1e6:,.1f}M"


def _px(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:,.4g}"


def _who(x: dict, rank: int) -> str:
    name = f" \"{x['name']}\"" if x.get("name") else ""
    return f"#{rank}{name} ({'+' if x['pnl30'] >= 0 else '-'}{_m(x['pnl30'])} in 30 days)"


def consensus_line(c: dict) -> str:
    if not c["seen"]:
        return "Top 20 on BTC: no positions read yet."
    s = (f"Top {c['n']} on BTC: {c['short']} short · {c['long']} long · {c['flat']} flat"
         + (f" ({c['seen']} answered)" if c["seen"] < c["n"] else ""))
    if c["long"] + c["short"]:
        lean = "SHORT" if c["short"] > c["long"] else "LONG" if c["long"] > c["short"] else "split"
        s += f" → {lean} {c['share']:.0%} of holders" if lean != "split" else " → split"
        s += f". Net {_m(c['net_usd'])} {'long' if c['net_usd'] >= 0 else 'short'}"
        gross = c["long_usd"] + c["short_usd"]
        if gross and c["biggest"] / gross > 0.4:
            s += f" (one account is {c['biggest'] / gross:.0%} of the dollars)"
        entries = [f"{k}s {_px(c[f'{k}_entry'])}" for k in ("short", "long") if c.get(f"{k}_entry")]
        if entries:
            s += f". Avg entry: {', '.join(entries)}" + (f"; BTC {_px(c['mark'])} now" if c.get("mark") else "")
    return s + "."


def _move(entry: float, exit_: float, was_long: bool) -> str:
    """Price move in the position's favour, e.g. +1.3% for a short that bought back lower."""
    return f"{(exit_ / entry - 1) * (100 if was_long else -100):+.1f}%"


def _signed(usd: float) -> str:
    return f"{'+' if usd >= 0 else '-'}{_m(usd)}"


def _exit_text(f: dict | None, entry: float | None, was_long: bool, what: str) -> str:
    """'Exit ~79,950 vs entry 81,000 (+1.3%), realised +$2.7M over 37 fills in 25 min.'"""
    if not f or not f["exit_px"]:
        return ""
    vs = f" vs entry {_px(entry)} ({_move(entry, f['exit_px'], was_long)})" if entry else ""
    span = ""
    if f["first"] and f["last"] and f["last"] - f["first"] >= timedelta(minutes=2):
        span = f" in {int((f['last'] - f['first']).total_seconds() // 60)} min"
    return (f"\n{what} ~{_px(f['exit_px'])}{vs}, realised {_signed(f['pnl'])} before fees "
            f"({f['n']} fills{span}{', maybe more' if f['partial'] else ''}).")


def format_event(ev: dict, name: str) -> str:
    x, coin, p, v = ev["trader"], _label(ev["coin"]), ev["pos"], ev["usd"]
    who = _who(x, ev["rank"])
    side = lambda u: "LONG" if u > 0 else "SHORT"
    prev, f = ev.get("prev") or {}, ev.get("fills")
    flip = ev["kind"] == "open" and ev["last"] * v < 0
    if ev["kind"] == "close":
        size = prev.get("usd") or ev["last"]
        rest = f" ({_m(v)} left)" if v else ""
        exit_ = _exit_text(f, prev.get("entry"), size > 0, "Exit")
        if not exit_ and prev.get("entry"):   # fills unavailable: the last thing we saw
            exit_ = (f"\nEntry {_px(prev['entry'])}; last check it was {_signed(prev['upnl'])} unrealised"
                     + (f" at {_px(prev['mark'])}" if prev.get("mark") else "") + ".")
        head = f"🐳 {name}: Hyperliquid {who} closed their {_m(size)} {coin} {side(size)}{rest}.{exit_}"
    else:
        if ev["kind"] == "add":
            what = f"added {_m(abs(v) - abs(ev['last']))} to their {coin} {side(v)}: now {_m(v)}"
            if f and f["open_px"]:
                what += f"\nAdded at ~{_px(f['open_px'])}" + (f", average entry was {_px(prev['entry'])}"
                                                            if prev.get("entry") else "")
        elif ev["kind"] == "cut":
            what = (f"cut their {coin} {side(v)} by {_m(abs(ev['last']) - abs(v))}: now {_m(v)}"
                    + _exit_text(f, p["entry"], v > 0, "Closed part at"))
        elif flip:
            what = (f"flipped {coin} from {side(ev['last'])} to a {_m(v)} {side(v)}"
                    + _exit_text(f, prev.get("entry"), ev["last"] > 0, f"Closed the {side(ev['last'])} at"))
        else:
            what = f"opened a {_m(v)} {coin} {side(v)}"
        lev = f"{p['lev']}x, " if p.get("lev") else ""
        liq = f", liquidation {_px(p['liq'])}" if p.get("liq") else ""
        now_px = f" (now {_px(p['mark'])})" if p.get("mark") else ""
        head = (f"🐳 {name}: Hyperliquid {who} {what}\n{lev}average entry {_px(p['entry'])}{now_px}{liq}. "
                f"Account {_m(x['acct'])}.")
    lines = [head]
    if coin == "BTC":
        lines.append(consensus_line(consensus()))
    if ":" in ev["coin"]:
        lines.append(f"{ev['coin'].split(':')[0]} = Hyperliquid's stock/commodity futures. They trade 24/7 but the real "
                     "market doesn't, so off-hours prices are thin and can jump - and a big short may be a hedge on "
                     "shares held elsewhere.")
    else:
        lines.append("Copying them isn't a proven edge - three paper wallets are testing it (/hl).")
    lines.append(EXPLORER + ev["addr"])
    return "\n".join(lines)


def refresh_text(ch: dict) -> str:
    """Daily roster change message."""
    if ch["first"]:
        return f"🐳 Now tracking the Hyperliquid top {len(top())} by 30-day profit. /hl to see them."
    if not ch["joined"] and not ch["left"]:
        return ""
    left = ", ".join(f"{x['addr'][:8]} (7d {'+' if x['pnl7'] >= 0 else '-'}{_m(x['pnl7'])})" for x in ch["left"])
    joined = ", ".join(f"{x['addr'][:8]} (+{_m(x['pnl30'])} 30d)" for x in ch["joined"])
    return (f"🐳 Hyperliquid top {len(top())} refreshed: {len(ch['joined'])} in, {len(ch['left'])} out.\n"
            + (f"Out: {left}\n" if left else "") + (f"In: {joined}" if joined else ""))


def call_section(c: dict, model_dir: str) -> str:
    """For the long-or-short reply: what the top 20 are doing next to Amadeus's own lean."""
    if not top():
        return "🐳 Hyperliquid top 20: list not loaded yet."
    lines = ["🐳 Hyperliquid top 20 (untested)", "  " + consensus_line(c)]
    if c.get("stale"):
        lines.append("  ⚠ Couldn't read fresh positions - not used.")
    elif c["side"] is None:
        lines.append(f"  No clear majority (needs {CONSENSUS_SHARE:.0%}+ of {CONSENSUS_MIN}+ holders), "
                     "so the copy wallet sits this hour out.")
    elif model_dir in ("LONG", "SHORT"):
        if c["side"] == model_dir:
            lines.append(f"  They agree with my lean ({model_dir}). Two unproven signals agreeing still isn't an edge.")
        else:
            lines.append(f"  They disagree with my lean ({model_dir}) - one more reason to treat direction as a coin flip.")
    else:
        lines.append(f"  They lean {c['side']}; I have no lean.")
    if c.get("avg_x") is not None:
        lines.append(f"  Their average BTC exposure: {_x_text(c['avg_x'])} of their accounts (the mirror wallet holds this).")
    lines.append("  " + wallets_line())
    return "\n".join(lines)


def wallets_line() -> str:
    """'Wallets: mine $980 · copy $1,040 (3) · follow $1,000 (0) · mirror $1,012' - the race in one line."""
    parts = []
    for b, name in (("amadeus", "mine"), ("copy", "copy"), ("follow", "follow")):
        w = pl.wallet(b)
        parts.append(f"{name} ${w['balance']:,.0f}" + (f" ({len(w['done'])})" if b != "amadeus" else ""))
    st = mirror_state()
    parts.append(f"mirror ${st['eq']:,.0f}" if st else "mirror -")
    n = min(len(pl.wallet(b)["done"]) for b in ("copy", "follow"))
    return "Wallets: " + " · ".join(parts) + (" - too few trades to judge yet (aim for 50+)" if n < 50 else "")


def stocks_line(n: int = 6) -> str:
    """The top 20's biggest net stock/commodity future positions, added up across traders:
    'Stocks & commodities (xyz): GOLD SHORT $57.9M (9 traders), GOOGL SHORT $32.2M (8), ...'"""
    net, who = {}, {}
    for a, pos in state["positions"].items():
        for c, p in pos.items():
            if ":" in c and abs(p["usd"]) >= MIN_POS_USD:
                net[c] = net.get(c, 0.0) + p["usd"]
                who[c] = who.get(c, 0) + 1
    if not net:
        return ""
    best = sorted(net, key=lambda c: -abs(net[c]))[:n]
    gross = sum(abs(v) for v in net.values())
    items = [f"{c.split(':', 1)[1]} {'LONG' if net[c] > 0 else 'SHORT'} {_m(net[c])} ({who[c]})" for c in best]
    return (f"Stocks & commodities ({', '.join(HL_DEXES)}), net across the 20 (traders): " + ", ".join(items)
            + f". {_m(gross)} in {len(net)} markets.")


def liq_levels(positions: dict | None = None, tr: list | None = None) -> dict | None:
    """Where the top 20's BTC positions get liquidated, nearest first. Real numbers: Hyperliquid
    reports each position's liquidation price. It is None when the account has so much spare
    margin that no BTC price would do it; those are counted, not placed."""
    positions = state["positions"] if positions is None else positions
    rows, safe, mark = [], 0, None
    for rank, x in enumerate(tr if tr is not None else top(), 1):
        b = (positions.get(x["addr"]) or {}).get("BTC")
        if not b or abs(b["usd"]) < MIN_POS_USD:
            continue
        mark = mark or b.get("mark")
        if b.get("liq") and b.get("mark"):
            rows.append({"rank": rank, "side": "long" if b["usd"] > 0 else "short", "usd": abs(b["usd"]),
                         "liq": b["liq"], "dist": b["liq"] / b["mark"] - 1})
        else:
            safe += 1
    if not rows and not safe:
        return None
    rows.sort(key=lambda r: abs(r["dist"]))
    return {"mark": mark, "safe": safe, "short": [r for r in rows if r["side"] == "short"],
            "long": [r for r in rows if r["side"] == "long"]}


def liq_line(n: int = LIQ_SHOWN) -> str:
    """'BTC liquidation prices: shorts above - 82,400 (+3.0%, #4 $50.0M), ...; longs below - none.'"""
    v = liq_levels()
    if not v:
        return ""
    def side(rows, where):
        if not rows:
            return "none"
        near = [r for r in rows if abs(r["dist"]) <= LIQ_NEAR]
        txt = ", ".join(f"{_px(r['liq'])} ({r['dist'] * 100:+.1f}%, #{r['rank']} {_m(r['usd'])})" for r in rows[:n])
        more = f", +{len(rows) - n} further" if len(rows) > n else ""
        return (f"{txt}{more}. Within {LIQ_NEAR * 100:.0f}% {where}: "
                + (f"{_m(sum(r['usd'] for r in near))} from {len(near)}" if near else "nothing"))
    return (f"BTC liquidation prices (theirs, as Hyperliquid reports them; BTC {_px(v['mark'])} now):\n"
            f"• Shorts, liquidated if BTC rises to: {side(v['short'], 'above')}.\n"
            f"• Longs, liquidated if BTC falls to: {side(v['long'], 'below')}.\n"
            + (f"• {v['safe']} more hold BTC with no liquidation price (too much spare margin).\n" if v["safe"] else "")
            + "These move when they add margin, resize or their other positions change, and big accounts usually "
              "cut before getting there. It shows where they'd be forced out, not where price is going.")


def status_text(on: bool, tz) -> str:
    tr = top()
    if not tr:
        return "🐳 Hyperliquid watch: the top-20 list hasn't loaded yet (it downloads once a day)."
    t = top_refreshed()
    lines = [f"🐳 Hyperliquid top {len(tr)} by 30-day profit (refreshed {t.astimezone(tz):%d %b %H:%M}). "
             f"Pings {'ON' if on else 'OFF'} (/hl {'off' if on else 'on'}): opening/closing {_m(HL_PING_USD)}+ BTC "
             f"or {_m(HL_PING_ALT_USD)}+ on any other coin"
             + (f" or stock/commodity future ({', '.join(HL_DEXES)})" if HL_DEXES else "") + ".", ""]
    for i, x in enumerate(tr, 1):
        pos = state["positions"].get(x["addr"])
        if pos is None:
            btc = "?" if x["addr"] in state["errors"] else "not read yet"
        else:
            b = pos.get("BTC")
            btc = (f"BTC {'LONG' if b['usd'] > 0 else 'SHORT'} {_m(b['usd'])}" + (f" {b['lev']}x" if b.get("lev") else "")
                   + (f" @ {_px(b['entry'])} ({_signed(b.get('upnl', 0.0))})" if b.get("entry") else "")) \
                if b and abs(b["usd"]) >= MIN_POS_USD else "no BTC"
            others = sorted(((c, p["usd"]) for c, p in pos.items()
                             if c != "BTC" and ":" not in c and abs(p["usd"]) >= MIN_POS_USD),
                            key=lambda cp: -abs(cp[1]))[:2]   # crypto here; stock futures get their own section
            if others:
                btc += " · " + ", ".join(f"{c} {'L' if u > 0 else 'S'} {_m(u)}" for c, u in others)
        lines.append(f"{i}. {x['addr'][:8]}{' ' + x['name'] if x.get('name') else ''}: "
                     f"+{_m(x['pnl30'])} 30d ({x['roi30'] * 100:+.0f}%), {_m(x['acct'])} acct · {btc}")
    lines += ["", consensus_line(consensus())]
    stocks = stocks_line()
    if stocks:
        lines.append(stocks)
    if state["errors"]:
        lines.append(f"⚠ {len(state['errors'])} didn't answer the last check.")
    lines.append("")
    lines += [mirror_text(), "", pl.wallet_text("copy"), "", pl.wallet_text("follow")]
    lines += ["", "Three ways of copying them, racing my own wallet: mirror = hold their average exposure now; "
                  "copy = their majority side every hour; follow = go with a fresh big BTC entry. Copy and follow "
                  "use my stops, 4h exit, leverage and fees, so only the choice of side differs. A good month on "
                  "the leaderboard can be luck or leverage; this is how we find out."]
    return "\n".join(lines)
