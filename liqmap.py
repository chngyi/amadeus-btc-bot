"""Hyperliquid BTC liquidation map: how many dollars of BTC positions get force-closed at each
price, added up over the biggest accounts on Hyperliquid (hl.MAP_ACCOUNTS, picked daily in hl.refresh_top).

Every number is real: Hyperliquid reports each position's liquidation price. Nothing is estimated,
which is also the limit: it covers one exchange (a small slice of all BTC futures) and only its
biggest accounts. Nothing can be said about Binance, OKX or Bybit, where positions are private.

`poll_batch` runs every minute and reads the next MAP_BATCH accounts, so the whole map is
10 minutes old at worst. Description only: no pings, never fed to the model, no
effect on any call or wallet (CLAUDE.md rules 3-4). The hourly row in hl_liqmap is there so
"does price get pulled towards the big clusters?" can be tested once, later.
"""
from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import hyperliquid as hl
import paper_log as pl

log = logging.getLogger("liqmap")

MAP_BATCH = 100           # accounts read per minute (weight 200 of Hyperliquid's ~1200 a minute per IP)
MAP_BAND = 500.0          # $ price band the liquidations are added up over
MAP_RANGE = 0.20          # the map covers price +-20%; beyond that is only totalled
MAP_NEAR = 0.05           # "close" = within 5% of price
MAP_STALE_MIN = 45        # a reading older than this is left out (a full round takes 10 min)
MAP_MIN_READ = 200        # accounts read before a map is drawn (after a restart)
MAP_TOP = 3               # biggest clusters named per side
MAP_WORKERS = 10          # parallel requests (100 accounts took ~23s with 5)

SCHEMA = """CREATE TABLE IF NOT EXISTS hl_liqmap (
    t TEXT, mark REAL, n_read INTEGER, n_btc INTEGER, long_usd REAL, short_usd REAL, bands TEXT)"""   # hourly

state: dict = {"pos": {}, "cursor": 0, "logged": None}   # pos: addr -> {"t", "usd", "liq", "mark"}; usd 0 = no BTC


def accounts() -> list[str]:
    raw = pl.get_kv("hl_map")
    return json.loads(raw)["addrs"] if raw else []


def _btc(addr: str) -> dict:
    """One account's BTC position on the main exchange (usd 0 if it holds none)."""
    st = hl._post({"type": "clearinghouseState", "user": addr})
    for p in st.get("assetPositions", []):
        p = p["position"]
        szi = float(p["szi"])
        if p["coin"] == "BTC" and szi:
            value = abs(float(p["positionValue"]))
            return {"t": hl._now(), "usd": value * (1 if szi > 0 else -1), "mark": value / abs(szi),
                    "liq": float(p["liquidationPx"]) if p.get("liquidationPx") else None}
    return {"t": hl._now(), "usd": 0.0, "mark": None, "liq": None}


def poll_batch(n: int = MAP_BATCH) -> dict:
    """Read the next n accounts on the list (round robin). One try each; a failure keeps that
    account's last reading until it goes stale. Raises if most of the batch failed."""
    addrs = accounts()
    if not addrs:
        return {"read": 0, "failed": 0}
    keep = set(addrs)
    for a in [a for a in state["pos"] if a not in keep]:   # dropped off the list
        del state["pos"][a]
    i = state["cursor"] % len(addrs)
    batch = (addrs + addrs)[i:i + min(n, len(addrs))]
    state["cursor"] = (i + len(batch)) % len(addrs)

    def one(a):
        try:
            return a, _btc(a)
        except Exception as e:
            return a, e
    with ThreadPoolExecutor(MAP_WORKERS) as ex:
        got = dict(ex.map(one, batch))
    bad = [r for r in got.values() if isinstance(r, Exception)]
    for a, r in got.items():
        if not isinstance(r, Exception):
            state["pos"][a] = r
    if len(bad) > len(batch) / 2:
        raise RuntimeError(f"{len(bad)} of {len(batch)} accounts failed, e.g. {type(bad[0]).__name__}: {bad[0]}"[:200])
    m = build()
    if m and m["n_read"] >= MAP_MIN_READ:
        _log_map(m)
    return {"read": len(batch) - len(bad), "failed": len(bad)}


def build(now: datetime | None = None) -> dict | None:
    """Add up the fresh readings. None until anyone holding BTC has been read."""
    now = now or hl._now()
    fresh = {a: p for a, p in state["pos"].items() if now - p["t"] <= timedelta(minutes=MAP_STALE_MIN)}
    held = [p for p in fresh.values() if p["usd"]]
    if not held:
        return None
    mark = max(held, key=lambda p: p["t"])["mark"]   # the newest price seen
    m = {"t": now, "oldest": min(p["t"] for p in fresh.values()), "mark": mark, "n_accts": len(accounts()),
         "n_read": len(fresh), "n_btc": len(held), "usd": {"long": 0.0, "short": 0.0},
         "bands": {"long": {}, "short": {}}, "near": {"long": 0.0, "short": 0.0},
         "far": {"long": 0.0, "short": 0.0}, "safe": {"long": 0.0, "short": 0.0}}
    for p in held:
        side, usd = ("long" if p["usd"] > 0 else "short"), abs(p["usd"])
        m["usd"][side] += usd
        dist = p["liq"] / mark - 1 if p["liq"] else None
        if dist is None or (dist > 0) != (side == "short"):
            m["safe"][side] += usd      # no liquidation price (or one already behind price): can't be placed
        elif abs(dist) > MAP_RANGE:
            m["far"][side] += usd
        else:
            band = (p["liq"] // MAP_BAND) * MAP_BAND
            m["bands"][side][band] = m["bands"][side].get(band, 0.0) + usd
            if abs(dist) <= MAP_NEAR:
                m["near"][side] += usd
    return m


def _log_map(m: dict) -> None:
    """One row an hour."""
    hour = m["t"].replace(minute=0, second=0, microsecond=0)
    if state["logged"] == hour:
        return
    with pl._db() as con:
        con.execute(SCHEMA)
        con.execute("INSERT INTO hl_liqmap VALUES (?,?,?,?,?,?,?)",
                    (m["t"].isoformat(), m["mark"], m["n_read"], m["n_btc"], m["usd"]["long"], m["usd"]["short"],
                     json.dumps({s: [[b, round(u)] for b, u in sorted(m["bands"][s].items())] for s in m["bands"]})))
    state["logged"] = hour


def _top(m: dict, side: str, n: int = MAP_TOP) -> list[tuple[float, float]]:
    return sorted(m["bands"][side].items(), key=lambda x: -x[1])[:n]


def progress_text() -> str:
    n, total = len(state["pos"]), len(accounts())
    if not total:
        return "I don't have the account list yet (it comes with the Hyperliquid leaderboard download)."
    left = max(1, -(-(min(MAP_MIN_READ, total) - n) // MAP_BATCH)) + (n == 0)   # +1: the next read is up to a minute off
    return (f"Still reading: {n:,} of {total:,} accounts so far ({MAP_BATCH} a minute). "
            f"Ask again in about {left} minute{'s' if left > 1 else ''}.")


def ready(m: dict | None) -> bool:
    return bool(m) and m["n_read"] >= min(MAP_MIN_READ, m["n_accts"] or MAP_MIN_READ)


def summary(m: dict, tz) -> str:
    """The caption under the picture (Telegram allows 1024 characters there)."""
    def side(s, word, where):
        top = "; ".join(f"{b:,.0f}-{b + MAP_BAND:,.0f} {hl._m(u)} ({((b + MAP_BAND / 2) / m['mark'] - 1) * 100:+.1f}%)"
                        for b, u in _top(m, s))
        return (f"• {word} liquidated {where}: {hl._m(m['near'][s])} within {MAP_NEAR * 100:.0f}%, "
                f"{hl._m(sum(m['bands'][s].values()))} within {MAP_RANGE * 100:.0f}%."
                + (f" Biggest: {top}." if top else ""))
    part = "" if m["n_read"] >= m["n_accts"] else f" (read {m['n_read']:,} so far)"
    return "\n".join([
        f"Hyperliquid BTC liquidation map, BTC {hl._px(m['mark'])}, {m['t'].astimezone(tz):%d %b %H:%M %Z}.",
        f"The {m['n_accts']:,} biggest accounts{part}: {m['n_btc']} hold BTC, {hl._m(m['usd']['long'])} long "
        f"and {hl._m(m['usd']['short'])} short.",
        side("short", "Shorts", "above"), side("long", "Longs", "below"),
        f"• Not on the map: {hl._m(m['safe']['long'] + m['safe']['short'])} with no liquidation price (too much "
        f"spare margin), {hl._m(m['far']['long'] + m['far']['short'])} further than {MAP_RANGE * 100:.0f}% away.",
        "Real liquidation prices, but Hyperliquid only: a small slice of all BTC futures. It shows where these "
        "accounts would be forced out, not where price is going. Context only, never part of my call."])


# Chart colours: the dataviz reference palette (categorical slots 1-2, validated together).
C_LONG, C_SHORT = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def draw(m: dict, tz) -> bytes:
    """The map as a PNG: price up the side, $ liquidated per band across. One axis, two series
    (shorts above price, longs below), the biggest clusters labelled."""
    from matplotlib.figure import Figure   # imported here: slow to load, and only needed for the picture
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    mark = m["mark"]
    fig = Figure(figsize=(8, 10), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0.13, 0.085, 0.80, 0.775], facecolor=SURFACE)
    biggest = max([u for s in m["bands"] for u in m["bands"][s].values()] or [1.0])
    for s, color in (("short", C_SHORT), ("long", C_LONG)):
        b = m["bands"][s]
        ax.barh([x + MAP_BAND / 2 for x in b], [u / 1e6 for u in b.values()], height=MAP_BAND * 0.72,
                color=color, linewidth=0, zorder=3)
        for x, u in _top(m, s):
            ax.annotate(hl._m(u), (u / 1e6, x + MAP_BAND / 2), xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=9, color=INK, zorder=5)
    ax.axhline(mark, color=INK, linewidth=1.2, zorder=4)
    ax.annotate(f"BTC {hl._px(mark)}", (1, mark), xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", ha="right", va="bottom", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(mark * (1 - MAP_RANGE), mark * (1 + MAP_RANGE))
    ax.set_xlim(0, biggest / 1e6 * 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}M" if biggest >= 5e6 else f"${v:,.1f}M"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.get_major_locator().set_params(nbins=12)
    ax.xaxis.get_major_locator().set_params(nbins=6)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    pct = ax.secondary_yaxis("right", functions=(lambda p: (p / mark - 1) * 100, lambda d: mark * (1 + d / 100)))
    pct.set_yticks(range(-15, 16, 5))
    pct.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else ""))
    pct.tick_params(colors=MUTED, labelsize=9, length=0)
    pct.spines["right"].set_visible(False)
    for s, y, va, word in (("short", 0.985, "top", "above"), ("long", 0.015, "bottom", "below")):
        if m["far"][s]:   # most big accounts run low leverage, so a lot sits beyond the map: say how much
            ax.text(0.985, y, f"{hl._m(m['far'][s])} more {s}s liquidate further {word}", transform=ax.transAxes,
                    ha="right", va=va, fontsize=9, color=INK2, zorder=5)
    ax.set_xlabel(f"positions force-closed per ${MAP_BAND:,.0f} of price", color=INK2, fontsize=9.5, labelpad=8)

    fig.text(0.04, 0.965, "Hyperliquid BTC liquidation map", fontsize=17, fontweight="bold", color=INK, va="top")
    part = "" if m["n_read"] >= m["n_accts"] else f" ({m['n_read']:,} read so far)"
    fig.text(0.04, 0.928, f"Where the {m['n_accts']:,} biggest accounts{part} get liquidated, if BTC goes there",
             fontsize=10.5, color=INK2, va="top")
    fig.legend(handles=[Patch(color=C_SHORT, label=f"Shorts, liquidated if BTC rises ({hl._m(sum(m['bands']['short'].values()))})"),
                        Patch(color=C_LONG, label=f"Longs, liquidated if BTC falls ({hl._m(sum(m['bands']['long'].values()))})")],
               loc="upper left", bbox_to_anchor=(0.03, 0.905), ncols=2, frameon=False, fontsize=9.5,
               labelcolor=INK2, handlelength=1.1, handleheight=0.9, columnspacing=1.6)
    fig.text(0.04, 0.022, f"Real liquidation prices from Hyperliquid, read {m['oldest'].astimezone(tz):%H:%M}-"
                          f"{m['t'].astimezone(tz):%H:%M %Z %d %b}. One exchange only. Not a forecast.",
             fontsize=8.5, color=MUTED, va="bottom")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    return buf.getvalue()
