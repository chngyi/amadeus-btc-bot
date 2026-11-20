"""Weekly report card (/report, and by itself once a week): how Amadeus did over the last 7 days.

It only reads what is already logged in paper_log.db - no network, no new numbers. It starts with
how much of the week she was actually running, because every figure after that only covers that time.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

import hyperliquid as hl
import paper_log as pl
import whales

ENOUGH_TRADES = 50        # the copy and follow wallets get their one verdict at this many trades
MIRROR_DAYS = 30          # the mirror wallet gets its one verdict after this long


def uptime(since: datetime, now: datetime) -> float | None:
    """Share of minutes since `since` with an order-book reading logged (the whale job logs one a minute)."""
    try:
        with pl._db() as con:
            n = con.execute("SELECT COUNT(*) FROM whale_depth WHERE t >= ?", (since.isoformat(),)).fetchone()[0]
    except Exception:
        return None   # no table yet
    return min(1.0, n / max(1.0, (now - since).total_seconds() / 60))


def _vol(since: datetime | None) -> dict | None:
    with pl._db() as con:
        df = pd.read_sql_query("SELECT * FROM vol_forecasts WHERE actual IS NOT NULL", con)
    if since is not None:
        df = df[pd.to_datetime(df["created_utc"], format="ISO8601", utc=True) >= since]
    if df.empty:
        return None
    return {"n": len(df), "inside": float(((df["actual"] >= df["range_lo"]) & (df["actual"] <= df["range_hi"])).mean()),
            "forecast": float(df["range_med"].median()), "actual": float(df["actual"].median())}


def text(tz, days: int = 7, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    lines = [f"🗒 Report card · {since.astimezone(tz):%d %b} to {now.astimezone(tz):%d %b}"]

    up = uptime(since, now)
    if up is not None:
        lines += ["", f"Running: {up:.0%} of the time." + (" Everything below only covers the hours I was on, "
                                                         "and most of my tests need far more than that." if up < 0.8 else "")]

    wk, ever = _vol(since), _vol(None)
    if wk:
        lines += ["", f"Volatility forecasts: {wk['n']} checked this week. {wk['inside']:.0%} of outcomes landed inside "
                      f"my 80% band (aim: 80%). Typical forecast {wk['forecast']:.2f}% vs actual {wk['actual']:.2f}%."]
        if ever and ever["n"] > wk["n"]:
            lines.append(f"  Since the start: {ever['n']} checked, {ever['inside']:.0%} inside the band.")
    else:
        lines += ["", "Volatility forecasts: none checked this week."]

    lines += ["", "Wallets (each started at ${:,.0f}):".format(pl.wallet("amadeus")["start"])]
    for book, name in (("amadeus", "Mine"), ("copy", "Copy"), ("follow", "Follow")):
        w = pl.wallet(book)
        done = w["done"]
        recent = done[pd.to_datetime(done["exit_utc"], format="ISO8601", utc=True) >= since] if len(done) else done
        pnl = float(recent["pnl_usd"].sum()) if len(recent) else 0.0
        week = (f"{'-' if pnl < 0 else '+'}${abs(pnl):,.0f} this week on {len(recent)} trades ({int((recent['pnl_usd'] > 0).sum())} won)"
                if len(recent) else "no trades closed this week")
        verdict = "" if book == "amadeus" else f" · {len(done)} of {ENOUGH_TRADES} trades towards its verdict"
        lines.append(f"• {name}: ${w['balance']:,.0f} · {week}{verdict}" + (" · BROKE" if w["broke"] else ""))
    m = hl.mirror_state()
    if m:
        start = datetime.fromisoformat(m["start_t"])
        due = start + timedelta(days=MIRROR_DAYS)
        lines.append(f"• Mirror: ${m['eq']:,.0f} ({m['eq'] / m['start_eq'] - 1:+.0%} since {start.astimezone(tz):%d %b}; "
                     f"BTC itself {m['px'] / m['start_px'] - 1:+.0%}) · "
                     + (f"verdict due {due.astimezone(tz):%d %b}" if due > now else "its month is up: time for the verdict"))

    try:
        with pl._db() as con:
            kinds = dict(con.execute("SELECT kind, COUNT(*) FROM whale_events WHERE t >= ? GROUP BY kind", (since.isoformat(),)).fetchall())
    except Exception:
        kinds = {}
    r = whales.wall_record(days)
    walls = (f" Of the walls I reported: {r['hit']} hit, {r['pulled']} pulled before price got there."
             if r and r["reported"] else "")
    lines += ["", f"Whale log this week: {kinds.get('burst', 0)} buying/selling bursts, {kinds.get('liq', 0)} liquidation "
                  f"cascades, {kinds.get('wall', 0)} walls.{walls}"]
    lines += ["", "Nothing here is tuned on these results: each wallet and each log gets judged once, when it has enough data."]
    return "\n".join(lines)
