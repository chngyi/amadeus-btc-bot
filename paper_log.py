"""Paper-trade log: every call is recorded, then scored against what BTC actually did."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import (DB_PATH, FEE_ROUND_TRIP, FIXED_LEVERAGE, HORIZON_H, MAINT_MARGIN, MIN_NOTIONAL,
                    START_EQUITY)

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    source TEXT NOT NULL,          -- 'auto' (hourly scan), 'alert', 'asked'; Hyperliquid: 'copy', 'follow'
    decision TEXT NOT NULL,        -- LONG / SHORT
    strength TEXT,
    p_up REAL,
    entry REAL NOT NULL,
    tp REAL NOT NULL,              -- near end of the TP zone (conservative fill)
    tp_far REAL,
    sl REAL NOT NULL,
    expires_utc TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',   -- open / target / stop / time (the unleveraged outcome)
    exit_price REAL,
    exit_utc TEXT,
    ret_net REAL,                  -- unleveraged price return after fees
    news_score REAL,
    news_label TEXT,
    leverage REAL,                 -- paper leverage (FIXED_LEVERAGE when opened)
    liq REAL,                      -- isolated-margin liquidation price
    funding REAL,                  -- perp funding paid while open, fraction of notional (<0 = received)
    liquidated INTEGER,            -- 1 if the leveraged copy hit its liquidation price first
    ret_acct REAL,                 -- leveraged return on the margin, after fees x leverage and funding
    margin REAL,                   -- $ the wallet put in (whole balance); NULL = wallet sat this one out
    pnl_usd REAL                   -- $ the wallet made/lost on it (margin x ret_acct)
);
CREATE TABLE IF NOT EXISTS vol_forecasts (
    created_utc TEXT PRIMARY KEY,  -- bar close the forecast was made at (one per hour)
    expires_utc TEXT NOT NULL,
    price REAL NOT NULL,
    range_med REAL NOT NULL,       -- forecast next-4h high-low range, % (same units as vol_model)
    range_lo REAL,                 -- 80% band
    range_hi REAL,
    p_big2 REAL,                   -- forecast P(range > 2%)
    regime TEXT,
    percentile REAL,
    actual REAL                    -- realised range, %; NULL until the 4h are up
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(trades)")}
    for col, typ in (("news_score", "REAL"), ("news_label", "TEXT"), ("leverage", "REAL"),
                     ("liq", "REAL"), ("funding", "REAL"), ("liquidated", "INTEGER"), ("ret_acct", "REAL"),
                     ("margin", "REAL"), ("pnl_usd", "REAL")):
        if col not in cols:  # trades from before a column existed keep NULL (e.g. never in the wallet)
            con.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")
    return con


def liq_price(entry: float, decision: str, lev: float) -> float | None:
    """Isolated-margin liquidation price: the position is closed once the loss eats the
    margin (1/lev of the position) down to the maintenance margin. None at 1x or less."""
    if lev <= 1:
        return None
    d = 1 if decision == "LONG" else -1
    return entry * (1 - d * (1 / lev - MAINT_MARGIN))


def get_kv(k: str, default: str | None = None) -> str | None:
    with _db() as con:
        r = con.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def set_kv(k: str, v: str) -> None:
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, v))


# Wallets that race each other on the same trade mechanics (entry, stops, 4h, leverage, fees):
# Amadeus's own calls; 'copy' takes the Hyperliquid top 20's BTC majority every hour; 'follow'
# goes with a top trader's fresh big BTC entry. (The proportional mirror wallet holds positions
# instead of 4h trades, so it lives in hyperliquid.py.)
BOOKS = {"amadeus": ("wallet_since", "wallet_start"), "copy": ("copy_wallet_since", "copy_wallet_start"),
         "follow": ("follow_wallet_since", "follow_wallet_start")}
TITLES = {"amadeus": "💰 Amadeus's wallet", "copy": "🐳 Copy wallet (HL top 20 majority)",
          "follow": "🐳 Follow wallet (HL fresh entries)"}


def book_of(source: str) -> str:
    return source if source in BOOKS else "amadeus"


def _book_sql(book: str) -> tuple[str, tuple]:
    """WHERE clause selecting one book's trades (Amadeus's = every source that isn't another book)."""
    if book == "amadeus":
        others = [b for b in BOOKS if b != "amadeus"]
        return f"source NOT IN ({','.join('?' * len(others))})", tuple(others)
    return "source = ?", (book,)


def log_signal(s: dict, source: str) -> int | None:
    """Record a LONG/SHORT call. Skips duplicates (same direction already open in the same book).
    If that book's wallet is free (no wallet trade open, not broke) it goes all-in on this one."""
    tid = _insert_trade(s, source)
    if tid:
        w = wallet(book_of(source))
        if w["open"] is None and not w["broke"]:
            with _db() as con:
                con.execute("UPDATE trades SET margin=? WHERE id=?", (w["balance"], tid))
    return tid


def get_trade(tid: int) -> dict:
    with _db() as con:
        con.row_factory = sqlite3.Row
        return dict(con.execute("SELECT * FROM trades WHERE id=?", (tid,)).fetchone())


def _insert_trade(s: dict, source: str) -> int | None:
    if s["decision"] not in ("LONG", "SHORT"):
        return None
    with _db() as con:
        where, args = _book_sql(book_of(source))
        dup = con.execute(f"SELECT id FROM trades WHERE status='open' AND decision=? AND {where}",
                          (s["decision"], *args)).fetchone()
        if dup:
            return None
        created = s["time"].astimezone(timezone.utc)
        cur = con.execute(
            "INSERT INTO trades (created_utc, source, decision, strength, p_up, entry, tp, tp_far, sl,"
            " expires_utc, news_score, news_label, leverage, liq) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (created.isoformat(), source, s["decision"], s["strength"], s["p_up"], s["price"],
             s["levels"]["tp_near"], s["levels"]["tp_far"], s["levels"]["sl"],
             (created + timedelta(hours=HORIZON_H)).isoformat(),
             (s.get("news") or {}).get("score"), (s.get("news") or {}).get("label"),
             FIXED_LEVERAGE, liq_price(s["price"], s["decision"], FIXED_LEVERAGE)))
        return cur.lastrowid


def resolve_open(fetch_5m, fetch_funding) -> list[dict]:
    """Close open trades whose target/stop/liquidation was hit or whose 4h are up.
    fetch_5m(start_datetime) -> DataFrame of 5m bars indexed by bar CLOSE time.
    fetch_funding(start_datetime) -> Series of perp funding rates indexed by funding time.
    Returns the trades closed in this pass."""
    with _db() as con:
        rows = con.execute("SELECT id, created_utc, source, decision, entry, tp, sl, expires_utc,"
                           " leverage, liq, margin FROM trades WHERE status='open'").fetchall()
    if not rows:
        return []
    # Download once for all open trades, before anything is written: a network error
    # can't leave a trade closed in the DB but never announced.
    start = min(datetime.fromisoformat(r[1]) for r in rows)
    all_bars = fetch_5m(start)
    funding = fetch_funding(start)
    closed = []
    for tid, created, source, decision, entry, tp, sl, expires, lev, liq, margin in rows:
        lev = lev or FIXED_LEVERAGE
        t0, t1 = datetime.fromisoformat(created), datetime.fromisoformat(expires)
        bars = all_bars[(all_bars.index > pd.Timestamp(t0)) & (all_bars.index <= pd.Timestamp(t1))]
        d = 1 if decision == "LONG" else -1
        outcome = None
        for ts, b in bars.iterrows():
            hit_sl = b["low"] <= sl if d > 0 else b["high"] >= sl
            hit_tp = b["high"] >= tp if d > 0 else b["low"] <= tp
            if hit_sl:  # if both in the same 5 minutes, assume the stop (pessimistic)
                outcome = ("stop", sl, ts)
                break
            if hit_tp:
                outcome = ("target", tp, ts)
                break
        if outcome is None and len(bars) and bars.index[-1] >= pd.Timestamp(t1):
            outcome = ("time", float(bars["close"].iloc[-1]), bars.index[-1])
        if outcome:
            status, px, ts = outcome
            ret = d * (px / entry - 1) - FEE_ROUND_TRIP
            # The leveraged copy is liquidated if price touched the liquidation level before
            # the exit (same 5 minutes as a target counts as liquidated: pessimistic).
            # The unleveraged result above is unaffected - at 1x there's nothing to liquidate.
            liq_ts = None
            if liq is not None:
                path = bars[bars.index <= ts]
                hit = path[(path["low"] <= liq) if d > 0 else (path["high"] >= liq)]
                liq_ts = hit.index[0] if len(hit) else None
            end = liq_ts if liq_ts is not None else ts
            # Perp funding at each funding time the trade was open for: longs pay a positive rate, shorts receive it.
            paid = funding[(funding.index > pd.Timestamp(t0)) & (funding.index.floor("min") <= pd.Timestamp(end))] \
                if len(funding) else funding
            fund = float(d * paid.sum())
            # Losing the whole margin is the floor: an isolated position can't cost more than the account.
            ret_acct = -1.0 if liq_ts is not None else max(-1.0, lev * (ret - fund))
            pnl = margin * ret_acct if margin is not None else None
            with _db() as con:
                con.execute("UPDATE trades SET status=?, exit_price=?, exit_utc=?, ret_net=?, leverage=?,"
                            " funding=?, ret_acct=?, liquidated=?, pnl_usd=? WHERE id=?",
                            (status, px, pd.Timestamp(ts).isoformat(), ret, lev, fund, ret_acct,
                             int(liq_ts is not None), pnl, tid))
            closed.append({"id": tid, "source": source, "decision": decision, "status": status,
                           "entry": entry, "exit": px, "ret_net": ret, "leverage": lev, "liq": liq,
                           "liquidated": liq_ts is not None, "funding": fund, "ret_acct": ret_acct,
                           "margin": margin, "pnl_usd": pnl})
    return closed


def wallet(book: str = "amadeus") -> dict:
    """A paper wallet since the last /wallet reset: the starting money plus the $ result
    of every trade it went all-in on, one at a time. book: a key of BOOKS."""
    k_since, k_start = BOOKS[book]
    since = int(get_kv(k_since, "0"))
    start = float(get_kv(k_start, str(START_EQUITY)))
    with _db() as con:
        where, args = _book_sql(book)
        df = pd.read_sql_query(f"SELECT * FROM trades WHERE margin IS NOT NULL AND id > ? AND {where} ORDER BY id",
                               con, params=(since, *args))
    done = df[df["status"] != "open"]   # id order = the order the wallet went through them
    eq = start + done["pnl_usd"].cumsum()
    peaks = np.maximum.accumulate(np.r_[start, eq.values])
    balance = float(eq.iloc[-1]) if len(eq) else start
    op = df[df["status"] == "open"]
    return {"start": start, "balance": balance, "done": done,
            "open": op.iloc[0].to_dict() if len(op) else None,
            "peak": float(peaks.max()), "max_dd": float(min(0.0, (eq.values / peaks[1:] - 1).min())) if len(eq) else 0.0,
            "broke": balance * FIXED_LEVERAGE < MIN_NOTIONAL}


def wallet_reset(books=tuple(BOOKS)) -> None:
    """Start fresh wallets with START_EQUITY (both by default, so the race restarts level).
    Old trades stay in the log, just not in the balance."""
    with _db() as con:
        last = con.execute("SELECT COALESCE(MAX(id), 0) FROM trades").fetchone()[0]
    for b in books:
        k_since, k_start = BOOKS[b]
        set_kv(k_since, str(last))
        set_kv(k_start, str(START_EQUITY))


def wallet_text(book: str = "amadeus") -> str:
    w = wallet(book)
    bal, start, d, lev = w["balance"], w["start"], w["done"], FIXED_LEVERAGE
    lines = [f"{TITLES[book]}: ${bal:,.2f}  (started with ${start:,.0f}, {(bal / start - 1) * 100:+.1f}%)"]
    if len(d):
        notional = d["margin"] * d["leverage"]
        fees = float((notional * FEE_ROUND_TRIP).sum())
        fund = float((notional * d["funding"].fillna(0)).sum())
        won = int((d["pnl_usd"] > 0).sum())
        liqd = int(d["liquidated"].fillna(0).sum())
        lines.append(f"{len(d)} trades: {won} won, {len(d) - won} lost" + (f", {liqd} liquidated 💥" if liqd else "")
                     + f". Peak ${w['peak']:,.2f}, worst drop {w['max_dd'] * 100:.1f}% from a peak.")
        lines.append(f"Paid ${fees:,.2f} in fees; funding {'paid' if fund >= 0 else 'received'} ${abs(fund):,.2f}.")
    o = w["open"]
    if o is not None:
        liq = f", liquidation {o['liq']:,.0f}" if o["liq"] is not None and pd.notna(o["liq"]) else ""
        lines.append(f"In a trade: #{o['id']} {o['decision']} @ {o['entry']:,.0f} - ${o['margin']:,.2f} × "
                     f"{o['leverage']:g}x = ${o['margin'] * o['leverage']:,.0f} position{liq}.")
    elif w["broke"]:
        lines.append(f"💸 Broke: ${bal:,.2f} × {lev:g}x is below the ${MIN_NOTIONAL:,.0f} minimum order. "
                     f"/wallet reset starts again with ${START_EQUITY:,.0f}.")
    elif book == "follow":
        lines.append(f"Waiting for a top-20 trader to open or add a big BTC position. All-in at {lev:g}x, "
                     "same stops and fees as mine.")
    elif book == "copy":
        lines.append(f"Waiting for 60%+ of the top 20's BTC holders to agree (checked hourly). All-in at {lev:g}x, "
                     "same stops and fees as mine.")
    else:
        lines.append(f"Waiting for the next lean. She goes all-in at {lev:g}x: liquidation "
                     f"{(1 / lev - MAINT_MARGIN) * 100:.1f}% away, fees {lev * FEE_ROUND_TRIP * 100:.1f}% "
                     "of the wallet per trade.")
    return "\n".join(lines)


def log_vol(s: dict) -> bool:
    """Record the volatility forecast. One row per hourly bar; repeats are ignored."""
    v = s.get("vol")
    if not v:
        return False
    created = s["time"].astimezone(timezone.utc)
    with _db() as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO vol_forecasts (created_utc, expires_utc, price, range_med, range_lo,"
            " range_hi, p_big2, regime, percentile) VALUES (?,?,?,?,?,?,?,?,?)",
            (created.isoformat(), (created + timedelta(hours=HORIZON_H)).isoformat(), s["price"],
             v["range_med"], v["range_lo"], v["range_hi"], v["p_big"].get(2.0), v["regime"],
             v["percentile"]))
        return cur.rowcount > 0


def resolve_vol(fetch_5m) -> int:
    """Fill in the actual range for forecasts whose 4h are up. Same definition as
    vol_model.vol_target: log(highest high / lowest low) over the window, in %.
    Forecasts older than 7 days that never got data are left alone."""
    now = datetime.now(timezone.utc)
    with _db() as con:
        rows = con.execute("SELECT created_utc, expires_utc FROM vol_forecasts"
                           " WHERE actual IS NULL ORDER BY created_utc").fetchall()
    rows = [(datetime.fromisoformat(c), datetime.fromisoformat(e)) for c, e in rows]
    rows = [(t0, t1) for t0, t1 in rows if t1 <= now and now - t0 < timedelta(days=7)]
    if not rows:
        return 0
    bars = fetch_5m(rows[0][0])
    done = 0
    for t0, t1 in rows:
        w = bars[(bars.index > pd.Timestamp(t0)) & (bars.index <= pd.Timestamp(t1))]
        if not len(w) or bars.index[-1] < pd.Timestamp(t1):
            continue
        actual = float(np.log(w["high"].max() / w["low"].min()) * 100)
        with _db() as con:
            con.execute("UPDATE vol_forecasts SET actual=? WHERE created_utc=?", (actual, t0.isoformat()))
        done += 1
    return done


def trades_df() -> pd.DataFrame:
    with _db() as con:
        return pd.read_sql_query("SELECT * FROM trades ORDER BY id", con)


def stats_text() -> str:
    df = trades_df()
    if df.empty:
        return "No paper trades logged yet."
    copy = df[(df["source"] == "copy") & (df["status"] != "open")]
    follow = df[(df["source"] == "follow") & (df["status"] != "open")]
    df = df[~df["source"].isin(["copy", "follow"])]   # Amadeus's own calls; the Hyperliquid books get their own lines
    done = df[df["status"] != "open"].sort_values("exit_utc")
    lines = [f"Paper log: {len(df)} calls ({(df['status']=='open').sum()} still open)"]
    if done.empty:
        return "\n".join(lines + ["None finished yet - check back in a few hours."])

    def block(name, g):
        if g.empty:
            return f"{name}: -"
        wins = (g["ret_net"] > 0).mean()
        return (f"{name}: {len(g)} trades, win {wins:.0%}, avg {g['ret_net'].mean()*100:+.2f}%, "
                f"total {g['ret_net'].sum()*100:+.2f}%  "
                f"[TP {(g['status']=='target').sum()} / SL {(g['status']=='stop').sum()} / time {(g['status']=='time').sum()}]")

    lines.append(block("All", done))
    lines.append(block("Longs", done[done["decision"] == "LONG"]))
    lines.append(block("Shorts", done[done["decision"] == "SHORT"]))
    lines.append(block("Auto (hourly)", done[done["source"] == "auto"]))
    lines.append(block("Alerts (pings)", done[done["source"] == "alert"]))
    lines.append(block("Asked", done[done["source"] == "asked"]))
    lines.append(block("Copy (HL top 20)", copy))
    lines.append(block("Follow (HL fresh entries)", follow))
    if "news_score" in done and done["news_score"].notna().any():
        dirn = done["decision"].map({"LONG": 1, "SHORT": -1})
        agree = done[(done["news_score"] * dirn) >= 1]
        against = done[(done["news_score"] * dirn) <= -1]
        lines.append(block("News agreed", agree))
        lines.append(block("News against", against))
    if len(done) < 30:
        lines.append(f"\nOnly {len(done)} finished trades - too few to judge. Aim for 50+.")
    lines.append("These are per trade, unleveraged, after 0.10% fees - is the call itself any good? "
                 "The wallet above is what the same calls did with real-looking leverage.")
    return "\n".join(lines)


def vol_stats_text() -> str:
    """How the live volatility forecasts compare with what BTC actually did."""
    with _db() as con:
        df = pd.read_sql_query("SELECT * FROM vol_forecasts WHERE actual IS NOT NULL", con)
    if df.empty:
        return "Vol forecasts: none checked yet (each one needs its 4h to pass)."
    inside = ((df["actual"] >= df["range_lo"]) & (df["actual"] <= df["range_hi"])).mean()
    lines = [f"Vol forecasts: {len(df)} checked. Median forecast {df['range_med'].median():.2f}% "
             f"vs actual {df['actual'].median():.2f}%",
             f"  Inside the 80% band: {inside:.0%} (should be ~80%)"]
    if df["p_big2"].notna().any():
        lines.append(f"  Range > 2%: forecast {df['p_big2'].mean():.0%}, actual {(df['actual'] > 2).mean():.0%}")
    if len(df) < 100:
        # Hourly forecasts over 4h windows overlap, so n hours ~ n/4 independent checks.
        lines.append("  Too few to judge yet - aim for 100+ (about 4 days).")
    return "\n".join(lines)


def counts() -> dict:
    """Quick totals for /status."""
    with _db() as con:
        open_, done = con.execute("SELECT SUM(status='open'), SUM(status!='open') FROM trades").fetchone()
        logged, checked, last = con.execute(
            "SELECT COUNT(*), COUNT(actual), MAX(created_utc) FROM vol_forecasts").fetchone()
    return {"open": open_ or 0, "done": done or 0, "vol_logged": logged, "vol_checked": checked,
            "vol_last": datetime.fromisoformat(last) if last else None}


def open_text() -> str:
    df = trades_df()
    op = df[df["status"] == "open"]
    if op.empty:
        return "No open paper trades."
    out = []
    for _, r in op.iterrows():
        lev = f"  {r['leverage']:g}x" if pd.notna(r["leverage"]) else ""
        tag = f" [{r['source']}]" if r["source"] in ("copy", "follow") else ""
        liq = f" (liq {r['liq']:,.0f})" if pd.notna(r["liq"]) else ""
        out.append(f"#{r['id']}{tag} {r['decision']} @ {r['entry']:,.0f}  TP {r['tp']:,.0f}  SL {r['sl']:,.0f}{lev}{liq}  "
                   f"expires {datetime.fromisoformat(r['expires_utc']):%d %b %H:%M} UTC")
    return "\n".join(out)


def export_csv(path) -> None:
    trades_df().to_csv(path, index=False)
