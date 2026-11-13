"""Why did BTC pump / dump?

1. Find the biggest move in the window you asked about.
2. Check what else happened during exactly that stretch: other cryptos,
   stocks, the dollar, gold, oil, open interest, buy/sell pressure,
   headlines, scheduled US data.
3. Give the most likely cause - or say it's unclear when nothing lines up.

    python explain.py "why did btc dump in the last 24h"
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import news
from config import ANTHROPIC_API_KEY, BOT_NAME, DISPLAY_TZ
from data import cached_yahoo, klines, open_interest

TZ = ZoneInfo(DISPLAY_TZ)
DOWN_WORDS = (r"crash\w*|dump\w*|drop\w*|fall\w*|fell|plung\w*|tank\w*|sell-?offs?|dip|dipped|dipping"
              r"|bleed\w*|nuke\w*|slid|slide|sliding|red|down")
UP_WORDS = (r"spik\w*|pump\w*|rall\w*|rise|rises|rising|rose|surg\w*|moon\w*|jump\w*|ripp\w*|rip"
            r"|green|up|bounc\w*|soar\w*|squeez\w*")
_MOVE = rf"{DOWN_WORDS}|{UP_WORDS}|move|moved|moving|price|btc|bitcoin|market|happened|volatil\w*"
WHY_PATTERN = re.compile(rf"(\bwhy\b|what caused|\breason\b).*\b({_MOVE})\b"
                         rf"|\bwhat happened\b|\b({_MOVE})\b.*\bwhy\b", re.I)


# --------------------------------------------------------------------------
# Understanding the question
# --------------------------------------------------------------------------

def parse_window(text: str) -> int:
    t = text.lower()
    m = re.search(r"(\d+)\s*(h|hr|hrs|hours?|d|days?|w|wk|weeks?)\b", t)
    if m:
        n, u = int(m.group(1)), m.group(2)
        hours = n if u.startswith("h") else n * 24 if u.startswith("d") else n * 168
    elif "week" in t:
        hours = 168
    elif "yesterday" in t:
        hours = 48
    elif re.search(r"\b(hour|just now|right now|minutes?|earlier)\b", t):
        hours = 3
    elif "weekend" in t:
        hours = 72
    else:
        hours = 24
    return max(1, min(168, hours))


def parse_side(text: str) -> str | None:
    t = text.lower()
    dn, up = re.search(rf"\b({DOWN_WORDS})\b", t), re.search(rf"\b({UP_WORDS})\b", t)
    if dn and not up:
        return "down"
    if up and not dn:
        return "up"
    return None


# --------------------------------------------------------------------------
# Finding the move
# --------------------------------------------------------------------------

def biggest_legs(df: pd.DataFrame, bar: pd.Timedelta) -> dict:
    """Largest peak->trough drop and trough->peak rise (using highs/lows)."""
    hi, lo, idx = df["high"].values, df["low"].values, df.index
    dn, up = (0.0, 0, 0), (0.0, 0, 0)
    pk = tr = 0
    for j in range(len(df)):
        if hi[j] > hi[pk]:
            pk = j
        if lo[j] < lo[tr]:
            tr = j
        d = lo[j] / hi[pk] - 1
        u = hi[j] / lo[tr] - 1
        if d < dn[0]:
            dn = (d, pk, j)
        if u > up[0]:
            up = (u, tr, j)

    def leg(x, kind):
        pct, i, j = x
        return {"kind": kind, "pct": pct * 100,
                "t0": idx[i] - bar, "t1": idx[j],
                "p0": float(hi[i] if kind == "down" else lo[i]),
                "p1": float(lo[j] if kind == "down" else hi[j])}

    return {"down": leg(dn, "down"), "up": leg(up, "up")}


def _pct(series: pd.Series | None, t0, t1) -> float | None:
    if series is None or len(series) == 0:
        return None
    s = series.dropna()
    a, b = s[s.index <= t0], s[s.index <= t1]
    if a.empty or b.empty:
        return None
    if not ((s.index > t0) & (s.index <= t1)).any():
        return None  # market closed / no update during the move
    return float(b.iloc[-1] / a.iloc[-1] - 1) * 100


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

def gather(text: str) -> dict:
    hours = parse_window(text)
    side = parse_side(text)
    now = datetime.now(timezone.utc)
    use_15m = hours <= 72
    interval = "15m" if use_15m else "1h"
    bar = pd.Timedelta(minutes=15) if use_15m else pd.Timedelta(hours=1)
    start_ms = int((now - timedelta(hours=hours + 72)).timestamp() * 1000)

    btc = klines("BTCUSDT", interval, start_ms)
    win = btc[btc.index > pd.Timestamp(now - timedelta(hours=hours))]
    if len(win) < 2:
        raise RuntimeError("Not enough price data for that window.")
    base = btc[btc.index <= win.index[0]]

    legs = biggest_legs(win, bar)
    if side:
        main = legs[side]
    else:
        main = legs["down"] if abs(legs["down"]["pct"]) >= legs["up"]["pct"] else legs["up"]
    other = legs["up" if main["kind"] == "down" else "down"]
    t0, t1 = main["t0"], main["t1"]

    ev: dict = {
        "hours": hours, "asked_side": side, "now": now,
        "price_now": float(win["close"].iloc[-1]),
        "change_window_pct": float(win["close"].iloc[-1] / win["open"].iloc[0] - 1) * 100,
        "range": (float(win["low"].min()), float(win["high"].max())),
        "main": main, "other": other,
    }

    leg_bars = win[(win.index > t0) & (win.index <= t1)]
    if len(base) > 20 and len(leg_bars):
        ev["volume_x"] = float(leg_bars["volume"].mean() / base["volume"].tail(len(base)).mean())
    if len(leg_bars) and leg_bars["volume"].sum() > 0:
        ev["sell_share"] = float(1 - leg_bars["taker_buy_base"].sum() / leg_bars["volume"].sum())

    # other crypto
    for sym, key in (("ETHUSDT", "eth"), ("SOLUSDT", "sol")):
        try:
            k = klines(sym, interval, int((t0 - bar * 2).timestamp() * 1000))
            ev[key] = _pct(k["close"], t0, t1)
        except Exception as e:
            print(f"  [explain] {sym}: {e}")

    # traditional markets (hourly: give the window an hour of slack each side)
    try:
        cross = cached_yahoo(60, max_age_min=20)
        ev["cross"] = {name: _pct(cross.get(name), t0 - pd.Timedelta(hours=1), t1 + pd.Timedelta(hours=1))
                       for name in ("es", "nq", "dxy", "gold", "oil", "us10y")}
    except Exception as e:
        print(f"  [explain] yahoo: {e}")
        ev["cross"] = {}

    # leverage
    try:
        oi = open_interest(period="15m" if hours <= 120 else "1h")
        ev["oi_pct"] = _pct(oi, t0, t1)
    except Exception as e:
        print(f"  [explain] open interest: {e}")

    # headlines: from 3h before the move until 3h after (after = reporters explaining it)
    try:
        ev["headlines"] = news.headlines_between(t0 - pd.Timedelta(hours=3), t1 + pd.Timedelta(hours=3))
    except Exception as e:
        print(f"  [explain] headlines: {e}")
        ev["headlines"] = []
    try:
        ev["events"] = news.events_between(t0 - pd.Timedelta(hours=1), t1)
    except Exception:
        ev["events"] = []
    return ev


# --------------------------------------------------------------------------
# Writing it up
# --------------------------------------------------------------------------

def _t(dt) -> str:
    return pd.Timestamp(dt).tz_convert(TZ).strftime("%a %H:%M")


def _dur(td) -> str:
    m = int(td.total_seconds() // 60)
    return f"{m // 60}h {m % 60:02d}m" if m >= 60 else f"{m}m"


def _f(x, d=1) -> str:
    return "n/a" if x is None else f"{x:+.{d}f}%"


NAMES = {"es": "S&P fut", "nq": "Nasdaq fut", "dxy": "Dollar", "gold": "Gold", "oil": "Oil", "us10y": "10y yield"}


def ranked_headlines(ev: dict) -> list[dict]:
    m = ev["main"]
    sign = -1 if m["kind"] == "down" else 1
    out = []
    for h in ev.get("headlines", []):
        lag = (pd.Timestamp(h["published"]) - m["t0"]).total_seconds() / 3600
        crypto = bool(re.search(r"\b(bitcoin|btc|crypto|ether|etf)\b", h["title"], re.I))
        score = h["impact"] * (1.5 if h["direction"] == sign else 1.0 if h["direction"] == 0 else 0.3)
        score += 0.8 if crypto else 0
        score -= 0.15 * abs(lag)
        if h["impact"] > 0 or crypto:
            out.append({**h, "lag_h": lag, "rank": score, "after": h["published"] > m["t1"]})
    return sorted(out, key=lambda x: -x["rank"])


def rule_read(ev: dict) -> str:
    m = ev["main"]
    down = m["kind"] == "down"
    sign = -1 if down else 1
    causes, amplifiers = [], []

    heads = [h for h in ranked_headlines(ev) if h["impact"] >= 2 and h["direction"] == sign
             and h["lag_h"] <= 1.0]
    if heads:
        h = heads[0]
        when = "just before the move" if h["lag_h"] < 0 else "as it started"
        causes.append(f"news - \"{h['title'][:80]}\" ({h['source']}) came out {when}")

    cr = ev.get("cross", {})
    stocks = [v for v in (cr.get("es"), cr.get("nq")) if v is not None]
    if stocks and np.sign(np.mean(stocks)) == sign and abs(np.mean(stocks)) >= 0.5:
        causes.append(f"a broad risk-{'off' if down else 'on'} move - stocks went {'down' if down else 'up'} too")
    elif cr.get("dxy") is not None and -sign * cr["dxy"] >= 0.4 and \
            not (stocks and np.sign(np.mean(stocks)) == -sign and abs(np.mean(stocks)) >= 0.5):
        causes.append(f"the dollar {'strengthening' if down else 'weakening'}")
    if cr.get("oil") is not None and abs(cr["oil"]) >= 2:
        amplifiers.append(f"oil moved {cr['oil']:+.1f}%")

    for e in ev.get("events", []):
        causes.append(f"scheduled US data ({e['title']} at {_t(e['time'])})")

    o = ev["other"]
    if o["t1"] <= m["t0"] + pd.Timedelta(hours=1) and abs(o["pct"]) >= 0.8 * abs(m["pct"]):
        if down:
            causes.append(f"a pullback after the earlier {o['pct']:+.1f}% run-up (profit-taking)")
        else:
            causes.append(f"a rebound after the earlier {o['pct']:+.1f}% drop "
                          "(dip-buyers stepping in, shorts taking profit)")

    eth = ev.get("eth")
    if eth is not None:
        if np.sign(eth) == sign and abs(eth) >= 0.6 * abs(m["pct"]):
            scope = "the whole crypto market moved together"
        else:
            scope = "it was mostly BTC-specific (ETH didn't follow much) - think ETF flows or big BTC players"
    else:
        scope = None

    oi = ev.get("oi_pct")
    if oi is not None:
        if oi <= -3:
            amplifiers.append("forced liquidations - open interest fell "
                              f"{abs(oi):.1f}%, i.e. leveraged {'longs' if down else 'shorts'} were wiped out")
        elif oi >= 3:
            amplifiers.append(f"new leverage piling in (open interest {oi:+.1f}%)")
    vx = ev.get("volume_x")
    if vx and vx >= 2.5:
        amplifiers.append(f"heavy volume ({vx:.1f}x normal)")

    if causes:
        read = "Most likely: " + "; plus ".join(causes) + "."
    else:
        read = ("No clear outside trigger lined up in time. Moves like this are often large orders "
                "or a chain of stop-losses and liquidations rather than news.")
    if amplifiers:
        read += " Made bigger by " + " and ".join(amplifiers) + "."
    if scope:
        read += f" Scope: {scope}."
    return read


def claude_read(ev: dict, facts: str) -> str:
    heads = ranked_headlines(ev)[:30]
    hl = "\n".join(f"- [{_t(h['published'])}, {h['lag_h']:+.1f}h vs move start] {h['title']} ({h['source']})"
                   for h in heads) or "(none found)"
    prompt = f"""You are {BOT_NAME}, a sharp, honest crypto market analyst in a Telegram chat.
The user asked: "{ev.get('question', '')}"

Measured facts about the move (times in {DISPLAY_TZ}):
{facts}

Headlines published from 3h before to 3h after the move:
{hl}

Explain why BTC moved, using ONLY this evidence.
- First line: the most likely cause, in one sentence.
- Then 2-4 short bullets with the supporting evidence (timing, what else moved, leverage).
- Headlines published AFTER the move are often reporters explaining it; use them as clues, not proof.
- If nothing clearly lines up in time, say the trigger is unclear instead of guessing.
- Never invent facts, numbers or events. Plain text, no markdown headers, under 150 words."""
    import llm
    # Raises if cut off or declined - explain() then uses the rule-based read, never half an answer.
    return llm.ask(prompt, max_tokens=2000)


def facts_text(ev: dict) -> str:
    m, o = ev["main"], ev["other"]
    arrow = "▼" if m["kind"] == "down" else "▲"
    word = "drop" if m["kind"] == "down" else "rise"
    lines = [
        f"BTC now {ev['price_now']:,.0f} ({_f(ev['change_window_pct'])} over the last {ev['hours']}h, "
        f"range {ev['range'][0]:,.0f} - {ev['range'][1]:,.0f})",
        "",
        f"Biggest {word}: {arrow} {m['pct']:+.1f}%  {m['p0']:,.0f} → {m['p1']:,.0f}",
        f"  {_t(m['t0'])} → {_t(m['t1'])} ({_dur(m['t1'] - m['t0'])})"
        + (f", volume {ev['volume_x']:.1f}x normal" if ev.get("volume_x") else ""),
    ]
    if not ev["asked_side"] and abs(o["pct"]) >= 0.6 * abs(m["pct"]) and abs(o["pct"]) >= 1:
        lines.append(f"  (also a {o['pct']:+.1f}% {'rise' if o['kind']=='up' else 'drop'} "
                     f"{_t(o['t0'])} → {_t(o['t1'])})")
    lines.append("")
    lines.append("During that stretch:")
    crypto = [f"ETH {_f(ev.get('eth'))}", f"SOL {_f(ev.get('sol'))}"]
    lines.append("  " + " · ".join(crypto))
    cr = ev.get("cross") or {}
    trad = [f"{NAMES[k]} {_f(v)}" for k, v in cr.items() if v is not None]
    closed = [NAMES[k] for k, v in cr.items() if v is None]
    if trad:
        lines.append("  " + " · ".join(trad))
    if closed and len(closed) == len(cr):
        lines.append("  (stock/FX markets were closed)")
    lev = []
    if ev.get("oi_pct") is not None:
        lev.append(f"Open interest {_f(ev['oi_pct'])}")
    if ev.get("sell_share") is not None:
        ss = ev["sell_share"]
        lev.append(f"{max(ss, 1-ss):.0%} of volume was aggressive {'selling' if ss >= 0.5 else 'buying'}")
    if lev:
        lines.append("  " + " · ".join(lev))
    for e in ev.get("events", []):
        lines.append(f"  📅 {e['title']} released {_t(e['time'])}")
    heads = ranked_headlines(ev)[:4]
    if heads:
        lines.append("")
        lines.append("Headlines around then:")
        for h in heads:
            tag = " - after the move" if h["after"] else ""
            lines.append(f"  {news._arrow(h['direction'])} {h['title'][:95]} ({h['source']}, {_t(h['published'])}{tag})")
    return "\n".join(lines)


def explain(text: str) -> str:
    ev = gather(text)
    ev["question"] = text
    facts = facts_text(ev)
    read = None
    if ANTHROPIC_API_KEY:
        try:
            read = claude_read(ev, facts)
        except Exception as e:
            print(f"  [explain] Claude failed, using rules: {e}")
    if not read:
        read = rule_read(ev)
    small = abs(ev["main"]["pct"]) < 1.5
    head = f"{BOT_NAME}: what moved BTC (last {ev['hours']}h, times {DISPLAY_TZ.split('/')[-1]})"
    note = ("\n\n(Honestly a fairly quiet stretch - moves this small usually have no single cause.)"
            if small else "")
    return f"{head}\n\n{facts}\n\nMy read:\n{read}{note}"


if __name__ == "__main__":
    print(explain(" ".join(sys.argv[1:]) or "why did btc move in the last 24h"))
