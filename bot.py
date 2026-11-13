"""Telegram bot: ask for a call any time, get pinged on strong setups,
and have every call paper-traded automatically.

    python bot.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import (Application, CommandHandler, ContextTypes, MessageHandler,
                          filters)

import explain
import hyperliquid as hl
import liqmap
import news
import paper_log as pl
import requests

import whales
from config import (ALLOWED_CHAT_ID, ANTHROPIC_API_KEY, BOT_NAME, DISPLAY_TZ, FEE_ROUND_TRIP, FIXED_LEVERAGE, HORIZON_H,
                    HEARTBEAT_URL, HL_PING_ALT_USD, HL_PING_USD, MIN_NOTIONAL, MODEL_PATH, REPORT_PATH, START_EQUITY, TELEGRAM_TOKEN)
from data import btc_5m_since, funding_since
from signal_engine import compute_signal, format_signal, levels_for, load_model

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

ASK_PATTERN = re.compile(r"\b(long|short|btc|bitcoin|buy|sell|position|call|signal|should i|volatil\w*|risk\w*|stop|calm|vol)\b", re.I)
NEWS_PATTERN = re.compile(r"\b(news|headlines?|happening|war|oil|fed|cpi)\b", re.I)
STATUS_PATTERN = re.compile(r"\b(status|alive|are you (there|up|on|awake))\b", re.I)
WHALE_PATTERN = re.compile(r"\b(whales?|liquidations?|walls?|spoof\w*|order ?books?|liquidity|clusters?|depth)\b", re.I)
MAP_PATTERN = re.compile(r"\b(heat ?map|liq\w* ?map)\b", re.I)
HL_PATTERN = re.compile(r"\b(hyperliquid|hl|top (traders?|wallets?|20)|smart money|copy ?(trad\w*|wallet))\b", re.I)
WALLET_PATTERN = re.compile(r"\b(wallet|balance|money (left|do you have)|how much (money|cash|have you))\b", re.I)
# "ping me if...", "alert me when...", "notify me", "let me know if..." - asking for pings, not for a call now.
ALERT_REQ_PATTERN = re.compile(r"\b((ping|alert|notify|message|text|warn) me|(tell me|let me know) (when|if)"
                               r"|notifications?)\b", re.I)

_model = {"art": None, "mtime": 0.0}
_last_sigma: dict = {"sig": None, "t": None}   # latest expected 4h move (fraction), for follow-trade stops

# Background-job health since this process started (in memory: a restart clears it).
_started = datetime.now(timezone.utc)
_health: dict[str, dict] = {}   # job name -> {"ok": time, "err": (time, text), "fails": n, "last": ...}
SCAN_FAIL_PING = 3              # ping after this many hourly scans fail in a row


def _job_ok(name: str) -> int:
    """Mark a job run as successful. Returns how many runs had failed in a row before it."""
    h = _health.setdefault(name, {"fails": 0})
    prev, h["ok"], h["fails"] = h["fails"], datetime.now(timezone.utc), 0
    return prev


def _job_failed(name: str, e: Exception) -> int:
    """Record a failed job run. Returns how many runs have now failed in a row."""
    h = _health.setdefault(name, {"fails": 0})
    h["err"] = (datetime.now(timezone.utc), f"{type(e).__name__}: {e}"[:200])
    h["fails"] += 1
    return h["fails"]


def _opened_note(tid: int) -> str:
    t = pl.get_trade(tid)
    note = f"\n\n📝 Paper trade #{tid} opened."
    if t["margin"] is not None:
        liq = f", liquidation {t['liq']:,.0f}" if t["liq"] is not None else ""
        return note + (f"\n💰 Wallet all-in: ${t['margin']:,.2f} × {t['leverage']:g}x = "
                       f"${t['margin'] * t['leverage']:,.0f} position{liq}.")
    w = pl.wallet(pl.book_of(t["source"]))
    why = "is broke - /wallet reset" if w["broke"] else f"is already in #{w['open']['id']}"
    return note + f" (The wallet {why}, so this one is logged unleveraged only.)"


def _closed_text(c: dict, balance: float) -> str:
    how = {"target": "take-profit hit ✅", "stop": "stop-loss hit ❌", "time": f"{HORIZON_H}h time stop ⏱"}[c["status"]]
    book = {"copy": "Copy wallet", "follow": "Follow wallet"}.get(c["source"])
    text = ((f"🐳 {book}: " if book else "") + f"Paper trade #{c['id']} {c['decision']} closed: {how}\n"
            f"{c['entry']:,.0f} → {c['exit']:,.0f}  ({c['ret_net']*100:+.2f}% after fees)")
    if c["margin"] is None:
        return text
    lev, notional = c["leverage"], c["margin"] * c["leverage"]
    if c["liquidated"]:
        text += (f"\n💥 At {lev:g}x the wallet was liquidated at {c['liq']:,.0f} before that - "
                 f"lost the whole ${c['margin']:,.2f}.")
    else:
        text += (f"\n💰 At {lev:g}x: {c['ret_acct']*100:+.1f}% → {'+' if c['pnl_usd'] >= 0 else '-'}"
                 f"${abs(c['pnl_usd']):,.2f} (fees ${notional * FEE_ROUND_TRIP:,.2f}, "
                 f"funding {'-' if c['funding'] >= 0 else '+'}${abs(notional * c['funding']):,.2f}).")
    text += f"\n{book or 'Wallet'} now ${balance:,.2f}."
    if balance * FIXED_LEVERAGE < MIN_NOTIONAL:
        text += " 💸 That's too little to open another position - /wallet reset to start again."
    return text


def _dur(td: timedelta) -> str:
    m = max(0, int(td.total_seconds() // 60))
    if m < 60:
        return f"{m}m"
    if m < 48 * 60:
        return f"{m // 60}h {m % 60}m"
    return f"{m // 1440}d {m % 1440 // 60}h"


def model() -> dict:
    """Load the model, reloading automatically after a retrain."""
    mt = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
    if _model["art"] is None or mt != _model["mtime"]:
        _model["art"], _model["mtime"] = load_model(), mt
    return _model["art"]


def model_age_note() -> str:
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(model()["trained_at"])
    except Exception:
        return ""
    if age > timedelta(days=14):
        return f"\n\n(Model is {age.days} days old - send /retrain.)"
    return ""


def authorised(update: Update) -> bool:
    return bool(ALLOWED_CHAT_ID) and str(update.effective_chat.id) == str(ALLOWED_CHAT_ID)


async def guard(update: Update) -> bool:
    if authorised(update):
        return True
    if not ALLOWED_CHAT_ID:
        await update.effective_message.reply_text(
            f"Hi! Your chat id is {update.effective_chat.id}.\n"
            f"Put  ALLOWED_CHAT_ID={update.effective_chat.id}  in the .env file and restart me.")
    else:
        await update.effective_message.reply_text("This is a private bot.")
    return False


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

HELP = (f"I'm {BOT_NAME}. I tell you how much BTC is likely to move in the next {HORIZON_H} hours, "
        "where to put stops, and why it moved.\n\n"
        "Just ask: \"long or short?\" or \"how volatile is it?\"  (or /signal)\n"
        "/stats - how my paper-traded calls and volatility forecasts have done\n"
        "/open - paper trades still running\n"
        f"/wallet - how much of my paper ${START_EQUITY:,.0f} is left (I go all-in at {FIXED_LEVERAGE:g}x); "
        "/wallet reset starts over\n"
        "/status - am I alive, and are my hourly scan, trade checker and news reader working?\n"
        "Ask \"why did btc dump today?\" (or /why 24h) - I'll find the move and what caused it\n"
        "/news - headlines that matter for BTC + upcoming US data\n"
        "/alerts - what I ping you about (and what I don't); /alerts on|off - volatility + US data pings\n"
        "/breaking on|off - pings on major breaking news\n"
        "/whales - whale watch (Binance buying/selling bursts + liquidations, order walls on 7 exchanges); "
        "/whales on|off\n"
        "/hl - the Hyperliquid top 20 traders: what they hold, and the three wallets that copy them; /hl on|off\n"
        "/heatmap - a picture of where the 1,000 biggest Hyperliquid accounts' BTC positions get liquidated\n"
        "/model - how the model did in its backtest\n"
        "/retrain - refresh the model with the latest data (~2-5 min)\n\n"
        "Every hour I quietly paper-trade my direction lean and log my volatility forecast, "
        "so we can keep checking if they're any good. "
        "Not financial advice.")


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await guard(update):
        await update.message.reply_text(HELP)


async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    msg = await update.effective_message.reply_text("Looking at the market...")
    try:
        s = await asyncio.to_thread(compute_signal, model())
    except Exception as e:
        log.exception("signal failed")
        await msg.edit_text(f"Couldn't compute a call: {e}")
        return
    tid = pl.log_signal(s, "asked")
    text = format_signal(s, header=f"{BOT_NAME}:")
    try:
        c = await asyncio.to_thread(hl.fresh_consensus)
        text += "\n\n" + hl.call_section(c, s["decision"])
    except Exception as e:
        log.warning("hyperliquid consensus failed: %s", e)
    books = whales.call_section()
    if books:
        text += "\n\n" + books
    if tid:
        text += _opened_note(tid)
    await msg.edit_text((text + model_age_note())[:4000])


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await guard(update):
        await update.effective_message.reply_text(
            (_all_wallets_text() + "\n\n" + pl.stats_text() + "\n\n" + pl.vol_stats_text())[:4000])


async def cmd_wallet(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    if ctx.args and ctx.args[0].lower() == "reset":
        before = hl.wallets_line()
        pl.wallet_reset()
        hl.mirror_reset()
        await update.message.reply_text(
            f"All wallets reset to ${START_EQUITY:,.0f}, so the race starts level.\nBefore: {before}\n"
            "The old trades stay in /stats; the next trades start the new wallets.")
        return
    await update.effective_message.reply_text(_all_wallets_text()[:4000])


def _all_wallets_text() -> str:
    return "\n\n".join([pl.wallet_text(), hl.mirror_text(), pl.wallet_text("copy"), pl.wallet_text("follow")])


def _news_scorer_status(now: datetime) -> str:
    if not ANTHROPIC_API_KEY:
        return "keywords (no ANTHROPIC_API_KEY)."
    st = pl.get_kv("claude_news")
    if not st:
        return "Claude - no new headlines rated yet."
    st = json.loads(st)
    ago = _dur(now - datetime.fromisoformat(st["t"]))
    if st["ok"]:
        return f"Claude, working (last batch {ago} ago)."
    return f"⚠ Claude failed {ago} ago, so those headlines used keywords: {st['error']}"


JOB_LABELS = {"hourly_scan": "Hourly scan", "resolve_job": "Trade/forecast checker", "news_job": "News reader",
              "whale_job": "Whale watch", "hl_job": "Hyperliquid watch", "hl_refresh_job": "Hyperliquid leaderboard", "liqmap_job": "Liquidation map"}
if HEARTBEAT_URL:
    JOB_LABELS["heartbeat_job"] = "Heartbeat"


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Is the bot alive, and are its background jobs actually working? No network calls,
    so it answers instantly even when Binance is down."""
    if not await guard(update):
        return
    now = datetime.now(timezone.utc)
    tz = ZoneInfo(DISPLAY_TZ)
    lines = [f"✅ {BOT_NAME} is alive on {platform.node() or 'this machine'}. "
             f"Up {_dur(now - _started)} (since {_started.astimezone(tz):%d %b %H:%M %Z})."]

    lines.append("")
    next_t = {j.name: j.next_t for j in ctx.job_queue.jobs()} if ctx.job_queue else {}
    for name, label in JOB_LABELS.items():
        h = _health.get(name, {})
        parts = [f"last OK {_dur(now - h['ok'])} ago" if h.get("ok") else "no successful run yet"]
        if next_t.get(name):
            parts.append(f"next in {_dur(next_t[name] - now)}")
        elif name not in next_t:
            parts.append("NOT SCHEDULED")
        lines.append(f"{'⚠' if h.get('fails') or name not in next_t else '•'} {label}: {', '.join(parts)}")
        if h.get("fails"):
            t, err = h["err"]
            lines.append(f"    failed {h['fails']}x in a row, last {_dur(now - t)} ago: {err}")

    last = _health.get("hourly_scan", {}).get("last")
    if last:
        lines.append("")
        call = last["decision"] if last["decision"] == "NO TRADE" else f"{last['decision']} ({last['strength']})"
        vol = f", vol {last['regime']} ~{last['range_med']:.1f}%" if last.get("regime") else ""
        lines.append(f"Last scan: {call}{vol}, BTC {last['price']:,.0f}")
        bar_age = now - last["bar"]
        if bar_age > timedelta(hours=2):
            lines.append(f"⚠ Its price data was {_dur(bar_age)} old - Binance may be lagging or unreachable.")

    lines.append("")
    try:
        art = model()
        age = now - datetime.fromisoformat(art["trained_at"])
        verdict = "no direction edge" if not any(art["stats"].get(k, {}).get("edge") for k in ("long", "short")) \
            else "direction edge found"
        lines.append(f"Model: trained {_dur(age)} ago, {verdict}, vol model {'on' if art.get('vol') else 'MISSING'}."
                     + (" ⚠ Over 14 days old - send /retrain." if age > timedelta(days=14) else ""))
    except Exception as e:
        lines.append(f"⚠ Model: {e}")

    c = pl.counts()
    w = pl.wallet()
    lines.append(f"Paper trades: {c['open']} open, {c['done']} finished.")
    lines.append(f"💰 Wallet: ${w['balance']:,.2f} at {FIXED_LEVERAGE:g}x"
                 + (" (broke - /wallet reset)" if w["broke"] else
                    f" (in #{w['open']['id']})" if w["open"] is not None else " (waiting for a lean)") + ".")
    lines.append("🐳 " + hl.wallets_line().split(" - ")[0] + ".")
    vol_last = f", last logged {_dur(now - c['vol_last'])} ago" if c["vol_last"] else ""
    lines.append(f"Vol forecasts: {c['vol_logged']} logged, {c['vol_checked']} checked{vol_last}.")
    lines.append(f"Alerts: {pl.get_kv('alerts', 'on').upper()} · Breaking news: {pl.get_kv('breaking', 'on').upper()}")
    lines.append("News scorer: " + _news_scorer_status(now))
    lines.append("Liquidation feed: " + (f"connected {_dur(now - whales.feed['since'])}" if whales.feed["connected"]
                                         else f"⚠ down ({whales.feed['error'] or 'not started'})"))
    t = hl.top_refreshed()
    lines.append(f"Hyperliquid: tracking {len(hl.top())}" + (f" (list refreshed {_dur(now - t)} ago)" if t else
                                                             " - ⚠ list not loaded yet")
                 + (f", {len(hl.state['errors'])} not answering" if hl.state["errors"] else "") + ".")
    await update.effective_message.reply_text("\n".join(lines)[:4000])


async def cmd_open(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if await guard(update):
        await update.message.reply_text(pl.open_text())


def alerts_text() -> str:
    """What Amadeus pings about by herself, what she doesn't and why, and the current switches."""
    alerts, breaking = pl.get_kv("alerts", "on") == "on", pl.get_kv("breaking", "on") == "on"
    onoff = lambda b: "ON" if b else "OFF"
    lines = ["Here's what I ping you about by myself:", "",
             f"⚡ Volatility rising or calming down - when my {HORIZON_H}h move forecast enters ELEVATED "
             "(top 25% of the past year) or HIGH (top 8%), and when it drops back. At most one every "
             f"3h, except a first jump to HIGH. [{onoff(alerts)}]",
             f"📅 US data heads-up - about 2h before CPI, jobs, FOMC and other big US releases, "
             f"when BTC often jumps either way. [{onoff(alerts)}]",
             f"📰 Breaking news - major headlines, checked every 10 min. [{onoff(breaking)}]",
             f"🐋 Whale watch - big market buying/selling and liquidation cascades on Binance, big order walls "
             f"on Binance, Coinbase, OKX, Kraken, Bitfinex and Bitstamp (/whales for details). "
             f"[{onoff(pl.get_kv('whales', 'on') == 'on')}]",
             f"🐳 Hyperliquid top 20 - when one of the 20 most profitable traders of the last 30 days opens, "
             f"flips or closes a {hl._m(HL_PING_USD)}+ BTC position ({hl._m(HL_PING_ALT_USD)}+ on other coins and "
             f"stock/commodity futures), or adds "
             f"or cuts a big piece of one. "
             f"Also when the mirror wallet switches between long, short and flat. "
             f"Context, not a signal: copying them is being tested by three paper wallets. "
             f"[{onoff(pl.get_kv('hl', 'on') == 'on')}]",
             "⚠ If my hourly scan keeps failing. [always on]", ""]
    try:
        art = model()
        sides = [k.upper() for k in ("long", "short") if art["stats"].get(k, {}).get("edge")]
        auc = f"AUC {art['auc']:.2f}, 0.50 = coin flip"
    except Exception:
        sides, auc = [], "no model loaded"
    if sides:
        lines.append(f"🟢🔴 Strong {' / '.join(sides)} setups - my latest backtest found an edge after fees "
                     f"on that side, so I ping the strongest calls. [{onoff(alerts)}]")
        if len(sides) == 2:
            lines.append("")
    if len(sides) < 2:
        side = "either side" if not sides else f"the {'SHORT' if sides == ['LONG'] else 'LONG'} side"
        where = "in calling direction" if not sides else "on that side"
        lines.append(f"What I don't ping: \"go long/short now\" opportunities on {side}. My backtest found no "
                     f"edge after fees {where} ({auc}), so a ping would be a coin flip dressed "
                     "up as a setup. If a /retrain ever finds an edge, those pings switch on by themselves.")
        lines.append("")
    lines.append("Turn them on/off: /alerts on|off (volatility + US data), /breaking on|off (news). "
                 "Ask \"long or short?\" any time for my current read.")
    return "\n".join(lines)


async def cmd_alerts(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    if ctx.args and ctx.args[0].lower() in ("on", "off"):
        pl.set_kv("alerts", ctx.args[0].lower())
        await update.message.reply_text(f"Alerts are {ctx.args[0].upper()}. (/alerts on or /alerts off)")
        return
    await update.effective_message.reply_text(alerts_text())


async def cmd_news(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    msg = await update.message.reply_text("Reading the news...")
    try:
        await asyncio.to_thread(news.refresh_if_stale, 5)
        text = await asyncio.to_thread(news.news_text)
    except Exception as e:
        log.exception("news failed")
        text = f"Couldn't read the news: {e}"
    await msg.edit_text(text[:4000], disable_web_page_preview=True)


async def cmd_why(update: Update, ctx: ContextTypes.DEFAULT_TYPE, text: str | None = None):
    if not await guard(update):
        return
    q = text or ("why did btc move " + " ".join(ctx.args or []))
    msg = await update.effective_message.reply_text("Digging into the move... (charts, stocks, leverage, headlines)")
    try:
        out = await asyncio.to_thread(explain.explain, q)
    except Exception as e:
        log.exception("explain failed")
        out = f"Couldn't work it out: {e}"
    await msg.edit_text(out[:4000], disable_web_page_preview=True)


async def cmd_breaking(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    if ctx.args and ctx.args[0].lower() in ("on", "off"):
        pl.set_kv("breaking", ctx.args[0].lower())
    state = pl.get_kv("breaking", "on")
    await update.message.reply_text(f"Breaking-news pings are {state.upper()}. (/breaking on or /breaking off)")


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    if not REPORT_PATH.exists():
        await update.message.reply_text("No training report yet. Run /retrain.")
        return
    rep = REPORT_PATH.read_text(encoding="utf-8")
    # Telegram's message limit is 4096 characters; send the summary part.
    cut = rep.split("Final model weights")[0]
    if "=== Volatility" in rep:
        cut += "\n=== Volatility" + rep.split("=== Volatility")[1]
    await update.message.reply_text(cut[:4000])


async def cmd_retrain(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    await update.message.reply_text("Retraining with fresh data... this takes a few minutes.")
    try:
        import train
        await asyncio.to_thread(train.run, False)
        rep = REPORT_PATH.read_text(encoding="utf-8")
        verdicts = "\n".join(l for l in rep.splitlines() if l.startswith(("Verdict", "Direction accuracy", "R²")))
        await update.message.reply_text(f"Done. Model reloaded.\n{verdicts}")
    except Exception as e:
        log.exception("retrain failed")
        await update.message.reply_text(f"Retrain failed: {e}")


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    text = update.message.text or ""
    if STATUS_PATTERN.search(text):
        await cmd_status(update, ctx)
    elif MAP_PATTERN.search(text):
        await cmd_heatmap(update, ctx)
    elif HL_PATTERN.search(text) and not re.search(r"\b(long or short|should i)\b", text, re.I):
        await cmd_hl(update, ctx)
    elif WALLET_PATTERN.search(text):
        await cmd_wallet(update, ctx)
    elif WHALE_PATTERN.search(text) and not re.search(r"\b(why|long or short|should i)\b", text, re.I):
        await cmd_whales(update, ctx)
    elif ALERT_REQ_PATTERN.search(text) and not re.search(r"\b(should i|long or short)\b", text, re.I):
        await update.message.reply_text(alerts_text())
    elif explain.WHY_PATTERN.search(text) and not re.search(r"\b(should|long or short)\b", text, re.I):
        await cmd_why(update, ctx, text)
    elif NEWS_PATTERN.search(text) and not re.search(r"\b(long|short)\b", text, re.I):
        await cmd_news(update, ctx)
    elif ASK_PATTERN.search(text):
        await cmd_signal(update, ctx)
    else:
        await update.message.reply_text("Ask me \"long or short?\" any time, or /help.")


# --------------------------------------------------------------------------
# Background jobs
# --------------------------------------------------------------------------

REGIME_RANK = {"CALM": 0, "NORMAL": 1, "ELEVATED": 2, "HIGH": 3}
VOL_PING_COOLDOWN_H = 3


async def _send(ctx, text: str):
    await ctx.bot.send_message(chat_id=int(ALLOWED_CHAT_ID), text=text[:4000],
                               disable_web_page_preview=True)


def _vol_ping(s: dict, now: datetime) -> str | None:
    """Message for a volatility regime change (or None), and update the stored regime.
    At most one ping per VOL_PING_COOLDOWN_H hours, so a forecast hovering on a regime
    boundary can't ping every hour. Exception: the first step up to HIGH always gets through."""
    v = s["vol"]
    prev = pl.get_kv("vol_regime")
    rank = REGIME_RANK[v["regime"]]
    rising = rank >= 2 and (prev is None or REGIME_RANK.get(prev, 0) < rank)
    calming = prev in ("ELEVATED", "HIGH") and rank <= 1
    if rising or calming:
        last_t = pl.get_kv("vol_ping_t")
        cooling = last_t is not None and \
            now - datetime.fromisoformat(last_t) < timedelta(hours=VOL_PING_COOLDOWN_H)
        if cooling and not (v["regime"] == "HIGH" and pl.get_kv("vol_ping_regime") != "HIGH"):
            # Keep the old regime stored: if the change lasts, it's announced after the cooldown.
            log.info("vol ping held (cooldown): %s -> %s", prev, v["regime"])
            return None
        pl.set_kv("vol_ping_t", now.isoformat())
        pl.set_kv("vol_ping_regime", v["regime"])
    pl.set_kv("vol_regime", v["regime"])
    if rising:
        return format_signal(s, header=f"⚡ {BOT_NAME}: volatility rising ({prev or '?'} → {v['regime']})")
    if calming:
        return (f"😌 {BOT_NAME}: things are calming down - next {HORIZON_H}h range "
                f"~{v['range_med']:.1f}% ({v['regime']}).")
    return None


def _copy_signal(s: dict, c: dict) -> dict:
    """Amadeus's call with the direction swapped for the top 20's majority: same price, time,
    stops and targets for that side, so the only difference between the wallets is who picked the side."""
    holders = c["long"] + c["short"]
    return {**s, "decision": c["side"], "strength": "copy", "p_up": c["long"] / holders,
            "levels": s["both_levels"][c["side"]], "news": None}


async def hourly_scan(ctx: ContextTypes.DEFAULT_TYPE):
    try:
        s = await asyncio.to_thread(compute_signal, model())
    except Exception as e:
        log.exception("hourly scan failed")
        # Sent even with /alerts off: that switch is for market pings, not for "I'm broken".
        if _job_failed("hourly_scan", e) == SCAN_FAIL_PING:
            await _send(ctx, f"⚠ {BOT_NAME}: my hourly scan has failed {SCAN_FAIL_PING} times in a row, "
                             f"so no forecasts or paper trades are being logged.\n{_health['hourly_scan']['err'][1]}\n"
                             "Send /status for details.")
        return
    if s.get("sigma_pct"):
        _last_sigma.update(sig=s["sigma_pct"] / 100, t=datetime.now(timezone.utc))
    v = s.get("vol")
    log.info("scan: %s %s p_up=%.3f vol=%s", s["decision"], s["strength"], s["p_up"],
             v and f"{v['regime']} {v['range_med']:.2f}%")
    now = datetime.now(timezone.utc)
    alerts = pl.get_kv("alerts", "on") == "on"

    # Close whatever expired on this hour first, so the wallet is free to take the new trade
    # (otherwise a trade ending at HH:00 would block it until the 10-minute checker ran).
    await resolve_job(ctx)

    # Record-keeping runs even with alerts off: /alerts only silences the pings.
    # 1) Log the vol forecast; resolve_job scores it against the actual range after 4h.
    pl.log_vol(s)

    # 2) Paper-trade the direction lean every hour (lean or strong; NO TRADE logs nothing).
    #    A strong call with a backtested edge gets pinged, so it's logged as 'alert' instead.
    last_dir, last_t = pl.get_kv("last_alert_dir"), pl.get_kv("last_alert_t")
    repeat = last_dir == s["decision"] and last_t is not None and \
        now - datetime.fromisoformat(last_t) < timedelta(hours=HORIZON_H)
    ping_dir = alerts and s["strength"] == "strong" and s["edge"] and not repeat
    tid = pl.log_signal(s, "alert" if ping_dir else "auto")
    if tid:
        log.info("paper trade #%s opened (%s %s)", tid, s["decision"], s["strength"])

    # 2b) The copy wallet: the Hyperliquid top 20's BTC majority, same stops/leverage/fees as mine.
    try:
        c = await asyncio.to_thread(hl.fresh_consensus)
        if c["side"]:
            cid = pl.log_signal(_copy_signal(s, c), "copy")
            if cid:
                log.info("copy trade #%s opened (%s, %d long / %d short)", cid, c["side"], c["long"], c["short"])
    except Exception:
        log.exception("copy trade failed")

    if _job_ok("hourly_scan") >= SCAN_FAIL_PING:
        await _send(ctx, f"✅ {BOT_NAME}: hourly scan is working again.")
    _health["hourly_scan"]["last"] = {
        "bar": s["time"].astimezone(timezone.utc), "price": s["price"], "decision": s["decision"],
        "strength": s["strength"], "regime": v and v["regime"], "range_med": v and v["range_med"]}
    if not alerts:
        return

    # 3) Volatility stepping up (calm/normal -> elevated/high, or elevated -> high) or calming down
    if v:
        text = _vol_ping(s, now)
        if text:
            await _send(ctx, text)

    # 4) Heads-up before scheduled US data (CPI, jobs, FOMC...)
    try:
        upcoming = await asyncio.to_thread(news.upcoming_events, 2, 0)
    except Exception:
        upcoming = []
    for e in upcoming:
        key = f"event_{e['title']}_{e['time'].isoformat()}"
        if pl.get_kv(key) or e["mins"] < 0:
            continue
        pl.set_kv(key, "1")
        rng = f" Right now I expect a ~{v['range_med']:.1f}% range over {HORIZON_H}h without it." if v else ""
        await _send(ctx, f"📅 {BOT_NAME}: {news.fmt_event(e)}.\nBTC often jumps on US data releases "
                         f"- either direction.{rng}\nWiden stops, size down, or wait until it's out.")

    # 5) Direction pings - only if the latest backtest found a real edge
    if ping_dir:
        pl.set_kv("last_alert_dir", s["decision"])
        pl.set_kv("last_alert_t", now.isoformat())
        emoji = "🟢" if s["decision"] == "LONG" else "🔴"
        text = format_signal(s, header=f"{emoji} {BOT_NAME}: strong {s['decision'].lower()} setup")
        if tid:
            text += _opened_note(tid)
        await _send(ctx, text)


async def resolve_job(ctx: ContextTypes.DEFAULT_TYPE):
    # Separate tries: trades are already marked closed in the DB by resolve_open,
    # so a vol-resolve failure must not stop them being announced.
    err = None
    try:
        closed = await asyncio.to_thread(pl.resolve_open, btc_5m_since, funding_since)
    except Exception as e:
        log.exception("resolve failed")
        err, closed = e, []
    try:
        await asyncio.to_thread(pl.resolve_vol, btc_5m_since)
    except Exception as e:
        log.exception("vol resolve failed")
        err = err or e
    if err:
        _job_failed("resolve_job", err)
    else:
        _job_ok("resolve_job")
    balances = {b: pl.wallet(b)["balance"] for b in pl.BOOKS} if closed else {}
    for c in closed:
        # Hourly, copy and follow trades close quietly unless the wallet was in them - then the money moved.
        if c["source"] in ("auto", "copy", "follow") and c["margin"] is None:
            continue
        await _send(ctx, _closed_text(c, balances[pl.book_of(c["source"])]))


async def whale_job(ctx: ContextTypes.DEFAULT_TYPE):
    try:
        events = await asyncio.to_thread(whales.check_all)   # runs (and logs) even when muted
        _job_ok("whale_job")
    except Exception as e:
        log.exception("whale job failed")
        _job_failed("whale_job", e)
        return
    if pl.get_kv("whales", "on") != "on":
        return
    for ev in events:
        await _send(ctx, whales.format_event(ev, BOT_NAME, ZoneInfo(DISPLAY_TZ)))


async def cmd_whales(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    if ctx.args and ctx.args[0].lower() in ("on", "off"):
        pl.set_kv("whales", ctx.args[0].lower())
    on = pl.get_kv("whales", "on") == "on"
    await update.effective_message.reply_text(whales.status_text(on, ZoneInfo(DISPLAY_TZ)))


async def _follow_trade(ev: dict) -> int | None:
    """The follow wallet: a fresh big BTC open/add by a top-20 trader becomes a 4h paper trade their
    way, entered now at the current price with the same stops/targets/leverage as my own trades."""
    side = "LONG" if ev["usd"] > 0 else "SHORT"
    now = datetime.now(timezone.utc)
    sig = _last_sigma["sig"]
    if sig is None or now - _last_sigma["t"] > timedelta(hours=2):
        s = await asyncio.to_thread(compute_signal, model())
        sig = s["sigma_pct"] / 100
        _last_sigma.update(sig=sig, t=now)
    price = await asyncio.to_thread(hl.btc_price)
    return pl.log_signal({"time": now, "price": price, "decision": side, "strength": "follow", "p_up": None,
                          "levels": levels_for(price, sig)[side], "news": None}, "follow")


async def hl_job(ctx: ContextTypes.DEFAULT_TYPE):
    try:
        events = await asyncio.to_thread(hl.check_all)    # runs (and logs) even when muted
        _job_ok("hl_job")
    except Exception as e:
        log.exception("hyperliquid job failed")
        _job_failed("hl_job", e)
        return
    on = pl.get_kv("hl", "on") == "on"
    for ev in events:
        text = hl.format_event(ev, BOT_NAME)
        if ev["coin"] == "BTC" and ev["kind"] in ("open", "add"):
            try:   # paper-traded even when the pings are muted
                tid = await _follow_trade(ev)
                if tid:
                    t = pl.get_trade(tid)
                    text += (f"\n📝 Follow wallet: paper {t['decision']} #{tid} at {t['entry']:,.0f}, "
                             f"stop {t['sl']:,.0f}, target {t['tp']:,.0f}, {HORIZON_H}h"
                             + (f", all-in ${t['margin']:,.0f} × {t['leverage']:g}x." if t["margin"] else
                                " (wallet busy - logged unleveraged)."))
            except Exception:
                log.exception("follow trade failed")
        if on:
            await _send(ctx, text)
    try:
        mev = await asyncio.to_thread(hl.mirror_check, funding_since)
    except Exception:
        log.exception("mirror wallet step failed")
        return
    if mev and mev["side_change"] and on:
        await _send(ctx, hl.mirror_event_text(mev, BOT_NAME))


async def liqmap_job(ctx: ContextTypes.DEFAULT_TYPE):
    """Reads the next 100 of the biggest Hyperliquid accounts. Never pings: /heatmap shows the result."""
    try:
        await asyncio.to_thread(liqmap.poll_batch)
        _job_ok("liqmap_job")
    except Exception as e:
        if _job_failed("liqmap_job", e) == 1:
            log.warning("liquidation map read failed: %s", e)


async def cmd_heatmap(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    if not liqmap.accounts():   # normally saved by the daily leaderboard download; fetch it now if that hasn't worked
        msg = await update.effective_message.reply_text("Downloading the Hyperliquid leaderboard (~40MB) for the account list...")
        try:
            await asyncio.to_thread(hl.refresh_top, True)
        except Exception as e:
            log.exception("leaderboard download for the map failed")
            await msg.edit_text(f"Couldn't download the Hyperliquid leaderboard: {e}. Try /heatmap again in a few minutes.")
            return
        await msg.edit_text(liqmap.progress_text())
        return
    m = liqmap.build()
    if not liqmap.ready(m):
        await update.effective_message.reply_text(liqmap.progress_text())
        return
    tz = ZoneInfo(DISPLAY_TZ)
    try:
        png = await asyncio.to_thread(liqmap.draw, m, tz)
    except Exception as e:
        log.exception("liquidation map drawing failed")
        await update.effective_message.reply_text(f"Couldn't draw the map ({e}). The numbers:\n\n{liqmap.summary(m, tz)}")
        return
    text = liqmap.summary(m, tz)
    if len(text) <= 1024:   # Telegram's limit for a caption
        await update.effective_message.reply_photo(png, caption=text)
    else:
        await update.effective_message.reply_photo(png)
        await update.effective_message.reply_text(text[:4000])


async def hl_refresh_job(ctx: ContextTypes.DEFAULT_TYPE):
    """Hourly check; re-picks the top 20 once the list is a day old (one ~40MB download)."""
    try:
        ch = await asyncio.to_thread(hl.refresh_top)
        _job_ok("hl_refresh_job")
    except Exception as e:
        log.exception("hyperliquid refresh failed")
        _job_failed("hl_refresh_job", e)
        if ctx.job and ctx.job.name == "hl_refresh_job":   # the hourly run failed: one retry in 5 minutes
            ctx.job_queue.run_once(hl_refresh_job, 300, name="hl_refresh_retry")
        return
    if ch and pl.get_kv("hl", "on") == "on":
        text = hl.refresh_text(ch)
        if text:
            await _send(ctx, text)


async def heartbeat_job(ctx: ContextTypes.DEFAULT_TYPE):
    """Dead-man's switch: tells HEARTBEAT_URL she's alive. The pings stopping is the alert,
    so a failed ping is only logged (once per outage), never sent to Telegram."""
    try:
        r = await asyncio.to_thread(requests.get, HEARTBEAT_URL, timeout=10)
        r.raise_for_status()
        _job_ok("heartbeat_job")
    except Exception as e:
        if _job_failed("heartbeat_job", e) == 1:
            log.warning("heartbeat ping failed: %s", e)


async def cmd_hl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    if ctx.args and ctx.args[0].lower() in ("on", "off"):
        pl.set_kv("hl", ctx.args[0].lower())
    on = pl.get_kv("hl", "on") == "on"
    msg = None
    if not hl.top():
        msg = await update.effective_message.reply_text("Downloading the Hyperliquid leaderboard (~40MB)...")
        try:
            await asyncio.to_thread(hl.refresh_top, True)
        except Exception as e:
            await msg.edit_text(f"Couldn't load the Hyperliquid leaderboard: {e}")
            return
    if hl.state["polled"] is None:
        await asyncio.to_thread(hl.poll)
    text = hl.status_text(on, ZoneInfo(DISPLAY_TZ))[:4000]
    if msg:
        await msg.edit_text(text, disable_web_page_preview=True)
    else:
        await update.effective_message.reply_text(text, disable_web_page_preview=True)
    liqs = hl.liq_line()   # its own message: the list above is already close to Telegram's 4096 limit
    if liqs:
        await update.effective_message.reply_text(liqs)


async def news_job(ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await asyncio.to_thread(news.refresh)
        hits = await asyncio.to_thread(news.breaking_to_ping)
        _job_ok("news_job")
    except Exception as e:
        log.exception("news job failed")
        _job_failed("news_job", e)
        return
    if not hits or pl.get_kv("breaking", "on") != "on":
        return
    for it in hits:
        lean = {-1: "likely risk-off / bearish for BTC", 1: "likely risk-on / bullish for BTC",
                0: "direction unclear - expect volatility"}[it["direction"]]
        text = (f"📰 {BOT_NAME}: breaking\n{it['title']}\n- {it['source']}\n\n"
                f"My read: {lean}.")
        if it.get("reason") and not it["reason"].startswith("keyword"):
            text += f"\n({it['reason']})"
        text += "\n\nAsk me \"long or short?\" for an updated call."
        if it.get("link"):
            text += f"\n{it['link']}"
        await ctx.bot.send_message(chat_id=int(ALLOWED_CHAT_ID), text=text,
                                   disable_web_page_preview=True)


async def on_startup(app: Application):
    if not ALLOWED_CHAT_ID:
        log.warning("ALLOWED_CHAT_ID not set: message the bot once to get your chat id.")
        return
    now = datetime.now(timezone.utc)
    # Run 90s after each hour, once Binance has closed the hourly bar.
    nxt = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, seconds=90))
    app.job_queue.run_repeating(hourly_scan, interval=3600, first=(nxt - now).total_seconds())
    app.job_queue.run_repeating(resolve_job, interval=600, first=30)
    app.job_queue.run_repeating(news_job, interval=600, first=10)
    # Whale watch: every minute, 5s after Binance closes the 1-minute bar; liquidations stream live.
    nxt_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1, seconds=5)
    app.job_queue.run_repeating(whale_job, interval=60, first=(nxt_min - now).total_seconds())
    app.bot_data["liq_task"] = asyncio.create_task(whales.liquidation_listener())
    # Hyperliquid: the list is re-picked daily (checked hourly); positions polled every minute, 30s off the whale job.
    app.job_queue.run_repeating(hl_refresh_job, interval=3600, first=15)
    app.job_queue.run_repeating(hl_job, interval=60, first=(nxt_min - now).total_seconds() + 30)
    app.job_queue.run_repeating(liqmap_job, interval=60, first=(nxt_min - now).total_seconds() + 45)
    if HEARTBEAT_URL:
        app.job_queue.run_repeating(heartbeat_job, interval=60, first=5)
    try:
        model()
        note = f"Model loaded (trained {model()['trained_at'][:10]})."
    except Exception as e:
        note = f"⚠ {e}"
    await app.bot.send_message(chat_id=int(ALLOWED_CHAT_ID),
                               text=f"{BOT_NAME} is online. {note}\nAlerts: {pl.get_kv('alerts', 'on').upper()} · "
                                    f"Breaking news: {pl.get_kv('breaking', 'on').upper()}. "
                                    f"Next scan at {nxt:%H:%M} UTC.")


async def on_shutdown(app: Application):
    task = app.bot_data.get("liq_task")
    if task:
        task.cancel()


def main():
    if not TELEGRAM_TOKEN:
        raise SystemExit("TELEGRAM_TOKEN missing - copy .env.example to .env and fill it in.")
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(on_startup).post_shutdown(on_shutdown).build()
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler(["signal", "call", "vol"], cmd_signal))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("open", cmd_open))
    app.add_handler(CommandHandler(["wallet", "balance"], cmd_wallet))
    app.add_handler(CommandHandler(["status", "ping"], cmd_status))
    app.add_handler(CommandHandler("alerts", cmd_alerts))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("why", cmd_why))
    app.add_handler(CommandHandler("breaking", cmd_breaking))
    app.add_handler(CommandHandler(["whales", "whale"], cmd_whales))
    app.add_handler(CommandHandler(["hl", "hyperliquid", "top", "copy"], cmd_hl))
    app.add_handler(CommandHandler(["heatmap", "liqmap", "map"], cmd_heatmap))
    app.add_handler(CommandHandler("retrain", cmd_retrain))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("%s running. Ctrl+C to stop.", BOT_NAME)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
