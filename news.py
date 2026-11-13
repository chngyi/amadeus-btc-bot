"""News + economic calendar.

Headlines come from free RSS feeds (crypto, world, markets, the Fed). Each one
is rated for its likely BTC impact:
  - by Claude (if ANTHROPIC_API_KEY is set in .env) - understands context,
    e.g. "rules out rate hike" or a routine update on an ongoing war;
  - otherwise by keyword rules - crude but free.

News is NOT part of the trained model (there's no free headline archive to
backtest on). It's used as a risk overlay: it can downgrade a call, warn you,
and ping you on major breaking stories. It never upgrades a call.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from calendar import timegm
from datetime import datetime, timedelta, timezone

import requests

from config import ANTHROPIC_API_KEY, DATA_DIR, DB_PATH

FEEDS = {
    # crypto
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "The Block": "https://www.theblock.co/rss.xml",
    "Decrypt": "https://decrypt.co/feed",
    # world / geopolitics
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Google News (geopolitics)": "https://news.google.com/rss/search?q=(%22declares+war%22+OR+%22missile+strike%22+OR+%22military+strike%22+OR+%22oil+embargo%22+OR+%22new+sanctions%22+OR+invades+OR+ceasefire)+when:1d&hl=en-US&gl=US&ceid=US:en",
    # markets / macro
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "CNBC Markets": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "Google News (oil & macro)": "https://news.google.com/rss/search?q=(oil+OR+OPEC+OR+Fed+OR+inflation+OR+tariffs)+when:1d&hl=en-US&gl=US&ceid=US:en",
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
}

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id TEXT PRIMARY KEY,
    title TEXT, source TEXT, link TEXT,
    published_utc TEXT, first_seen_utc TEXT,
    impact INTEGER, direction INTEGER, category TEXT, reason TEXT, scorer TEXT,
    pinged INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Keyword scorer:  (pattern, impact 1-3, direction -1/+1, category)
# --------------------------------------------------------------------------
RULES = [
    (r"\b(declares? war|war (breaks out|erupts)|(military|full[- ]scale|ground) invasion|invasion of|invades?|full[- ]scale attack)\b", 3, -1, "war"),
    (r"\b(airstrikes?|missiles?|drone strikes?|military strikes?|bombing|shelling|troops)\b", 2, -1, "war"),
    (r"\bnuclear (test|strike|threat|weapons?|attack)\b", 3, -1, "war"),
    (r"\b((oil )?embargo|blockade|strait of hormuz|oil (supply )?(shock|disruption))\b", 3, -1, "oil"),
    (r"\bopec\+? (cuts?|slashes|reduces?)\b|\boil (prices? )?(surges?|spikes?|soars?|jumps?)\b", 2, -1, "oil"),
    (r"\bsanctions?\b", 2, -1, "sanctions"),
    (r"\btariffs?\b|\btrade war\b", 2, -1, "trade"),
    (r"\b(ceasefire|truce|peace (deal|agreement|talks))\b", 2, +1, "peace"),
    (r"\b(emergency rate|surprise (rate )?(hike|cut))\b", 3, 0, "fed"),
    (r"\b(rate hikes?|raises? (interest )?rates|hawkish)\b", 2, -1, "fed"),
    (r"\b(rate cuts?|cuts? (interest )?rates|dovish)\b", 2, +1, "fed"),
    (r"\b(inflation|cpi|pce)\b.*\b(hotter|higher than expected|accelerat\w*|jumps?|surges?)\b", 2, -1, "inflation"),
    (r"\b(inflation|cpi|pce)\b.*\b(cool\w*|eases?|slows?|lower than expected)\b", 2, +1, "inflation"),
    (r"\b(recession|bank (run|failure|collapse)|debt default|credit crunch)\b", 2, -1, "macro"),
    (r"\b(insolven\w*|bankrupt\w*|withdrawals? (halted|paused|frozen|suspended)|depeg\w*)\b", 3, -1, "crypto-crisis"),
    (r"\b(hack(ed)?|exploit(ed)?|drained|stolen|breach)\b", 2, -1, "crypto-hack"),
    (r"\b(sec (sues|charges)|lawsuit|crackdown|bans? (crypto|bitcoin)|delist\w*)\b", 2, -1, "regulation"),
    (r"\bliquidations?\b", 1, -1, "crypto-market"),
    # descriptions of market moves - shown as context, never treated as a cause
    (r"\b(crash\w*|plung\w*|sell-?offs?|rout|tumbl\w*|meltdown|slump\w*|carry trade)\b", 1, -1, "market-move"),
    (r"\b(rall(y|ies|ied)|soar\w*|record high|all-time high|surg\w*)\b", 1, +1, "market-move"),
    (r"\betf (approv\w*|inflows?)\b|\brecord inflows?\b", 2, +1, "crypto-flows"),
    (r"\betf outflows?\b", 1, -1, "crypto-flows"),
    (r"\b(strategic (bitcoin )?reserve|adopts? bitcoin|legal tender)\b", 2, +1, "adoption"),
]
# Local crime / courts / politics that trip war-ish keywords but don't move markets.
NOT_MARKET = re.compile(r"\b(home invasion|police|cops|stabb\w*|shot by|shooting|murder\w*|arrest\w*|suspect"
                        r"|prosecutors?|icc|trial|lawsuit|insider look|inside look|adviser|celebrity)\b", re.I)
GEO_CATS = {"war", "sanctions", "peace", "oil", "trade"}
NEGATION = re.compile(r"\b(rules? out|denies|denied|no plans?|unlikely|won't|will not|averts?|avoids?)\b", re.I)
_COMPILED = [(re.compile(p, re.I), i, d, c) for p, i, d, c in RULES]

# War/sanctions/peace words only mean geopolitics when a state or military actor is named:
# "Jokic declares war on fans" and "a burn scar invades the brain" are not market news.
NEEDS_ACTOR = {"war", "sanctions", "peace"}
GEO_ENTITIES = {
    "us": r"(?-i:US|U\.S\.?)|USA|united states|america\w*|washington|pentagon|white house|trump",
    "russia": r"russia\w*|moscow|kremlin|putin",
    "ukraine": r"ukrain\w*|kyiv|kiev|zelensk\w*",
    "china": r"china|chinese|beijing|xi jinping",
    "taiwan": r"taiwan\w*",
    "israel": r"israel\w*|idf|netanyahu",
    "gaza": r"gaza|hamas|palestin\w*|west bank",
    "lebanon": r"leban\w*|hezbollah|beirut",
    "iran": r"iran\w*|tehran",
    "yemen": r"yemen\w*|houthis?|red sea",
    "gulf": r"saudi\w*|uae|qatar\w*|hormuz|persian gulf|opec\+?",
    "syria": r"syria\w*|damascus",
    "korea": r"korea\w*|pyongyang|seoul|kim jong[- ]un",
    "south asia": r"india\w*|pakistan\w*|kashmir",
    "europe": r"nato|eu|european|europe|britain|british|uk|france|french|german\w*|poland|polish|baltic",
    "turkey": r"turkey|turkish|t[uü]rkiye|ankara",
    "venezuela": r"venezuela\w*|caracas",
    "japan": r"japan\w*|tokyo",
    "military": r"military|army|navy|troops|armed forces|warships?|fighter jets?|defen[cs]e (ministry|minister|forces)"
                r"|security council|united nations|\bun\b",
}
_GEO = {k: re.compile(rf"(?<!\w)(?:{p})(?!\w)", re.I) for k, p in GEO_ENTITIES.items()}
GENERIC_ENTITIES = {"us", "europe", "military"}   # too common to show two headlines are the same story
_STOP = {"with", "from", "that", "this", "after", "over", "into", "says", "said", "will", "amid", "more", "than",
         "about", "their", "have", "been", "what", "when", "news", "live", "update", "updates", "report", "reports"}


def geo_entities(title: str) -> set[str]:
    return {k for k, rx in _GEO.items() if rx.search(title)}


def _words(title: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", title.lower()) if len(w) >= 4 and w not in _STOP}


def same_story(a: dict, b: dict) -> bool:
    """Two headlines are the same story if they share a specific place/actor plus another
    key word or actor ("Israel strikes Iran" / "Israeli jets hit Iran nuclear sites")."""
    ea, eb = geo_entities(a["title"]), geo_entities(b["title"])
    specific = (ea & eb) - GENERIC_ENTITIES
    if not specific and a["category"] in NEEDS_ACTOR:
        return False
    return len(ea & eb) + len(_words(a["title"]) & _words(b["title"])) >= 2


def corroborated(it: dict, pool: list[dict]) -> bool:
    """Reported by 2+ different outlets - about the same story, not just the same category."""
    return any(x["source"] != it["source"] and x["category"] == it["category"] and same_story(it, x)
               for x in pool)


def keyword_score(title: str) -> tuple[int, int, str, str]:
    best = (0, 0, "", "")
    for rx, imp, d, cat in _COMPILED:
        if rx.search(title) and imp > best[0]:
            best = (imp, d, cat, f"keyword: {cat}")
    imp, d, cat, why = best
    if imp and cat in GEO_CATS and NOT_MARKET.search(title):
        return 0, 0, "", "not market news"
    if imp and cat in NEEDS_ACTOR and not geo_entities(title):
        return 0, 0, "", "no country or military named"
    if imp and NEGATION.search(title):
        # "rules out rate hike", "averts war": the scary thing is NOT happening
        imp, d, why = imp - 1, -d, why + " (negated)"
    return imp, d, cat, why


# --------------------------------------------------------------------------
# Claude scorer (optional)
# --------------------------------------------------------------------------
PROMPT = """You rate news headlines for their likely effect on the Bitcoin price over the next few hours.
For each headline give:
- impact 0-3: 0 = irrelevant, or a routine update on an ongoing story with nothing new;
  1 = minor; 2 = notable, could move crypto ~0.5-1%; 3 = major NEW shock (war breaking out,
  oil embargo, surprise Fed move, big exchange collapse, major hack).
- direction for BTC: -1 bearish / risk-off, 0 unclear, +1 bullish / risk-on.
- category: one or two words.
- reason: under 12 words.
Reply with ONLY a JSON array: [{"id": "...", "impact": 0, "direction": 0, "category": "...", "reason": "..."}]

Headlines:
"""


CLAUDE_BATCH = 25   # headlines per request: small enough that the reply never runs out of room


def claude_score(items: list[dict]) -> dict[str, tuple[int, int, str, str]]:
    """Raises if the reply is cut off, declined or not valid JSON - the caller then uses keywords."""
    import llm
    lines = "\n".join(f'{it["id"]}: [{it["source"]}] {it["title"]}' for it in items)
    text = llm.ask(PROMPT + lines, max_tokens=8000)
    arr = json.loads(text[text.index("["): text.rindex("]") + 1])
    out = {}
    for a in arr:
        try:
            out[str(a["id"])] = (max(0, min(3, int(a["impact"]))), max(-1, min(1, int(a["direction"]))),
                                 str(a.get("category", ""))[:30], str(a.get("reason", ""))[:120])
        except (KeyError, ValueError, TypeError):
            continue
    return out


# --------------------------------------------------------------------------
# Fetch + store
# --------------------------------------------------------------------------

def _hid(title: str) -> str:
    norm = re.sub(r"[^a-z0-9 ]", "", title.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:12]


def fetch_headlines(max_age_h: int = 24, feeds: dict | None = None, per_feed: int = 60) -> list[dict]:
    import feedparser

    cutoff = _now() - timedelta(hours=max_age_h)
    out, seen = [], set()
    for source, url in (feeds or FEEDS).items():
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (amadeus-bot)"})
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as e:
            print(f"  [news] {source} failed: {e}")
            continue
        for e in feed.entries[:per_feed]:
            title = re.sub(r"\s+", " ", e.get("title", "")).strip()
            if not title:
                continue
            # Google News appends " - Publisher"
            src = source
            if source.startswith("Google News") and " - " in title:
                title, src = title.rsplit(" - ", 1)
            st = e.get("published_parsed") or e.get("updated_parsed")
            pub = datetime.fromtimestamp(timegm(st), timezone.utc) if st else None
            if pub and pub < cutoff:
                continue
            hid = _hid(title)
            if hid in seen:
                continue
            seen.add(hid)
            out.append({"id": hid, "title": title, "source": src, "link": e.get("link", ""),
                        "published": pub})
    return out


def _gnews(q: str, days: int) -> str:
    from urllib.parse import quote_plus
    return (f"https://news.google.com/rss/search?q={quote_plus(q)}+when:{days}d"
            "&hl=en-US&gl=US&ceid=US:en")


def headlines_between(t0: datetime, t1: datetime) -> list[dict]:
    """Headlines published in [t0, t1]: everything the bot has stored, plus a
    fresh Google News search (so it works even if the bot just started)."""
    days = max(1, min(7, int((_now() - t0).total_seconds() // 86400) + 1))
    searches = {
        "Google News (crypto)": _gnews("bitcoin OR btc OR crypto", days),
        "Google News (markets)": _gnews("stocks OR oil OR Fed OR inflation OR tariffs OR dollar", days),
        "Google News (world)": _gnews("war OR sanctions OR embargo OR missile OR invasion OR ceasefire", days),
    }
    items = {}
    for it in fetch_headlines(max_age_h=days * 24 + 24, feeds=searches, per_feed=100):
        if it["published"] and t0 <= it["published"] <= t1:
            imp, d, cat, why = keyword_score(it["title"])
            it.update(impact=imp, direction=d, category=cat, reason=why, scorer="keyword")
            items[it["id"]] = it
    with _db() as con:
        rows = con.execute("SELECT id, title, source, link, published_utc, impact, direction, category,"
                           " reason, scorer FROM news WHERE published_utc BETWEEN ? AND ?",
                           (t0.isoformat(), t1.isoformat())).fetchall()
    for r in rows:  # stored ones may carry better (Claude) ratings
        items[r[0]] = dict(zip(["id", "title", "source", "link", "published", "impact", "direction",
                                "category", "reason", "scorer"], r))
        items[r[0]]["published"] = datetime.fromisoformat(r[4])
    return sorted(items.values(), key=lambda x: x["published"])


KEYWORD_VERSION = "3"


def _rescore_if_rules_changed() -> None:
    """Re-rate stored keyword headlines when the rules are updated."""
    with _db() as con:
        r = con.execute("SELECT v FROM kv WHERE k='kw_version'").fetchone()
        if r and r[0] == KEYWORD_VERSION:
            return
        rows = con.execute("SELECT id, title FROM news WHERE scorer='keyword'").fetchall()
        for hid, title in rows:
            imp, d, cat, why = keyword_score(title)
            con.execute("UPDATE news SET impact=?, direction=?, category=?, reason=? WHERE id=?",
                        (imp, d, cat, why, hid))
        con.execute("INSERT OR REPLACE INTO kv VALUES ('kw_version', ?)", (KEYWORD_VERSION,))


def refresh() -> list[dict]:
    """Fetch feeds, score headlines not seen before, store them. Returns the new ones."""
    _rescore_if_rules_changed()
    items = fetch_headlines()
    with _db() as con:
        known = {r[0] for r in con.execute("SELECT id FROM news")}
    new = [it for it in items if it["id"] not in known]
    scores: dict = {}
    now = _now().isoformat()
    if new and ANTHROPIC_API_KEY:
        # Batch by batch: one failed or cut-off batch falls back to keywords on its own,
        # without throwing away the batches Claude did rate.
        errors = []
        for i in range(0, len(new), CLAUDE_BATCH):
            batch = new[i:i + CLAUDE_BATCH]
            try:
                scores.update(claude_score(batch))
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}"[:200])
                print(f"  [news] Claude scoring failed for {len(batch)} headlines, using keywords: {errors[-1]}")
        with _db() as con:
            con.execute("INSERT OR REPLACE INTO kv VALUES ('claude_news', ?)",
                        (json.dumps({"t": now, "ok": not errors, "error": errors[-1] if errors else None}),))
    with _db() as con:
        for it in new:
            imp, d, cat, why = scores.get(it["id"]) or keyword_score(it["title"])
            sc = "claude" if it["id"] in scores else "keyword"
            it.update(impact=imp, direction=d, category=cat, reason=why, scorer=sc)
            con.execute("INSERT OR IGNORE INTO news VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
                        (it["id"], it["title"], it["source"], it["link"],
                         (it["published"] or _now()).isoformat(), now, imp, d, cat, why, sc))
        con.execute("INSERT OR REPLACE INTO kv VALUES ('news_refreshed', ?)", (now,))
        # keep the table small
        con.execute("DELETE FROM news WHERE first_seen_utc < ?", ((_now() - timedelta(days=14)).isoformat(),))
    return new


def refresh_if_stale(max_age_min: int = 15) -> None:
    _rescore_if_rules_changed()
    with _db() as con:
        r = con.execute("SELECT v FROM kv WHERE k='news_refreshed'").fetchone()
    if not r or _now() - datetime.fromisoformat(r[0]) > timedelta(minutes=max_age_min):
        refresh()


def recent(hours: float = 6, min_impact: int = 1) -> list[dict]:
    since = (_now() - timedelta(hours=hours)).isoformat()
    with _db() as con:
        rows = con.execute(
            "SELECT id, title, source, link, published_utc, impact, direction, category, reason, scorer, pinged"
            " FROM news WHERE published_utc >= ? AND impact >= ? ORDER BY published_utc DESC",
            (since, min_impact)).fetchall()
    keys = ["id", "title", "source", "link", "published", "impact", "direction", "category", "reason",
            "scorer", "pinged"]
    out = [dict(zip(keys, r)) for r in rows]
    for o in out:
        o["published"] = datetime.fromisoformat(o["published"])
        o["age_h"] = (_now() - o["published"]).total_seconds() / 3600
    return out


def news_state(hours: float = 6) -> dict:
    """Overall news backdrop. score < 0 = risk-off, > 0 = risk-on.
    Recent and high-impact stories count most (half-life 3h). Each category
    is capped so twenty articles about one story don't count twenty times."""
    items = recent(hours)
    # Keyword ratings are noisy for world news, so a geopolitical story only
    # counts once 2+ different outlets report the same story. Headlines that merely
    # describe market moves ("stocks slide") are effects, not causes.
    pool = items
    items = [it for it in items if it["category"] != "market-move" and (
        it["scorer"] == "claude" or it["category"] not in GEO_CATS or corroborated(it, pool))]
    by_cat: dict[str, float] = {}
    for it in items:
        w = it["impact"] * 0.5 ** (it["age_h"] / 3)
        it["weight"] = w
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + it["direction"] * w
    score = sum(max(-4.0, min(4.0, v)) for v in by_cat.values())
    if score <= -3:
        label = "strongly risk-off"
    elif score <= -1:
        label = "risk-off"
    elif score >= 3:
        label = "strongly risk-on"
    elif score >= 1:
        label = "risk-on"
    else:
        label = "neutral"
    shock = [it for it in items if it["impact"] >= 3 and it["age_h"] <= 2]
    top = sorted(items, key=lambda x: -x["weight"])[:3]
    return {"score": score, "label": label, "shock": shock, "top": top, "n": len(items)}


def breaking_to_ping() -> list[dict]:
    """Major (impact 3) stories from the last hour not yet pinged.
    In keyword mode, require 2+ different outlets reporting the same story, so one
    dramatic headline (or three unrelated ones in the same category) doesn't set it off."""
    items = [it for it in recent(1, min_impact=3) if not it["pinged"]]
    pool = recent(1, min_impact=2)
    out = [it for it in items if it["scorer"] == "claude" or corroborated(it, pool)]
    # one ping per category
    uniq: dict[str, dict] = {}
    for it in out:
        uniq.setdefault(it["category"], it)
    if items:
        with _db() as con:
            con.executemany("UPDATE news SET pinged=1 WHERE id=?",
                            [(it["id"],) for it in out] if out else [])
    return list(uniq.values())


# --------------------------------------------------------------------------
# Economic calendar (US high-impact releases: CPI, jobs, FOMC...)
# --------------------------------------------------------------------------

def _calendar() -> list:
    path = DATA_DIR / "calendar.json"
    data = None
    if path.exists() and time.time() - path.stat().st_mtime < 3 * 3600:
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            r = requests.get(CALENDAR_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            data = r.json()
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            print(f"  [calendar] failed: {e}")
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
    return data or []


def events_between(t0: datetime, t1: datetime, min_impact: str = "High") -> list[dict]:
    """US scheduled releases between t0 and t1 (this week's calendar only)."""
    out = []
    for ev in _calendar():
        if ev.get("country") != "USD" or ev.get("impact") != min_impact:
            continue
        try:
            t = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
        except (KeyError, ValueError):
            continue
        if t0 <= t <= t1:
            out.append({"title": ev.get("title", ""), "time": t, "mins": int((t - _now()).total_seconds() // 60),
                        "forecast": ev.get("forecast", ""), "previous": ev.get("previous", "")})
    return sorted(out, key=lambda e: e["time"])


def upcoming_events(within_h: float = 4, include_recent_min: int = 30) -> list[dict]:
    data = _calendar()
    if not data:
        return []
    now = _now()
    out = []
    for ev in data:
        if ev.get("country") != "USD" or ev.get("impact") != "High":
            continue
        try:
            t = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
        except (KeyError, ValueError):
            continue
        if now - timedelta(minutes=include_recent_min) <= t <= now + timedelta(hours=within_h):
            out.append({"title": ev.get("title", ""), "time": t,
                        "mins": int((t - now).total_seconds() // 60),
                        "forecast": ev.get("forecast", ""), "previous": ev.get("previous", "")})
    return sorted(out, key=lambda e: e["time"])


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def _age(h: float) -> str:
    return f"{int(h * 60)}m ago" if h < 1 else f"{h:.0f}h ago"


def _arrow(d: int) -> str:
    return {-1: "🔻", 0: "•", 1: "🔺"}[d]


def fmt_event(e: dict) -> str:
    when = f"in {e['mins'] // 60}h {e['mins'] % 60:02d}m" if e["mins"] >= 0 else f"{-e['mins']}m ago"
    extra = f" (forecast {e['forecast']}, prev {e['previous']})" if e["forecast"] else ""
    return f"{e['title']} {when}{extra}"


def news_text(limit: int = 10) -> str:
    st = news_state()
    lines = [f"News backdrop (last 6h): {st['label'].upper()}  (score {st['score']:+.1f})"]
    items = recent(12)
    if not items:
        lines.append("No market-relevant headlines in the last 12h.")
    for it in items[:limit]:
        lines.append(f"{_arrow(it['direction'])} [{it['impact']}] {it['title'][:110]} "
                     f"- {it['source']}, {_age(it['age_h'])}")
    ev = upcoming_events(24)
    if ev:
        lines.append("\nUS high-impact events, next 24h:")
        lines += [f"  📅 {fmt_event(e)}" for e in ev]
    scorer = "Claude" if ANTHROPIC_API_KEY else "keyword rules"
    lines.append(f"\n[impact 1-3] · 🔺 bullish 🔻 bearish for BTC · rated by {scorer}")
    return "\n".join(lines)
