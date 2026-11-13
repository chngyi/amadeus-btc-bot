"""Compute the current call: direction, probability, drivers, exit levels."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd

from zoneinfo import ZoneInfo

from config import (BOT_NAME, DISPLAY_TZ, HORIZON_H, RISK_PCT, MODEL_PATH, NEWS_OVERLAY, SL_MULT, TP_FAR,
                    TP_NEAR)
from data import load_bundle
from features import LABELS, build_features, sigma_4h


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise RuntimeError("No trained model yet - run:  python train.py")
    return joblib.load(MODEL_PATH)


def levels_for(price: float, sig: float) -> dict:
    """Take-profit zone and stop for both sides, from the expected 4h move sig (a fraction)."""
    def lv(dd):
        return {"tp_near": price * np.exp(dd * TP_NEAR * sig),
                "tp_far": price * np.exp(dd * TP_FAR * sig),
                "sl": price * np.exp(-dd * SL_MULT * sig)}
    return {"LONG": lv(1), "SHORT": lv(-1)}


def compute_signal(art: dict | None = None) -> dict:
    art = art or load_model()
    bundle = load_bundle(days=40, live=True)
    X = build_features(bundle)
    feats = art["features"]

    missing = [f for f in feats if f not in X or pd.isna(X[f].iloc[-1])]
    row = X.reindex(columns=feats).iloc[[-1]].copy()
    for f in missing:  # neutral value instead of crashing
        row[f] = art["fill"][f]

    model = art["model"]
    p_up = float(model.predict_proba(row)[0, 1])

    # Why: each feature's push on the log-odds.
    z = model.named_steps["scale"].transform(row)[0]
    coef = model.named_steps["lr"].coef_[0]
    contrib = pd.Series(z * coef, index=feats).drop(labels=missing, errors="ignore")
    contrib = contrib.drop(labels=["hour_cos"], errors="ignore")
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:4]
    drivers = [(LABELS.get(k, k), float(v), float(z[feats.index(k)])) for k, v in top.items()]

    th = art["thresholds"]
    if p_up >= th["long_alert"]:
        decision, strength = "LONG", "strong"
    elif p_up <= th["short_alert"]:
        decision, strength = "SHORT", "strong"
    elif p_up >= th["long_lean"]:
        decision, strength = "LONG", "lean"
    elif p_up <= th["short_lean"]:
        decision, strength = "SHORT", "lean"
    else:
        decision, strength = "NO TRADE", "none"

    model_strength = strength
    news_info, events, notes = None, [], []
    if NEWS_OVERLAY:
        news_info, events, notes = _news_overlay(decision, strength)
        if notes and strength == "strong" and any(n.startswith("↓") for n in notes):
            strength = "lean"

    btc = bundle["btc"]
    price = float(btc["close"].iloc[-1])
    vol = None
    if art.get("vol"):
        from vol_model import predict_vol, vol_features
        try:
            vol = predict_vol(art["vol"], vol_features(bundle))
        except Exception as e:
            print(f"  [vol] failed: {e}")
    # Stops/targets scale with the forecast move size (falls back to recent volatility).
    sig = vol["sigma"] if vol else float(sigma_4h(btc).iloc[-1])

    both = levels_for(price, sig)
    levels = both.get(decision)
    stop_pct = SL_MULT * sig * 100
    sizing = {"risk_pct": RISK_PCT, "stop_pct": stop_pct,
              "position_x": (RISK_PCT / stop_pct) if stop_pct > 0 else None}
    key = "long" if decision == "LONG" else "short"
    if strength == "lean":
        key += "_lean"
    hist = art["stats"].get(key) if decision != "NO TRADE" else None

    return {
        "time": btc.index[-1].to_pydatetime(),
        "price": price,
        "p_up": p_up,
        "decision": decision,
        "strength": strength,
        "sigma_pct": sig * 100,
        "levels": levels,
        "drivers": drivers,
        "missing": missing,
        "hist": hist,
        "edge": bool(art["stats"].get(key.split("_")[0], {}).get("edge", False)) if decision != "NO TRADE" else False,
        "model_strength": model_strength,
        "vol": vol,
        "both_levels": both,
        "sizing": sizing,
        "any_edge": any(art["stats"].get(k, {}).get("edge", False) for k in ("long", "short")),
        "news": news_info,
        "events": events,
        "notes": notes,
        "trained_at": art["trained_at"],
        "auc": art["auc"],
    }


def _news_overlay(decision: str, strength: str):
    """News can only make a call MORE cautious, never more aggressive -
    it isn't backtested, so it doesn't get to add risk."""
    import news
    notes: list[str] = []
    try:
        news.refresh_if_stale()
        st = news.news_state()
    except Exception as e:
        print(f"  [news] unavailable: {e}")
        st = None
    try:
        events = news.upcoming_events(HORIZON_H)
    except Exception:
        events = []
    d = {"LONG": 1, "SHORT": -1}.get(decision, 0)
    if st and d:
        if st["score"] * d <= -2:
            notes.append(f"↓ News backdrop is {st['label']}, against this call.")
        elif st["score"] * d >= 2:
            notes.append(f"News backdrop ({st['label']}) agrees with this call.")
    if st and st["shock"]:
        notes.append("⚡ Major news in the last 2h - expect bigger, jumpier moves. "
                     "Stops can slip; consider smaller size.")
    if events and d:
        notes.append(f"↓ High-impact US data inside this {HORIZON_H}h window - "
                     "price can jump either way on the release.")
    if strength != "strong":  # downgrades only matter for strong calls
        notes = [n.replace("↓ ", "") if n.startswith("↓") else n for n in notes]
    return st, events, notes


BINARY = {
    "weekend": ("it's the weekend", "it's a weekday"),
    "US futures trading": ("US futures are trading", "US futures are closed"),
}


def _describe(name: str, z: float) -> str:
    if name in BINARY:
        return BINARY[name][0] if z > 0 else BINARY[name][1]
    if name == "time of day":
        return "time of day"
    level = "very " if abs(z) > 1.5 else ""
    return f"{name}: {level}{'high' if z > 0 else 'low'}"


def _fmt_price(x: float) -> str:
    return f"{x:,.0f}"


def _news_lines(s: dict, direction_matters: bool) -> list[str]:
    import news as _n
    out = []
    nw = s.get("news")
    if nw is not None and (nw["top"] or nw["label"] != "neutral"):
        out.append(f"News (6h): {nw['label']}")
        for it in nw["top"]:
            out.append(f"  {_n._arrow(it['direction'])} {it['title'][:90]} - {it['source']}, {_n._age(it['age_h'])}")
    for e in s.get("events") or []:
        out.append(f"  📅 {_n.fmt_event(e)}")
    for note in s.get("notes") or []:
        if not direction_matters and "this call" in note:
            continue
        out.append(note.replace("↓ ", "⚠ "))
    return out


def _format_vol_first(s: dict, lines: list[str]) -> str:
    """Layout when the direction model has no proven edge: lead with what
    IS predictable (how much BTC will move), be honest about direction."""
    from vol_model import fmt_vol
    v = s["vol"]
    lines.append("")
    lines.append("⚡ Volatility")
    lines += fmt_vol(v)
    if s.get("events"):
        lines.append("(Scheduled US data in this window - the forecast doesn't include it, expect more.)")

    lines.append("")
    up = s["p_up"]
    if s["decision"] == "NO TRADE":
        lines.append(f"🧭 Direction: no lean ({up:.0%} up / {1-up:.0%} down) - coin flip.")
    else:
        h = s["hist"] or {}
        track = (f" In testing, calls like this were right {h['hit_rate']:.0%} and made "
                 f"{h['avg_net_exits_pct']:+.2f}%/trade after fees." if h.get("n") else "")
        p_dir = up if s["decision"] == "LONG" else 1 - up
        lines.append(f"🧭 Direction: no proven edge. Model leans {s['decision']} ({p_dir:.0%}).{track} "
                     "Treat it as a coin flip.")

    b, sz = s["both_levels"], s["sizing"]
    lines.append("")
    lines.append(f"If you trade anyway (next {HORIZON_H}h):")
    lines.append(f"  Long:  stop {_fmt_price(b['LONG']['sl'])}  ·  target {_fmt_price(b['LONG']['tp_near'])}-{_fmt_price(b['LONG']['tp_far'])}")
    lines.append(f"  Short: stop {_fmt_price(b['SHORT']['sl'])}  ·  target {_fmt_price(b['SHORT']['tp_near'])}-{_fmt_price(b['SHORT']['tp_far'])}")
    if sz["position_x"]:
        lines.append(f"  Stop is {sz['stop_pct']:.2f}% away → risking {sz['risk_pct']:g}% of your account "
                     f"means a position of ~{sz['position_x']:.1f}x your account"
                     + (" (no leverage needed)" if sz["position_x"] <= 1 else f" ({sz['position_x']:.1f}x leverage)") + ".")
    if v["regime"] in ("ELEVATED", "HIGH"):
        lines.append("  Busy market: wider stops mean a smaller position for the same risk.")

    extra = _news_lines(s, direction_matters=False)
    if extra:
        lines.append("")
        lines += extra
    if s["missing"] or v.get("missing"):
        lines.append("")
        lines.append("(Some data missing right now - treated as neutral.)")
    return "\n".join(lines)


def format_signal(s: dict, header: str | None = None) -> str:
    lines = []
    if header:
        lines.append(header)
    t = s["time"].astimezone(timezone.utc)
    tl = t.astimezone(ZoneInfo(DISPLAY_TZ))
    lines.append(f"BTC {_fmt_price(s['price'])}  ·  {tl:%d %b %H:%M %Z}  ·  next {HORIZON_H}h")
    if datetime.now(timezone.utc) - t > timedelta(hours=3):
        lines.append("⚠ Latest price data is over 3h old - treat this call as stale.")
    if s.get("vol") and not s["edge"]:
        return _format_vol_first(s, lines)
    lines.append("")
    up = s["p_up"]
    if s["decision"] == "NO TRADE":
        lines.append(f"Call: NO TRADE  (P(up) {up:.0%} / P(down) {1-up:.0%})")
        lines.append("Nothing stands out right now - staying flat is the call.")
    else:
        word = "STRONG " if s["strength"] == "strong" else "LEAN "
        if s.get("model_strength") == "strong" and s["strength"] != "strong":
            word = "LEAN (downgraded from strong) "
        p_dir = up if s["decision"] == "LONG" else 1 - up
        lines.append(f"Call: {word}{s['decision']}  (model P({'up' if s['decision']=='LONG' else 'down'}) {p_dir:.0%})")
        h = s["hist"] or {}
        if h.get("n"):
            lines.append(f"Signals like this were right {h['hit_rate']:.0%} of the time "
                         f"({h['n']} past cases), avg {h['avg_net_exits_pct']:+.2f}% per trade after fees.")
        lv = s["levels"]
        lines.append("")
        lines.append(f"Take profit zone: {_fmt_price(lv['tp_near'])} - {_fmt_price(lv['tp_far'])}")
        lines.append(f"Stop loss:        {_fmt_price(lv['sl'])}")
        lines.append(f"Time stop: close after {HORIZON_H}h if neither is hit.")
        lines.append(f"(Typical {HORIZON_H}h move right now: ±{s['sigma_pct']:.2f}%)")
        if not s["edge"]:
            lines.append("")
            lines.append("⚠ Backtest found no reliable edge after fees for this side. Paper-trade only.")
    lines.append("")
    lines.append("Why:")
    for name, c, z in s["drivers"]:
        push = "LONG" if c > 0 else "SHORT"
        lines.append(f"  • {_describe(name, z)}  → leans {push}")
    extra = _news_lines(s, direction_matters=True)
    if s.get("vol"):
        from vol_model import fmt_vol
        extra = ["⚡ Volatility"] + fmt_vol(s["vol"]) + ([""] + extra if extra else [])
    if extra:
        lines.append("")
        lines += extra
    if s["missing"]:
        lines.append("")
        lines.append(f"(Data missing right now, treated as neutral: {', '.join(LABELS.get(m, m) for m in s['missing'])})")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_signal(compute_signal(), header=f"{BOT_NAME}:"))
