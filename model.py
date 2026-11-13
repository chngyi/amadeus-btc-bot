"""Model training, honest walk-forward evaluation, and exit-level testing."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (ALERT_PCT, FEE_ROUND_TRIP, HORIZON_H, LEAN_PCT, SL_MULT, TP_NEAR)
from features import build_features, sigma_4h, targets

MIN_COVERAGE = 0.8  # drop a feature if it's missing in >20% of rows


def make_model() -> Pipeline:
    # Strong regularisation (small C): with noisy targets, a timid model
    # generalises better than a confident one.
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=0.02, max_iter=5000)),
    ])


def prepare(bundle: dict):
    X = build_features(bundle)
    T = targets(bundle)
    sig = sigma_4h(bundle["btc"]).rename("sigma")
    keep = [c for c in X if X[c].notna().mean() >= MIN_COVERAGE]
    dropped = [c for c in X if c not in keep]
    df = pd.concat([X[keep], T, sig], axis=1).dropna(subset=keep + ["y", "sigma"])
    return df, keep, dropped


def walk_forward(df: pd.DataFrame, feats: list[str], initial_days: int = 240,
                 step_days: int = 30) -> pd.Series:
    """Retrain every `step_days` on everything before, predict the next block.
    Every prediction is made by a model that never saw that data."""
    idx = df.index
    preds = pd.Series(np.nan, index=idx, name="p_up")
    t = idx[0] + pd.Timedelta(days=initial_days)
    purge = pd.Timedelta(hours=HORIZON_H)  # labels of the last 4 rows overlap the test block
    while t < idx[-1]:
        tr = idx < (t - purge)
        te = (idx >= t) & (idx < t + pd.Timedelta(days=step_days))
        if te.any() and tr.sum() > 500:
            m = make_model().fit(df.loc[tr, feats], df.loc[tr, "y"])
            preds[te] = m.predict_proba(df.loc[te, feats])[:, 1]
        t += pd.Timedelta(days=step_days)
    return preds


def simulate_exit(row: pd.Series, direction: int) -> tuple[float, str]:
    """Take-profit at TP_NEAR*sigma, stop at SL_MULT*sigma, else exit after 4h.
    Uses hourly highs/lows; if both levels are touched in the same hour we
    assume the stop hit first (pessimistic)."""
    tp, sl = TP_NEAR * row["sigma"], SL_MULT * row["sigma"]
    for k in range(1, HORIZON_H + 1):
        hi, lo = row[f"hi{k}"], row[f"lo{k}"]
        if np.isnan(hi) or np.isnan(lo):
            break
        adverse = -lo if direction > 0 else hi
        favour = hi if direction > 0 else -lo
        if adverse >= sl:
            return -sl - FEE_ROUND_TRIP, "stop"
        if favour >= tp:
            return tp - FEE_ROUND_TRIP, "target"
    return direction * row["fwd_ret"] - FEE_ROUND_TRIP, "time"


def bucket_stats(df: pd.DataFrame, mask: pd.Series, direction: int) -> dict:
    """Stats for trades taken when `mask` is true, non-overlapping
    (no new trade while one is open - like the alert cooldown)."""
    sel = df[mask]
    taken, last_t = [], None
    for t, row in sel.iterrows():
        if last_t is not None and t < last_t + pd.Timedelta(hours=HORIZON_H):
            continue
        last_t = t
        taken.append((t, row))
    if not taken:
        return {"n": 0}
    hold = np.array([direction * r["fwd_ret"] - FEE_ROUND_TRIP for _, r in taken])
    ex = [simulate_exit(r, direction) for _, r in taken]
    exr = np.array([e[0] for e in ex])
    kinds = pd.Series([e[1] for e in ex]).value_counts(normalize=True).to_dict()
    days = max((taken[-1][0] - taken[0][0]).days, 1)
    return {
        "n": len(taken),
        "per_week": round(len(taken) / days * 7, 1),
        "hit_rate": float((np.array([direction * r["fwd_ret"] for _, r in taken]) > 0).mean()),
        "avg_net_hold_pct": float(hold.mean() * 100),
        "avg_net_exits_pct": float(exr.mean() * 100),
        "exit_target_pct": float(kinds.get("target", 0) * 100),
        "exit_stop_pct": float(kinds.get("stop", 0) * 100),
        "exit_time_pct": float(kinds.get("time", 0) * 100),
        "total_net_exits_pct": float(exr.sum() * 100),
    }


def train(bundle: dict, verbose: bool = True) -> tuple[dict, str]:
    df, feats, dropped = prepare(bundle)
    lines: list[str] = []
    say = lines.append
    say(f"Trained {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    say(f"Data: {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}  ({len(df):,} hourly rows)")
    say(f"Features used ({len(feats)}): {', '.join(feats)}")
    if dropped:
        say(f"Features dropped (missing data): {', '.join(dropped)}")

    p = walk_forward(df, feats)
    oos = df.assign(p_up=p).dropna(subset=["p_up"])
    base_up = oos["y"].mean()
    auc = roc_auc_score(oos["y"], oos["p_up"])
    acc = ((oos["p_up"] > 0.5) == (oos["y"] == 1)).mean()
    say("")
    say("=== Out-of-sample test (walk-forward, model never saw these hours) ===")
    say(f"Period: {oos.index[0]:%Y-%m-%d} -> {oos.index[-1]:%Y-%m-%d}  ({len(oos):,} hours)")
    say(f"Up-moves in period: {base_up:.1%}   (always guessing 'up' = {max(base_up, 1-base_up):.1%})")
    say(f"Direction accuracy: {acc:.1%}    AUC: {auc:.3f}   (0.500 = coin flip)")

    q = oos["p_up"].quantile([ALERT_PCT, LEAN_PCT, 1 - LEAN_PCT, 1 - ALERT_PCT]).values
    thr = {"short_alert": float(q[0]), "short_lean": float(q[1]),
           "long_lean": float(q[2]), "long_alert": float(q[3])}

    say("")
    say("Hit rate by signal strength (deciles of predicted P(up)):")
    oos["decile"] = pd.qcut(oos["p_up"], 10, labels=False, duplicates="drop")
    tab = oos.groupby("decile").agg(p_lo=("p_up", "min"), p_hi=("p_up", "max"),
                                    up_rate=("y", "mean"),
                                    avg_4h_ret_pct=("fwd_ret", lambda s: s.mean() * 100),
                                    n=("y", "size"))
    say(tab.round(3).to_string())

    stats = {
        "long": bucket_stats(oos, oos["p_up"] >= thr["long_alert"], +1),
        "short": bucket_stats(oos, oos["p_up"] <= thr["short_alert"], -1),
        "long_lean": bucket_stats(oos, (oos["p_up"] >= thr["long_lean"]) & (oos["p_up"] < thr["long_alert"]), +1),
        "short_lean": bucket_stats(oos, (oos["p_up"] <= thr["short_lean"]) & (oos["p_up"] > thr["short_alert"]), -1),
    }
    say("")
    say(f"Strong signals (top/bottom {ALERT_PCT:.0%}), one trade at a time, after {FEE_ROUND_TRIP:.2%} fees:")
    for k in ("long", "short", "long_lean", "short_lean"):
        s = stats[k]
        if not s["n"]:
            say(f"  {k:10s}: no trades")
            continue
        say(f"  {k:10s}: {s['n']:4d} trades (~{s['per_week']}/wk)  right {s['hit_rate']:.1%}  "
            f"avg net: hold-4h {s['avg_net_hold_pct']:+.3f}%  with TP/SL {s['avg_net_exits_pct']:+.3f}%  "
            f"[TP {s['exit_target_pct']:.0f}% / SL {s['exit_stop_pct']:.0f}% / time {s['exit_time_pct']:.0f}%]")
    for k in ("long", "short"):
        s = stats[k]
        s["edge"] = bool(s["n"] >= 30 and s["avg_net_exits_pct"] > 0 and s["hit_rate"] > 0.5)
    say("")
    for k in ("long", "short"):
        verdict = "EDGE after fees (in this test)" if stats[k]["edge"] else "NO reliable edge after fees"
        say(f"Verdict {k.upper():5s}: {verdict}")

    final = make_model().fit(df[feats], df["y"])
    coefs = pd.Series(final.named_steps["lr"].coef_[0], index=feats).sort_values()
    say("")
    say("Final model weights (standardised; + pushes LONG, - pushes SHORT):")
    say(coefs.round(3).to_string())

    from vol_model import train_vol
    vol_art, vol_lines = train_vol(bundle)
    lines.extend(vol_lines)

    artefact = {
        "vol": vol_art,
        "model": final,
        "features": feats,
        "fill": df[feats].median().to_dict(),
        "thresholds": thr,
        "stats": stats,
        "auc": float(auc),
        "accuracy": float(acc),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "oos_period": [str(oos.index[0]), str(oos.index[-1])],
    }
    report = "\n".join(lines)
    if verbose:
        print(report)
    return artefact, report
