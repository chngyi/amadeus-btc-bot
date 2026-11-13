"""Volatility model: how big will BTC's next-4h range be?

Unlike direction, this is genuinely predictable: calm and wild periods both
cluster, and activity follows the clock (US open, weekends). Target is the
next-4h high-low range; the model is a transparent linear regression on
log scales, tested walk-forward against the naive "recent volatility" guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import HORIZON_H

# Expected high-low range of a random walk is ~1.6x the std of its return.
RANGE_TO_SIGMA = 1.6
BIG_MOVES = (1.0, 2.0, 3.0)   # % ranges we quote probabilities for

LABELS = {
    "lrange_1": "last hour's activity",
    "lrange_4": "last 4h activity",
    "lrange_24": "last 24h activity",
    "lrange_168": "last week's activity",
    "volume_surge": "trading volume vs normal",
    "absmom_24h": "size of the 24h trend",
    "absmom_72h": "size of the 3-day trend",
    "drawdown_7d": "distance below the 7-day high",
    "abs_funding": "funding rate extremes",
    "abs_basis": "perp premium extremes",
    "vix": "VIX (stock-market fear)",
    "es_absmove_24h": "size of the S&P move (24h)",
    "us_mkt_open": "US futures trading",
    "weekend": "weekend",
    "time": "time of day",
}


def vol_features(bundle: dict) -> pd.DataFrame:
    btc = bundle["btc"]
    c = btc["close"]
    r1 = np.log(c).diff()
    lr = np.log(btc["high"] / btc["low"]).replace(0, np.nan).ffill()
    X = pd.DataFrame(index=btc.index)
    for w in (1, 4, 24, 168):
        X[f"lrange_{w}"] = np.log(lr.rolling(w, min_periods=max(1, w // 2)).mean())
    ew = np.sqrt((r1 ** 2).ewm(halflife=24, min_periods=24).mean())
    v = btc["volume"]
    X["volume_surge"] = np.log(v.rolling(4).mean() / v.rolling(168, min_periods=120).mean())
    X["absmom_24h"] = (np.abs(np.log(c / c.shift(24))) / (ew * np.sqrt(24))).clip(0, 6)
    X["absmom_72h"] = (np.abs(np.log(c / c.shift(72))) / (ew * np.sqrt(72))).clip(0, 6)
    X["drawdown_7d"] = np.log(c / btc["high"].rolling(168, min_periods=120).max())

    fund = bundle.get("funding")
    if fund is not None and len(fund):
        fr = fund.reindex(X.index.union(fund.index)).ffill().reindex(X.index) * 1e4
        X["abs_funding"] = fr.abs().clip(0, 10)
    perp = bundle.get("perp")
    if perp is not None and len(perp):
        pc = perp["close"].reindex(X.index).ffill(limit=2)
        X["abs_basis"] = ((pc / c - 1).abs() * 1e4).clip(0, 50)
    cross = bundle.get("cross")
    if cross is not None and len(cross):
        x = cross.reindex(X.index.union(cross.index)).ffill(limit=72).reindex(X.index)
        if "vix" in x:
            X["vix"] = np.log(x["vix"])
        if "es" in x:
            X["es_absmove_24h"] = (np.abs(np.log(x["es"] / x["es"].shift(24))) * 100).clip(0, 8)
            last_upd = cross["es"].dropna().index.to_series()
            last_upd = last_upd.reindex(X.index.union(last_upd.index)).ffill().reindex(X.index)
            X["us_mkt_open"] = ((X.index.to_series() - last_upd) <= pd.Timedelta(hours=2)).astype(float)

    hr = X.index.hour
    for k in range(24):          # the next 4h start at this UTC hour
        X[f"hour_{k:02d}"] = (hr == k).astype(float)
    X["sat"] = (X.index.dayofweek == 5).astype(float)
    X["sun"] = (X.index.dayofweek == 6).astype(float)
    return X.replace([np.inf, -np.inf], np.nan)


def vol_target(bundle: dict) -> pd.Series:
    btc = bundle["btc"]
    hi = pd.concat([btc["high"].shift(-k) for k in range(1, HORIZON_H + 1)], axis=1).max(axis=1)
    lo = pd.concat([btc["low"].shift(-k) for k in range(1, HORIZON_H + 1)], axis=1).min(axis=1)
    rng = np.log(hi / lo)
    rng[btc["close"].shift(-HORIZON_H).isna()] = np.nan
    return np.log(rng).rename("y")


def _make() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("lr", Ridge(alpha=10.0))])


def _walk_forward(df, cols, make, initial_days=240, step_days=30):
    p = pd.Series(np.nan, index=df.index)
    t = df.index[0] + pd.Timedelta(days=initial_days)
    while t < df.index[-1]:
        tr = df.index < t - pd.Timedelta(hours=HORIZON_H)
        te = (df.index >= t) & (df.index < t + pd.Timedelta(days=step_days))
        if te.any():
            p[te] = make().fit(df.loc[tr, cols], df.loc[tr, "y"]).predict(df.loc[te, cols])
        t += pd.Timedelta(days=step_days)
    return p


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def train_vol(bundle: dict) -> tuple[dict, list[str]]:
    X = vol_features(bundle)
    y = vol_target(bundle)
    feats = [c for c in X if X[c].notna().mean() >= 0.8]
    df = pd.concat([X[feats], y], axis=1).dropna()
    # naive baseline = what we used before: recent volatility only
    ew = np.log(np.sqrt((np.log(bundle["btc"]["close"]).diff() ** 2).ewm(halflife=24).mean()))
    df["ewma"] = ew.reindex(df.index)
    df = df.dropna()

    from sklearn.linear_model import LinearRegression
    naive = _walk_forward(df, ["ewma"], LinearRegression)
    pred = _walk_forward(df, feats, _make)
    o = df.assign(naive=naive, pred=pred).dropna(subset=["naive", "pred"])
    resid = (o["y"] - o["pred"]).values
    actual_pct = np.exp(o["y"]) * 100
    pred_pct = np.exp(o["pred"]) * 100

    lines = ["", "=== Volatility model (next-4h high-low range), walk-forward ==="]
    lines.append(f"R² (share of range-size explained): model {_r2(o['y'], o['pred']):.2f}  vs  "
                 f"recent-volatility-only {_r2(o['y'], o['naive']):.2f}")
    by_year = ", ".join(f"{yr}: {_r2(g['y'], g['pred']):.2f}" for yr, g in o.groupby(o.index.year))
    lines.append(f"By year: {by_year}")
    q = pd.qcut(o["pred"], 10, labels=False, duplicates="drop")
    tab = pd.DataFrame({"predicted": pred_pct.groupby(q).median(),
                        "actual_median": actual_pct.groupby(q).median(),
                        "actual_p90": actual_pct.groupby(q).quantile(0.9)}).round(2)
    tab.index.name = "decile"
    lines.append("Predicted vs actual 4h range (%), by forecast decile:")
    lines.append(tab.to_string())
    lo10, hi90 = np.quantile(resid, [0.1, 0.9])
    cover = ((o["y"] >= o["pred"] + lo10) & (o["y"] <= o["pred"] + hi90)).mean()
    lines.append(f"80% band coverage (should be ~80%): {cover:.0%}")
    for b in BIG_MOVES:
        thr = np.log(b / 100)
        p_hat = np.array([(resid > thr - p).mean() for p in o["pred"].values[::25]])
        act = (o["y"].values[::25] > thr).mean()
        lines.append(f"P(range > {b:.0f}%): avg forecast {p_hat.mean():.0%}, actual {act:.0%}")

    final = _make().fit(df[feats], df["y"])
    hist_pred = pd.Series(final.predict(df[feats].tail(24 * 365)), index=df.index[-24 * 365:])
    art = {
        "model": final, "features": feats, "fill": df[feats].median().to_dict(),
        "resid_q": np.quantile(resid, np.linspace(0.005, 0.995, 199)),
        "hist_pred": np.sort(hist_pred.values),     # past year's forecasts, for percentiles
        "r2": float(_r2(o["y"], o["pred"])), "r2_naive": float(_r2(o["y"], o["naive"])),
    }
    return art, lines


def _group(name: str) -> str:
    if name.startswith("hour_"):
        return "time"
    if name in ("sat", "sun"):
        return "weekend"
    return name


def predict_vol(art: dict, X: pd.DataFrame) -> dict:
    feats = art["features"]
    row = X.reindex(columns=feats).iloc[[-1]].copy()
    missing = [f for f in feats if pd.isna(row[f].iloc[0])]
    for f in missing:
        row[f] = art["fill"][f]
    m = art["model"]
    mu = float(m.predict(row)[0])
    rq = art["resid_q"]
    med = np.exp(mu + np.median(rq)) * 100
    lo, hi = np.exp(mu + np.quantile(rq, 0.1)) * 100, np.exp(mu + np.quantile(rq, 0.9)) * 100
    p_big = {b: float((rq > np.log(b / 100) - mu).mean()) for b in BIG_MOVES}
    pct = float(np.searchsorted(art["hist_pred"], mu) / len(art["hist_pred"]))
    if pct < 0.25:
        regime = "CALM"
    elif pct < 0.75:
        regime = "NORMAL"
    elif pct < 0.92:
        regime = "ELEVATED"
    else:
        regime = "HIGH"

    z = m.named_steps["scale"].transform(row)[0]
    contrib = pd.Series(z * m.named_steps["lr"].coef_, index=feats).drop(labels=missing, errors="ignore")
    grouped = contrib.groupby([_group(f) for f in contrib.index]).sum()
    top = grouped.reindex(grouped.abs().sort_values(ascending=False).index)[:3]
    drivers = [(LABELS.get(k, k), float(v)) for k, v in top.items() if abs(v) > 0.03]
    return {"range_med": med, "range_lo": lo, "range_hi": hi, "p_big": p_big,
            "percentile": pct, "regime": regime, "sigma": med / 100 / RANGE_TO_SIGMA,
            "drivers": drivers, "missing": missing}


REGIME_WORDS = {"CALM": "quieter than usual", "NORMAL": "about normal",
                "ELEVATED": "busier than usual", "HIGH": "very volatile"}


def fmt_vol(v: dict, horizon: int = HORIZON_H) -> list[str]:
    lines = [
        f"Next {horizon}h: expect a ~{v['range_med']:.1f}% range "
        f"(80% chance {v['range_lo']:.1f}-{v['range_hi']:.1f}%)",
        f"Regime: {v['regime']} - {REGIME_WORDS[v['regime']]} "
        f"(busier than {v['percentile']:.0%} of the past year)",
        f"Chance of a 2%+ swing: {v['p_big'][2.0]:.0%}  ·  3%+: {v['p_big'][3.0]:.0%}",
    ]
    if v["drivers"]:
        d = " · ".join(f"{name} ({'adds' if c > 0 else 'calms'})" for name, c in v["drivers"])
        lines.append(f"Why: {d}")
    return lines
