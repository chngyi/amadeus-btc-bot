"""Turn raw market data into model features.

Every row is indexed by decision time t (the close of an hourly BTC bar) and
uses only information known at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HORIZON_H

# Plain-English names, used when explaining a call.
LABELS = {
    "btc_mom_1h": "BTC 1h move",
    "btc_mom_4h": "BTC 4h momentum",
    "btc_mom_12h": "BTC 12h momentum",
    "btc_mom_24h": "BTC 24h momentum",
    "btc_mom_72h": "BTC 3-day trend",
    "btc_vs_ma7d": "BTC vs its 7-day average",
    "btc_range_pos": "BTC position in 24h range",
    "vol_regime": "volatility vs normal",
    "volume_surge": "BTC volume vs normal",
    "taker_imb_4h": "aggressive buying vs selling (4h)",
    "taker_imb_24h": "aggressive buying vs selling (24h)",
    "eth_vs_btc_24h": "ETH vs BTC (24h)",
    "sol_vs_btc_24h": "SOL vs BTC (24h)",
    "perp_basis": "perp premium over spot",
    "perp_taker_imb_4h": "perp aggressive buying (4h)",
    "funding": "funding rate",
    "funding_chg": "funding rate change",
    "es_ret_4h": "S&P futures 4h",
    "es_ret_24h": "S&P futures 24h",
    "nq_ret_4h": "Nasdaq futures 4h",
    "nq_ret_24h": "Nasdaq futures 24h",
    "dxy_ret_24h": "US dollar index 24h",
    "gold_ret_24h": "gold 24h",
    "vix_chg_24h": "VIX change 24h",
    "us10y_chg_24h": "10y yield change 24h",
    "btc_es_corr": "BTC-stocks correlation (7d)",
    "us_mkt_open": "US futures trading",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
    "weekend": "weekend",
}


def _logret(s: pd.Series, k: int) -> pd.Series:
    return np.log(s / s.shift(k))


def build_features(bundle: dict) -> pd.DataFrame:
    btc = bundle["btc"]
    c = btc["close"]
    r1 = np.log(c).diff()
    vol24 = r1.rolling(24, min_periods=20).std()
    vol168 = r1.rolling(168, min_periods=120).std()

    f = pd.DataFrame(index=btc.index)
    for k in (1, 4, 12, 24, 72):
        f[f"btc_mom_{k}h"] = (_logret(c, k) / (vol24 * np.sqrt(k))).clip(-5, 5)
    f["btc_vs_ma7d"] = (np.log(c / c.rolling(168, min_periods=120).mean())
                        / (vol24 * np.sqrt(24))).clip(-6, 6)
    hi, lo = btc["high"].rolling(24).max(), btc["low"].rolling(24).min()
    f["btc_range_pos"] = ((c - lo) / (hi - lo).replace(0, np.nan)) - 0.5
    f["vol_regime"] = np.log(vol24 / vol168)
    v = btc["volume"]
    f["volume_surge"] = np.log((v.rolling(4).sum() / 4) / v.rolling(168, min_periods=120).mean())

    if "taker_buy_base" in btc and btc["taker_buy_base"].notna().any():
        tb = btc["taker_buy_base"]
        f["taker_imb_4h"] = 2 * tb.rolling(4).sum() / v.rolling(4).sum() - 1
        f["taker_imb_24h"] = 2 * tb.rolling(24).sum() / v.rolling(24).sum() - 1

    # --- crypto breadth ---------------------------------------------------
    for name in ("eth", "sol"):
        alt = bundle.get(name)
        if alt is not None and len(alt):
            ac = alt["close"].reindex(f.index).ffill(limit=2)
            f[f"{name}_vs_btc_24h"] = ((_logret(ac, 24) - _logret(c, 24))
                                       / (vol24 * np.sqrt(24))).clip(-6, 6)

    # --- derivatives positioning -----------------------------------------
    perp = bundle.get("perp")
    if perp is not None and len(perp):
        pc = perp["close"].reindex(f.index).ffill(limit=2)
        f["perp_basis"] = ((pc / c - 1) * 1e4).clip(-50, 50)  # bps
        pv = perp["volume"].reindex(f.index)
        ptb = perp["taker_buy_base"].reindex(f.index)
        f["perp_taker_imb_4h"] = 2 * ptb.rolling(4).sum() / pv.rolling(4).sum() - 1
    fund = bundle.get("funding")
    if fund is not None and len(fund):
        fr = fund.reindex(f.index.union(fund.index)).ffill().reindex(f.index) * 1e4  # bps / 8h
        f["funding"] = fr.clip(-10, 10)
        f["funding_chg"] = (fr - fr.shift(24)).clip(-10, 10)

    # --- traditional markets ---------------------------------------------
    cross = bundle.get("cross")
    if cross is not None and len(cross):
        x = cross.reindex(f.index.union(cross.index)).ffill(limit=72).reindex(f.index)
        for name in ("es", "nq"):
            if name in x:
                f[f"{name}_ret_4h"] = _logret(x[name], 4) * 100
                f[f"{name}_ret_24h"] = _logret(x[name], 24) * 100
        for name in ("dxy", "gold"):
            if name in x:
                f[f"{name}_ret_24h"] = _logret(x[name], 24) * 100
        if "vix" in x:
            f["vix_chg_24h"] = _logret(x["vix"], 24) * 100
        if "us10y" in x:
            f["us10y_chg_24h"] = (x["us10y"] - x["us10y"].shift(24)) * 10  # bps
        if "es" in x:
            # Is the futures market live right now? (Last real update within 2h.)
            last_upd = cross["es"].dropna().index.to_series()
            last_upd = last_upd.reindex(f.index.union(last_upd.index)).ffill().reindex(f.index)
            f["us_mkt_open"] = ((f.index.to_series() - last_upd) <= pd.Timedelta(hours=2)).astype(float)
            es_r = np.log(x["es"]).diff()
            f["btc_es_corr"] = r1.rolling(168, min_periods=60).corr(es_r.where(es_r != 0))
        # Fixed clips (not data-dependent) so training and live use identical scaling.
        for col in [k for k in f if k.startswith(("es_", "nq_", "dxy_", "gold_"))]:
            f[col] = f[col].clip(-8, 8)
        if "vix_chg_24h" in f:
            f["vix_chg_24h"] = f["vix_chg_24h"].clip(-60, 60)
        if "us10y_chg_24h" in f:
            f["us10y_chg_24h"] = f["us10y_chg_24h"].clip(-40, 40)

    # --- calendar ---------------------------------------------------------
    hr = f.index.hour + f.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    f["weekend"] = (f.index.dayofweek >= 5).astype(float)

    return f.replace([np.inf, -np.inf], np.nan)


def targets(bundle: dict) -> pd.DataFrame:
    """Forward 4h log return from the decision bar close, plus path extremes
    (for testing take-profit / stop-loss levels)."""
    btc = bundle["btc"]
    c = btc["close"]
    fwd = np.log(c.shift(-HORIZON_H) / c)
    out = pd.DataFrame({"fwd_ret": fwd, "y": (fwd > 0).astype(float)}, index=btc.index)
    out.loc[fwd.isna(), "y"] = np.nan
    for k in range(1, HORIZON_H + 1):
        out[f"hi{k}"] = np.log(btc["high"].shift(-k) / c)
        out[f"lo{k}"] = np.log(btc["low"].shift(-k) / c)
    return out


def sigma_4h(btc: pd.DataFrame) -> pd.Series:
    """Expected size of a 4h move (log-return std), from recent hourly volatility."""
    r1 = np.log(btc["close"]).diff()
    return np.sqrt((r1 ** 2).ewm(halflife=24, min_periods=24).mean() * HORIZON_H)
