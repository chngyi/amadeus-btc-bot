"""Settings. Secrets live in .env (copy .env.example -> .env)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_env() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

# --- Telegram -------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
ALLOWED_CHAT_ID = os.getenv("ALLOWED_CHAT_ID", "")  # only this chat may use the bot
BOT_NAME = os.getenv("BOT_NAME", "Amadeus")
DISPLAY_TZ = os.getenv("DISPLAY_TZ", "Asia/Tokyo")  # times in messages
# Dead-man's switch: she opens this URL every minute (e.g. a free healthchecks.io check), which
# alerts you when the pings stop. Only the request itself is sent, no data. Empty = off.
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "")

# --- News (optional: Claude rates headlines; without a key, keyword rules do) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
NEWS_OVERLAY = os.getenv("NEWS_OVERLAY", "1") == "1"

# --- Strategy -------------------------------------------------------------
HORIZON_H = 4                 # each call is about the next 4 hours
FEE_ROUND_TRIP = float(os.getenv("FEE_ROUND_TRIP", "0.001"))  # 0.05% taker in + out

RISK_PCT = float(os.getenv("RISK_PCT", "1.0"))  # % of account you'd lose if the stop hits

# Amadeus's paper wallet: starts with WALLET_START dollars and goes all-in (isolated margin =
# whole balance) at FIXED_LEVERAGE on the perp, one trade at a time. Paper only - no real orders.
FIXED_LEVERAGE = float(os.getenv("FIXED_LEVERAGE", "40"))
MAINT_MARGIN = float(os.getenv("MAINT_MARGIN", "0.004"))  # Binance BTCUSDT perp, smallest tier
START_EQUITY = float(os.getenv("WALLET_START", "1000"))
MIN_NOTIONAL = 100.0          # Binance's smallest BTCUSDT perp order: below this the wallet is broke

# Exit levels as multiples of expected 4h move (sigma).
TP_NEAR = float(os.getenv("TP_NEAR", "0.75"))
TP_FAR = float(os.getenv("TP_FAR", "1.25"))
SL_MULT = float(os.getenv("SL_MULT", "1.0"))

# A signal is "strong" (alert-worthy) if it is in the top / bottom ALERT_PCT
# of the model's own out-of-sample history; "lean" if in top / bottom LEAN_PCT.
ALERT_PCT = float(os.getenv("ALERT_PCT", "0.10"))
LEAN_PCT = float(os.getenv("LEAN_PCT", "0.25"))

# --- Whale watch (Binance BTCUSDT; context pings, not trade signals) ----------
WHALE_BURST_USD = float(os.getenv("WHALE_BURST_USD", "100e6"))  # net market buying/selling in 5 min, perp + spot
WHALE_LIQ_USD = float(os.getenv("WHALE_LIQ_USD", "5e6"))        # longs or shorts liquidated in 5 min (a lower bound)
# Walls: in a 12-min sample (2026-10-27) the biggest spot band was $1-8M per exchange and $33M on the OKX perp.
WHALE_WALL_USD = float(os.getenv("WHALE_WALL_USD", "5e6"))      # resting spot orders in one $50 price band
WHALE_WALL_PERP_USD = float(os.getenv("WHALE_WALL_PERP_USD", "30e6"))  # same on a perp book (10-30x deeper)

# --- Hyperliquid top traders (positions are public there; context + a paper copy wallet) ---
HL_TOP_N = int(os.getenv("HL_TOP_N", "20"))                      # best 30-day profit, re-picked every HL_REFRESH_H
HL_REFRESH_H = float(os.getenv("HL_REFRESH_H", "24"))
HL_PING_USD = float(os.getenv("HL_PING_USD", "5e6"))          # a BTC position opened/closed this big
HL_PING_ALT_USD = float(os.getenv("HL_PING_ALT_USD", "10e6"))  # same on any other coin or stock/commodity future
# Extra Hyperliquid exchanges to watch (HIP-3). "xyz" = stock/commodity futures: on 2026-10-29 the top 20
# held $403M there and ~$1M on all 9 others combined. Comma-separated; empty = crypto only.
HL_DEXES = tuple(d.strip() for d in os.getenv("HL_DEXES", "xyz").split(",") if d.strip())

HISTORY_DAYS =int(os.getenv("HISTORY_DAYS", "729"))  # Yahoo hourly data stops at ~730d

# --- Files ----------------------------------------------------------------
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "model.joblib"
REPORT_PATH = ROOT / "training_report.txt"
DB_PATH = ROOT / "paper_log.db"
DATA_DIR.mkdir(exist_ok=True)
