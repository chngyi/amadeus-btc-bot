"""Print the current call in the terminal (no Telegram needed).

    python check.py
"""
from config import BOT_NAME
from signal_engine import compute_signal, format_signal

if __name__ == "__main__":
    print(format_signal(compute_signal(), header=f"{BOT_NAME}:"))
