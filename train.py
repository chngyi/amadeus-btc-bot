"""Download history, test the model honestly, and save it.

    python train.py
"""
from __future__ import annotations

import time

import joblib

from config import HISTORY_DAYS, MODEL_PATH, REPORT_PATH
from data import load_bundle
from model import train


def run(verbose: bool = True) -> str:
    t0 = time.time()
    if verbose:
        print(f"Downloading ~{HISTORY_DAYS} days of hourly data (first run takes a few minutes)...")
    bundle = load_bundle(HISTORY_DAYS, live=False)
    if verbose:
        for k, v in bundle.items():
            n = 0 if v is None else len(v)
            print(f"  {k:8s} {n:>7,} rows")
        print("Training + walk-forward test...\n")
    art, report = train(bundle, verbose=verbose)
    joblib.dump(art, MODEL_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8")
    if verbose:
        print(f"\nSaved model -> {MODEL_PATH.name}, report -> {REPORT_PATH.name}  ({time.time()-t0:.0f}s)")
    return report


if __name__ == "__main__":
    run()
