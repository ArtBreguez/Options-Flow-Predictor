#!/usr/bin/env python3
"""Live inference using trained model artifacts + options-flow gates."""

from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime


from live_options_polling import fetch_symbol_snapshot, snapshot_signature


def _load_yfinance():
    import yfinance as yf  # type: ignore
    return yf


def build_live_row(symbol: str, snap):
    import pandas as pd  # type: ignore
    yf = _load_yfinance()
    t = yf.Ticker(symbol)
    px = t.history(period="30d")
    close = px["Close"]

    row = {
        "close_price": float(close.iloc[-1]),
        "volume": float(px["Volume"].iloc[-1]),
        "return_1d": float(close.pct_change().iloc[-1]),
        "return_5d": float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0,
        "ma_10": float(close.rolling(10).mean().iloc[-1]),
        "ma_20": float(close.rolling(20).mean().iloc[-1]),
        "vol_20": float(close.pct_change().rolling(20).std().iloc[-1]),
        "pcr_volume": snap.pcr_volume,
        "pcr_open_interest": snap.pcr_oi,
        "uoa_ratio": snap.uoa_ratio,
    }
    return pd.DataFrame([row]).fillna(0)


def decide(ml_pred: float, pcr_signal: str, unusual_volume: bool, min_abs_pred: float) -> str:
    ml_signal = "bullish" if ml_pred > 0 else "bearish"
    aligned = (ml_signal == pcr_signal)
    if abs(ml_pred) < min_abs_pred:
        return "NO_TRADE"
    if aligned and ml_signal == "bullish":
        return "LONG"
    if aligned and ml_signal == "bearish":
        return "SHORT"
    if unusual_volume:
        return "NO_TRADE_VOLATILITY"
    return "NO_TRADE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/live_model.pkl")
    ap.add_argument("--symbols", default="SPY,QQQ,IWM")
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--min-abs-pred", type=float, default=0.0003)
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    rf = bundle["models"]["random_forest"]
    feature_cols = bundle["feature_cols"]

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    last = {}

    while True:
        print(f"\n[{datetime.now().isoformat()}] ml polling {', '.join(symbols)}")
        for s in symbols:
            try:
                snap = fetch_symbol_snapshot(s)
                if not snap:
                    print(f"- {s}: no snapshot")
                    continue
                sig = snapshot_signature(snap)
                x = build_live_row(s, snap)[feature_cols]
                ml_pred = float(rf.predict(x)[0])
                action = decide(ml_pred, snap.pcr_signal, snap.unusual_volume, args.min_abs_pred)

                if last.get(s) != (sig, action):
                    print(
                        f"- {s}: action={action} ml_pred={ml_pred:.6f} pcr={snap.pcr_signal} "
                        f"uoa={snap.uoa_ratio:.3f} unusual={snap.unusual_volume}"
                    )
                    last[s] = (sig, action)
            except Exception as exc:
                print(f"- {s}: error={exc}")

        if args.once:
            return 0
        time.sleep(max(args.poll_seconds, 30))


if __name__ == "__main__":
    raise SystemExit(main())
