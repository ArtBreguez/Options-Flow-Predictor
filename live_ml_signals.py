#!/usr/bin/env python3
"""Live inference using trained model artifacts + notebook-like gating."""

from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime

from live_options_polling import fetch_symbol_snapshot, get_vix_metrics, snapshot_signature


def _load_yfinance():
    import yfinance as yf  # type: ignore

    return yf


def _rsi(close, window: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean().replace(0, 1e-9)
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _macd(close):
    return close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()


def _bb_position(close, window: int = 20):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std().replace(0, 1e-9)
    return (close - ma) / (2 * std)


def build_live_row(symbol: str, snap, signal_timeframe: str = "1m"):
    import pandas as pd  # type: ignore

    yf = _load_yfinance()
    t = yf.Ticker(symbol)
    period_for_interval = "7d" if signal_timeframe == "1m" else "1mo"
    px = t.history(period=period_for_interval, interval=signal_timeframe)
    close = px["Close"]
    ret = close.pct_change()

    vix, vix_term = get_vix_metrics()
    vix_val = float(vix) if vix is not None else 20.0

    row = {
        "close_price": float(close.iloc[-1]),
        "volume": float(px["Volume"].iloc[-1]),
        "return_1d": float(ret.iloc[-1]),
        "return_5d": float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0,
        "rsi": float(_rsi(close).iloc[-1]) if len(close) > 14 else 50.0,
        "macd": float(_macd(close).iloc[-1]),
        "bb_position": float(_bb_position(close).iloc[-1]) if len(close) > 20 else 0.0,
        "volatility_forecast": float(ret.rolling(30).std().iloc[-1]) if len(close) > 30 else float(ret.std() or 0.0),
        "ma_10": float(close.rolling(10).mean().iloc[-1]) if len(close) > 10 else float(close.iloc[-1]),
        "ma_20": float(close.rolling(20).mean().iloc[-1]) if len(close) > 20 else float(close.iloc[-1]),
        "vol_20": float(ret.rolling(20).std().iloc[-1]) if len(close) > 20 else 0.0,
        "pcr_volume": snap.pcr_volume,
        "pcr_open_interest": snap.pcr_oi,
        "uoa_ratio": snap.uoa_ratio,
        "pcr_signal_numeric": 1 if snap.pcr_signal == "bullish" else -1 if snap.pcr_signal == "bearish" else 0,
        "unusual_volume_signal": 1 if snap.unusual_volume else 0,
        "VIX": vix_val,
        "VIX_Z_Score": 0.0,
        "VIX_Term_Structure": float(vix_term) if vix_term is not None else 0.0,
    }

    return pd.DataFrame([row]).fillna(0)


def decide(ml_pred: float, pcr_signal: str, unusual_volume: bool, min_abs_pred: float) -> str:
    ml_signal = "bullish" if ml_pred > 0 else "bearish"
    aligned = ml_signal == pcr_signal
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
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--signal-timeframe", default="1m", help="Price timeframe for live features (default: 1m)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--min-abs-pred", type=float, default=0.0003)
    ap.add_argument("--print-all", action="store_true", help="Print every polling cycle, not only when state changes")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    rf = bundle["models"]["random_forest"]
    xgb = bundle["models"].get("xgboost")
    feature_cols = bundle["feature_cols"]

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    last = {}

    while True:
        print(f"\n[{datetime.now().isoformat()}] ml polling {', '.join(symbols)}")
        for s in symbols:
            try:
                snap = fetch_symbol_snapshot(s, price_interval=args.signal_timeframe)
                if not snap:
                    print(f"- {s}: no snapshot")
                    continue

                sig = snapshot_signature(snap)
                x = build_live_row(s, snap, signal_timeframe=args.signal_timeframe)[feature_cols]

                rf_pred = float(rf.predict(x)[0])
                if xgb is not None:
                    xgb_pred = float(xgb.predict(x)[0])
                    ml_pred = (rf_pred + xgb_pred) / 2
                else:
                    xgb_pred = None
                    ml_pred = rf_pred

                action = decide(ml_pred, snap.pcr_signal, snap.unusual_volume, args.min_abs_pred)
                current = (sig, action, round(ml_pred, 7))

                if args.print_all or last.get(s) != current:
                    msg = (
                        f"- {s}: action={action} ml_pred={ml_pred:.6f} rf_pred={rf_pred:.6f} "
                        f"pcr={snap.pcr_signal} uoa={snap.uoa_ratio:.3f} unusual={snap.unusual_volume}"
                    )
                    if xgb_pred is not None:
                        msg += f" xgb_pred={xgb_pred:.6f}"
                    print(msg)
                    last[s] = current
            except Exception as exc:
                print(f"- {s}: error={exc}")

        if args.once:
            return 0
        time.sleep(max(args.poll_seconds, 30))


if __name__ == "__main__":
    raise SystemExit(main())
