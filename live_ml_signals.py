#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime

from feature_pipeline import FEATURE_COLS, bollinger_position, macd_series, rsi_series, temporal_options_features
from live_options_polling import fetch_symbol_snapshot, get_vix_metrics, snapshot_signature


def _load_yfinance():
    import yfinance as yf  # type: ignore
    return yf


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

    base_opt = {
        "pcr_volume": snap.pcr_volume,
        "pcr_open_interest": snap.pcr_oi,
        "uoa_ratio": snap.uoa_ratio,
        "net_gamma_exposure": 0.0,
        "total_gamma": 0.0,
        "atm_iv_average": 0.2,
        "call_put_iv_spread": 0.0,
    }
    opt_ts = temporal_options_features(px, base_opt)
    opt_last = opt_ts.iloc[-1]

    row = {
        "close_price": float(close.iloc[-1]),
        "volume": float(px["Volume"].iloc[-1]),
        "return_1d": float(ret.iloc[-1]),
        "return_5d": float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0,
        "rsi": float(rsi_series(close).iloc[-1]) if len(close) > 14 else 50.0,
        "macd": float(macd_series(close).iloc[-1]),
        "bb_position": float(bollinger_position(close).iloc[-1]) if len(close) > 20 else 0.0,
        "volatility_forecast": float(ret.rolling(30).std().iloc[-1]) if len(close) > 30 else float(ret.std() or 0.0),
        "ma_10": float(close.rolling(10).mean().iloc[-1]) if len(close) > 10 else float(close.iloc[-1]),
        "ma_20": float(close.rolling(20).mean().iloc[-1]) if len(close) > 20 else float(close.iloc[-1]),
        "vol_20": float(ret.rolling(20).std().iloc[-1]) if len(close) > 20 else 0.0,
        "pcr_volume": float(opt_last["pcr_volume"]),
        "pcr_open_interest": float(opt_last["pcr_open_interest"]),
        "uoa_ratio": float(opt_last["uoa_ratio"]),
        "pcr_signal_numeric": int(opt_last["pcr_signal_numeric"]),
        "unusual_volume_signal": int(opt_last["unusual_volume_signal"]),
        "net_gamma_exposure": float(opt_last["net_gamma_exposure"]),
        "total_gamma": float(opt_last["total_gamma"]),
        "atm_iv_average": float(opt_last["atm_iv_average"]),
        "call_put_iv_spread": float(opt_last["call_put_iv_spread"]),
        "VIX": vix_val,
        "VIX_Z_Score": 0.0,
        "VIX_Term_Structure": float(vix_term) if vix_term is not None else 0.0,
        "vix_regime": 1 if vix_val < 15 else -1 if vix_val > 25 else 0,
    }
    return pd.DataFrame([row]).fillna(0)


def decide(ml_pred: float, pcr_signal: str, unusual_volume: bool, min_abs_pred: float, entry_mode: str = "balanced") -> str:
    ml_signal = "bullish" if ml_pred > 0 else "bearish"
    aligned = ml_signal == pcr_signal

    if entry_mode == "conservative":
        thresh = min_abs_pred * 1.5
    elif entry_mode == "aggressive":
        thresh = min_abs_pred * 0.7
    else:
        thresh = min_abs_pred

    if abs(ml_pred) < thresh:
        return "NO_TRADE"

    if entry_mode == "aggressive":
        if ml_signal == "bullish":
            return "LONG"
        return "SHORT"

    if aligned and ml_signal == "bullish":
        return "LONG"
    if aligned and ml_signal == "bearish":
        return "SHORT"

    if unusual_volume and entry_mode == "balanced":
        return "NO_TRADE_VOLATILITY"
    return "NO_TRADE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/live_model.pkl")
    ap.add_argument("--symbols", default="SPY,QQQ,IWM")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--signal-timeframe", default="1m")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--min-abs-pred", type=float, default=0.0003)
    ap.add_argument("--print-all", action="store_true")
    ap.add_argument("--entry-mode", choices=["conservative", "balanced", "aggressive"], default="balanced")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    rf = bundle["models"]["random_forest"]
    xgb = bundle["models"].get("xgboost")
    model_cols = bundle.get("feature_cols", FEATURE_COLS)
    model_tf = bundle.get("train_timeframe")
    if model_tf and model_tf != args.signal_timeframe:
        print(f"[warning] model trained on timeframe={model_tf}, running live on timeframe={args.signal_timeframe}")

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
                x = build_live_row(s, snap, signal_timeframe=args.signal_timeframe)
                x = x.reindex(columns=model_cols, fill_value=0)

                rf_pred = float(rf.predict(x)[0])
                if xgb is not None:
                    xgb_pred = float(xgb.predict(x)[0])
                    ml_pred = (rf_pred + xgb_pred) / 2
                else:
                    xgb_pred = None
                    ml_pred = rf_pred

                action = decide(ml_pred, snap.pcr_signal, snap.unusual_volume, args.min_abs_pred, entry_mode=args.entry_mode)
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
