#!/usr/bin/env python3
"""Train a lightweight live model aligned with the notebook flow.

It builds historical features from yfinance data and trains RF (+ optional XGBoost)
for 1-day return prediction, then saves artifacts for live inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle



def _load_yfinance():
    import yfinance as yf  # type: ignore
    return yf


def _maybe_xgb():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None




def _load_core_ml():
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    from sklearn.metrics import r2_score  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore
    return np, pd, RandomForestRegressor, r2_score, TimeSeriesSplit


def fetch_symbol_frame(symbol: str, period: str = "2y"):
    _, pd, _, _, _ = _load_core_ml()
    yf = _load_yfinance()
    t = yf.Ticker(symbol)
    px = t.history(period=period)
    if px.empty:
        return pd.DataFrame()

    # options snapshot features (latest snapshot reused for simplicity)
    put_v = call_v = put_oi = call_oi = 0.0
    expiries = t.options[:3]
    for exp in expiries:
        ch = t.option_chain(exp)
        if not ch.calls.empty:
            call_v += float(ch.calls["volume"].fillna(0).sum())
            call_oi += float(ch.calls["openInterest"].fillna(0).sum())
        if not ch.puts.empty:
            put_v += float(ch.puts["volume"].fillna(0).sum())
            put_oi += float(ch.puts["openInterest"].fillna(0).sum())

    pcr_volume = put_v / max(call_v, 1.0)
    pcr_oi = put_oi / max(call_oi, 1.0)
    uoa_ratio = (put_v + call_v) / max(put_oi + call_oi, 1.0)

    df = pd.DataFrame(index=px.index)
    df["symbol"] = symbol
    df["close_price"] = px["Close"]
    df["volume"] = px["Volume"]
    df["return_1d"] = px["Close"].pct_change()
    df["return_5d"] = px["Close"].pct_change(5)
    df["ma_10"] = px["Close"].rolling(10).mean()
    df["ma_20"] = px["Close"].rolling(20).mean()
    df["vol_20"] = px["Close"].pct_change().rolling(20).std()
    df["pcr_volume"] = pcr_volume
    df["pcr_open_interest"] = pcr_oi
    df["uoa_ratio"] = uoa_ratio

    # notebook-like target
    df["target_1d"] = df["close_price"].pct_change().shift(-1)
    return df.reset_index(names="date")


def prepare_dataset(symbols: list[str]):
    np, pd, _, _, _ = _load_core_ml()
    frames = [fetch_symbol_frame(s) for s in symbols]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = df.dropna(subset=["target_1d"]).copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def train(df) -> dict:
    _, _, RandomForestRegressor, r2_score, TimeSeriesSplit = _load_core_ml()
    feature_cols = [
        "close_price",
        "volume",
        "return_1d",
        "return_5d",
        "ma_10",
        "ma_20",
        "vol_20",
        "pcr_volume",
        "pcr_open_interest",
        "uoa_ratio",
    ]
    X = df[feature_cols]
    y = df["target_1d"]

    tscv = TimeSeriesSplit(n_splits=3)
    train_idx, test_idx = next(tscv.split(X))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = float(r2_score(y_test, rf_pred))

    models = {"random_forest": rf}
    metrics = {"random_forest_r2": rf_r2}

    xgb_mod = _maybe_xgb()
    if xgb_mod is not None:
        xgb = xgb_mod.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
        xgb.fit(X_train, y_train, verbose=False)
        xgb_pred = xgb.predict(X_test)
        metrics["xgboost_r2"] = float(r2_score(y_test, xgb_pred))
        models["xgboost"] = xgb

    return {"models": models, "feature_cols": feature_cols, "metrics": metrics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,IWM")
    ap.add_argument("--out", default="artifacts/live_model.pkl")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    df = prepare_dataset(symbols)
    if df.empty:
        print("No training data.")
        return 1

    bundle = train(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)

    print("Saved model:", args.out)
    print(json.dumps(bundle["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
