#!/usr/bin/env python3
"""Robust training pipeline aligned to the notebook flow.

Improvements over the prior MVP:
- richer feature set (technical + options-flow + VIX regime context)
- walk-forward TimeSeriesSplit validation across all folds
- RF + optional XGBoost + ensemble evaluation
- fit final models on full data and persist artifacts for live inference
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


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
    from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore

    return np, pd, RandomForestRegressor, r2_score, mean_squared_error, TimeSeriesSplit


def rsi_series(close, window: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def macd_series(close):
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    return ema_fast - ema_slow


def bollinger_position(close, window: int = 20):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std().replace(0, 1e-9)
    return (close - ma) / (2 * std)


def get_vix_context(period: str):
    _, pd, _, _, _, _ = _load_core_ml()
    yf = _load_yfinance()

    vix = yf.Ticker("^VIX").history(period=period)
    if vix.empty:
        return pd.DataFrame(columns=["VIX", "VIX_Z_Score", "VIX_Term_Structure"])

    out = pd.DataFrame(index=vix.index)
    out["VIX"] = vix["Close"]

    vix9d = yf.Ticker("^VIX9D").history(period=period)
    if not vix9d.empty:
        out["VIX9D"] = vix9d["Close"].reindex(out.index).ffill()
        out["VIX_Term_Structure"] = out["VIX"] - out["VIX9D"]
    else:
        out["VIX_Term_Structure"] = 0.0

    out["VIX_MA_50"] = out["VIX"].rolling(50).mean()
    out["VIX_Z_Score"] = (out["VIX"] - out["VIX_MA_50"]) / out["VIX"].rolling(50).std().replace(0, 1e-9)
    return out[["VIX", "VIX_Z_Score", "VIX_Term_Structure"]].fillna(0)


def get_options_snapshot_features(ticker):
    put_v = call_v = put_oi = call_oi = 0.0
    expiries = ticker.options[:3]
    for exp in expiries:
        chain = ticker.option_chain(exp)
        if not chain.calls.empty:
            call_v += float(chain.calls["volume"].fillna(0).sum())
            call_oi += float(chain.calls["openInterest"].fillna(0).sum())
        if not chain.puts.empty:
            put_v += float(chain.puts["volume"].fillna(0).sum())
            put_oi += float(chain.puts["openInterest"].fillna(0).sum())

    pcr_volume = put_v / max(call_v, 1.0)
    pcr_oi = put_oi / max(call_oi, 1.0)
    uoa_ratio = (put_v + call_v) / max(put_oi + call_oi, 1.0)

    pcr_signal = 1 if pcr_volume < 0.7 else -1 if pcr_volume > 1.0 else 0
    unusual_volume = 1 if uoa_ratio > 1.25 else 0

    return {
        "pcr_volume": pcr_volume,
        "pcr_open_interest": pcr_oi,
        "uoa_ratio": uoa_ratio,
        "pcr_signal_numeric": pcr_signal,
        "unusual_volume_signal": unusual_volume,
    }


def fetch_symbol_frame(symbol: str, period: str = "max"):
    np, pd, _, _, _, _ = _load_core_ml()
    yf = _load_yfinance()

    t = yf.Ticker(symbol)
    px = t.history(period=period)
    if px.empty:
        return pd.DataFrame()

    df = pd.DataFrame(index=px.index)
    df["symbol"] = symbol
    df["close_price"] = px["Close"]
    df["volume"] = px["Volume"]
    df["return_1d"] = px["Close"].pct_change()
    df["return_5d"] = px["Close"].pct_change(5)

    df["rsi"] = rsi_series(px["Close"]).fillna(50)
    df["macd"] = macd_series(px["Close"]).fillna(0)
    df["bb_position"] = bollinger_position(px["Close"]).fillna(0)

    ret = px["Close"].pct_change()
    df["volatility_forecast"] = ret.rolling(30).std().fillna(ret.std() if not np.isnan(ret.std()) else 0)

    df["ma_10"] = px["Close"].rolling(10).mean().fillna(px["Close"])
    df["ma_20"] = px["Close"].rolling(20).mean().fillna(px["Close"])
    df["vol_20"] = ret.rolling(20).std().fillna(0)

    vix_ctx = get_vix_context(period)
    df = df.join(vix_ctx.reindex(df.index).ffill().fillna(0), how="left")

    opt = get_options_snapshot_features(t)
    for k, v in opt.items():
        df[k] = v

    df["target_1d"] = df["close_price"].pct_change().shift(-1)
    return df.reset_index(names="date")


def prepare_dataset(symbols: list[str], period: str):
    np, pd, _, _, _, _ = _load_core_ml()
    frames = [fetch_symbol_frame(s, period=period) for s in symbols]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = df.dropna(subset=["target_1d"]).copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df


def evaluate_walk_forward(X, y, n_splits: int, use_xgb: bool):
    np, _, RandomForestRegressor, r2_score, mean_squared_error, TimeSeriesSplit = _load_core_ml()
    xgb_mod = _maybe_xgb() if use_xgb else None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        row = {
            "fold": fold,
            "rf_r2": float(r2_score(y_test, rf_pred)),
            "rf_rmse": float(np.sqrt(mean_squared_error(y_test, rf_pred))),
        }

        if xgb_mod is not None:
            xgb = xgb_mod.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            xgb.fit(X_train, y_train, verbose=False)
            xgb_pred = xgb.predict(X_test)
            ens = (rf_pred + xgb_pred) / 2

            row["xgb_r2"] = float(r2_score(y_test, xgb_pred))
            row["xgb_rmse"] = float(np.sqrt(mean_squared_error(y_test, xgb_pred)))
            row["ensemble_r2"] = float(r2_score(y_test, ens))
            row["ensemble_rmse"] = float(np.sqrt(mean_squared_error(y_test, ens)))

        fold_metrics.append(row)

    return fold_metrics


def fit_final_models(X, y, use_xgb: bool):
    _, pd, RandomForestRegressor, _, _, _ = _load_core_ml()
    xgb_mod = _maybe_xgb() if use_xgb else None

    models = {}

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    models["random_forest"] = rf

    if xgb_mod is not None:
        xgb = xgb_mod.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        xgb.fit(X, y, verbose=False)
        models["xgboost"] = xgb

    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return models, fi.head(20).to_dict()


def summarize_metrics(fold_metrics):
    np, _, _, _, _, _ = _load_core_ml()
    keys = sorted({k for row in fold_metrics for k in row.keys() if k != "fold"})
    out = {}
    for k in keys:
        vals = [row[k] for row in fold_metrics if k in row]
        out[k + "_mean"] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
    return out


def train(df, n_splits: int = 5, use_xgb: bool = True) -> dict:
    feature_cols = [
        "close_price",
        "volume",
        "return_1d",
        "return_5d",
        "rsi",
        "macd",
        "bb_position",
        "volatility_forecast",
        "ma_10",
        "ma_20",
        "vol_20",
        "pcr_volume",
        "pcr_open_interest",
        "uoa_ratio",
        "pcr_signal_numeric",
        "unusual_volume_signal",
        "VIX",
        "VIX_Z_Score",
        "VIX_Term_Structure",
    ]

    X = df[feature_cols]
    y = df["target_1d"]

    fold_metrics = evaluate_walk_forward(X, y, n_splits=n_splits, use_xgb=use_xgb)
    summary = summarize_metrics(fold_metrics)
    models, feature_importance = fit_final_models(X, y, use_xgb=use_xgb)

    return {
        "models": models,
        "feature_cols": feature_cols,
        "fold_metrics": fold_metrics,
        "metrics_summary": summary,
        "feature_importance_top20": feature_importance,
        "n_rows": int(len(df)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,IWM")
    ap.add_argument("--out", default="artifacts/live_model.pkl")
    ap.add_argument("--period", default="max", help="yfinance history period for training (default: max)")
    ap.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit folds for robust walk-forward validation")
    ap.add_argument("--disable-xgb", action="store_true", help="Disable XGBoost even if installed")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    df = prepare_dataset(symbols, period=args.period)
    if df.empty:
        print("No training data.")
        return 1

    bundle = train(df, n_splits=args.cv_splits, use_xgb=not args.disable_xgb)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)

    print("Saved model:", args.out)
    print("Training period:", args.period)
    print("Rows:", bundle["n_rows"])
    print(json.dumps(bundle["metrics_summary"], indent=2))
    print("Top RF features:")
    for k, v in bundle["feature_importance_top20"].items():
        print(f"  - {k}: {v:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
