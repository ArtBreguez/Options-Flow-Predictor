#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from feature_pipeline import FEATURE_COLS, bollinger_position, macd_series, options_snapshot_features, rsi_series, vix_context


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

    vix = vix_context(yf, period)
    df = df.join(vix.reindex(df.index).ffill().fillna(0), how="left")

    opt = options_snapshot_features(t, float(px["Close"].iloc[-1]))
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
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def evaluate_walk_forward(X, y, n_splits: int, use_xgb: bool):
    np, _, RandomForestRegressor, r2_score, mean_squared_error, TimeSeriesSplit = _load_core_ml()
    xgb_mod = _maybe_xgb() if use_xgb else None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []

    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        rf_pred = rf.predict(Xte)
        row = {
            "fold": fold,
            "rf_r2": float(r2_score(yte, rf_pred)),
            "rf_rmse": float(np.sqrt(mean_squared_error(yte, rf_pred))),
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
            xgb.fit(Xtr, ytr, verbose=False)
            xgb_pred = xgb.predict(Xte)
            ens = (rf_pred + xgb_pred) / 2
            row.update(
                {
                    "xgb_r2": float(r2_score(yte, xgb_pred)),
                    "xgb_rmse": float(np.sqrt(mean_squared_error(yte, xgb_pred))),
                    "ensemble_r2": float(r2_score(yte, ens)),
                    "ensemble_rmse": float(np.sqrt(mean_squared_error(yte, ens))),
                }
            )
        metrics.append(row)
    return metrics


def summarize_metrics(folds):
    import numpy as np  # type: ignore
    keys = sorted({k for f in folds for k in f.keys() if k != "fold"})
    out = {}
    for k in keys:
        vals = [f[k] for f in folds if k in f]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def fit_final_models(X, y, use_xgb: bool):
    _, pd, RandomForestRegressor, _, _, _ = _load_core_ml()
    xgb_mod = _maybe_xgb() if use_xgb else None

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    models = {"random_forest": rf}

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


def train(df, n_splits: int, use_xgb: bool):
    X = df[FEATURE_COLS]
    y = df["target_1d"]
    folds = evaluate_walk_forward(X, y, n_splits=n_splits, use_xgb=use_xgb)
    summary = summarize_metrics(folds)
    models, fi = fit_final_models(X, y, use_xgb=use_xgb)
    return {
        "models": models,
        "feature_cols": FEATURE_COLS,
        "fold_metrics": folds,
        "metrics_summary": summary,
        "feature_importance_top20": fi,
        "n_rows": int(len(df)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,IWM")
    ap.add_argument("--out", default="artifacts/live_model.pkl")
    ap.add_argument("--period", default="max")
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--disable-xgb", action="store_true")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
