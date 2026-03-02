#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

from feature_pipeline import (
    FEATURE_COLS,
    bollinger_position,
    macd_series,
    options_snapshot_features,
    rsi_series,
    temporal_options_features,
    vix_context,
)


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


def _period_for_interval(period: str, interval: str) -> str:
    if interval == "1m":
        return "7d"
    if interval in {"2m", "5m", "15m", "30m"}:
        return "60d" if period == "max" else period
    if interval in {"60m", "90m", "1h"}:
        return "730d" if period == "max" else period
    return period


def fetch_symbol_frame(symbol: str, period: str = "max", train_timeframe: str = "5m", target_bars: int = 1):
    np, pd, _, _, _, _ = _load_core_ml()
    yf = _load_yfinance()

    t = yf.Ticker(symbol)
    yf_period = _period_for_interval(period, train_timeframe)
    px = t.history(period=yf_period, interval=train_timeframe)
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

    # VIX stays daily; forward-filled onto intraday index
    vix = vix_context(yf, period="1y")
    df = df.join(vix.reindex(df.index).ffill().fillna(0), how="left")

    opt = options_snapshot_features(t, float(px["Close"].iloc[-1]))
    opt_ts = temporal_options_features(px, opt)
    df = df.join(opt_ts, how="left")

    df["target_return"] = df["close_price"].pct_change(target_bars).shift(-target_bars)
    return df.reset_index(names="date")


def prepare_dataset(symbols: list[str], period: str, train_timeframe: str, target_bars: int):
    np, pd, _, _, _, _ = _load_core_ml()
    frames = [fetch_symbol_frame(s, period=period, train_timeframe=train_timeframe, target_bars=target_bars) for s in symbols]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = df.dropna(subset=["target_return"]).copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def evaluate_walk_forward(X, y, n_splits: int, use_xgb: bool, n_estimators: int, max_depth: int | None):
    np, _, RandomForestRegressor, r2_score, mean_squared_error, TimeSeriesSplit = _load_core_ml()
    xgb_mod = _maybe_xgb() if use_xgb else None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []

    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        rf_pred = rf.predict(Xte)
        row = {
            "fold": fold,
            "rf_r2": float(r2_score(yte, rf_pred)),
            "rf_rmse": float(np.sqrt(mean_squared_error(yte, rf_pred))),
        }

        if xgb_mod is not None:
            xgb = xgb_mod.XGBRegressor(
                n_estimators=n_estimators,
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


def notebook_style_split(X, y):
    _, _, _, _, _, TimeSeriesSplit = _load_core_ml()
    tr, te = next(TimeSeriesSplit(n_splits=3).split(X))
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]


def compute_baselines(y_train, y_test):
    import numpy as np  # type: ignore
    from sklearn.metrics import mean_squared_error, r2_score  # type: ignore

    pred_zero = np.zeros(len(y_test))
    mu = float(np.mean(y_train))
    pred_mean = np.full(len(y_test), mu)

    return {
        "baseline_zero_r2": float(r2_score(y_test, pred_zero)),
        "baseline_zero_rmse": float(np.sqrt(mean_squared_error(y_test, pred_zero))),
        "baseline_mean_r2": float(r2_score(y_test, pred_mean)),
        "baseline_mean_rmse": float(np.sqrt(mean_squared_error(y_test, pred_mean))),
    }


def notebook_baseline_eval(X, y, use_xgb: bool, n_estimators: int, max_depth: int | None):
    import numpy as np  # type: ignore

    _, _, RandomForestRegressor, r2_score, mean_squared_error, _ = _load_core_ml()
    xgb_mod = _maybe_xgb() if use_xgb else None
    Xtr, Xte, ytr, yte = notebook_style_split(X, y)

    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    out = {
        "rf_r2": float(r2_score(yte, rf_pred)),
        "rf_rmse": float(np.sqrt(mean_squared_error(yte, rf_pred))),
    }

    if xgb_mod is not None:
        xgb = xgb_mod.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        xgb.fit(Xtr, ytr, verbose=False)
        xgb_pred = xgb.predict(Xte)
        ens = (rf_pred + xgb_pred) / 2
        out.update(
            {
                "xgb_r2": float(r2_score(yte, xgb_pred)),
                "xgb_rmse": float(np.sqrt(mean_squared_error(yte, xgb_pred))),
                "ensemble_r2": float(r2_score(yte, ens)),
                "ensemble_rmse": float(np.sqrt(mean_squared_error(yte, ens))),
            }
        )

    out.update(compute_baselines(ytr, yte))
    return out


def fit_final_models(X, y, use_xgb: bool, n_estimators: int, max_depth: int | None):
    _, pd, RandomForestRegressor, _, _, _ = _load_core_ml()
    xgb_mod = _maybe_xgb() if use_xgb else None

    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    models = {"random_forest": rf}

    if xgb_mod is not None:
        xgb = xgb_mod.XGBRegressor(
            n_estimators=n_estimators,
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


def train(df, n_splits: int, use_xgb: bool, notebook_mode: bool, n_estimators: int, max_depth: int | None, save_models: bool = True):
    X = df[FEATURE_COLS]
    y = df["target_return"]

    baseline_eval = notebook_baseline_eval(X, y, use_xgb=use_xgb, n_estimators=n_estimators, max_depth=max_depth)
    folds = evaluate_walk_forward(X, y, n_splits=n_splits, use_xgb=use_xgb, n_estimators=n_estimators, max_depth=max_depth)
    summary = summarize_metrics(folds)
    models, fi = fit_final_models(X, y, use_xgb=use_xgb, n_estimators=n_estimators, max_depth=max_depth) if save_models else ({}, {})

    metrics_summary = baseline_eval if notebook_mode else {**baseline_eval, **summary}
    return {
        "models": models,
        "feature_cols": FEATURE_COLS,
        "target_col": "target_return",
        "fold_metrics": folds,
        "notebook_baseline_metrics": baseline_eval,
        "metrics_summary": metrics_summary,
        "feature_importance_top20": fi,
        "n_rows": int(len(df)),
    }




def evaluate_config(symbols, period, timeframe, target_bars, cv_splits, use_xgb, notebook_mode, n_estimators, max_depth):
    df = prepare_dataset(symbols, period=period, train_timeframe=timeframe, target_bars=target_bars)
    if df.empty:
        return None
    bundle = train(df, n_splits=cv_splits, use_xgb=use_xgb, notebook_mode=notebook_mode, n_estimators=n_estimators, max_depth=max_depth, save_models=False)
    metrics = bundle["notebook_baseline_metrics"] if notebook_mode else bundle["metrics_summary"]
    score = metrics.get("rf_r2", metrics.get("rf_r2_mean", -1e9))
    baseline = metrics.get("baseline_mean_r2", -1e9)
    edge = score - baseline
    return {
        "timeframe": timeframe,
        "target_bars": target_bars,
        "bundle": bundle,
        "metrics": metrics,
        "score": score,
        "baseline": baseline,
        "edge": edge,
    }


def search_best_config(symbols, period, timeframes, target_bars_list, cv_splits, use_xgb, notebook_mode, n_estimators, max_depth):
    candidates = []
    for tf in timeframes:
        for bars in target_bars_list:
            print(f"[search] evaluating timeframe={tf}, target_bars={bars}")
            res = evaluate_config(symbols, period, tf, bars, cv_splits, use_xgb, notebook_mode, n_estimators, max_depth)
            if res is None:
                print("[search] skipped (no data)")
                continue
            print(f"[search] rf_r2={res['score']:.6f} baseline_mean_r2={res['baseline']:.6f} edge={res['edge']:.6f}")
            candidates.append(res)

    if not candidates:
        return None, []

    candidates.sort(key=lambda x: (x["edge"], x["score"]), reverse=True)
    return candidates[0], candidates

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,IWM")
    ap.add_argument("--out", default="artifacts/live_model.pkl")
    ap.add_argument("--period", default="max")
    ap.add_argument("--train-timeframe", default="5m", help="yfinance interval for training data (e.g., 5m)")
    ap.add_argument("--target-bars", type=int, default=1, help="Forward bars for target_return (5m timeframe: 1=+5m)")
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--disable-xgb", action="store_true")
    ap.add_argument("--notebook-mode", action="store_true", help="Report primary metrics using notebook-style first split")
    ap.add_argument("--n-estimators", type=int, default=120, help="Tree count (smaller = lighter model artifact)")
    ap.add_argument("--max-depth", type=int, default=8, help="Max tree depth (smaller = lighter artifact)")
    ap.add_argument("--search", action="store_true", help="Search best timeframe/target-bars combo against baseline")
    ap.add_argument("--search-timeframes", default="1m,2m,5m,15m", help="Comma list for --search")
    ap.add_argument("--search-target-bars", default="1,2,3", help="Comma list for --search")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.search:
        tfs = [x.strip() for x in args.search_timeframes.split(",") if x.strip()]
        bars_list = [int(x.strip()) for x in args.search_target_bars.split(",") if x.strip()]
        best, all_candidates = search_best_config(
            symbols=symbols,
            period=args.period,
            timeframes=tfs,
            target_bars_list=bars_list,
            cv_splits=args.cv_splits,
            use_xgb=not args.disable_xgb,
            notebook_mode=args.notebook_mode,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        if best is None:
            print("No training data for searched configs.")
            return 1
        # retrain best config with models enabled for deployment
        df_best = prepare_dataset(symbols, period=args.period, train_timeframe=best["timeframe"], target_bars=best["target_bars"])
        bundle = train(df_best, n_splits=args.cv_splits, use_xgb=not args.disable_xgb, notebook_mode=args.notebook_mode, n_estimators=args.n_estimators, max_depth=args.max_depth, save_models=True)
        bundle["search_results"] = [
            {"timeframe": c["timeframe"], "target_bars": c["target_bars"], "score": c["score"], "baseline": c["baseline"], "edge": c["edge"]}
            for c in all_candidates
        ]
        bundle["train_timeframe"] = best["timeframe"]
        bundle["target_bars"] = best["target_bars"]
        print(f"Selected best config timeframe={best['timeframe']} target_bars={best['target_bars']} edge={best['edge']:.6f}")
    else:
        df = prepare_dataset(symbols, period=args.period, train_timeframe=args.train_timeframe, target_bars=args.target_bars)
        if df.empty:
            print("No training data.")
            return 1
        bundle = train(df, n_splits=args.cv_splits, use_xgb=not args.disable_xgb, notebook_mode=args.notebook_mode, n_estimators=args.n_estimators, max_depth=args.max_depth, save_models=True)
        bundle["train_timeframe"] = args.train_timeframe
        bundle["target_bars"] = args.target_bars

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.out.endswith(".gz"):
        with gzip.open(args.out, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.out, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved model:", args.out)
    print("Training period:", args.period)
    print("Train timeframe:", bundle.get("train_timeframe", args.train_timeframe))
    print("Target bars:", bundle.get("target_bars", args.target_bars))
    print("Rows:", bundle["n_rows"])
    print(json.dumps(bundle["metrics_summary"], indent=2))
    m = bundle.get("metrics_summary", {})
    if "rf_r2" in m and "baseline_mean_r2" in m:
        edge = float(m["rf_r2"]) - float(m["baseline_mean_r2"])
        print(f"Edge vs baseline_mean_r2: {edge:.6f}")
        if edge <= 0:
            print("[warning] Model is not beating baseline_mean in this config.")
    if not args.notebook_mode:
        print("Notebook-style baseline metrics:")
        print(json.dumps(bundle["notebook_baseline_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
