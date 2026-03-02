#!/usr/bin/env python3
"""Shared feature engineering for train/live scripts.

Designed to mirror notebook concepts with a consistent schema:
- technical indicators (RSI, MACD, Bollinger position)
- volatility context
- options flow (PCR/UOA)
- dealer proxy (net gamma exposure)
- IV structure (ATM IV + call/put spread)
- VIX regime context
"""

from __future__ import annotations

import math
from dataclasses import dataclass


FEATURE_COLS = [
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
    "net_gamma_exposure",
    "total_gamma",
    "atm_iv_average",
    "call_put_iv_spread",
    "VIX",
    "VIX_Z_Score",
    "VIX_Term_Structure",
    "vix_regime",
]


def rsi_series(close, window: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean().replace(0, 1e-9)
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd_series(close):
    return close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()


def bollinger_position(close, window: int = 20):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std().replace(0, 1e-9)
    return (close - ma) / (2 * std)


def _gamma_black_scholes(s: float, k: float, t: float, iv: float, r: float = 0.02) -> float:
    if s <= 0 or k <= 0:
        return 0.0
    t = max(t, 1 / 365)
    iv = max(iv, 0.01)
    d1 = (math.log(s / k) + (r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    pdf = math.exp(-(d1**2) / 2) / math.sqrt(2 * math.pi)
    gamma = pdf / (s * iv * math.sqrt(t))
    if math.isfinite(gamma):
        return float(gamma)
    return 0.0


def options_snapshot_features(ticker, spot_price: float):
    put_v = call_v = put_oi = call_oi = 0.0
    total_gamma = 0.0
    call_gex = 0.0
    put_gex = 0.0
    atm_call_ivs = []
    atm_put_ivs = []

    expiries = ticker.options[:3]
    for exp in expiries:
        chain = ticker.option_chain(exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        if not calls.empty:
            call_v += float(calls["volume"].fillna(0).sum())
            call_oi += float(calls["openInterest"].fillna(0).sum())
        if not puts.empty:
            put_v += float(puts["volume"].fillna(0).sum())
            put_oi += float(puts["openInterest"].fillna(0).sum())

        # gamma / gex proxies
        for side, df in (("call", calls), ("put", puts)):
            if df.empty:
                continue
            for _, row in df.iterrows():
                strike = float(row.get("strike", 0) or 0)
                iv = float(row.get("impliedVolatility", 0) or 0)
                oi = float(row.get("openInterest", 0) or 0)
                m = strike / max(spot_price, 1e-9)
                texp = 30 / 365  # proxy for short-dated options
                g = _gamma_black_scholes(spot_price, strike, texp, iv)
                gex = g * oi * 100 * (spot_price**2)
                total_gamma += g
                if side == "call":
                    call_gex += gex
                    if abs(m - 1.0) < 0.05 and iv > 0:
                        atm_call_ivs.append(iv)
                else:
                    put_gex += gex
                    if abs(m - 1.0) < 0.05 and iv > 0:
                        atm_put_ivs.append(iv)

    pcr_volume = put_v / max(call_v, 1.0)
    pcr_oi = put_oi / max(call_oi, 1.0)
    uoa_ratio = (put_v + call_v) / max(put_oi + call_oi, 1.0)

    pcr_signal = 1 if pcr_volume < 0.7 else -1 if pcr_volume > 1.0 else 0
    unusual_volume = 1 if uoa_ratio > 1.25 else 0

    atm_call_iv = sum(atm_call_ivs) / len(atm_call_ivs) if atm_call_ivs else 0.2
    atm_put_iv = sum(atm_put_ivs) / len(atm_put_ivs) if atm_put_ivs else 0.2

    return {
        "pcr_volume": pcr_volume,
        "pcr_open_interest": pcr_oi,
        "uoa_ratio": uoa_ratio,
        "pcr_signal_numeric": pcr_signal,
        "unusual_volume_signal": unusual_volume,
        "net_gamma_exposure": call_gex - put_gex,
        "total_gamma": total_gamma,
        "atm_iv_average": (atm_call_iv + atm_put_iv) / 2,
        "call_put_iv_spread": atm_call_iv - atm_put_iv,
    }


def vix_context(yf, period: str):
    _, pd = _load_np_pd()
    vix = yf.Ticker("^VIX").history(period=period)
    if vix.empty:
        return pd.DataFrame(columns=["VIX", "VIX_Z_Score", "VIX_Term_Structure", "vix_regime"])

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
    out["vix_regime"] = out["VIX"].apply(lambda x: 1 if x < 15 else -1 if x > 25 else 0)
    return out[["VIX", "VIX_Z_Score", "VIX_Term_Structure", "vix_regime"]].fillna(0)


def _load_np_pd():
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    return np, pd


def temporal_options_features(px, base_opt: dict):
    """Create time-varying options-flow proxies per date.

    Yahoo free data does not provide full historical option chains per day,
    so we derive progressive proxies from price/volume dynamics and blend with
    a static options snapshot anchor.
    """
    _, pd = _load_np_pd()
    close = px["Close"]
    vol = px["Volume"].replace(0, 1)
    ret = close.pct_change().fillna(0)

    down_vol = (vol * (ret < 0).astype(int)).rolling(10, min_periods=1).sum()
    up_vol = (vol * (ret > 0).astype(int)).rolling(10, min_periods=1).sum().replace(0, 1)
    flow_pcr_proxy = (down_vol / up_vol).clip(0.2, 5.0)

    oi_proxy = vol.rolling(20, min_periods=1).mean().replace(0, 1)
    uoa_dynamic = (vol / oi_proxy).clip(0, 5.0)

    pcr_base = float(base_opt.get("pcr_volume", 1.0))
    pcr_oi_base = float(base_opt.get("pcr_open_interest", 1.0))
    uoa_base = float(base_opt.get("uoa_ratio", 1.0))

    # blend static anchor with dynamic proxy to get per-date variation
    pcr_volume = (0.5 * pcr_base + 0.5 * flow_pcr_proxy).clip(0.2, 5.0)
    pcr_oi = (0.7 * pcr_oi_base + 0.3 * flow_pcr_proxy).clip(0.2, 5.0)
    uoa_ratio = (0.4 * uoa_base + 0.6 * uoa_dynamic).clip(0, 5.0)

    pcr_signal = pcr_volume.apply(lambda x: 1 if x < 0.7 else -1 if x > 1.0 else 0)
    unusual = (uoa_ratio > 1.25).astype(int)

    # vary dealer/iv proxies with realized vol/moneyness pressure
    realized_vol = ret.rolling(20, min_periods=1).std().fillna(0)
    gex_scale = (1 + (realized_vol * 100)).clip(0.5, 3.0)

    out = pd.DataFrame(index=px.index)
    out["pcr_volume"] = pcr_volume
    out["pcr_open_interest"] = pcr_oi
    out["uoa_ratio"] = uoa_ratio
    out["pcr_signal_numeric"] = pcr_signal
    out["unusual_volume_signal"] = unusual
    out["net_gamma_exposure"] = float(base_opt.get("net_gamma_exposure", 0.0)) * gex_scale
    out["total_gamma"] = float(base_opt.get("total_gamma", 0.0)) * gex_scale
    out["atm_iv_average"] = float(base_opt.get("atm_iv_average", 0.2)) + realized_vol
    out["call_put_iv_spread"] = float(base_opt.get("call_put_iv_spread", 0.0)) + (ret.rolling(5, min_periods=1).mean())
    return out.fillna(0)
