#!/usr/bin/env python3
"""MVP live monitoring for options flow using free data sources.

Strategy (zero-cost MVP):
- Poll Yahoo Finance data (via yfinance) every 5 minutes.
- Pull spot price + options chains + VIX.
- Recompute rule-based signals (put/call + unusual volume).
- Print signals to terminal (no webhook/notification delivery).

Notes:
- This is near-real-time polling, not direct exchange feed.
- Best effort only: Yahoo data can be delayed/intermittent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


def _load_yfinance():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except Exception as exc:
        raise RuntimeError("yfinance not installed. Install with: pip install yfinance") from exc


@dataclass
class SignalSnapshot:
    symbol: str
    price: float
    pcr_volume: float
    pcr_oi: float
    uoa_ratio: float
    unusual_volume: bool
    pcr_signal: str
    vix: Optional[float]
    vix_term_structure: Optional[float]
    timestamp: str


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return default


def get_vix_metrics() -> tuple[Optional[float], Optional[float]]:
    yf = _load_yfinance()
    vix = yf.Ticker("^VIX").history(period="5d")
    if vix.empty:
        return None, None
    vix_last = safe_float(vix["Close"].iloc[-1], None)

    vix9d = yf.Ticker("^VIX9D").history(period="5d")
    if vix9d.empty:
        return vix_last, None
    vix9d_last = safe_float(vix9d["Close"].iloc[-1], None)
    if vix_last is None or vix9d_last is None:
        return vix_last, None
    return vix_last, vix_last - vix9d_last


def fetch_symbol_snapshot(symbol: str, expiries_to_use: int = 3, price_interval: str = "1m") -> Optional[SignalSnapshot]:
    yf = _load_yfinance()
    ticker = yf.Ticker(symbol)
    # 1m candles are only available for limited history windows on Yahoo
    period_for_interval = "7d" if price_interval == "1m" else "1mo"
    px_hist = ticker.history(period=period_for_interval, interval=price_interval)
    if px_hist.empty:
        return None

    price = safe_float(px_hist["Close"].iloc[-1])

    total_put_volume = 0.0
    total_call_volume = 0.0
    total_put_oi = 0.0
    total_call_oi = 0.0

    expiries = ticker.options[:expiries_to_use]
    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts

        total_call_volume += safe_float(calls["volume"].fillna(0).sum()) if not calls.empty else 0.0
        total_put_volume += safe_float(puts["volume"].fillna(0).sum()) if not puts.empty else 0.0

        total_call_oi += safe_float(calls["openInterest"].fillna(0).sum()) if not calls.empty else 0.0
        total_put_oi += safe_float(puts["openInterest"].fillna(0).sum()) if not puts.empty else 0.0

    pcr_volume = total_put_volume / max(total_call_volume, 1.0)
    pcr_oi = total_put_oi / max(total_call_oi, 1.0)

    total_volume = total_put_volume + total_call_volume
    total_oi = total_put_oi + total_call_oi
    uoa_ratio = total_volume / max(total_oi, 1.0)

    pcr_signal = "neutral"
    if pcr_volume < 0.7:
        pcr_signal = "bullish"
    elif pcr_volume > 1.0:
        pcr_signal = "bearish"

    unusual_volume = uoa_ratio > 1.25

    vix, vix_term = get_vix_metrics()

    return SignalSnapshot(
        symbol=symbol,
        price=price,
        pcr_volume=pcr_volume,
        pcr_oi=pcr_oi,
        uoa_ratio=uoa_ratio,
        unusual_volume=unusual_volume,
        pcr_signal=pcr_signal,
        vix=vix,
        vix_term_structure=vix_term,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def snapshot_signature(s: SignalSnapshot) -> str:
    return json.dumps(
        {
            "symbol": s.symbol,
            "pcr_signal": s.pcr_signal,
            "unusual_volume": s.unusual_volume,
            "bucket_pred": "bullish" if s.pcr_signal == "bullish" and s.unusual_volume else s.pcr_signal,
        },
        sort_keys=True,
    )


def format_snapshot(s: SignalSnapshot) -> str:
    return (
        f"symbol={s.symbol}\n"
        f"time_utc={s.timestamp}\n"
        f"price={s.price:.2f}\n"
        f"pcr_volume={s.pcr_volume:.3f}\n"
        f"pcr_oi={s.pcr_oi:.3f}\n"
        f"uoa_ratio={s.uoa_ratio:.3f}\n"
        f"unusual_volume={s.unusual_volume}\n"
        f"pcr_signal={s.pcr_signal}\n"
        f"vix={s.vix if s.vix is not None else 'n/a'}\n"
        f"vix_term_structure={s.vix_term_structure if s.vix_term_structure is not None else 'n/a'}"
    )


def run_self_check(symbols: list[str], signal_timeframe: str) -> int:
    print("Running self-check...")
    try:
        _load_yfinance()
        print("- yfinance import: OK")
    except Exception as exc:
        print(f"- yfinance import: FAIL ({exc})")
        return 2

    for symbol in symbols:
        try:
            snap = fetch_symbol_snapshot(symbol, price_interval=signal_timeframe)
            if snap is None:
                print(f"- {symbol}: FAIL (no data)")
                return 3
            print(f"- {symbol}: OK (price={snap.price:.2f}, pcr={snap.pcr_volume:.3f}, uoa={snap.uoa_ratio:.3f})")
        except Exception as exc:
            print(f"- {symbol}: FAIL ({exc})")
            return 4

    print("Self-check passed.")
    return 0


def run_loop(symbols: list[str], poll_seconds: int, signal_timeframe: str = "1m", once: bool = False) -> int:
    last_state: dict[str, str] = {}

    while True:
        print(f"\n[{datetime.now().isoformat()}] polling {', '.join(symbols)}")
        for symbol in symbols:
            try:
                snap = fetch_symbol_snapshot(symbol, price_interval=signal_timeframe)
                if not snap:
                    print(f"  - {symbol}: no data")
                    continue

                sig = snapshot_signature(snap)
                payload = format_snapshot(snap)
                print(f"  - {symbol}: signal={snap.pcr_signal}, unusual_volume={snap.unusual_volume}, uoa={snap.uoa_ratio:.3f}")

                if last_state.get(symbol) != sig:
                    print("    -> change detected")
                    print(payload)
                    last_state[symbol] = sig
            except Exception as err:
                print(f"  - {symbol}: error={err}")

        if once:
            return 0
        time.sleep(max(poll_seconds, 30))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Free near-real-time options flow polling monitor")
    parser.add_argument("--symbols", default="SPY,QQQ,IWM", help="Comma-separated symbols")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval in seconds (default 60 for 1min signals)")
    parser.add_argument("--signal-timeframe", default="1m", help="Price timeframe for signal refresh (default: 1m)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--self-check", action="store_true", help="Validate dependencies and fetch one snapshot per symbol")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("No symbols configured.")
        return 1

    if args.self_check:
        return run_self_check(symbols, args.signal_timeframe)

    print("Starting free MVP live monitor...")
    print("Output mode: terminal only (no webhooks/email).")
    return run_loop(symbols=symbols, poll_seconds=args.poll_seconds, signal_timeframe=args.signal_timeframe, once=args.once)


if __name__ == "__main__":
    sys.exit(main())
