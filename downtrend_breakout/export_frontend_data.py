"""
export_frontend_data.py
Export MINIMO para la UI de downtrend_breakout: velas + señales historicas SHORT
con sus criterios (regime, donchian, adx, atr, volumen), para visualizar GRAFICAMENTE
por que entra cada señal.

NO incluye backtest/equity/cartera (alcance intermedio, decidido 2026-11-12): el objetivo
es ENTENDER cada criterio de señal, no re-mostrar el backtest.

Salida: downtrend.latest.json (candles + historicalSignals + currentSignal + config).

Uso:
    python export_frontend_data.py --out-dir <dir>
    python export_frontend_data.py --days 365 --candle-cap 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TIMEFRAME, DAYS, DEFAULT_SYMBOLS, MODE_DEFAULT
from main import _resolve_symbol_params, _evaluate_downtrend_conditions
from breakout_momentum.data_loader import fetch_ohlcv
from strategy import add_features, generate_signals

SCHEMA_VERSION = "downtrend_breakout.frontend.v1"
STRATEGY_NAME = "downtrend_breakout"


def _to_iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC").isoformat() if value.tzinfo else value.tz_localize("UTC").isoformat()
    if isinstance(value, np.datetime64):
        return pd.to_datetime(value, utc=True).isoformat()
    return str(value)


def _sf(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _of(value: Any):
    """Optional float: None si NaN."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _build_conditions(df: pd.DataFrame, i: int, params: dict) -> tuple[list[dict], str | None]:
    """Reusa _evaluate_downtrend_conditions sobre la fila i (no la ultima)."""
    sub = df.iloc[: i + 1]
    checks, blocking = _evaluate_downtrend_conditions(sub, params, MODE_DEFAULT)
    conditions = [{"name": n, "passed": bool(ok), "detail": d} for n, ok, d in checks]
    return conditions, (None if blocking == "none" else blocking)


def _process_symbol(symbol: str, days: int, candle_cap: int, btc_df) -> dict[str, Any]:
    params = _resolve_symbol_params(symbol)
    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=days)
    df = add_features(df, btc_df=btc_df)
    df = generate_signals(df, symbol_params=params, mode=MODE_DEFAULT)
    df = df.reset_index(drop=True)

    capped = df.tail(max(50, int(candle_cap))).copy()

    candles = [
        {
            "timestamp": _to_iso(r.get("timestamp")),
            "open": _sf(r.get("open")), "high": _sf(r.get("high")),
            "low": _sf(r.get("low")), "close": _sf(r.get("close")),
            "volume": _of(r.get("volume")),
        }
        for _, r in capped.iterrows()
    ]

    # Señales historicas SHORT (solo las del rango capado, para no inflar el JSON)
    historical = []
    for i, r in capped.iterrows():
        signal = r.get("signal", "NO TRADE")
        mapped = "SHORT" if signal == "SHORT" else "NO_SIGNAL"
        entry = {
            "timestamp": _to_iso(r.get("timestamp")),
            "signal": mapped,
            "close": _sf(r.get("close")),
            "ret_24h_pct": _sf(r.get("ret_24h")) * 100,
            "ret_6h_pct": _sf(r.get("ret_6h")) * 100,
            "adx14": _of(r.get("adx14")),
            "atr_pct_rank_90d": _of(r.get("atr_pct_rank_90d")),
            "relative_volume_20": _of(r.get("relative_volume_20")),
            "donchian_low": _of(r.get("donchian_low")),
            "regime_bearish": bool(r.get("regime_bearish", False)),
            "probability": _of(r.get("probability")),
            "entry_reason": r.get("entry_reason", "") or None,
        }
        # Solo adjuntamos criterios detallados a las señales SHORT (ahorra tamaño)
        if mapped == "SHORT":
            conds, _ = _build_conditions(df, i, params)
            entry["conditions"] = conds
        historical.append(entry)

    # Señal actual (ultima vela, con criterios siempre)
    last_i = len(df) - 1
    last = df.iloc[last_i]
    conds, blocking = _build_conditions(df, last_i, params)
    is_short = last.get("signal", "NO TRADE") == "SHORT"
    current = {
        "signal": "SHORT" if is_short else "NO_SIGNAL",
        "timestamp": _to_iso(last.get("timestamp")),
        "close": _sf(last.get("close")),
        "ret_24h_pct": _sf(last.get("ret_24h")) * 100,
        "adx14": _of(last.get("adx14")),
        "regime_bearish": bool(last.get("regime_bearish", False)),
        "blocked_by": None if is_short else blocking,
        "conditions": conds,
        "entry_reason": last.get("entry_reason", "") or None,
    }

    return {
        "symbol": symbol,
        "candles": candles,
        "historicalSignals": historical,
        "currentSignal": current,
        "signalParams": {
            "short_ret24_threshold": _sf(params.get("short_ret24_threshold")),
            "adx_min": _sf(params.get("adx_min")),
            "donchian_window": int(_sf(params.get("donchian_window"))),
            "relative_volume_min": _sf(params.get("relative_volume_min")),
            "atr_pct_rank_max": _sf(params.get("atr_pct_rank_max")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export minimo para la UI de downtrend_breakout")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--candle-cap", type=int, default=2000)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "exports" / "downtrend_breakout"))
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS
    print(f"\n{'='*60}\n  DOWNTREND_BREAKOUT FRONTEND EXPORT (minimo)\n{'='*60}")
    print(f"  Symbols: {len(symbols)} | days: {args.days} | cap: {args.candle_cap}")

    try:
        btc_df = fetch_ohlcv("BTC/USDT", timeframe=TIMEFRAME, days=args.days)
    except Exception as e:
        print(f"  (aviso) BTC context no disponible: {e}")
        btc_df = None

    results = []
    for sym in symbols:
        try:
            print(f"Processing {sym}...")
            results.append(_process_symbol(sym, args.days, args.candle_cap, btc_df))
        except Exception as e:
            print(f"  error on {sym}: {e}")

    if not results:
        raise RuntimeError("No data exported.")

    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "strategy": STRATEGY_NAME,
        "source": "python",
        "runId": f"dtb-{uuid.uuid4().hex[:12]}",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "symbolResults": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "downtrend.latest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print(f"\nWrote {path}\nExport complete.")


if __name__ == "__main__":
    main()
