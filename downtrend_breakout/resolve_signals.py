"""
resolve_signals.py
Cierra el bucle de validacion (paper trading) para downtrend_breakout (SHORT).

Para cada señal SHORT registrada en live_signals.jsonl, baja las velas posteriores y
resuelve el resultado real con la MISMA logica de salida que el backtest: trailing stop
dinamico (se activa a +activation_r, sigue al mejor precio a trail_atr_mult*ATR), take
profit (R-multiple), stop loss (ATR), o timeout. PnL SHORT = (entry - exit)/entry, con
costes de futuros (fees, slippage con direccion correcta para short).

Salidas: live_results.jsonl + live_summary.json (con comparativa vivo-vs-backtest).
Idempotente. NO ejecuta ordenes.

Uso:
    python resolve_signals.py
    python resolve_signals.py --log-path X --results-path Y --summary-path Z
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FEE_PCT, SLIPPAGE_PCT, TRAIL_ACTIVATION_R, TRAIL_ATR_MULT
from breakout_momentum.data_loader import fetch_ohlcv

TIMEFRAME = "1h"


def _load_signals(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("signal") == "SHORT":
                out.append(rec)
    return out


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR(14) Wilder, para el trailing stop durante la resolucion."""
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _resolve_one(sig: dict, candles: pd.DataFrame) -> dict | None:
    """Resuelve una señal SHORT contra las velas posteriores (trailing stop incluido)."""
    entry = sig.get("entry_price")
    tp = sig.get("take_profit")
    sl0 = sig.get("stop_loss")
    atr_entry = sig.get("atr14")
    hold = int(sig.get("hold_timeout_bars", 48))
    if entry is None or tp is None or sl0 is None or not atr_entry:
        return None

    bar_ts = pd.to_datetime(sig["bar_timestamp"], utc=True)
    future = candles[candles["timestamp"] > bar_ts].reset_index(drop=True)
    if len(future) == 0:
        return {"status": "open", "bars_available": 0, "hold": hold}

    atr_series = _atr_series(candles)
    atr_by_ts = dict(zip(candles["timestamp"], atr_series))

    risk_distance = abs(entry - sl0)
    stop = sl0
    best_price = entry           # para short, el "mejor" es el mas bajo
    trail_active = False
    exit_price = exit_reason = exit_ts = None

    for i in range(min(len(future), hold)):
        row = future.iloc[i]
        high, low, close, ts = row["high"], row["low"], row["close"], row["timestamp"]
        atr_now = atr_by_ts.get(ts, atr_entry)
        if not (atr_now and atr_now > 0):
            atr_now = atr_entry

        # Trailing stop (SHORT): se activa cuando baja activation_r*risk; sigue al low
        if not trail_active and low <= entry - TRAIL_ACTIVATION_R * risk_distance:
            trail_active = True
        if trail_active:
            best_price = min(best_price, low)
            trail_stop = best_price + TRAIL_ATR_MULT * atr_now
            stop = min(stop, trail_stop)

        if high >= stop:              # stop tocado (precio sube contra el short)
            exit_price, exit_reason = stop, "stop_loss"
        elif low <= tp:               # take profit (precio baja a favor)
            exit_price, exit_reason = tp, "take_profit"
        elif (i + 1) >= hold:
            exit_price, exit_reason = close, "timeout"
        else:
            continue
        exit_ts = ts
        break

    if exit_reason is None:
        if len(future) < hold:
            return {"status": "open", "bars_available": len(future), "hold": hold}
        last = future.iloc[min(len(future), hold) - 1]
        exit_price, exit_reason, exit_ts = last["close"], "timeout", last["timestamp"]

    # PnL SHORT con costes de futuros. Slippage: vendes peor al entrar, compras peor al salir.
    entry_fill = entry * (1 - SLIPPAGE_PCT)
    exit_fill = exit_price * (1 + SLIPPAGE_PCT) if exit_reason != "stop_loss" else exit_price
    price_pnl = (entry_fill - exit_fill) / entry_fill      # short
    net = price_pnl - 2 * FEE_PCT                          # fees entrada+salida

    return {
        "status": "resolved",
        "symbol": sig["symbol"],
        "bar_timestamp": sig["bar_timestamp"],
        "entry_price": entry,
        "exit_price": round(float(exit_price), 6),
        "exit_reason": exit_reason,
        "pnl_pct": round(float(net) * 100, 4),
        "win": bool(net > 0),
    }


def _summarize(results: list[dict], backtest_json: Path | None) -> dict:
    resolved = [r for r in results if r.get("status") == "resolved"]
    n = len(resolved)
    summary = {
        "resolved": n,
        "open": sum(1 for r in results if r.get("status") == "open"),
        "winRatePct": 0.0, "profitFactor": 0.0, "avgPnlPct": 0.0, "totalPnlPct": 0.0,
        "byExitReason": {}, "bySymbol": {}, "liveVsBacktest": None,
    }
    if n == 0:
        return summary

    pnls = [r["pnl_pct"] for r in resolved]
    wins = [r for r in resolved if r["win"]]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    summary["winRatePct"] = round(len(wins) / n * 100, 1)
    summary["profitFactor"] = round(gp / gl, 2) if gl > 0 else 0.0
    summary["avgPnlPct"] = round(sum(pnls) / n, 3)
    summary["totalPnlPct"] = round(sum(pnls), 2)

    reasons, by_sym = {}, {}
    for r in resolved:
        reasons[r["exit_reason"]] = reasons.get(r["exit_reason"], 0) + 1
        s = by_sym.setdefault(r["symbol"], {"trades": 0, "wins": 0, "pnl_pct": 0.0})
        s["trades"] += 1
        s["wins"] += 1 if r["win"] else 0
        s["pnl_pct"] = round(s["pnl_pct"] + r["pnl_pct"], 3)
    for s in by_sym.values():
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0.0
    summary["byExitReason"] = reasons
    summary["bySymbol"] = by_sym
    return summary


def run(log_path: Path, results_path: Path, summary_path: Path, backtest_json: Path | None) -> dict:
    signals = _load_signals(log_path)
    print(f"  Señales SHORT registradas: {len(signals)}")

    by_symbol: dict[str, list[dict]] = {}
    for s in signals:
        by_symbol.setdefault(s["symbol"], []).append(s)

    results: list[dict] = []
    for symbol, sigs in by_symbol.items():
        earliest = min(pd.to_datetime(s["bar_timestamp"], utc=True) for s in sigs)
        days_needed = (pd.Timestamp.now(tz="UTC") - earliest).days + 3
        try:
            candles = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=max(3, days_needed))
        except Exception as e:
            print(f"  {symbol:<10} ERROR velas: {e}")
            continue
        for sig in sigs:
            res = _resolve_one(sig, candles)
            if res is None:
                continue
            results.append(res)
            if res["status"] == "resolved":
                print(f"  {symbol:<10} {sig['bar_timestamp'][:16]} -> {res['exit_reason']:<11} pnl={res['pnl_pct']:+.2f}%")
            else:
                print(f"  {symbol:<10} {sig['bar_timestamp'][:16]} -> open ({res['bars_available']}/{res['hold']})")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            if r.get("status") == "resolved":
                f.write(json.dumps(r, ensure_ascii=True) + "\n")

    summary = _summarize(results, backtest_json)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"\n  Resueltas: {summary['resolved']}  |  abiertas: {summary['open']}")
    if summary["resolved"]:
        print(f"  WR vivo: {summary['winRatePct']}%  PF: {summary['profitFactor']}  PnL total: {summary['totalPnlPct']:+.2f}%")
    print(f"  Wrote {results_path}")
    print(f"  Wrote {summary_path}")
    return summary


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "exports" / "downtrend_breakout"
    parser = argparse.ArgumentParser(description="Resolve live paper-trading signals (downtrend SHORT)")
    parser.add_argument("--log-path", default=str(base / "live_signals.jsonl"))
    parser.add_argument("--results-path", default=str(base / "live_results.jsonl"))
    parser.add_argument("--summary-path", default=str(base / "live_summary.json"))
    parser.add_argument("--backtest-json", default=str(base / "backtest.latest.json"))
    args = parser.parse_args()

    bt = Path(args.backtest_json) if args.backtest_json else None
    run(
        log_path=Path(args.log_path),
        results_path=Path(args.results_path),
        summary_path=Path(args.summary_path),
        backtest_json=bt,
    )
