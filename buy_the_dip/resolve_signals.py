"""
resolve_signals.py
Cierra el bucle de validacion (paper trading): toma cada señal LONG registrada en
live_signals.jsonl y determina que paso DE VERDAD en el mercado despues de la entrada.

Para cada señal:
  - Baja las velas 1H desde la vela de entrada hasta entry + hold_timeout_hours.
  - Recorre con la MISMA logica de salida que el backtest (liquidacion -> SL -> TP
    -> timeout, peor caso primero) para ser consistente.
  - Calcula el PnL neto con los mismos costes de futuros (slippage, funding aprox, fees,
    leverage) reutilizando _finalize_trade.
  - Si aun no han pasado suficientes velas, la deja como "open" (no resuelta).

Salidas:
  - live_results.jsonl: una linea por señal resuelta (outcome + pnl).
  - live_summary.json: agregado (WR, PF, PnL) + comparativa vivo-vs-backtest para la UI.

NO ejecuta ordenes. Idempotente: re-resuelve y reescribe results/summary cada corrida.

Uso:
    python resolve_signals.py
    python resolve_signals.py --log-path X --results-path Y --summary-path Z
    python resolve_signals.py --backtest-json <backtest.latest.json>  # para comparar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FEE_PCT, SLIPPAGE_PCT
from breakout_momentum.data_loader import fetch_ohlcv
from backtest import _finalize_trade, liquidation_price_long

TIMEFRAME = "1h"
# Buffer de velas extra tras el timeout para asegurar que la ventana esta disponible.
FETCH_BUFFER_HOURS = 6


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
            if rec.get("signal") == "LONG":
                out.append(rec)
    return out


def _resolve_one(sig: dict, candles: pd.DataFrame) -> dict | None:
    """
    Resuelve una señal contra las velas posteriores a su entrada.
    Devuelve dict de resultado, o {'status':'open'} si aun no hay datos suficientes,
    o None si falta info basica.
    """
    entry_price = sig.get("entry_price")
    tp = sig.get("take_profit")
    sl = sig.get("stop_loss")
    hold = int(sig.get("hold_timeout_hours", 24))
    lev = float(sig.get("leverage", 1.0) or 1.0)
    if entry_price is None or tp is None or sl is None:
        return None

    bar_ts = pd.to_datetime(sig["bar_timestamp"], utc=True)
    # Velas DESPUES de la vela de entrada (la entrada es al close de bar_ts)
    future = candles[candles["timestamp"] > bar_ts].reset_index(drop=True)

    liq = liquidation_price_long(entry_price, lev)

    pos = {
        "entry_price": entry_price,
        "entry_time": bar_ts,
        "allocated": 1.0,          # normalizado: pnl_after_fees sera el % sobre margen
        "bars_held": 0,
        "stop_loss": sl,
        "take_profit": tp,
        "liquidation_price": liq,
        "max_rebound_pct": 0.0,
        "dip_size": sig.get("past_return_pct", 0.0) / 100.0,
    }

    exit_price = None
    exit_reason = None
    exit_ts = None
    liquidated = False

    for i in range(min(len(future), hold)):
        row = future.iloc[i]
        pos["bars_held"] += 1
        high, low, close, ts = row["high"], row["low"], row["close"], row["timestamp"]

        reb = (high - entry_price) / entry_price
        if reb > pos["max_rebound_pct"]:
            pos["max_rebound_pct"] = reb

        if liq > 0 and low <= liq:
            exit_price, exit_reason, liquidated = liq, "liquidation", True
        elif low <= sl:
            exit_price, exit_reason = sl, "stop_loss"
        elif high >= tp:
            exit_price, exit_reason = tp, "take_profit"
        elif pos["bars_held"] >= hold:
            exit_price, exit_reason = close, "timeout"
        else:
            continue
        exit_ts = ts
        break

    # ¿Se resolvio?
    if exit_reason is None:
        # No hubo salida. ¿Es porque faltan velas (sigue abierta) o llego al final?
        if len(future) < hold:
            return {"status": "open", "bars_available": len(future), "hold": hold}
        # Hubo velas suficientes pero ninguna condicion -> cerrar por timeout en la ultima
        last = future.iloc[min(len(future), hold) - 1]
        exit_price, exit_reason, exit_ts = last["close"], "timeout", last["timestamp"]

    trade, _ = _finalize_trade(
        pos, exit_price, exit_reason, exit_ts,
        fee_pct=FEE_PCT, slippage_pct=SLIPPAGE_PCT, leverage=lev,
        funding_series=None, liquidated=liquidated,  # funding aprox via fallback en _finalize
    )

    return {
        "status": "resolved",
        "symbol": sig["symbol"],
        "bar_timestamp": sig["bar_timestamp"],
        "entry_price": entry_price,
        "exit_price": trade["exit_price"],
        "exit_reason": trade["exit_reason"],
        "pnl_pct": round(float(trade["pnl_after_fees"]) * 100, 4),   # % neto sobre margen (con leverage)
        "duration_bars": int(trade["duration_bars"]),
        "leverage": lev,
        "win": bool(trade["pnl_after_fees"] > 0),
    }


def _summarize(results: list[dict], backtest_json: Path | None) -> dict:
    resolved = [r for r in results if r.get("status") == "resolved"]
    n = len(resolved)
    summary = {
        "resolved": n,
        "open": sum(1 for r in results if r.get("status") == "open"),
        "winRatePct": 0.0,
        "profitFactor": 0.0,
        "avgPnlPct": 0.0,
        "totalPnlPct": 0.0,
        "byExitReason": {},
        "bySymbol": {},
        "liveVsBacktest": None,
    }
    if n == 0:
        return summary

    wins = [r for r in resolved if r["win"]]
    pnls = [r["pnl_pct"] for r in resolved]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))

    summary["winRatePct"] = round(len(wins) / n * 100, 1)
    summary["profitFactor"] = round(gp / gl, 2) if gl > 0 else 0.0
    summary["avgPnlPct"] = round(sum(pnls) / n, 3)
    summary["totalPnlPct"] = round(sum(pnls), 2)

    reasons: dict[str, int] = {}
    by_sym: dict[str, dict] = {}
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

    # Comparativa vivo vs backtest (si se paso el backtest.latest.json)
    if backtest_json and backtest_json.exists():
        try:
            bt = json.loads(backtest_json.read_text())
            bt_wr, bt_pf, cnt = 0.0, 0.0, 0
            for sr in bt.get("symbolResults", []):
                m = sr.get("metrics", {})
                bt_wr += m.get("win_rate", 0) * 100
                bt_pf += m.get("profit_factor", 0)
                cnt += 1
            if cnt:
                summary["liveVsBacktest"] = {
                    "backtestWinRatePct": round(bt_wr / cnt, 1),
                    "liveWinRatePct": summary["winRatePct"],
                    "backtestProfitFactor": round(bt_pf / cnt, 2),
                    "liveProfitFactor": summary["profitFactor"],
                }
        except Exception:
            pass

    return summary


def run(log_path: Path, results_path: Path, summary_path: Path,
        backtest_json: Path | None) -> dict:
    signals = _load_signals(log_path)
    print(f"  Señales LONG registradas: {len(signals)}")
    if not signals:
        print("  (nada que resolver)")

    # Agrupar por simbolo para bajar velas una vez por simbolo
    by_symbol: dict[str, list[dict]] = {}
    for s in signals:
        by_symbol.setdefault(s["symbol"], []).append(s)

    results: list[dict] = []
    for symbol, sigs in by_symbol.items():
        max_hold = max(int(s.get("hold_timeout_hours", 24)) for s in sigs)
        earliest = min(pd.to_datetime(s["bar_timestamp"], utc=True) for s in sigs)
        days_needed = (pd.Timestamp.now(tz="UTC") - earliest).days + 2
        try:
            candles = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=max(2, days_needed))
        except Exception as e:
            print(f"  {symbol:<10} ERROR bajando velas: {e}")
            continue

        for sig in sigs:
            res = _resolve_one(sig, candles)
            if res is None:
                continue
            results.append(res)
            if res["status"] == "resolved":
                print(f"  {symbol:<10} {sig['bar_timestamp'][:16]} -> "
                      f"{res['exit_reason']:<11} pnl={res['pnl_pct']:+.2f}%")
            else:
                print(f"  {symbol:<10} {sig['bar_timestamp'][:16]} -> open "
                      f"({res['bars_available']}/{res['hold']} velas)")

    # Escribir results (solo resueltas) y summary
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            if r.get("status") == "resolved":
                f.write(json.dumps(r, ensure_ascii=True) + "\n")

    summary = _summarize(results, backtest_json)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"\n  Resueltas: {summary['resolved']}  |  abiertas: {summary['open']}")
    if summary["resolved"]:
        print(f"  WR vivo: {summary['winRatePct']}%  PF: {summary['profitFactor']}  "
              f"PnL total: {summary['totalPnlPct']:+.2f}%")
        lvb = summary.get("liveVsBacktest")
        if lvb:
            print(f"  vs backtest: WR {lvb['liveWinRatePct']}% (bt {lvb['backtestWinRatePct']}%)  "
                  f"PF {lvb['liveProfitFactor']} (bt {lvb['backtestProfitFactor']})")
    print(f"  Wrote {results_path}")
    print(f"  Wrote {summary_path}")
    return summary


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "exports" / "buy_the_dip"
    parser = argparse.ArgumentParser(description="Resolve live paper-trading signals")
    parser.add_argument("--log-path", default=str(base / "live_signals.jsonl"))
    parser.add_argument("--results-path", default=str(base / "live_results.jsonl"))
    parser.add_argument("--summary-path", default=str(base / "live_summary.json"))
    parser.add_argument("--backtest-json", default=str(base / "backtest.latest.json"),
                        help="backtest.latest.json para comparar vivo-vs-backtest")
    args = parser.parse_args()

    bt = Path(args.backtest_json) if args.backtest_json else None
    run(
        log_path=Path(args.log_path),
        results_path=Path(args.results_path),
        summary_path=Path(args.summary_path),
        backtest_json=bt,
    )
