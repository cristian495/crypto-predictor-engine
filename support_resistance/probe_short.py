"""
probe_short.py
SPIKE de validación: ¿tiene edge la estrategia SHORT en resistencia con rechazo bajista?

NO es la estrategia final. Es un prototipo minimo para decidir CON DATOS si vale la
pena montar toda la infraestructura (como buy_the_dip) o descartarla. Porta la ESENCIA
de la logica de la UI (detectShortSetups), no el scoring completo:

  1. Pivot highs (ventana) -> zonas de resistencia (clustering por tolerancia, min toques).
  2. Vela que toca una resistencia + mecha de rechazo bajista (upper wick / rango >= umbral).
  3. SHORT al close. Stop sobre la resistencia (+buffer). Target: soporte inferior mas cercano.
     Exige RR minimo. Exit por TP (soporte), SL (resistencia) o timeout.

Incluye costes de futuros (fees, slippage, funding, leverage) para que el PF sea comparable
con buy_the_dip. Short: el PnL es (entry - exit)/entry.

Uso:
    python probe_short.py --symbols BTC/USDT ETH/USDT SOL/USDT --days 1825
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from breakout_momentum.data_loader import fetch_ohlcv, fetch_funding_rates

# Costes de futuros (mismos que buy_the_dip para comparar justo)
FEE_PCT = 0.0004
SLIPPAGE_PCT = 0.0005
LEVERAGE = 3
FUNDING_INTERVAL_HOURS = 8
FUNDING_RATE_FALLBACK = 0.0001


def pivot_highs(df: pd.DataFrame, window: int) -> list[tuple[int, float]]:
    """Indices + precio de los máximos locales (high mayor que sus `window` vecinos)."""
    highs = df["high"].values
    out = []
    for i in range(window, len(df) - window):
        h = highs[i]
        if all(h > highs[i - o] and h > highs[i + o] for o in range(1, window + 1)):
            out.append((i, h))
    return out


def pivot_lows(df: pd.DataFrame, window: int) -> list[tuple[int, float]]:
    lows = df["low"].values
    out = []
    for i in range(window, len(df) - window):
        l = lows[i]
        if all(l < lows[i - o] and l < lows[i + o] for o in range(1, window + 1)):
            out.append((i, l))
    return out


def cluster_zones(pivots: list[tuple[int, float]], tol_pct: float, min_touches: int) -> list[dict]:
    """Agrupa pivots cercanos (dentro de tol_pct) en zonas. Devuelve zonas con min_touches."""
    if not pivots:
        return []
    prices = sorted(p[1] for p in pivots)
    zones = []
    cluster = [prices[0]]
    for price in prices[1:]:
        if abs(price - cluster[-1]) / cluster[-1] * 100 <= tol_pct:
            cluster.append(price)
        else:
            if len(cluster) >= min_touches:
                zones.append({"min": min(cluster), "max": max(cluster),
                              "mid": sum(cluster) / len(cluster), "touches": len(cluster)})
            cluster = [price]
    if len(cluster) >= min_touches:
        zones.append({"min": min(cluster), "max": max(cluster),
                      "mid": sum(cluster) / len(cluster), "touches": len(cluster)})
    return zones


def bearish_rejection(row, threshold: float) -> bool:
    """Vela con mecha superior significativa (rechazo bajista)."""
    upper_wick = max(0.0, row["high"] - max(row["open"], row["close"]))
    body = abs(row["close"] - row["open"])
    full_range = max(0.0, row["high"] - row["low"])
    if full_range <= 0:
        return False
    wick_ratio = upper_wick / full_range
    return wick_ratio >= threshold or upper_wick > body


def funding_cost(funding_series, entry_ts, exit_ts) -> float:
    """Funding acumulado. Para SHORT, funding positivo va A FAVOR (cobras); negativo en contra."""
    if funding_series is not None and len(funding_series) > 0:
        try:
            mask = (funding_series.index >= entry_ts) & (funding_series.index < exit_ts)
            # short cobra funding positivo -> resta del coste (signo invertido vs long)
            return -float(funding_series.loc[mask].sum())
        except Exception:
            pass
    hours = max(0.0, (exit_ts - entry_ts) / pd.Timedelta(hours=1))
    return -(int(hours // FUNDING_INTERVAL_HOURS) * FUNDING_RATE_FALLBACK)


def backtest_short(df: pd.DataFrame, funding_series, *, pivot_window: int, tol_pct: float,
                   min_touches: int, rejection_thr: float, min_rr: float,
                   stop_buffer_pct: float, hold_timeout: int, lookback_zones: int) -> dict:
    """
    Backtest SHORT en resistencia con rechazo. Usa SOLO zonas formadas ANTES de cada vela
    (lookback de `lookback_zones` velas) para evitar lookahead.
    """
    trades = []
    n = len(df)
    in_position = False
    pos = None

    for i in range(lookback_zones, n):
        row = df.iloc[i]

        # ── gestionar salida ──
        if in_position:
            pos["bars"] += 1
            hit_sl = row["high"] >= pos["stop"]
            hit_tp = row["low"] <= pos["target"]
            exit_price = None
            reason = None
            if hit_sl:  # peor caso primero
                exit_price, reason = pos["stop"], "stop"
            elif hit_tp:
                exit_price, reason = pos["target"], "target"
            elif pos["bars"] >= hold_timeout:
                exit_price, reason = row["close"], "timeout"
            if exit_price is not None:
                entry_fill = pos["entry"] * (1 - SLIPPAGE_PCT)   # short: vendes peor (mas bajo)
                exit_fill = exit_price * (1 + SLIPPAGE_PCT) if reason != "stop" else exit_price
                price_pnl = (entry_fill - exit_fill) / entry_fill  # short
                fund = funding_cost(funding_series, pos["ts"], row["timestamp"])
                net = (price_pnl - FEE_PCT - fund) * LEVERAGE
                trades.append({"pnl": net, "reason": reason})
                in_position = False
                pos = None
            continue

        # ── buscar entrada ──
        # Zonas formadas con datos ANTERIORES a i (sin lookahead)
        hist = df.iloc[max(0, i - lookback_zones):i]
        res = cluster_zones(pivot_highs(hist, pivot_window), tol_pct, min_touches)
        sup = cluster_zones(pivot_lows(hist, pivot_window), tol_pct, min_touches)
        if not res or not sup:
            continue

        # ¿toca alguna resistencia por encima y rechaza?
        near = [z for z in res if row["high"] >= z["min"] * (1 - 0.003) and row["close"] <= z["max"] * 1.003]
        if not near:
            continue
        if not bearish_rejection(row, rejection_thr):
            continue

        zone = min(near, key=lambda z: abs(row["high"] - z["mid"]))
        entry = row["close"]
        stop = zone["max"] * (1 + stop_buffer_pct / 100)
        risk = stop - entry
        if risk <= 0:
            continue
        # target: soporte mas cercano por debajo
        below = [z for z in sup if z["mid"] < entry]
        if not below:
            continue
        target = max(below, key=lambda z: z["mid"])["mid"]
        reward = entry - target
        if reward <= 0 or reward / risk < min_rr:
            continue

        in_position = True
        pos = {"entry": entry, "stop": stop, "target": target, "ts": row["timestamp"], "bars": 0}

    if not trades:
        return {"trades": 0, "pf": 0.0, "win_rate": 0.0, "total_pnl_pct": 0.0, "avg_pnl_pct": 0.0}

    pnls = np.array([t["pnl"] for t in trades])
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls < 0].sum())
    return {
        "trades": len(trades),
        "pf": round(float(gp / gl), 2) if gl > 0 else 0.0,
        "win_rate": round(float((pnls > 0).mean()) * 100, 1),
        "total_pnl_pct": round(float(pnls.sum()) * 100, 1),
        "avg_pnl_pct": round(float(pnls.mean()) * 100, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Spike: edge de SHORT en resistencia")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--pivot-window", type=int, default=3)
    parser.add_argument("--tol-pct", type=float, default=0.6)
    parser.add_argument("--min-touches", type=int, default=2)
    parser.add_argument("--rejection-thr", type=float, default=0.35)
    parser.add_argument("--min-rr", type=float, default=1.4)
    parser.add_argument("--stop-buffer-pct", type=float, default=0.15)
    parser.add_argument("--hold-timeout", type=int, default=24)
    parser.add_argument("--lookback-zones", type=int, default=200)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  SPIKE — SHORT en resistencia con rechazo bajista (costes de futuros x3)")
    print(f"{'='*70}")
    print(f"  pivot_window={args.pivot_window} tol={args.tol_pct}% touches>={args.min_touches} "
          f"rej>={args.rejection_thr} RR>={args.min_rr} hold={args.hold_timeout}h")
    print(f"{'='*70}")
    print(f"  {'Symbol':<12} {'Trades':>7} {'WR':>7} {'PF':>7} {'TotalPnL':>10} {'AvgPnL':>9}")
    print(f"  {'-'*58}")

    total_pf = []
    for sym in args.symbols:
        try:
            df = fetch_ohlcv(sym, timeframe="1h", days=args.days)
        except Exception as e:
            print(f"  {sym:<12} ERROR: {e}")
            continue
        fser = None
        try:
            fdf = fetch_funding_rates(sym, days=args.days)
            if len(fdf):
                fser = fdf.set_index("timestamp")["funding_rate"].sort_index()
        except Exception:
            pass
        r = backtest_short(
            df, fser, pivot_window=args.pivot_window, tol_pct=args.tol_pct,
            min_touches=args.min_touches, rejection_thr=args.rejection_thr, min_rr=args.min_rr,
            stop_buffer_pct=args.stop_buffer_pct, hold_timeout=args.hold_timeout,
            lookback_zones=args.lookback_zones,
        )
        print(f"  {sym:<12} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['pf']:>7.2f} "
              f"{r['total_pnl_pct']:>9.1f}% {r['avg_pnl_pct']:>8.3f}%")
        if r["trades"] >= 20:
            total_pf.append(r["pf"])

    print(f"  {'-'*58}")
    if total_pf:
        print(f"  PF medio (simbolos con >=20 trades): {np.mean(total_pf):.2f}")
        print(f"\n  Veredicto: PF>1.2 en la mayoria = hay señal; PF~1 o <1 = sin edge tras costes.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
