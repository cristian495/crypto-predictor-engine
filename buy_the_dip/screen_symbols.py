"""
screen_symbols.py
Screening de pares candidatos para la cesta de Buy-the-Dip.

METODO (el que validó la cesta actual, documentado 2026-08-10):
  1. Candidatos: top-N pares USDT spot por volumen en Binance (excluye los ya en cesta).
  2. Parametros GLOBALES (sin tuning por moneda) — a proposito: tunear por moneda en el
     screening es curve-fitting. El tuning fino viene DESPUES, solo para los que pasen.
  3. Partir la historia en DOS mitades temporales disjuntas. Exigir que AMBAS sean
     rentables (PF > 1). Un par que solo gana en una mitad no tiene edge estable.
  4. Rankear por el PEOR PF de las dos mitades (criterio robusto, no por el retorno total).
  5. Añadir el walk-forward stability score como segunda señal de robustez.

Un par que pasa aqui NO entra automatico a la cesta: hay que correr DESPUES el backtest
de CARTERA (portfolio_backtest.py) para ver si APORTA con capital compartido. Un par bueno
en solitario puede restar en cartera (fue el caso de INJ).

Uso:
    python screen_symbols.py                      # top 40 por volumen, 5 años
    python screen_symbols.py --top 60 --days 1825
    python screen_symbols.py --symbols SOL/USDT AVAX/USDT LINK/USDT   # lista manual
    python screen_symbols.py --min-history-days 900 --min-trades 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ccxt
import numpy as np
import pandas as pd

from config import (
    DAYS,
    TIMEFRAME,
    DEFAULT_SYMBOLS,
    DEFAULT_DIP_THRESHOLD,
    USE_VOLUME_FILTER,
    USE_RSI_FILTER,
    MIN_VOLUME_MULT,
    RSI_MIN,
    RSI_MAX,
    TAKE_PROFIT_PCT,
    STOP_LOSS_PCT,
    HOLD_TIMEOUT_HOURS,
    MAX_POSITIONS,
    LEVERAGE,
    INITIAL_CAPITAL,
    FEE_PCT,
    SLIPPAGE_PCT,
)
from breakout_momentum.data_loader import fetch_ohlcv, fetch_funding_rates
from strategy import add_dip_features, generate_signals
from backtest import run_backtest
from walk_forward import walk_forward_validation, analyze_results


def top_usdt_symbols_by_volume(top_n: int, exclude: set[str]) -> list[str]:
    """Top-N pares .../USDT spot por volumen quote de 24h en Binance (ccxt)."""
    exchange = ccxt.binance({"enableRateLimit": True})
    tickers = exchange.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        # Solo spot USDT normalizado "BASE/USDT", con volumen quote disponible
        if not sym.endswith("/USDT"):
            continue
        qv = t.get("quoteVolume")
        if qv is None:
            continue
        rows.append((sym, float(qv)))
    rows.sort(key=lambda x: x[1], reverse=True)

    out = []
    for sym, _ in rows:
        if sym in exclude:
            continue
        out.append(sym)
        if len(out) >= top_n:
            break
    return out


def _half_backtest(df: pd.DataFrame, funding_series) -> dict:
    """Backtest de una mitad con parametros GLOBALES + costes de futuros."""
    d = generate_signals(
        df.copy(),
        dip_threshold=DEFAULT_DIP_THRESHOLD,
        use_volume_filter=USE_VOLUME_FILTER,
        use_rsi_filter=USE_RSI_FILTER,
        min_volume_mult=MIN_VOLUME_MULT,
        rsi_min=RSI_MIN,
        rsi_max=RSI_MAX,
    )
    r = run_backtest(
        d,
        initial_capital=INITIAL_CAPITAL,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT,
        hold_timeout=HOLD_TIMEOUT_HOURS,
        max_positions=MAX_POSITIONS,
        fee_pct=FEE_PCT,
        slippage_pct=SLIPPAGE_PCT,
        leverage=LEVERAGE,
        funding_series=funding_series,
    )
    return r["metrics"]


def screen_symbol(symbol: str, days: int, min_history_days: int, min_trades: int) -> dict | None:
    """
    Evalua un simbolo: 2 mitades temporales + walk-forward. Devuelve dict de resultado
    o None si no tiene historia suficiente.
    """
    try:
        df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=days)
    except Exception as e:
        print(f"  {symbol:<12} ERROR ohlcv: {e}")
        return None

    df = add_dip_features(df)
    n = len(df)
    history_days = n / 24.0
    if history_days < min_history_days:
        print(f"  {symbol:<12} SKIP (solo {history_days:.0f}d de historia, min {min_history_days})")
        return None

    # Funding real (para que los PF sean netos de futuros, como la cesta)
    funding_series = None
    try:
        fdf = fetch_funding_rates(symbol, days=days)
        if len(fdf):
            funding_series = fdf.set_index("timestamp")["funding_rate"].sort_index()
    except Exception:
        pass

    # Dos mitades temporales disjuntas
    mid = n // 2
    h1, h2 = df.iloc[:mid].copy(), df.iloc[mid:].copy()
    m1 = _half_backtest(h1, funding_series)
    m2 = _half_backtest(h2, funding_series)

    pf1, pf2 = m1["profit_factor"], m2["profit_factor"]
    tr1, tr2 = m1["total_trades"], m2["total_trades"]
    worst_pf = min(pf1, pf2)
    both_profitable = (pf1 > 1.0 and pf2 > 1.0)
    enough_trades = (tr1 >= min_trades and tr2 >= min_trades)

    # Walk-forward stability (señal secundaria de robustez)
    try:
        wf_rows = walk_forward_validation(symbol, total_days=days)
        _, stability = analyze_results(wf_rows, symbol) if wf_rows else (None, 0.0)
    except Exception:
        stability = 0.0

    passed = both_profitable and enough_trades

    return {
        "symbol": symbol,
        "pf_h1": round(pf1, 2),
        "pf_h2": round(pf2, 2),
        "worst_pf": round(worst_pf, 2),
        "trades_h1": tr1,
        "trades_h2": tr2,
        "ret_full_pct": round((m1["total_return"] + m2["total_return"]) * 100, 1),
        "stability": round(float(stability), 1),
        "history_days": round(history_days),
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Screening de pares para la cesta de Buy-the-Dip")
    parser.add_argument("--top", type=int, default=40, help="Top-N pares por volumen a evaluar")
    parser.add_argument("--days", type=int, default=DAYS, help=f"Dias de historia (default {DAYS})")
    parser.add_argument("--symbols", nargs="+", default=None, help="Lista manual de pares (ignora --top)")
    parser.add_argument("--min-history-days", type=int, default=730, help="Minimo de historia exigido")
    parser.add_argument("--min-trades", type=int, default=20, help="Minimo de trades por mitad")
    args = parser.parse_args()

    in_basket = set(DEFAULT_SYMBOLS)
    if args.symbols:
        candidates = [s.upper() for s in args.symbols]
    else:
        print(f"Obteniendo top {args.top} pares USDT por volumen (excluye los {len(in_basket)} ya en cesta)...")
        candidates = top_usdt_symbols_by_volume(args.top, exclude=in_basket)

    print(f"\n{'='*78}")
    print(f"  SCREENING DE PARES — Buy-the-Dip (params GLOBALES, costes de futuros)")
    print(f"{'='*78}")
    print(f"  Candidatos: {len(candidates)}  |  dias: {args.days}  |  min historia: {args.min_history_days}d")
    print(f"  Params: th={DEFAULT_DIP_THRESHOLD*100:.0f}% TP={TAKE_PROFIT_PCT*100:.0f}% "
          f"SL={STOP_LOSS_PCT*100:.0f}% hold={HOLD_TIMEOUT_HOURS}h x{LEVERAGE}")
    print(f"{'='*78}\n")

    results = []
    for i, sym in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {sym}")
        r = screen_symbol(sym, args.days, args.min_history_days, args.min_trades)
        if r:
            results.append(r)

    if not results:
        print("\nNingun candidato con historia suficiente.")
        return

    # Rankear: primero los que pasan, luego por peor_PF (criterio robusto)
    results.sort(key=lambda r: (r["passed"], r["worst_pf"], r["stability"]), reverse=True)

    print(f"\n{'='*78}")
    print(f"  RESULTADOS (ordenados por robustez: pasa 2 mitades + peor PF)")
    print(f"{'='*78}")
    print(f"  {'Symbol':<12} {'PF_h1':>6} {'PF_h2':>6} {'worstPF':>8} {'Trades':>12} "
          f"{'RetFull':>9} {'Stab':>6} {'PASA':>6}")
    print(f"  {'-'*74}")
    for r in results:
        mark = "SI" if r["passed"] else "no"
        trades = f"{r['trades_h1']}/{r['trades_h2']}"
        print(f"  {r['symbol']:<12} {r['pf_h1']:>6.2f} {r['pf_h2']:>6.2f} {r['worst_pf']:>8.2f} "
              f"{trades:>12} {r['ret_full_pct']:>8.1f}% {r['stability']:>6.1f} {mark:>6}")

    passed = [r for r in results if r["passed"]]
    print(f"\n  {len(passed)}/{len(results)} pasaron (ambas mitades rentables + trades suficientes).")
    if passed:
        print(f"\n  CANDIDATOS A LA CESTA (verificar DESPUES en backtest de cartera):")
        print(f"    {', '.join(r['symbol'] for r in passed)}")
        print(f"\n  OJO: pasar aqui NO garantiza aportar en cartera. Corre portfolio_backtest.py")
        print(f"       con la cesta actual + cada candidato antes de añadirlo a DEFAULT_SYMBOLS.")
    print(f"\n{'='*78}\n")


if __name__ == "__main__":
    main()
