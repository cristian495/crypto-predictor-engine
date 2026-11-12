"""
probe_regime_wf.py
Walk-forward del FILTRO DE REGIMEN: ¿la mejora se sostiene en el tiempo o solo en
tramos concretos (suerte / sobreajuste)?

Metodo: fija SMA=110 (medio de la meseta hallada en probe_regime_filter). Parte la
historia en ventanas temporales disjuntas. En CADA ventana corre la cartera CON y SIN
filtro. Si el filtro mejora el drawdown (y no hunde el Sharpe) en la MAYORIA de ventanas,
es robusto. Si solo mejora en 1-2 ventanas que dominan el promedio, es suerte.

Un solo parametro fijo (SMA) => no hay "optimizacion por ventana" que validar; lo que se
valida es la CONSISTENCIA temporal del efecto. Es el OOS honesto para este caso.

Uso:
    python probe_regime_wf.py --sma 110 --window-months 6 --days 1825
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import DEFAULT_SYMBOLS, get_symbol_signal_params
from breakout_momentum.data_loader import fetch_ohlcv, fetch_funding_rates
from strategy import add_dip_features, generate_signals
from portfolio_backtest import run_portfolio_backtest
from probe_regime_filter import btc_regime, apply_regime_filter


def prep_symbol(symbol, days):
    sp = get_symbol_signal_params(symbol)
    df = fetch_ohlcv(symbol, timeframe="1h", days=days)
    df = add_dip_features(df)
    df = generate_signals(
        df, dip_threshold=sp["dip_threshold"], use_volume_filter=sp["use_volume_filter"],
        use_rsi_filter=sp["use_rsi_filter"], min_volume_mult=sp["min_volume_mult"],
        rsi_min=sp["rsi_min"], rsi_max=sp["rsi_max"],
        one_entry_per_dip_event=sp["one_entry_per_dip_event"], dip_event_reset_pct=sp["dip_event_reset_pct"],
    )
    if "signal_direction" not in df.columns:
        df["signal_direction"] = 0
    fser = None
    try:
        fdf = fetch_funding_rates(symbol, days=days)
        if len(fdf):
            fser = fdf.set_index("timestamp")["funding_rate"].sort_index()
    except Exception:
        pass
    return sp, df, fser


def run_window(frames, params, funding):
    r = run_portfolio_backtest(frames, params, funding_by_symbol=funding, max_positions=4, position_pct=0.20)
    return r["metrics"]


def slice_frames(frames_full, start, end):
    """Recorta cada df al rango [start, end)."""
    out = {}
    for s, df in frames_full.items():
        mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
        out[s] = df.loc[mask].reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Walk-forward del filtro de regimen")
    parser.add_argument("--sma", type=int, default=110)
    parser.add_argument("--regime-tf", default="1d")
    parser.add_argument("--window-months", type=int, default=6)
    parser.add_argument("--days", type=int, default=1825)
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD DEL FILTRO DE REGIMEN (BTC > SMA{args.sma} en {args.regime_tf})")
    print(f"  Ventanas de {args.window_months} meses | {args.days} dias")
    print(f"{'='*72}")

    regime = btc_regime(args.days, args.sma, args.regime_tf)

    # Preparar todos los simbolos una vez (raw y filtrado), con rango temporal comun
    cols = ["timestamp", "high", "low", "close", "signal", "dip_size"]
    raw_full, filt_full, params, funding = {}, {}, {}, {}
    tmin, tmax = None, None
    for s in DEFAULT_SYMBOLS:
        sp, df, fser = prep_symbol(s, args.days)
        raw_full[s] = df[cols].copy()
        filt_full[s] = apply_regime_filter(df, regime)[cols].copy()
        params[s] = sp
        if fser is not None:
            funding[s] = fser
        lo, hi = df["timestamp"].min(), df["timestamp"].max()
        tmin = lo if tmin is None else max(tmin, lo)   # ventana comun: el mas tardio inicio
        tmax = hi if tmax is None else min(tmax, hi)

    # Ventanas temporales disjuntas
    step = pd.DateOffset(months=args.window_months)
    windows = []
    start = tmin
    while start < tmax:
        end = start + step
        if end > tmax:
            break
        windows.append((start, end))
        start = end

    print(f"\n  {'Ventana':<22}{'DD sin':>8}{'DD con':>8}{'Shr sin':>8}{'Shr con':>8}{'Trades':>12}{'DD mejor':>9}")
    print(f"  {'-'*74}")

    dd_better_count = 0
    sharpe_better_count = 0
    valid = 0
    for (ws, we) in windows:
        raw_w = slice_frames(raw_full, ws, we)
        filt_w = slice_frames(filt_full, ws, we)
        # saltar ventanas sin suficientes datos
        if min(len(d) for d in raw_w.values()) < 200:
            continue
        m_raw = run_window(raw_w, params, funding)
        m_filt = run_window(filt_w, params, funding)
        if m_raw["total_trades"] < 10:
            continue
        valid += 1
        dd_better = m_filt["max_drawdown"] > m_raw["max_drawdown"]  # menos negativo
        shr_better = m_filt["sharpe"] > m_raw["sharpe"]
        dd_better_count += 1 if dd_better else 0
        sharpe_better_count += 1 if shr_better else 0
        label = f"{ws.date()}->{we.date()}"
        trades = f"{m_raw['total_trades']}/{m_filt['total_trades']}"
        mark = "SI" if dd_better else "no"
        print(f"  {label:<22}{m_raw['max_drawdown']*100:>7.1f}%{m_filt['max_drawdown']*100:>7.1f}%"
              f"{m_raw['sharpe']:>8.2f}{m_filt['sharpe']:>8.2f}{trades:>12}{mark:>9}")

    print(f"  {'-'*74}")
    if valid:
        print(f"\n  Ventanas donde el filtro MEJORA el drawdown: {dd_better_count}/{valid} "
              f"({dd_better_count/valid*100:.0f}%)")
        print(f"  Ventanas donde el filtro MEJORA el Sharpe:   {sharpe_better_count}/{valid} "
              f"({sharpe_better_count/valid*100:.0f}%)")
        print(f"\n  Robusto si mejora en la MAYORIA de ventanas (>60%), no solo en 1-2.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
