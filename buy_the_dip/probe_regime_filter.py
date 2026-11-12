"""
probe_regime_filter.py
SPIKE: ¿mejora la cartera de buy_the_dip si solo operamos cuando el mercado NO
esta en tendencia bajista (BTC por encima de su media movil larga)?

Hipotesis: buy_the_dip es LONG-only; deberia sufrir en bajistas. Filtrar las señales
para operar solo con "regimen alcista/lateral" (BTC > SMA_N) deberia bajar el drawdown
evitando las peores rachas, al coste de perder algunas señales.

Metodo: baja BTC una vez, marca el regimen por timestamp (precio vs SMA). Para cada
simbolo, ANULA las señales LONG que caen en regimen bajista. Compara cartera CON vs SIN.
NO toca config ni codigo de produccion: es un experimento aislado.

Uso:
    python probe_regime_filter.py --sma 200 --days 1825
    python probe_regime_filter.py --sma 100 --regime-tf 1d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import DEFAULT_SYMBOLS, get_symbol_signal_params
from breakout_momentum.data_loader import fetch_ohlcv, fetch_funding_rates
from strategy import add_dip_features, generate_signals
from portfolio_backtest import run_portfolio_backtest


def btc_regime(days: int, sma: int, regime_tf: str) -> pd.Series:
    """
    Serie booleana indexada por timestamp: True = regimen alcista/lateral (BTC > SMA).
    Si regime_tf es '1d', calcula la SMA en diario y la reindexa a 1H (forward fill).
    """
    btc = fetch_ohlcv("BTC/USDT", timeframe=regime_tf, days=days)
    btc = btc.set_index("timestamp").sort_index()
    btc["sma"] = btc["close"].rolling(sma).mean()
    regime = (btc["close"] > btc["sma"]).rename("bullish")
    return regime


def apply_regime_filter(df: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """Anula (NO TRADE) las señales LONG que caen en regimen bajista."""
    d = df.copy()
    # Alinear el regimen a los timestamps de las velas del simbolo (forward fill)
    ts = pd.DatetimeIndex(d["timestamp"])
    aligned = regime.reindex(regime.index.union(ts)).sort_index().ffill().reindex(ts)
    mask_bear = (aligned.values == False)  # noqa: E712
    is_long = d["signal"].values == "LONG"
    block = is_long & mask_bear
    d.loc[block, "signal"] = "NO TRADE"
    d.loc[block, "signal_direction"] = 0
    return d


def prep_symbol(symbol: str, days: int):
    sp = get_symbol_signal_params(symbol)
    df = fetch_ohlcv(symbol, timeframe="1h", days=days)
    df = add_dip_features(df)
    df = generate_signals(
        df,
        dip_threshold=sp["dip_threshold"], use_volume_filter=sp["use_volume_filter"],
        use_rsi_filter=sp["use_rsi_filter"], min_volume_mult=sp["min_volume_mult"],
        rsi_min=sp["rsi_min"], rsi_max=sp["rsi_max"],
        one_entry_per_dip_event=sp["one_entry_per_dip_event"], dip_event_reset_pct=sp["dip_event_reset_pct"],
    )
    fser = None
    try:
        fdf = fetch_funding_rates(symbol, days=days)
        if len(fdf):
            fser = fdf.set_index("timestamp")["funding_rate"].sort_index()
    except Exception:
        pass
    return sp, df, fser


def portfolio(frames, params, funding):
    r = run_portfolio_backtest(frames, params, funding_by_symbol=funding, max_positions=4, position_pct=0.20)
    m = r["metrics"]
    return m


def main():
    parser = argparse.ArgumentParser(description="Spike: filtro de regimen de mercado")
    parser.add_argument("--sma", type=int, default=200, help="Periodo de la SMA de BTC")
    parser.add_argument("--regime-tf", default="1d", help="Timeframe del regimen (1d o 1h)")
    parser.add_argument("--days", type=int, default=1825)
    args = parser.parse_args()

    print(f"\n{'='*66}")
    print(f"  SPIKE — Filtro de regimen (BTC > SMA{args.sma} en {args.regime_tf})")
    print(f"{'='*66}")

    print(f"  Calculando regimen de BTC...")
    regime = btc_regime(args.days, args.sma, args.regime_tf)
    pct_bull = regime.mean() * 100
    print(f"  Mercado alcista/lateral: {pct_bull:.0f}% del tiempo  (bajista {100-pct_bull:.0f}%)")

    frames_raw, params, funding = {}, {}, {}
    frames_filt = {}
    for s in DEFAULT_SYMBOLS:
        sp, df, fser = prep_symbol(s, args.days)
        cols = ["timestamp", "high", "low", "close", "signal", "dip_size"]
        if "signal_direction" not in df.columns:
            df["signal_direction"] = 0
        frames_raw[s] = df[cols].copy()
        frames_filt[s] = apply_regime_filter(df, regime)[cols].copy()
        params[s] = sp
        if fser is not None:
            funding[s] = fser

    m_raw = portfolio(frames_raw, params, funding)
    m_filt = portfolio(frames_filt, params, funding)

    print(f"\n  {'':<16}{'Return':>10}{'MaxDD':>9}{'Sharpe':>8}{'PF':>6}{'Trades':>8}")
    print(f"  {'-'*57}")
    for label, m in [("SIN filtro", m_raw), ("CON filtro", m_filt)]:
        print(f"  {label:<16}{m['total_return']*100:>9.1f}%{m['max_drawdown']*100:>8.1f}%"
              f"{m['sharpe']:>8.2f}{m['profit_factor']:>6.2f}{m['total_trades']:>8}")

    # Veredicto
    dd_better = m_filt["max_drawdown"] > m_raw["max_drawdown"]  # menos negativo = mejor
    sharpe_better = m_filt["sharpe"] > m_raw["sharpe"]
    print(f"\n  {'-'*57}")
    print(f"  Drawdown: {'MEJORA' if dd_better else 'no mejora'} "
          f"({m_raw['max_drawdown']*100:.1f}% -> {m_filt['max_drawdown']*100:.1f}%)")
    print(f"  Sharpe:   {'MEJORA' if sharpe_better else 'no mejora'} "
          f"({m_raw['sharpe']:.2f} -> {m_filt['sharpe']:.2f})")
    trades_lost = m_raw['total_trades'] - m_filt['total_trades']
    print(f"  Trades filtrados: {trades_lost} ({trades_lost/max(1,m_raw['total_trades'])*100:.0f}% menos)")
    print(f"\n  Vale la pena si MEJORA Sharpe y Drawdown sin matar el retorno.")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
