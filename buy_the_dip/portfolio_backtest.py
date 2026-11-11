"""
portfolio_backtest.py
Backtest de CARTERA para Buy-the-Dip: un unico pozo de capital compartido
entre todos los simbolos.

Diferencia vs backtest.py (por simbolo):
- backtest.py simula cada simbolo con capital DEDICADO (como si tuvieras $10k
  por moneda). Infla el retorno y oculta el drawdown real.
- Aqui hay UN solo capital. Cuando varias señales coinciden en el tiempo
  (las cripto caen juntas), compiten por capital y slots globales. Las señales
  que no caben se PIERDEN y se contabilizan. El drawdown es el real de la cuenta.

Reutiliza los costes de futuros de backtest.py (_finalize_trade, liquidacion,
slippage, funding) para ser consistente con la vista por simbolo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL,
    POSITION_PCT,
    MAX_POSITIONS,
    FEE_PCT,
    SLIPPAGE_PCT,
    LEVERAGE,
    MAINTENANCE_MARGIN_RATE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    HOLD_TIMEOUT_HOURS,
)
from strategy import compute_exit_levels
from backtest import _finalize_trade, liquidation_price_long


def run_portfolio_backtest(
    symbol_frames: dict[str, pd.DataFrame],
    symbol_params: dict[str, dict],
    funding_by_symbol: dict[str, pd.Series] | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    position_pct: float = POSITION_PCT,
    max_positions: int = MAX_POSITIONS,
    fee_pct: float = FEE_PCT,
    slippage_pct: float = SLIPPAGE_PCT,
) -> dict:
    """
    Simula los simbolos a la vez sobre un timeline comun con capital compartido.

    Args:
        symbol_frames: {symbol: df con columnas timestamp, high, low, close, signal,
                        dip_size}. Deben compartir la malla de timestamps (1h).
        symbol_params: {symbol: signal_params dict} (usa stop_loss_pct,
                        take_profit_pct, hold_timeout_hours, leverage).
        funding_by_symbol: {symbol: pd.Series funding indexada por timestamp}.
        initial_capital: capital inicial UNICO para toda la cartera.
        position_pct: fraccion del capital por posicion (margen).
        max_positions: maximo de posiciones simultaneas GLOBALES (toda la cartera).
        fee_pct, slippage_pct: costes de futuros.

    Returns:
        dict con equity_curve, equity_timestamps, trades (con columna 'symbol'),
        metrics (de cartera), per_symbol (contribucion por simbolo),
        missed_signals (señales no tomadas por falta de capital/slots).
    """
    funding_by_symbol = funding_by_symbol or {}

    # 1. Timeline comun (union ordenada de timestamps)
    all_ts = sorted(set().union(*[set(df["timestamp"]) for df in symbol_frames.values()]))
    timeline = pd.DatetimeIndex(all_ts)

    # 2. Indexar cada df por timestamp para acceso O(1) por barra
    indexed = {sym: df.set_index("timestamp") for sym, df in symbol_frames.items()}

    # 3. Resolver parametros por simbolo (con defaults)
    def p(sym, key, default):
        return symbol_params.get(sym, {}).get(key, default)

    capital = float(initial_capital)
    positions = []  # cada pos lleva 'symbol'
    trades = []
    equity_curve = []
    equity_ts = []

    missed_signals = 0
    missed_by_symbol: dict[str, int] = {s: 0 for s in symbol_frames}
    taken_by_symbol: dict[str, int] = {s: 0 for s in symbol_frames}

    for ts in timeline:
        # ── 1. Marcar barra actual por simbolo ──
        bars = {}
        for sym, idf in indexed.items():
            if ts in idf.index:
                bars[sym] = idf.loc[ts]

        # ── 2. Revisar salidas de posiciones abiertas ──
        still_open = []
        for pos in positions:
            sym = pos["symbol"]
            bar = bars.get(sym)
            if bar is None:
                still_open.append(pos)
                continue

            pos["bars_held"] += 1
            high, low, close = bar["high"], bar["low"], bar["close"]

            reb = (high - pos["entry_price"]) / pos["entry_price"]
            if reb > pos.get("max_rebound_pct", 0):
                pos["max_rebound_pct"] = reb

            liq = pos.get("liquidation_price", 0.0)
            hold = pos["hold_timeout"]
            liquidated = False

            if liq > 0 and low <= liq:
                exit_price, reason, liquidated = liq, "liquidation", True
            elif low <= pos["stop_loss"]:
                exit_price, reason = pos["stop_loss"], "stop_loss"
            elif high >= pos["take_profit"]:
                exit_price, reason = pos["take_profit"], "take_profit"
            elif pos["bars_held"] >= hold:
                exit_price, reason = close, "timeout"
            else:
                still_open.append(pos)
                continue

            trade, capital_delta = _finalize_trade(
                pos, exit_price, reason, ts,
                fee_pct=fee_pct, slippage_pct=slippage_pct, leverage=pos["leverage"],
                funding_series=funding_by_symbol.get(sym), liquidated=liquidated,
            )
            trade["symbol"] = sym
            capital += capital_delta
            trades.append(trade)

        positions = still_open

        # ── 3. Recolectar señales de ESTA barra (compiten por capital) ──
        # Orden determinista: por tamaño del dip (mas profundo primero = mejor edge).
        candidates = []
        for sym, bar in bars.items():
            if bar.get("signal", "NO TRADE") == "LONG":
                candidates.append((sym, float(bar.get("dip_size", 0.0)), bar))
        candidates.sort(key=lambda x: x[1])  # dip_size mas negativo primero

        for sym, dip, bar in candidates:
            open_slots = max_positions - len(positions)
            if open_slots <= 0:
                missed_signals += 1
                missed_by_symbol[sym] += 1
                continue

            allocated = capital * position_pct
            if allocated <= 1 or capital <= 0:
                missed_signals += 1
                missed_by_symbol[sym] += 1
                continue

            lev = float(p(sym, "leverage", LEVERAGE) or LEVERAGE)
            sl_pct = float(p(sym, "stop_loss_pct", STOP_LOSS_PCT))
            tp_pct = float(p(sym, "take_profit_pct", TAKE_PROFIT_PCT))
            hold = int(p(sym, "hold_timeout_hours", HOLD_TIMEOUT_HOURS))

            entry_fee = allocated * lev * fee_pct
            capital -= allocated
            levels = compute_exit_levels(bar["close"], tp_pct, sl_pct)

            positions.append({
                "symbol": sym,
                "entry_price": bar["close"],
                "entry_time": ts,
                "allocated": allocated - entry_fee,
                "bars_held": 0,
                "stop_loss": levels["stop_loss"],
                "take_profit": levels["take_profit"],
                "liquidation_price": liquidation_price_long(bar["close"], lev),
                "leverage": lev,
                "hold_timeout": hold,
                "dip_size": dip,
                "max_rebound_pct": 0.0,
            })
            taken_by_symbol[sym] += 1

        # ── 4. Equity marcada a mercado ──
        equity = capital
        for pos in positions:
            bar = bars.get(pos["symbol"])
            if bar is not None:
                move = (bar["close"] - pos["entry_price"]) / pos["entry_price"]
                equity += max(0.0, pos["allocated"] * (1.0 + move * pos["leverage"]))
            else:
                equity += pos["allocated"]
        equity_curve.append(equity)
        equity_ts.append(ts)

    # ── Cerrar posiciones abiertas al final ──
    for pos in positions:
        sym = pos["symbol"]
        last_close = indexed[sym]["close"].iloc[-1]
        trade, capital_delta = _finalize_trade(
            pos, last_close, "end_of_data", timeline[-1],
            fee_pct=fee_pct, slippage_pct=slippage_pct, leverage=pos["leverage"],
            funding_series=funding_by_symbol.get(sym), liquidated=False,
        )
        trade["symbol"] = sym
        capital += capital_delta
        trades.append(trade)

    equity = np.array(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    metrics = _portfolio_metrics(trades_df, equity, initial_capital)
    metrics["missed_signals"] = missed_signals

    per_symbol = _per_symbol_contribution(
        trades_df, taken_by_symbol, missed_by_symbol
    )

    return {
        "equity_curve": equity,
        "equity_timestamps": np.array(timeline),
        "trades": trades_df,
        "metrics": metrics,
        "per_symbol": per_symbol,
        "missed_signals": missed_signals,
    }


def _portfolio_metrics(trades_df, equity, initial_capital) -> dict:
    """Metricas de cartera sobre la equity agregada."""
    if len(equity) == 0:
        return {"total_return": 0, "sharpe": 0, "max_drawdown": 0,
                "win_rate": 0, "profit_factor": 0, "total_trades": 0,
                "final_capital": initial_capital, "n_liquidations": 0}

    total_return = (equity[-1] - initial_capital) / initial_capital
    rets = np.diff(equity) / equity[:-1]
    sharpe = 0.0
    if len(rets) > 0 and np.std(rets) > 0:
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(8760)
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min()

    if len(trades_df) > 0:
        pnl = trades_df["pnl_after_fees"]
        win_rate = float((pnl > 0).mean())
        gp = pnl[pnl > 0].sum()
        gl = abs(pnl[pnl < 0].sum())
        pf = float(gp / gl) if gl > 0 else 0.0
        n_liq = int((trades_df["exit_reason"] == "liquidation").sum())
        total_trades = len(trades_df)
    else:
        win_rate = pf = 0.0
        n_liq = total_trades = 0

    return {
        "total_return": round(float(total_return), 4),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 2),
        "total_trades": total_trades,
        "final_capital": round(float(equity[-1]), 2),
        "n_liquidations": n_liq,
    }


def _per_symbol_contribution(trades_df, taken, missed) -> list[dict]:
    """Contribucion de cada simbolo al PnL de la cartera."""
    rows = []
    for sym in taken:
        if len(trades_df) > 0 and "symbol" in trades_df:
            st = trades_df[trades_df["symbol"] == sym]
            pnl_sum = float(st["pnl_after_fees"].sum()) if len(st) else 0.0
            wr = float((st["pnl_after_fees"] > 0).mean()) if len(st) else 0.0
        else:
            pnl_sum, wr = 0.0, 0.0
        rows.append({
            "symbol": sym,
            "trades_taken": taken.get(sym, 0),
            "signals_missed": missed.get(sym, 0),
            "pnl_sum_pct": round(pnl_sum * 100, 2),  # suma de % por trade (sobre margen)
            "win_rate": round(wr * 100, 1),
        })
    rows.sort(key=lambda r: r["pnl_sum_pct"], reverse=True)
    return rows
