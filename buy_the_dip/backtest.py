"""
backtest.py
Simple backtest engine for Buy-the-Dip strategy.

Uses fixed percentage SL/TP (no ATR complexity).
Simple and transparent - easy to validate.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "..")

from config import (
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    HOLD_TIMEOUT_HOURS,
    POSITION_PCT,
    MAX_POSITIONS,
    FEE_PCT,
    INITIAL_CAPITAL,
    FAIL_FAST_BARS,
    FAIL_FAST_MIN_REBOUND_PCT,
    SLIPPAGE_PCT,
    USE_FUNDING,
    FUNDING_INTERVAL_HOURS,
    FUNDING_RATE_FALLBACK,
    LEVERAGE,
    MAINTENANCE_MARGIN_RATE,
)
from strategy import generate_signals, add_dip_features, compute_exit_levels


def liquidation_price_long(entry_price: float, leverage: float,
                           maintenance_margin_rate: float = MAINTENANCE_MARGIN_RATE) -> float:
    """
    Precio de liquidacion aproximado para un LONG en futuros con margen aislado.

    Con apalancamiento L, el margen inicial cubre una caida de ~1/L del precio.
    La liquidacion ocurre un poco antes por el margen de mantenimiento (mmr):
        liq = entry * (1 - 1/L + mmr)

    Con L<=1 no hay liquidacion relevante -> se devuelve 0 (nunca se toca).
    """
    if leverage is None or leverage <= 1:
        return 0.0
    return entry_price * (1.0 - (1.0 / leverage) + maintenance_margin_rate)


def _funding_cost_pct(funding_series, entry_ts, exit_ts,
                      interval_hours: int = FUNDING_INTERVAL_HOURS,
                      fallback_rate: float = FUNDING_RATE_FALLBACK) -> float:
    """
    Coste de funding acumulado (como fraccion del notional) para un LONG abierto
    entre entry_ts y exit_ts. Positivo = el largo PAGA (resta del PnL).

    Usa los funding rates reales que caen dentro del intervalo [entry_ts, exit_ts).
    Si no hay serie, estima con fallback_rate por cada periodo de interval_hours.
    """
    if funding_series is not None and len(funding_series) > 0:
        try:
            mask = (funding_series.index >= entry_ts) & (funding_series.index < exit_ts)
            return float(funding_series.loc[mask].sum())
        except Exception:
            pass

    # Fallback: estimar nº de periodos de funding por duracion
    try:
        hours = (exit_ts - entry_ts) / pd.Timedelta(hours=1)
    except Exception:
        hours = 0.0
    n_periods = int(max(0.0, hours) // interval_hours)
    return n_periods * fallback_rate


def compute_metrics(trades_df: pd.DataFrame, equity_curve: np.ndarray,
                    initial_capital: float) -> dict:
    """Compute performance metrics."""
    if len(trades_df) == 0:
        return {
            "total_return": 0, "sharpe": 0, "max_drawdown": 0,
            "win_rate": 0, "profit_factor": 0, "total_trades": 0,
            "expectancy": 0, "final_capital": initial_capital,
            "gross_return": 0, "funding_paid_pct": 0, "slippage_paid_pct": 0,
            "n_liquidations": 0, "avg_win": 0, "avg_loss": 0,
        }

    # Returns
    total_return = (equity_curve[-1] - initial_capital) / initial_capital

    # Sharpe
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = 0
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(8760)

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = dd.min()

    # Win rate
    pnl_col = "pnl_after_fees"
    wins = trades_df[trades_df[pnl_col] > 0]
    win_rate = len(wins) / len(trades_df)

    # Profit factor
    gross_profit = trades_df.loc[trades_df[pnl_col] > 0, pnl_col].sum()
    gross_loss = abs(trades_df.loc[trades_df[pnl_col] < 0, pnl_col].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Expectancy
    avg_win = wins[pnl_col].mean() if len(wins) > 0 else 0
    losses = trades_df[trades_df[pnl_col] <= 0]
    avg_loss = abs(losses[pnl_col].mean()) if len(losses) > 0 else 0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    # ── Futures realism breakdown ──
    # gross = PnL de precio sin costes; net = tras fees+slippage+funding (ya en pnl_col).
    gross_pct_sum = float(trades_df["pnl_pct"].sum()) if "pnl_pct" in trades_df else 0.0
    net_pct_sum = float(trades_df[pnl_col].sum())
    funding_paid_pct = float(trades_df["funding_pct"].sum()) if "funding_pct" in trades_df else 0.0
    slippage_paid_pct = float(trades_df["slippage_pct"].sum()) if "slippage_pct" in trades_df else 0.0
    n_liquidations = int((trades_df["exit_reason"] == "liquidation").sum()) if "exit_reason" in trades_df else 0

    return {
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2),
        "total_trades": len(trades_df),
        "expectancy": round(expectancy, 6),
        "final_capital": round(equity_curve[-1], 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(-avg_loss, 4),
        # Desglose de costes de futuros (suma de % por trade, sobre el notional)
        "gross_return": round(gross_pct_sum, 4),
        "net_return_sum": round(net_pct_sum, 4),
        "funding_paid_pct": round(funding_paid_pct, 4),
        "slippage_paid_pct": round(slippage_paid_pct, 4),
        "n_liquidations": n_liquidations,
    }


def _finalize_trade(pos, exit_price, exit_reason, exit_ts,
                    fee_pct, slippage_pct, leverage,
                    funding_series, liquidated=False):
    """
    Cierra una posicion LONG aplicando costes reales de futuros y devuelve
    (trade_dict, capital_delta) donde capital_delta es lo que vuelve al capital
    (margen + PnL neto sobre el margen).

    Modelo:
    - entry_fill = entry_price * (1 + slippage)   (compras peor)
    - exit_fill  = exit_price  * (1 - slippage)   (vendes peor)  [salvo liquidacion]
    - price_pnl_pct = (exit_fill - entry_fill) / entry_fill   (sobre el notional)
    - costes (fees x2, funding) se restan como % del notional
    - el PnL sobre el MARGEN se amplifica por leverage
    - liquidacion => perdida del 100% del margen
    """
    entry_price = pos["entry_price"]
    margin = pos["allocated"]  # margen asignado (ya neto de fee de entrada)

    # Slippage en los fills. En liquidacion el fill ya es el peor caso (liq price).
    entry_fill = entry_price * (1.0 + slippage_pct)
    exit_fill = exit_price if liquidated else exit_price * (1.0 - slippage_pct)

    gross_pct = (exit_price - entry_price) / entry_price  # PnL de precio "ideal" (sin slip)
    price_pnl_pct = (exit_fill - entry_fill) / entry_fill  # PnL de precio con slippage

    # Slippage total como coste (diferencia entre ideal y con slippage, en el notional)
    slippage_pct_cost = gross_pct - price_pnl_pct

    # Fees: entrada ya descontada al abrir; aqui descontamos la de salida.
    exit_fee_pct = fee_pct

    # Funding acumulado durante la vida de la posicion (fraccion del notional)
    funding_pct = 0.0
    if USE_FUNDING:
        funding_pct = _funding_cost_pct(funding_series, pos["entry_time"], exit_ts)

    # PnL neto sobre el NOTIONAL (incluye salida; la fee de entrada ya se cobro)
    notional_pnl_pct = price_pnl_pct - exit_fee_pct - funding_pct

    if liquidated:
        # Se pierde todo el margen: PnL sobre margen = -100%
        margin_pnl = -margin
    else:
        # El PnL del notional se amplifica por leverage sobre el margen
        lev = leverage if leverage and leverage > 0 else 1.0
        margin_pnl = margin * notional_pnl_pct * lev
        # No puede perder mas que el margen (proteccion de modelo)
        if margin_pnl < -margin:
            margin_pnl = -margin

    capital_delta = margin + margin_pnl

    trade = {
        "entry_time": pos["entry_time"],
        "exit_time": exit_ts,
        "entry_price": entry_price,
        "exit_price": round(exit_price, 6),
        # pnl_pct / pnl_after_fees se expresan sobre el MARGEN (consistente con equity)
        "pnl_pct": round(gross_pct * (leverage or 1.0), 6),
        "pnl_after_fees": round(margin_pnl / margin if margin else 0.0, 6),
        "exit_reason": exit_reason,
        "duration_bars": pos["bars_held"],
        "dip_size": pos.get("dip_size", 0),
        "max_rebound_pct": round(pos.get("max_rebound_pct", 0), 6),
        # Desglose de costes (sobre el notional, en fraccion)
        "funding_pct": round(funding_pct, 6),
        "slippage_pct": round(slippage_pct_cost + exit_fee_pct, 6),
        "leverage": leverage or 1.0,
        "liquidation_price": round(pos.get("liquidation_price", 0.0), 6),
    }
    return trade, capital_delta


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    hold_timeout: int = HOLD_TIMEOUT_HOURS,
    position_pct: float = POSITION_PCT,
    max_positions: int = MAX_POSITIONS,
    fee_pct: float = FEE_PCT,
    fail_fast_bars: int | None = FAIL_FAST_BARS,
    fail_fast_min_rebound_pct: float = FAIL_FAST_MIN_REBOUND_PCT,
    slippage_pct: float = SLIPPAGE_PCT,
    leverage: float = LEVERAGE,
    funding_series=None,
) -> dict:
    """
    Run backtest on DataFrame with signals (futures-aware).

    Args:
        df: DataFrame with signals (signal, signal_direction columns)
        initial_capital: Starting capital
        stop_loss_pct: Stop loss percentage (sobre el PRECIO)
        take_profit_pct: Take profit percentage (sobre el PRECIO)
        hold_timeout: Max bars to hold
        position_pct: Fraction of capital per trade (margen)
        max_positions: Max simultaneous positions
        fee_pct: Fee per side (0.04% taker futuros)
        fail_fast_bars: Early-exit bars threshold (None to disable)
        fail_fast_min_rebound_pct: Minimum rebound required by fail-fast
        slippage_pct: Slippage por lado (conservador)
        leverage: Apalancamiento (define liquidacion y amplifica PnL sobre margen)
        funding_series: pd.Series indexada por timestamp con funding_rate real (opcional)

    Returns:
        dict with trades, equity_curve, metrics, sl_below_liquidation
    """
    capital = initial_capital
    positions = []  # list of position dicts
    trades = []
    equity_curve = []

    lev = leverage if leverage and leverage > 0 else 1.0

    # Alerta estructural: con este leverage y SL, ¿el SL cae por debajo de liquidacion?
    # liq_drop = 1/L - mmr  ;  si stop_loss_pct > liq_drop, te liquidan antes del SL.
    liq_drop = (1.0 / lev) - MAINTENANCE_MARGIN_RATE if lev > 1 else 1.0
    sl_below_liquidation = bool(lev > 1 and stop_loss_pct >= liq_drop)

    for i in range(len(df)):
        row = df.iloc[i]
        current_close = row["close"]
        current_high = row["high"]
        current_low = row["low"]
        ts = row["timestamp"]

        # ── 1. Check exits ──
        positions_to_close = []
        for j, pos in enumerate(positions):
            pos["bars_held"] += 1

            max_rebound_pct = (current_high - pos["entry_price"]) / pos["entry_price"]
            if max_rebound_pct > pos.get("max_rebound_pct", 0):
                pos["max_rebound_pct"] = max_rebound_pct

            liq_price = pos.get("liquidation_price", 0.0)
            liquidated = False

            # Peor caso primero: liquidacion y SL se evaluan antes que el TP.
            # Si una vela toca liquidacion, es lo prioritario (ocurre con caida).
            if liq_price > 0 and current_low <= liq_price:
                exit_price = liq_price
                exit_reason = "liquidation"
                liquidated = True
            # Check stop loss
            elif current_low <= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "stop_loss"
            # Check take profit
            elif current_high >= pos["take_profit"]:
                exit_price = pos["take_profit"]
                exit_reason = "take_profit"
            # Check fail-fast
            elif (
                fail_fast_bars is not None
                and int(fail_fast_bars) > 0
                and pos["bars_held"] >= int(fail_fast_bars)
                and pos.get("max_rebound_pct", 0) < fail_fast_min_rebound_pct
            ):
                exit_price = current_close
                exit_reason = "fail_fast"
            # Check timeout
            elif pos["bars_held"] >= hold_timeout:
                exit_price = current_close
                exit_reason = "timeout"
            else:
                continue

            trade, capital_delta = _finalize_trade(
                pos, exit_price, exit_reason, ts,
                fee_pct=fee_pct, slippage_pct=slippage_pct, leverage=lev,
                funding_series=funding_series, liquidated=liquidated,
            )
            capital += capital_delta
            trades.append(trade)
            positions_to_close.append(j)

        # Remove closed positions
        for j in sorted(positions_to_close, reverse=True):
            positions.pop(j)

        # ── 2. Check for new entries ──
        open_slots = max_positions - len(positions)

        if (
            open_slots > 0
            and capital > 0
            and row.get("signal", "NO TRADE") == "LONG"
        ):
            allocated = capital * position_pct

            if allocated > 1:
                # Fee de entrada sobre el NOTIONAL (margen * leverage)
                entry_fee = allocated * lev * fee_pct
                allocated_after_fee = allocated - entry_fee
                capital -= allocated

                levels = compute_exit_levels(
                    current_close, take_profit_pct, stop_loss_pct
                )

                positions.append({
                    "entry_price": current_close,
                    "entry_time": ts,
                    "allocated": allocated_after_fee,
                    "bars_held": 0,
                    "stop_loss": levels["stop_loss"],
                    "take_profit": levels["take_profit"],
                    "liquidation_price": liquidation_price_long(current_close, lev),
                    "dip_size": row.get("dip_size", 0),
                    "max_rebound_pct": 0.0,
                })

        # ── 3. Compute equity ──
        total_equity = capital
        for pos in positions:
            # Valor del margen marcado a mercado (PnL de precio amplificado por leverage)
            price_move = (current_close - pos["entry_price"]) / pos["entry_price"]
            pos_value = pos["allocated"] * (1.0 + price_move * lev)
            total_equity += max(0.0, pos_value)

        equity_curve.append(total_equity)

    # ── Close remaining positions ──
    for pos in positions:
        exit_price = df["close"].iloc[-1]
        trade, capital_delta = _finalize_trade(
            pos, exit_price, "end_of_data", df["timestamp"].iloc[-1],
            fee_pct=fee_pct, slippage_pct=slippage_pct, leverage=lev,
            funding_series=funding_series, liquidated=False,
        )
        capital += capital_delta
        trades.append(trade)

    # ── Build results ──
    equity = np.array(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    metrics = compute_metrics(trades_df, equity, initial_capital)
    metrics["leverage"] = lev
    metrics["sl_below_liquidation"] = sl_below_liquidation

    return {
        "trades": trades_df,
        "equity_curve": equity,
        "equity_timestamps": df["timestamp"].values,
        "metrics": metrics,
        "sl_below_liquidation": sl_below_liquidation,
    }
