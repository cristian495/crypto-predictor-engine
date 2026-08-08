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
)
from strategy import generate_signals, add_dip_features, compute_exit_levels


def compute_metrics(trades_df: pd.DataFrame, equity_curve: np.ndarray,
                    initial_capital: float) -> dict:
    """Compute performance metrics."""
    if len(trades_df) == 0:
        return {
            "total_return": 0, "sharpe": 0, "max_drawdown": 0,
            "win_rate": 0, "profit_factor": 0, "total_trades": 0,
            "expectancy": 0, "final_capital": initial_capital,
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
    }


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
) -> dict:
    """
    Run backtest on DataFrame with signals.

    Args:
        df: DataFrame with signals (signal, signal_direction columns)
        initial_capital: Starting capital
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        hold_timeout: Max bars to hold
        position_pct: Fraction of capital per trade
        max_positions: Max simultaneous positions
        fee_pct: Fee per side
        fail_fast_bars: Early-exit bars threshold (None to disable)
        fail_fast_min_rebound_pct: Minimum rebound required by fail-fast

    Returns:
        dict with trades, equity_curve, metrics
    """
    capital = initial_capital
    positions = []  # list of position dicts
    trades = []
    equity_curve = []

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

            # Check stop loss
            if current_low <= pos["stop_loss"]:
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

            # Calculate PnL (LONG only)
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
            pnl_after_fees = pnl_pct - fee_pct

            # Return capital
            capital_change = pos["allocated"] * pnl_after_fees
            capital += pos["allocated"] + capital_change

            trades.append({
                "entry_time": pos["entry_time"],
                "exit_time": ts,
                "entry_price": pos["entry_price"],
                "exit_price": round(exit_price, 6),
                "pnl_pct": round(pnl_pct, 6),
                "pnl_after_fees": round(pnl_after_fees, 6),
                "exit_reason": exit_reason,
                "duration_bars": pos["bars_held"],
                "dip_size": pos.get("dip_size", 0),
                "max_rebound_pct": round(pos.get("max_rebound_pct", 0), 6),
            })

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
                entry_fee = allocated * fee_pct
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
                    "dip_size": row.get("dip_size", 0),
                    "max_rebound_pct": 0.0,
                })

        # ── 3. Compute equity ──
        total_equity = capital
        for pos in positions:
            pos_value = pos["allocated"] * (current_close / pos["entry_price"])
            total_equity += pos_value

        equity_curve.append(total_equity)

    # ── Close remaining positions ──
    for pos in positions:
        exit_price = df["close"].iloc[-1]
        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        pnl_after_fees = pnl_pct - fee_pct

        capital_change = pos["allocated"] * pnl_after_fees
        capital += pos["allocated"] + capital_change

        trades.append({
            "entry_time": pos["entry_time"],
            "exit_time": df["timestamp"].iloc[-1],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 6),
            "pnl_after_fees": round(pnl_after_fees, 6),
            "exit_reason": "end_of_data",
            "duration_bars": pos["bars_held"],
            "dip_size": pos.get("dip_size", 0),
            "max_rebound_pct": round(pos.get("max_rebound_pct", 0), 6),
        })

    # ── Build results ──
    equity = np.array(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    metrics = compute_metrics(trades_df, equity, initial_capital)

    return {
        "trades": trades_df,
        "equity_curve": equity,
        "equity_timestamps": df["timestamp"].values,
        "metrics": metrics,
    }
