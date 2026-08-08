"""
Backtest engine for support_bounce_long.

Portfolio model is aligned with buy_the_dip:
- fixed capital allocation per trade (POSITION_PCT)
- max concurrent positions (MAX_POSITIONS)
- fee per side (FEE_PCT)

Trade exits come from setup-native levels:
- stop below support (with configured buffer)
- target at nearest resistance
- timeout in bars
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_metrics(
    trades_df: pd.DataFrame,
    equity_curve: np.ndarray,
    initial_capital: float,
) -> dict[str, Any]:
    if len(trades_df) == 0:
        final_capital = float(equity_curve[-1]) if len(equity_curve) else float(initial_capital)
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "expectancy": 0.0,
            "final_capital": round(final_capital, 2),
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    total_return = (float(equity_curve[-1]) - float(initial_capital)) / float(initial_capital)

    returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([])
    sharpe = 0.0
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(8760))

    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    pnl_col = "pnl_after_fees"
    wins = trades_df[trades_df[pnl_col] > 0]
    losses = trades_df[trades_df[pnl_col] <= 0]
    win_rate = float(len(wins) / len(trades_df)) if len(trades_df) else 0.0

    gross_profit = float(trades_df.loc[trades_df[pnl_col] > 0, pnl_col].sum())
    gross_loss = abs(float(trades_df.loc[trades_df[pnl_col] < 0, pnl_col].sum()))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    avg_win = float(wins[pnl_col].mean()) if len(wins) > 0 else 0.0
    avg_loss = abs(float(losses[pnl_col].mean())) if len(losses) > 0 else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    return {
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2),
        "total_trades": int(len(trades_df)),
        "expectancy": round(expectancy, 6),
        "final_capital": round(float(equity_curve[-1]), 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(-avg_loss, 4),
    }


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float,
    position_pct: float,
    max_positions: int,
    fee_pct: float,
    hold_timeout_bars: int,
) -> dict[str, Any]:
    """
    Execute backtest using LONG signals and setup-native stop/target levels.
    """
    if df.empty:
        equity = np.array([float(initial_capital)])
        return {
            "trades": pd.DataFrame(),
            "equity_curve": equity,
            "equity_timestamps": np.array([]),
            "metrics": compute_metrics(pd.DataFrame(), equity, initial_capital),
        }

    capital = float(initial_capital)
    hold_timeout = max(1, int(hold_timeout_bars))
    positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_curve: list[float] = []

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row.get("timestamp")
        current_close = float(row.get("close", 0.0))
        current_high = float(row.get("high", current_close))
        current_low = float(row.get("low", current_close))

        # 1) Exits
        positions_to_close: list[int] = []

        for j, pos in enumerate(positions):
            pos["bars_held"] += 1

            if current_low <= float(pos["stop_loss"]):
                exit_price = float(pos["stop_loss"])
                exit_reason = "stop_loss"
            elif current_high >= float(pos["take_profit"]):
                exit_price = float(pos["take_profit"])
                exit_reason = "take_profit"
            elif int(pos["bars_held"]) >= int(pos["hold_timeout_bars"]):
                exit_price = current_close
                exit_reason = "timeout"
            else:
                continue

            pnl_pct = (exit_price - float(pos["entry_price"])) / float(pos["entry_price"])
            pnl_after_fees = pnl_pct - float(fee_pct)

            capital_change = float(pos["allocated"]) * pnl_after_fees
            capital += float(pos["allocated"]) + capital_change

            trades.append(
                {
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "entry_price": float(pos["entry_price"]),
                    "exit_price": round(exit_price, 8),
                    "pnl_pct": round(pnl_pct, 6),
                    "pnl_after_fees": round(pnl_after_fees, 6),
                    "exit_reason": exit_reason,
                    "duration_bars": int(pos["bars_held"]),
                    "setup_id": pos.get("setup_id"),
                    "entry_setup": pos.get("entry_setup"),
                }
            )

            positions_to_close.append(j)

        for j in sorted(positions_to_close, reverse=True):
            positions.pop(j)

        # 2) Entries
        open_slots = int(max_positions) - len(positions)
        signal = str(row.get("signal", "NO_SIGNAL"))

        if open_slots > 0 and capital > 0 and signal == "LONG":
            setup = row.get("setup_data")
            if isinstance(setup, dict):
                stop_loss = float(setup.get("stop_price", 0.0))
                take_profit = float(setup.get("target_price", 0.0))
                entry_price = current_close

                if stop_loss > 0 and take_profit > 0 and entry_price > 0:
                    allocated = capital * float(position_pct)
                    if allocated > 1:
                        entry_fee = allocated * float(fee_pct)
                        allocated_after_fee = allocated - entry_fee
                        capital -= allocated

                        positions.append(
                            {
                                "entry_price": entry_price,
                                "entry_time": ts,
                                "allocated": allocated_after_fee,
                                "bars_held": 0,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "hold_timeout_bars": hold_timeout,
                                "setup_id": setup.get("id"),
                                "entry_setup": {
                                    "setup": setup,
                                    "conditions": list(row.get("conditions") or []),
                                    "summary_text": str(row.get("summary_text", "")),
                                    "blocked_by": row.get("blocked_by"),
                                },
                            }
                        )

        # 3) Equity mark-to-market
        total_equity = capital
        for pos in positions:
            pos_value = float(pos["allocated"]) * (current_close / float(pos["entry_price"]))
            total_equity += pos_value

        equity_curve.append(float(total_equity))

    # 4) Force close remaining positions at end of data
    last_close = float(df.iloc[-1].get("close", 0.0))
    last_ts = df.iloc[-1].get("timestamp")

    for pos in positions:
        pnl_pct = (last_close - float(pos["entry_price"])) / float(pos["entry_price"])
        pnl_after_fees = pnl_pct - float(fee_pct)

        capital_change = float(pos["allocated"]) * pnl_after_fees
        capital += float(pos["allocated"]) + capital_change

        trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": last_ts,
                "entry_price": float(pos["entry_price"]),
                "exit_price": round(last_close, 8),
                "pnl_pct": round(pnl_pct, 6),
                "pnl_after_fees": round(pnl_after_fees, 6),
                "exit_reason": "end_of_data",
                "duration_bars": int(pos["bars_held"]),
                "setup_id": pos.get("setup_id"),
                "entry_setup": pos.get("entry_setup"),
            }
        )

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity = np.array(equity_curve if equity_curve else [capital], dtype=float)

    metrics = compute_metrics(trades_df, equity, initial_capital)

    return {
        "trades": trades_df,
        "equity_curve": equity,
        "equity_timestamps": df["timestamp"].values,
        "metrics": metrics,
    }
