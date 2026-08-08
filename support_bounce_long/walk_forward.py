"""
Walk-forward validation for support_bounce_long.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breakout_momentum.data_loader import fetch_ohlcv
from backtest import run_backtest
from config import (
    DAYS,
    DEFAULT_SYMBOLS,
    FEE_PCT,
    INITIAL_CAPITAL,
    MAX_POSITIONS,
    POSITION_PCT,
    TIMEFRAME,
    WF_MIN_TRADES_FOR_PF,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
    get_symbol_params,
)
from strategy import generate_signals


def walk_forward_validation(
    symbol: str,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    total_days: int = DAYS,
    symbol_params: dict[str, Any] | None = None,
    timeframe: str = TIMEFRAME,
    preloaded_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Evaluate strategy on rolling out-of-sample windows.
    """
    params = dict(symbol_params or get_symbol_params(symbol))

    if preloaded_df is not None:
        df = preloaded_df.copy()
    else:
        if verbose:
            print(f"  Downloading {symbol} {timeframe} ({total_days}d)...")
        df = fetch_ohlcv(symbol, timeframe=timeframe, days=total_days)

    if len(df) < 200:
        return []

    warmup_size = int(train_months * 30 * 24)
    test_size = int(test_months * 30 * 24)
    step_size = int(step_months * 30 * 24)

    start_idx = warmup_size
    window_num = 1
    rows: list[dict[str, Any]] = []

    while True:
        test_end_idx = start_idx + test_size
        if test_end_idx > len(df):
            break

        train_start_idx = max(0, start_idx - warmup_size)
        train_data = df.iloc[train_start_idx:start_idx].copy()
        test_data = df.iloc[start_idx:test_end_idx].copy()

        if len(test_data) < 100:
            start_idx += step_size
            window_num += 1
            continue

        test_with_signals = generate_signals(test_data, params)
        n_signals = int((test_with_signals["signal"] == "LONG").sum())

        if n_signals <= 0:
            start_idx += step_size
            window_num += 1
            continue

        bt = run_backtest(
            test_with_signals,
            initial_capital=INITIAL_CAPITAL,
            position_pct=POSITION_PCT,
            max_positions=MAX_POSITIONS,
            fee_pct=FEE_PCT,
            hold_timeout_bars=int(params["hold_timeout_bars"]),
        )

        m = bt["metrics"]
        trades = int(m.get("total_trades", 0))
        if trades <= 0:
            start_idx += step_size
            window_num += 1
            continue

        rows.append(
            {
                "window": window_num,
                "train_start": train_data["timestamp"].min(),
                "train_end": train_data["timestamp"].max(),
                "test_start": test_data["timestamp"].min(),
                "test_end": test_data["timestamp"].max(),
                "total_return": float(m.get("total_return", 0.0)) * 100,
                "sharpe": float(m.get("sharpe", 0.0)),
                "max_dd": float(m.get("max_drawdown", 0.0)) * 100,
                "win_rate": float(m.get("win_rate", 0.0)) * 100,
                "profit_factor": float(m.get("profit_factor", 0.0)),
                "num_trades": trades,
                "expectancy": float(m.get("expectancy", 0.0)) * 100,
                "signals": n_signals,
            }
        )

        start_idx += step_size
        window_num += 1

    return rows


def analyze_results(
    results: list[dict[str, Any]],
    symbol: str,
    min_trades_for_pf: int = WF_MIN_TRADES_FOR_PF,
    verbose: bool = True,
) -> tuple[pd.DataFrame | None, float]:
    if not results:
        if verbose:
            print(f"  {symbol}: no valid walk-forward windows")
        return None, 0.0

    df_results = pd.DataFrame(results)

    positive_returns = int((df_results["total_return"] > 0).sum())
    sharpe_positive = int((df_results["sharpe"] > 0).sum())
    pf_positive = int(
        ((df_results["profit_factor"] > 1) & (df_results["num_trades"] >= int(min_trades_for_pf))).sum()
    )

    total_windows = len(df_results)
    positive_pct = (positive_returns / total_windows) * 100
    sharpe_pct = (sharpe_positive / total_windows) * 100
    pf_pct = (pf_positive / total_windows) * 100

    stability = positive_pct * 0.4 + sharpe_pct * 0.3 + pf_pct * 0.3

    if verbose:
        print(f"\n  WALK-FORWARD ANALYSIS: {symbol}")
        print(f"    windows: {total_windows}")
        print(f"    avg return: {df_results['total_return'].mean():.2f}%")
        print(f"    avg sharpe: {df_results['sharpe'].mean():.2f}")
        print(f"    avg win_rate: {df_results['win_rate'].mean():.1f}%")
        print(f"    positive windows: {positive_returns}/{total_windows} ({positive_pct:.1f}%)")
        print(f"    sharpe>0 windows: {sharpe_positive}/{total_windows} ({sharpe_pct:.1f}%)")
        print(
            "    pf>1 windows: "
            f"{pf_positive}/{total_windows} ({pf_pct:.1f}%) "
            f"[trades>={int(min_trades_for_pf)}]"
        )
        print(f"    stability score: {stability:.1f}/100")

    return df_results, float(stability)


def run_multi_symbol(
    symbols: list[str] | None = None,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    total_days: int = DAYS,
    min_trades_for_pf: int = WF_MIN_TRADES_FOR_PF,
    timeframe: str = TIMEFRAME,
    runtime_overrides: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    symbols = symbols or DEFAULT_SYMBOLS
    scores: dict[str, float] = {}

    if verbose:
        print(f"\n{'=' * 72}")
        print("  SUPPORT-BOUNCE-LONG WALK-FORWARD")
        print(f"{'=' * 72}")

    for symbol in symbols:
        try:
            params = get_symbol_params(symbol, overrides=runtime_overrides)
            rows = walk_forward_validation(
                symbol=symbol,
                train_months=train_months,
                test_months=test_months,
                step_months=step_months,
                total_days=total_days,
                symbol_params=params,
                timeframe=timeframe,
                verbose=verbose,
            )
            _, stability = analyze_results(
                rows,
                symbol,
                min_trades_for_pf=min_trades_for_pf,
                verbose=verbose,
            )
            scores[symbol] = stability
        except Exception as exc:
            if verbose:
                print(f"  Error in {symbol}: {exc}")

    if verbose and scores:
        print(f"\n{'-' * 72}")
        print("  FINAL STABILITY SUMMARY")
        print(f"{'-' * 72}")
        for symbol, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            label = "EXCELLENT" if score >= 70 else "GOOD" if score >= 55 else "MODERATE" if score >= 40 else "POOR"
            print(f"  {label:<10} {symbol:<12} {score:>6.2f}/100")
        print(f"  Average: {np.mean(list(scores.values())):.2f}/100")

    return scores
