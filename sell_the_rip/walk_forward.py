"""
Walk-forward validation for Sell-the-Rip.
"""

import argparse
import sys

sys.path.insert(0, "..")

import numpy as np
import pandas as pd

from breakout_momentum.data_loader import fetch_ohlcv
from backtest import run_backtest
from config import (
    DAYS,
    DEFAULT_SYMBOLS,
    FEE_PCT,
    INITIAL_CAPITAL,
    TIMEFRAME,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
    get_symbol_signal_params,
)
from strategy import add_rip_features, generate_signals


def walk_forward_validation(
    symbol: str,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    total_days: int = DAYS,
    rip_threshold: float = None,
):
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION: {symbol}")
    print(f"{'='*70}")
    print(f"  Warmup window: {train_months} months")
    print(f"  Test window:   {test_months} month(s)")
    print(f"  Step size:     {step_months} month(s)")
    print(f"{'='*70}\n")

    signal_params = get_symbol_signal_params(symbol, rip_threshold_override=rip_threshold)
    threshold = float(signal_params["rip_threshold"])
    print(f"  Rip threshold: +{threshold*100:.1f}%")
    print(
        "  Filters: "
        f"RSI={'on' if signal_params['use_rsi_filter'] else 'off'} "
        f"VOL={'on' if signal_params['use_volume_filter'] else 'off'} "
        f"TREND={'on' if signal_params['use_trend_filter'] else 'off'} "
        f"CT={'on' if signal_params.get('allow_countertrend_reversal', False) else 'off'}"
    )
    print(
        "  Risk: "
        f"TP={float(signal_params['take_profit_pct'])*100:.1f}% "
        f"SL={float(signal_params['stop_loss_pct'])*100:.1f}% "
        f"HOLD={int(signal_params['hold_timeout_hours'])}h "
        f"RSI=[{float(signal_params['rsi_min']):.0f},{float(signal_params['rsi_max']):.0f}] "
        f"VOL_MIN={float(signal_params['min_volume_mult']):.2f}"
    )

    print(f"\n  Downloading {symbol} data...")
    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=total_days)
    df = add_rip_features(df)

    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    results = []
    warmup_size = int(train_months * 30 * 24)
    test_size = int(test_months * 30 * 24)
    step_size = int(step_months * 30 * 24)

    start_idx = warmup_size
    window_num = 1

    while True:
        test_end_idx = start_idx + test_size
        if test_end_idx > len(df):
            break

        test_data = df.iloc[start_idx:test_end_idx].copy()
        if len(test_data) < 100:
            start_idx += step_size
            window_num += 1
            continue

        test_start = test_data["timestamp"].min()
        test_end = test_data["timestamp"].max()

        try:
            test_with_signals = generate_signals(
                test_data,
                rip_threshold=threshold,
                use_volume_filter=signal_params["use_volume_filter"],
                use_rsi_filter=signal_params["use_rsi_filter"],
                use_trend_filter=signal_params["use_trend_filter"],
                allow_countertrend_reversal=bool(signal_params.get("allow_countertrend_reversal", False)),
                countertrend_rip_lookback_hours=int(signal_params.get("countertrend_rip_lookback_hours", 8)),
                countertrend_min_drop_1h=float(signal_params.get("countertrend_min_drop_1h", 0.01)),
                countertrend_min_pullback_pct=float(signal_params.get("countertrend_min_pullback_pct", 0.012)),
                min_volume_mult=signal_params["min_volume_mult"],
                rsi_min=signal_params["rsi_min"],
                rsi_max=signal_params["rsi_max"],
            )

            n_signals = (test_with_signals["signal"] == "SHORT").sum()
            if n_signals == 0:
                start_idx += step_size
                window_num += 1
                continue

            bt = run_backtest(
                test_with_signals,
                initial_capital=INITIAL_CAPITAL,
                stop_loss_pct=float(signal_params["stop_loss_pct"]),
                take_profit_pct=float(signal_params["take_profit_pct"]),
                hold_timeout=int(signal_params["hold_timeout_hours"]),
                fee_pct=FEE_PCT,
            )
            m = bt["metrics"]

            results.append(
                {
                    "window": window_num,
                    "test_start": test_start,
                    "test_end": test_end,
                    "total_return": m.get("total_return", 0) * 100,
                    "sharpe": m.get("sharpe", 0),
                    "max_dd": m.get("max_drawdown", 0) * 100,
                    "win_rate": m.get("win_rate", 0) * 100,
                    "profit_factor": m.get("profit_factor", 0),
                    "num_trades": m.get("total_trades", 0),
                    "expectancy": m.get("expectancy", 0) * 100,
                }
            )

        except Exception as exc:
            print(f"    Error: {exc}")

        start_idx += step_size
        window_num += 1

    return results


def analyze_results(results: list, symbol: str, min_trades_for_pf: int = 5):
    if not results:
        print("\n  No results to analyze")
        return None, 0.0

    df_results = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD ANALYSIS: {symbol}")
    print(f"{'='*70}")

    print(f"\n  Summary ({len(results)} windows):")
    print("  ─────────────────────────────────────────")

    for metric in ["total_return", "sharpe", "win_rate", "profit_factor", "max_dd", "num_trades"]:
        vals = df_results[metric]
        print(f"\n  {metric}:")
        print(f"    Mean:   {vals.mean():>8.2f}")
        print(f"    Median: {vals.median():>8.2f}")
        print(f"    Std:    {vals.std():>8.2f}")
        print(f"    Min:    {vals.min():>8.2f}")
        print(f"    Max:    {vals.max():>8.2f}")

    positive_returns = (df_results["total_return"] > 0).sum()
    positive_pct = positive_returns / len(results) * 100

    sharpe_positive = (df_results["sharpe"] > 0).sum()
    sharpe_pct = sharpe_positive / len(results) * 100

    enough_trades = df_results["num_trades"] >= min_trades_for_pf
    low_trade_windows = (~enough_trades).sum()

    pf_above_1 = ((df_results["profit_factor"] > 1) & enough_trades).sum()
    pf_pct = pf_above_1 / len(results) * 100

    print("\n  Consistency:")
    print("  ─────────────────────────────────────────")
    print(f"    Positive returns: {positive_returns}/{len(results)} ({positive_pct:.1f}%)")
    print(f"    Sharpe > 0:       {sharpe_positive}/{len(results)} ({sharpe_pct:.1f}%)")
    print(f"    PF > 1 (trades>={min_trades_for_pf}): {pf_above_1}/{len(results)} ({pf_pct:.1f}%)")
    print(f"    Low-trade windows (<{min_trades_for_pf}): {low_trade_windows}/{len(results)}")

    stability = positive_pct * 0.4 + sharpe_pct * 0.3 + pf_pct * 0.3
    print(f"\n  STABILITY SCORE: {stability:.1f}/100")

    if stability >= 70:
        print("     EXCELLENT - Strategy is robust")
    elif stability >= 55:
        print("     GOOD - Reasonably stable")
    elif stability >= 40:
        print("     MODERATE - Some instability")
    else:
        print("     POOR - Likely not robust")

    print(f"\n{'='*70}\n")
    return df_results, float(stability)


def run_multi_symbol(
    symbols: list = None,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    total_days: int = DAYS,
    rip_threshold: float = None,
    min_trades_for_pf: int = 5,
):
    symbols = symbols or DEFAULT_SYMBOLS

    print(f"\n{'='*70}")
    print("  MULTI-SYMBOL WALK-FORWARD: SELL-THE-RIP")
    print(f"{'='*70}")

    all_scores = {}

    for symbol in symbols:
        try:
            results = walk_forward_validation(
                symbol,
                train_months=train_months,
                test_months=test_months,
                step_months=step_months,
                total_days=total_days,
                rip_threshold=rip_threshold,
            )
            _, stability = analyze_results(results, symbol, min_trades_for_pf=min_trades_for_pf)
            all_scores[symbol] = stability
        except Exception as exc:
            print(f"\n  Error with {symbol}: {exc}")

    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}\n")

    for symbol, score in all_scores.items():
        status = (
            "EXCELLENT" if score >= 70 else "GOOD" if score >= 55 else "MODERATE" if score >= 40 else "POOR"
        )
        print(f"  {status:<10} {symbol:<12} Stability: {score:>5.1f}/100")

    if all_scores:
        print(f"\n  Average: {np.mean(list(all_scores.values())):.1f}/100")

    print(f"\n{'='*70}\n")
    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward validation for Sell-the-Rip")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--min-trades", type=int, default=5)
    args = parser.parse_args()

    run_multi_symbol(
        symbols=args.symbols or DEFAULT_SYMBOLS,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        total_days=args.days,
        rip_threshold=args.threshold,
        min_trades_for_pf=args.min_trades,
    )
