"""
run_backtest.py
Run backtest on Buy-the-Dip strategy.
"""

import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
from breakout_momentum.data_loader import fetch_ohlcv
from strategy import add_dip_features, generate_signals
from backtest import run_backtest, compute_metrics
from config import (
    DEFAULT_SYMBOLS,
    TIMEFRAME,
    DAYS,
    INITIAL_CAPITAL,
    FEE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    get_symbol_signal_params,
)


def backtest_symbol(
    symbol: str,
    days: int = DAYS,
    dip_threshold: float = None,
    verbose: bool = True,
) -> dict:
    """
    Backtest a single symbol with Buy-the-Dip strategy.

    Args:
        symbol: Trading pair
        days: Historical data to fetch
        dip_threshold: Override dip threshold (e.g., -0.05, -0.07)
        verbose: Print details

    Returns:
        Backtest results dict
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  BUY-THE-DIP BACKTEST: {symbol}")
        print(f"{'='*60}")

    # 1. Fetch data
    if verbose:
        print(f"\n  Fetching {days} days of {TIMEFRAME} data...")

    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=days)

    if verbose:
        print(f"  Total rows: {len(df)}")
        print(f"  Date range: {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")

    # 2. Add features
    if verbose:
        print(f"\n  Computing dip features...")

    df = add_dip_features(df)

    # 3. Generate signals
    signal_params = get_symbol_signal_params(
        symbol, dip_threshold_override=dip_threshold
    )
    threshold = signal_params["dip_threshold"]

    if verbose:
        print(f"  Dip threshold: {threshold*100:.1f}%")
        print(
            "  Filters: "
            f"RSI={'on' if signal_params['use_rsi_filter'] else 'off'} "
            f"VOL={'on' if signal_params['use_volume_filter'] else 'off'} "
            f"TP={signal_params['take_profit_pct']*100:.1f}% "
            f"SL={signal_params['stop_loss_pct']*100:.1f}% "
            f"HLD={signal_params['hold_timeout_hours']}h "
            f"MAXP={signal_params['max_positions']} "
            f"RVmin={signal_params['min_volume_mult']:.2f} "
            f"RSI=[{signal_params['rsi_min']:.0f},{signal_params['rsi_max']:.0f}] "
            f"OED={'on' if signal_params['one_entry_per_dip_event'] else 'off'} "
            f"RST={signal_params['dip_event_reset_pct']*100:.1f}% "
            f"FF={signal_params['fail_fast_bars'] if signal_params['fail_fast_bars'] is not None else '-'} "
            f"FFmin={signal_params['fail_fast_min_rebound_pct']*100:.1f}%"
        )

    df = generate_signals(
        df,
        dip_threshold=threshold,
        use_volume_filter=signal_params["use_volume_filter"],
        use_rsi_filter=signal_params["use_rsi_filter"],
        min_volume_mult=signal_params["min_volume_mult"],
        rsi_min=signal_params["rsi_min"],
        rsi_max=signal_params["rsi_max"],
        one_entry_per_dip_event=signal_params["one_entry_per_dip_event"],
        dip_event_reset_pct=signal_params["dip_event_reset_pct"],
    )

    # Count signals
    signals = (df["signal"] == "LONG").sum()

    if verbose:
        print(f"  Total BUY signals: {signals}")
        print(f"  Avg signals/month: {signals / (days/30):.1f}")

    # 4. Run backtest
    if verbose:
        print(f"\n  Running backtest...")

    results = run_backtest(
        df,
        initial_capital=INITIAL_CAPITAL,
        stop_loss_pct=signal_params["stop_loss_pct"],
        take_profit_pct=signal_params["take_profit_pct"],
        hold_timeout=signal_params["hold_timeout_hours"],
        max_positions=signal_params["max_positions"],
        fee_pct=FEE_PCT,
        fail_fast_bars=signal_params["fail_fast_bars"],
        fail_fast_min_rebound_pct=signal_params["fail_fast_min_rebound_pct"],
    )

    # 5. Print results
    if verbose:
        m = results["metrics"]
        print(f"\n  ── Results ──────────────────────────────────")
        print(f"  Total Return:  {m.get('total_return', 0)*100:>7.2f}%")
        print(f"  Sharpe Ratio:  {m.get('sharpe', 0):>7.2f}")
        print(f"  Max Drawdown:  {m.get('max_drawdown', 0)*100:>7.2f}%")
        print(f"  Win Rate:      {m.get('win_rate', 0)*100:>7.2f}%")
        print(f"  Profit Factor: {m.get('profit_factor', 0):>7.2f}")
        print(f"  Total Trades:  {m.get('total_trades', 0):>7}")
        print(f"  Expectancy:    {m.get('expectancy', 0)*100:>7.2f}%")
        print(f"  Final Capital: ${m.get('final_capital', 0):>,.2f}")

    return results


def backtest_all_symbols(
    symbols: list = None,
    days: int = DAYS,
    dip_threshold: float = None,
) -> dict:
    """
    Backtest all symbols and aggregate results.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    print(f"\n{'='*60}")
    print(f"  BUY-THE-DIP BACKTEST - ALL SYMBOLS")
    print(f"{'='*60}")
    print(f"  Symbols:   {len(symbols)}")
    print(f"  Period:    {days} days")
    if dip_threshold is not None:
        print(f"  Threshold: {dip_threshold*100:.1f}% (global override)")
    else:
        print(f"  Threshold: per-symbol override")
    print(f"  SL/TP:     per-symbol overrides (default SL {STOP_LOSS_PCT*100:.1f}% / TP {TAKE_PROFIT_PCT*100:.1f}%)")
    print(f"{'='*60}")

    all_results = {}

    for symbol in symbols:
        try:
            results = backtest_symbol(
                symbol, days=days, dip_threshold=dip_threshold, verbose=True
            )
            all_results[symbol] = results
        except Exception as e:
            print(f"\n  Error with {symbol}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY - ALL SYMBOLS")
    print(f"{'='*60}")
    print(f"\n  {'Symbol':<12} {'Return':>8} {'Sharpe':>8} {'WR':>8} {'Trades':>8}")
    print(f"  {'-'*48}")

    total_return = 0
    total_sharpe = 0
    total_wr = 0
    count = 0

    for symbol, results in all_results.items():
        m = results["metrics"]
        ret = m.get("total_return", 0) * 100
        sharpe = m.get("sharpe", 0)
        wr = m.get("win_rate", 0) * 100
        trades = m.get("total_trades", 0)

        print(f"  {symbol:<12} {ret:>7.2f}% {sharpe:>8.2f} {wr:>7.1f}% {trades:>8}")

        total_return += ret
        total_sharpe += sharpe
        total_wr += wr
        count += 1

    if count > 0:
        print(f"  {'-'*48}")
        print(f"  {'AVERAGE':<12} {total_return/count:>7.2f}% {total_sharpe/count:>8.2f} {total_wr/count:>7.1f}%")

    print(f"\n{'='*60}\n")

    return all_results


def optimize_threshold(
    symbol: str,
    thresholds: list = None,
    days: int = DAYS,
) -> dict:
    """
    Test different dip thresholds to find optimal.
    """
    if thresholds is None:
        thresholds = [-0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10]

    print(f"\n{'='*60}")
    print(f"  THRESHOLD OPTIMIZATION: {symbol}")
    print(f"{'='*60}")

    results = {}

    for thresh in thresholds:
        try:
            res = backtest_symbol(symbol, days=days, dip_threshold=thresh, verbose=False)
            m = res["metrics"]
            results[thresh] = {
                "return": m.get("total_return", 0) * 100,
                "sharpe": m.get("sharpe", 0),
                "win_rate": m.get("win_rate", 0) * 100,
                "trades": m.get("total_trades", 0),
                "expectancy": m.get("expectancy", 0) * 100,
            }
        except Exception as e:
            print(f"  Error with threshold {thresh}: {e}")

    print(f"\n  {'Threshold':>10} {'Return':>8} {'Sharpe':>8} {'WR':>8} {'Trades':>8}")
    print(f"  {'-'*48}")

    best_thresh = None
    best_return = -999

    for thresh, r in sorted(results.items()):
        print(f"  {thresh*100:>9.1f}% {r['return']:>7.2f}% {r['sharpe']:>8.2f} {r['win_rate']:>7.1f}% {r['trades']:>8}")

        if r["return"] > best_return and r["trades"] >= 10:
            best_return = r["return"]
            best_thresh = thresh

    print(f"\n  Best threshold: {best_thresh*100:.1f}% (Return: {best_return:.2f}%)")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    # Run backtest on symbols with proven edge
    # Using -7% threshold (stronger edge)
    results = backtest_all_symbols(
        symbols=["BNB/USDT", "DOGE/USDT"],
        days=1825,
        dip_threshold=-0.07,
    )
