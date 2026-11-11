"""
main.py
CLI orchestrator for the Buy-the-Dip trading system.

Usage:
    python main.py --scan                       # Multi-crypto scan
    python main.py --scan --symbols BNB/USDT LINK/USDT DOGE/USDT
    python main.py --walk-forward               # Walk-forward validation
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, "..")

import pandas as pd

from config import (
    TIMEFRAME,
    DAYS,
    DEFAULT_SYMBOLS,
    TAKE_PROFIT_PCT,
    STOP_LOSS_PCT,
    HOLD_TIMEOUT_HOURS,
    POSITION_PCT,
    MAX_POSITIONS,
    FEE_PCT,
    INITIAL_CAPITAL,
    get_symbol_signal_params,
)
from breakout_momentum.data_loader import fetch_ohlcv
from strategy import add_dip_features, generate_signals, compute_exit_levels
from backtest import run_backtest

STRATEGY_NAME = Path(__file__).resolve().parent.name


def _evaluate_buy_filters(last: pd.Series, signal_params: dict) -> tuple[list[tuple[str, bool, str]], str]:
    """Evaluate current-bar entry filters and return condition lines + blocking reason."""
    threshold = float(signal_params["dip_threshold"])
    use_vol = bool(signal_params["use_volume_filter"])
    use_rsi = bool(signal_params["use_rsi_filter"])
    min_volume_mult = float(signal_params["min_volume_mult"])
    rsi_min = float(signal_params["rsi_min"])
    rsi_max = float(signal_params["rsi_max"])

    past_ret = last.get("past_return", float("nan"))
    rel_vol = last.get("relative_volume", float("nan"))
    rsi = last.get("rsi_14", float("nan"))

    checks: list[tuple[str, bool, str]] = []

    dip_ok = pd.notna(past_ret) and float(past_ret) < threshold
    checks.append(
        (
            "dip_threshold",
            bool(dip_ok),
            (
                f"past_return={float(past_ret)*100:+.2f}% "
                f"< th={threshold*100:.2f}%"
                if pd.notna(past_ret)
                else "past_return=NaN"
            ),
        )
    )

    if use_vol:
        vol_ok = pd.notna(rel_vol) and float(rel_vol) >= min_volume_mult
        checks.append(
            (
                "volume_filter",
                bool(vol_ok),
                (
                    f"relative_volume={float(rel_vol):.2f} "
                    f">= min={min_volume_mult:.2f}"
                    if pd.notna(rel_vol)
                    else "relative_volume=NaN"
                ),
            )
        )
    else:
        checks.append(("volume_filter", True, "disabled"))

    if use_rsi:
        rsi_ok = pd.notna(rsi) and (rsi_min <= float(rsi) <= rsi_max)
        checks.append(
            (
                "rsi_filter",
                bool(rsi_ok),
                (
                    f"rsi_14={float(rsi):.2f} in [{rsi_min}, {rsi_max}]"
                    if pd.notna(rsi)
                    else "rsi_14=NaN"
                ),
            )
        )
    else:
        checks.append(("rsi_filter", True, "disabled"))

    blocking = "none"
    for name, ok, detail in checks:
        if not ok:
            blocking = f"{name} ({detail})"
            break

    return checks, blocking


def print_current_signal(df: pd.DataFrame, symbol: str, threshold: float, signal_params: dict):
    """Print current signal status for a symbol."""
    last = df.iloc[-1]
    past_ret = last.get("past_return", 0) * 100
    signal = last.get("signal", "NO TRADE")
    checks, blocking = _evaluate_buy_filters(last, signal_params)

    if signal == "LONG":
        take_profit_pct = float(signal_params["take_profit_pct"])
        stop_loss_pct = float(signal_params["stop_loss_pct"])
        levels = compute_exit_levels(last["close"], take_profit_pct, stop_loss_pct)
        print(f"  {symbol:<12} BUY SIGNAL!")
        print(f"               24h Drop: {past_ret:.2f}%")
        print(f"               Entry:    ${last['close']:,.4f}")
        print(f"               TP:       ${levels['take_profit']:,.4f} (+{take_profit_pct*100:.1f}%)")
        print(f"               SL:       ${levels['stop_loss']:,.4f} (-{stop_loss_pct*100:.1f}%)")
        print(f"               Timeout:  {signal_params['hold_timeout_hours']}h")
    else:
        print(f"  {symbol:<12} NO SIGNAL  (24h: {past_ret:+.2f}%, need <{threshold*100:.0f}%)")

    print("               Conditions:")
    for name, ok, detail in checks:
        state = "PASS" if ok else "FAIL"
        print(f"                 - {name:<14} {state:<4} | {detail}")
    if signal != "LONG":
        print(f"               Blocked by: {blocking}")


def _build_signal_payload(df: pd.DataFrame, symbol: str, signal_params: dict) -> dict | None:
    """Build a structured signal dict for machine consumption."""
    last = df.iloc[-1]
    signal = last.get("signal", "NO TRADE")

    if signal != "LONG":
        return None

    past_ret = last.get("past_return", 0) * 100
    rsi = last.get("rsi_14", 0)
    entry = float(last.get("close", 0))
    take_profit_pct = float(signal_params["take_profit_pct"])
    stop_loss_pct = float(signal_params["stop_loss_pct"])
    levels = compute_exit_levels(entry, take_profit_pct, stop_loss_pct)

    extra = (
        f"Drop={past_ret:.2f}% RSI={rsi:.1f} "
        f"Entry={entry:.4f} TP={levels['take_profit']:.4f} "
        f"SL={levels['stop_loss']:.4f}"
    )

    return {
        "symbol": symbol,
        "signal": "LONG",
        "prob": 0.5,
        "extra": extra,
    }


def _write_signals_json(path: str, signals: list, strategy: str):
    """Write signals in a standard JSON format."""
    payload = {
        "strategy": strategy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    }
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=True)


# ── Multi-crypto scan mode ────────────────────────────────────


def main_scan(
    symbols: list = None,
    timeframe: str = TIMEFRAME,
    days: int = DAYS,
    dip_threshold: float = None,
    signals_json: str | None = None,
):
    """Multi-crypto pipeline: scan, signal, backtest."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    print("=" * 70)
    print("  BUY-THE-DIP — MULTI-CRYPTO SCANNER")
    print("=" * 70)
    print(f"  Mode:          Scanner")
    print(f"  Symbols:       {len(symbols)}")
    print(f"  Timeframe:     {timeframe}")
    print(f"  History:       {days} days")
    if dip_threshold is not None:
        print(f"  Dip threshold: {dip_threshold*100:.1f}% (global override)")
    else:
        print("  Dip threshold: per-symbol override")
    print(f"  Take Profit:   per-symbol override (default {TAKE_PROFIT_PCT*100:.1f}%)")
    print(f"  Stop Loss:     per-symbol override (default {STOP_LOSS_PCT*100:.1f}%)")
    print(f"  Hold timeout:  per-symbol (default {HOLD_TIMEOUT_HOURS}h)")
    print(f"  Max positions: per-symbol (default {MAX_POSITIONS})")
    print(f"  Fees:          {FEE_PCT:.2%} per side")
    print("=" * 70)

    # 1. Process each symbol
    print(f"\n[1/3] Processing symbols...")
    results = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(symbols)}] {symbol}")
        print('─' * 60)

        try:
            signal_params = get_symbol_signal_params(
                symbol, dip_threshold_override=dip_threshold
            )
            df = fetch_ohlcv(symbol, timeframe, days)
            df = add_dip_features(df)
            df = generate_signals(
                df,
                dip_threshold=signal_params["dip_threshold"],
                use_volume_filter=signal_params["use_volume_filter"],
                use_rsi_filter=signal_params["use_rsi_filter"],
                min_volume_mult=signal_params["min_volume_mult"],
                rsi_min=signal_params["rsi_min"],
                rsi_max=signal_params["rsi_max"],
                one_entry_per_dip_event=signal_params["one_entry_per_dip_event"],
                dip_event_reset_pct=signal_params["dip_event_reset_pct"],
            )

            n_signals = (df["signal"] == "LONG").sum()
            print(f"  Signals: {n_signals}")
            print(
                "  Config: "
                f"th={signal_params['dip_threshold']*100:.1f}% "
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

            bt = run_backtest(
                df,
                initial_capital=INITIAL_CAPITAL,
                stop_loss_pct=signal_params["stop_loss_pct"],
                take_profit_pct=signal_params["take_profit_pct"],
                hold_timeout=signal_params["hold_timeout_hours"],
                max_positions=signal_params["max_positions"],
                fee_pct=FEE_PCT,
                fail_fast_bars=signal_params["fail_fast_bars"],
                fail_fast_min_rebound_pct=signal_params["fail_fast_min_rebound_pct"],
                leverage=signal_params.get("leverage"),
            )

            m = bt["metrics"]
            print(f"  Return: {m['total_return']:.2%}  Sharpe: {m['sharpe']:.2f}  "
                  f"WR: {m['win_rate']:.1%}  Trades: {m['total_trades']}")

            results[symbol] = {
                "df": df,
                "backtest": bt,
                "metrics": m,
                "signal_params": signal_params,
            }

        except Exception as e:
            print(f"  ERROR: {e}")

    if not results:
        print("No symbols processed successfully.")
        return None

    # 2. Current signals
    print(f"\n[2/3] Current signals...")
    print(f"\n{'=' * 70}")
    print(f"  CURRENT SIGNALS — Buy-the-Dip")
    print(f"{'=' * 70}")

    signals_payload = []

    for symbol, data in results.items():
        threshold = data["signal_params"]["dip_threshold"]
        print_current_signal(data["df"], symbol, threshold, data["signal_params"])
        sig = _build_signal_payload(data["df"], symbol, data["signal_params"])
        if sig:
            signals_payload.append(sig)

    print(f"{'=' * 70}")

    # 3. Summary
    print(f"\n[3/3] Summary...")
    print(f"\n{'=' * 70}")
    print(f"  BACKTEST SUMMARY — ALL SYMBOLS")
    print(f"{'=' * 70}")
    print(f"\n  {'Symbol':<12} {'Return':>10} {'Sharpe':>8} {'WR':>8} {'PF':>8} {'Trades':>8}")
    print(f"  {'-'*56}")

    total_return = 0
    total_sharpe = 0
    total_wr = 0
    count = 0

    for symbol, data in results.items():
        m = data["metrics"]
        ret = m["total_return"] * 100
        sharpe = m["sharpe"]
        wr = m["win_rate"] * 100
        pf = m["profit_factor"]
        trades = m["total_trades"]

        print(f"  {symbol:<12} {ret:>9.2f}% {sharpe:>8.2f} {wr:>7.1f}% {pf:>8.2f} {trades:>8}")

        total_return += ret
        total_sharpe += sharpe
        total_wr += wr
        count += 1

    if count > 0:
        print(f"  {'-'*56}")
        print(f"  {'AVERAGE':<12} {total_return/count:>9.2f}% {total_sharpe/count:>8.2f} "
              f"{total_wr/count:>7.1f}%")

    print(f"\n{'=' * 70}")

    if signals_json:
        _write_signals_json(signals_json, signals_payload, STRATEGY_NAME)

    return results


# ── Walk-forward mode ─────────────────────────────────────────


def main_walk_forward(
    symbols: list = None,
    dip_threshold: float = None,
):
    """Run walk-forward validation."""
    from walk_forward import run_multi_symbol, analyze_results

    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    scores = run_multi_symbol(
        symbols=symbols,
        dip_threshold=dip_threshold,
    )

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Buy-the-Dip Trading System"
    )
    parser.add_argument("--timeframe", default=TIMEFRAME,
                        help=f"Candle timeframe (default: {TIMEFRAME})")
    parser.add_argument("--days", type=int, default=DAYS,
                        help=f"Days of history (default: {DAYS})")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Dip threshold (e.g., -0.07 for 7%%)")
    parser.add_argument("--scan", action="store_true",
                        help="Multi-crypto scanner mode")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols for scan/walk-forward mode")
    parser.add_argument("--signals-json", default=None,
                        help="Write current signals to JSON file")

    args = parser.parse_args()

    if args.walk_forward:
        main_walk_forward(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            dip_threshold=args.threshold,
        )
    elif args.scan:
        main_scan(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            timeframe=args.timeframe,
            days=args.days,
            dip_threshold=args.threshold,
            signals_json=args.signals_json,
        )
    else:
        parser.error("This strategy only supports scan mode. Use --scan.")
