"""
CLI orchestrator for Sell-the-Rip strategy.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "..")

import pandas as pd

from breakout_momentum.data_loader import fetch_ohlcv
from backtest import run_backtest
from config import (
    DAYS,
    DEFAULT_SYMBOLS,
    FEE_PCT,
    INITIAL_CAPITAL,
    MAX_POSITIONS,
    MATRIX_THRESHOLDS,
    MATRIX_MAX_CONFIGS_PER_SYMBOL,
    POSITION_PCT,
    TIMEFRAME,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
    get_symbol_signal_params,
)
from matrix_test import run_matrix
from strategy import add_rip_features, compute_exit_levels, generate_signals
from walk_forward import run_multi_symbol

STRATEGY_NAME = Path(__file__).resolve().parent.name


def _write_signals_json(path: str, signals: list, strategy: str):
    payload = {
        "strategy": strategy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)


def _build_signal_payload(df: pd.DataFrame, symbol: str) -> dict | None:
    last = df.iloc[-1]
    signal = last.get("signal", "NO TRADE")
    if signal != "SHORT":
        return None

    params = get_symbol_signal_params(symbol)
    take_profit_pct = float(params["take_profit_pct"])
    stop_loss_pct = float(params["stop_loss_pct"])

    past_ret = float(last.get("past_return", 0)) * 100
    rsi = float(last.get("rsi_14", 0))
    entry = float(last.get("close", 0))
    levels = compute_exit_levels(entry, take_profit_pct, stop_loss_pct)

    return {
        "symbol": symbol,
        "signal": "SHORT",
        "prob": 0.5,
        "extra": (
            f"Rip24h={past_ret:+.2f}% RSI={rsi:.1f} "
            f"Entry={entry:.4f} TP={levels['take_profit']:.4f} SL={levels['stop_loss']:.4f}"
        ),
    }


def _evaluate_filters(last: pd.Series, params: dict) -> tuple[list[tuple[str, bool, str]], str]:
    checks: list[tuple[str, bool, str]] = []
    rip_threshold = float(params["rip_threshold"])
    use_trend_filter = bool(params["use_trend_filter"])
    allow_countertrend_reversal = bool(params.get("allow_countertrend_reversal", False))
    countertrend_min_drop_1h = float(params.get("countertrend_min_drop_1h", 0.01))
    countertrend_min_pullback_pct = float(params.get("countertrend_min_pullback_pct", 0.012))

    past_ret = last.get("past_return", float("nan"))
    past_ret_recent_max = last.get("past_return_recent_max", float("nan"))
    ret_1h = last.get("ret_1h", float("nan"))
    pullback = last.get("pullback_from_recent_high", float("nan"))
    rel_vol = last.get("relative_volume", float("nan"))
    rsi = last.get("rsi_14", float("nan"))
    trend_bearish = bool(last.get("trend_bearish", False))

    rip_now = pd.notna(past_ret) and float(past_ret) > rip_threshold
    rip_recent = pd.notna(past_ret_recent_max) and float(past_ret_recent_max) > rip_threshold
    reversal_ok = (
        pd.notna(ret_1h)
        and pd.notna(pullback)
        and float(ret_1h) <= -countertrend_min_drop_1h
        and float(pullback) <= -countertrend_min_pullback_pct
    )

    trend_path_ok = rip_now
    countertrend_path_ok = (
        use_trend_filter
        and (not trend_bearish)
        and allow_countertrend_reversal
        and rip_recent
        and reversal_ok
    )
    if use_trend_filter:
        entry_ok = trend_path_ok if trend_bearish else countertrend_path_ok
    else:
        entry_ok = trend_path_ok

    checks.append(
        (
            "entry_gate",
            bool(entry_ok),
            f"trend_bearish={trend_bearish}, "
            f"rip_now={(float(past_ret)*100):+.2f}%>{rip_threshold*100:.2f}%, "
            f"rip_recent={(float(past_ret_recent_max)*100):+.2f}%>{rip_threshold*100:.2f}%, "
            f"ret_1h={(float(ret_1h)*100):+.2f}%<=-{countertrend_min_drop_1h*100:.2f}%, "
            f"pullback={(float(pullback)*100):+.2f}%<=-{countertrend_min_pullback_pct*100:.2f}%"
            if pd.notna(past_ret) and pd.notna(past_ret_recent_max) and pd.notna(ret_1h) and pd.notna(pullback)
            else "missing rip/reversal fields",
        )
    )

    if params["use_volume_filter"]:
        min_volume_mult = float(params["min_volume_mult"])
        vol_ok = pd.notna(rel_vol) and float(rel_vol) >= min_volume_mult
        checks.append(
            (
                "volume_filter",
                bool(vol_ok),
                (
                    f"rel_vol={float(rel_vol):.2f} >= {min_volume_mult:.2f}"
                    if pd.notna(rel_vol)
                    else "rel_vol=NaN"
                ),
            )
        )
    else:
        checks.append(("volume_filter", True, "disabled"))

    if params["use_rsi_filter"]:
        rsi_min = float(params["rsi_min"])
        rsi_max = float(params["rsi_max"])
        rsi_ok = pd.notna(rsi) and rsi_min <= float(rsi) <= rsi_max
        checks.append(
            (
                "rsi_filter",
                bool(rsi_ok),
                (
                    f"rsi_14={float(rsi):.2f} in [{rsi_min:.1f}, {rsi_max:.1f}]"
                    if pd.notna(rsi)
                    else "rsi_14=NaN"
                ),
            )
        )
    else:
        checks.append(("rsi_filter", True, "disabled"))

    if use_trend_filter:
        checks.append(
            (
                "trend_policy",
                bool(trend_bearish or allow_countertrend_reversal),
                f"trend_bearish={trend_bearish}, allow_countertrend_reversal={allow_countertrend_reversal}",
            )
        )
    else:
        checks.append(("trend_filter", True, "disabled"))

    blocking = "none"
    for name, ok, detail in checks:
        if not ok:
            blocking = f"{name} ({detail})"
            break
    return checks, blocking


def print_current_signal(df: pd.DataFrame, symbol: str, params: dict):
    last = df.iloc[-1]
    signal = last.get("signal", "NO TRADE")
    past_ret = float(last.get("past_return", 0)) * 100

    if signal == "SHORT":
        entry = float(last["close"])
        take_profit_pct = float(params["take_profit_pct"])
        stop_loss_pct = float(params["stop_loss_pct"])
        levels = compute_exit_levels(entry, take_profit_pct, stop_loss_pct)

        print(f"  {symbol:<12} SELL SIGNAL!")
        print(f"               24h Pump: {past_ret:+.2f}%")
        print(f"               Entry:    ${entry:,.4f}")
        print(f"               TP:       ${levels['take_profit']:,.4f} (-{take_profit_pct*100:.1f}%)")
        print(f"               SL:       ${levels['stop_loss']:,.4f} (+{stop_loss_pct*100:.1f}%)")
    else:
        print(f"  {symbol:<12} NO SIGNAL  (24h: {past_ret:+.2f}%, need >+{params['rip_threshold']*100:.0f}%)")

    checks, blocking = _evaluate_filters(last, params)
    print("               Conditions:")
    for name, ok, detail in checks:
        state = "PASS" if ok else "FAIL"
        print(f"                 - {name:<13} {state:<4} | {detail}")
    if signal != "SHORT":
        print(f"               Blocked by: {blocking}")


def main_scan(
    symbols=None,
    timeframe: str = TIMEFRAME,
    days: int = DAYS,
    threshold: float = None,
    signals_json: str | None = None,
):
    symbols = symbols or DEFAULT_SYMBOLS

    sample_params = get_symbol_signal_params(symbols[0]) if symbols else get_symbol_signal_params("SUI/USDT")

    print("=" * 70)
    print("  SELL-THE-RIP — MULTI-CRYPTO SCANNER")
    print("=" * 70)
    print("  Mode:          Scanner")
    print(f"  Symbols:       {len(symbols)}")
    print(f"  Timeframe:     {timeframe}")
    print(f"  History:       {days} days")
    print(f"  Rip threshold: {'global override' if threshold is not None else 'per-symbol override'}")
    print(f"  Take Profit:   {float(sample_params['take_profit_pct'])*100:.1f}%")
    print(f"  Stop Loss:     {float(sample_params['stop_loss_pct'])*100:.1f}%")
    print(f"  Hold timeout:  {int(sample_params['hold_timeout_hours'])}h")
    print(f"  Position pct:  {POSITION_PCT*100:.1f}%")
    print(f"  Max positions: {MAX_POSITIONS}")
    print(f"  Fees:          {FEE_PCT:.2%} per side")
    print(
        "  Countertrend: "
        f"{'on' if bool(sample_params.get('allow_countertrend_reversal', False)) else 'off'} "
        f"(lookback={int(sample_params.get('countertrend_rip_lookback_hours', 8))}h, "
        f"drop1h>={float(sample_params.get('countertrend_min_drop_1h', 0.01))*100:.1f}%, "
        f"pullback>={float(sample_params.get('countertrend_min_pullback_pct', 0.012))*100:.1f}%)"
    )
    print("=" * 70)

    print("\n[1/3] Processing symbols...")
    results = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(symbols)}] {symbol}")
        print("─" * 60)

        try:
            signal_params = get_symbol_signal_params(symbol, rip_threshold_override=threshold)
            df = fetch_ohlcv(symbol, timeframe, days)
            df = add_rip_features(df)
            df = generate_signals(
                df,
                rip_threshold=float(signal_params["rip_threshold"]),
                use_volume_filter=bool(signal_params["use_volume_filter"]),
                use_rsi_filter=bool(signal_params["use_rsi_filter"]),
                use_trend_filter=bool(signal_params["use_trend_filter"]),
                allow_countertrend_reversal=bool(signal_params.get("allow_countertrend_reversal", False)),
                countertrend_rip_lookback_hours=int(signal_params.get("countertrend_rip_lookback_hours", 8)),
                countertrend_min_drop_1h=float(signal_params.get("countertrend_min_drop_1h", 0.01)),
                countertrend_min_pullback_pct=float(signal_params.get("countertrend_min_pullback_pct", 0.012)),
                min_volume_mult=float(signal_params["min_volume_mult"]),
                rsi_min=float(signal_params["rsi_min"]),
                rsi_max=float(signal_params["rsi_max"]),
            )

            n_signals = int((df["signal"] == "SHORT").sum())
            print(f"  Signals: {n_signals}")
            print(
                "  Config: "
                f"th=+{float(signal_params['rip_threshold'])*100:.1f}% "
                f"RSI={'on' if signal_params['use_rsi_filter'] else 'off'} "
                f"VOL={'on' if signal_params['use_volume_filter'] else 'off'} "
                f"TREND={'on' if signal_params['use_trend_filter'] else 'off'} "
                f"CT={'on' if signal_params.get('allow_countertrend_reversal', False) else 'off'} "
                f"TP={float(signal_params['take_profit_pct'])*100:.1f}% "
                f"SL={float(signal_params['stop_loss_pct'])*100:.1f}% "
                f"HOLD={int(signal_params['hold_timeout_hours'])}h "
                f"RSI=[{float(signal_params['rsi_min']):.0f},{float(signal_params['rsi_max']):.0f}]"
            )

            bt = run_backtest(
                df,
                initial_capital=INITIAL_CAPITAL,
                stop_loss_pct=float(signal_params["stop_loss_pct"]),
                take_profit_pct=float(signal_params["take_profit_pct"]),
                hold_timeout=int(signal_params["hold_timeout_hours"]),
                fee_pct=FEE_PCT,
            )
            m = bt["metrics"]

            print(
                f"  Return: {m['total_return']:.2%}  Sharpe: {m['sharpe']:.2f}  "
                f"WR: {m['win_rate']:.1%}  PF: {m['profit_factor']:.2f}  Trades: {m['total_trades']}"
            )

            results[symbol] = {"df": df, "metrics": m, "signal_params": signal_params}
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not results:
        print("No symbols processed successfully.")
        return None

    print("\n[2/3] Current signals...")
    print(f"\n{'=' * 70}")
    print("  CURRENT SIGNALS — Sell-the-Rip")
    print(f"{'=' * 70}")

    payload = []
    for symbol, data in results.items():
        print_current_signal(data["df"], symbol, data["signal_params"])
        sig = _build_signal_payload(data["df"], symbol)
        if sig:
            payload.append(sig)

    print(f"{'=' * 70}")

    print("\n[3/3] Summary...")
    print(f"\n{'=' * 70}")
    print("  BACKTEST SUMMARY — ALL SYMBOLS")
    print(f"{'=' * 70}")
    print(f"\n  {'Symbol':<12} {'Return':>10} {'Sharpe':>8} {'WR':>8} {'PF':>8} {'Trades':>8}")
    print(f"  {'-'*56}")

    total_return = 0.0
    total_sharpe = 0.0
    total_wr = 0.0

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

    count = len(results)
    if count:
        print(f"  {'-'*56}")
        print(f"  {'AVERAGE':<12} {total_return/count:>9.2f}% {total_sharpe/count:>8.2f} {total_wr/count:>7.1f}%")

    print(f"\n{'=' * 70}")

    if signals_json:
        _write_signals_json(signals_json, payload, STRATEGY_NAME)

    return results


def main_walk_forward(
    symbols=None,
    threshold: float = None,
    days: int = DAYS,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    min_trades: int = 5,
):
    return run_multi_symbol(
        symbols=symbols or DEFAULT_SYMBOLS,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        total_days=days,
        rip_threshold=threshold,
        min_trades_for_pf=min_trades,
    )


def main_matrix(
    symbols=None,
    days: int = DAYS,
    thresholds: list[float] | None = None,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    robust_days: list[int] | None = None,
    max_configs: int = MATRIX_MAX_CONFIGS_PER_SYMBOL,
    shortlist_top_k: int = 40,
):
    symbols = symbols or DEFAULT_SYMBOLS
    thresholds = thresholds or MATRIX_THRESHOLDS

    all_best = {}
    out_dir = Path(__file__).resolve().parent / "matrix_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        print(f"\n{'='*72}\nRunning matrix for {symbol}\n{'='*72}")
        df = run_matrix(
            symbol=symbol,
            days=days,
            thresholds=thresholds,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            robust_days=robust_days,
            max_configs=max_configs,
            shortlist_top_k=shortlist_top_k,
        )

        safe = symbol.replace("/", "_")
        csv_path = out_dir / f"{safe}_matrix.csv"
        json_path = out_dir / f"{safe}_matrix.json"
        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=True, indent=2)

        if len(df):
            best = df.iloc[0].to_dict()
            all_best[symbol] = {
                "rip_threshold": float(best["threshold"]),
                "use_rsi_filter": bool(best["use_rsi_filter"]),
                "use_volume_filter": bool(best["use_volume_filter"]),
                "use_trend_filter": bool(best["use_trend_filter"]),
                "rsi_min": float(best["rsi_min"]),
                "rsi_max": float(best["rsi_max"]),
                "min_volume_mult": float(best["min_volume_mult"]),
                "take_profit_pct": float(best["take_profit_pct"]),
                "stop_loss_pct": float(best["stop_loss_pct"]),
                "hold_timeout_hours": int(best["hold_timeout_hours"]),
                "score": float(best["score"]),
            }

    if all_best:
        out_best = Path(__file__).resolve().parent / "matrix_overrides.json"
        merged_overrides = {}
        if out_best.exists():
            try:
                with open(out_best, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    merged_overrides.update(existing)
            except Exception:
                pass

        # Update only optimized symbols and keep previous overrides intact.
        merged_overrides.update(all_best)

        with open(out_best, "w", encoding="utf-8") as f:
            json.dump(merged_overrides, f, ensure_ascii=True, indent=2)
        print(
            f"\nSaved per-symbol best overrides: {out_best} "
            f"(updated {len(all_best)} symbol(s), total {len(merged_overrides)})"
        )

    return all_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sell-the-Rip strategy")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--signals-json", default=None)
    parser.add_argument("--train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--thresholds", default=",".join(str(x) for x in MATRIX_THRESHOLDS))
    parser.add_argument("--robust-days", default="")
    parser.add_argument("--max-configs", type=int, default=MATRIX_MAX_CONFIGS_PER_SYMBOL)
    parser.add_argument("--shortlist-top-k", type=int, default=40)
    args = parser.parse_args()

    if args.scan:
        main_scan(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            timeframe=args.timeframe,
            days=args.days,
            threshold=args.threshold,
            signals_json=args.signals_json,
        )
    elif args.walk_forward:
        main_walk_forward(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            threshold=args.threshold,
            days=args.days,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            min_trades=args.min_trades,
        )
    elif args.matrix:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
        robust_days = [int(x.strip()) for x in args.robust_days.split(",") if x.strip()] if args.robust_days else None
        main_matrix(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            days=args.days,
            thresholds=thresholds,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            robust_days=robust_days,
            max_configs=args.max_configs,
            shortlist_top_k=args.shortlist_top_k,
        )
    else:
        main_scan(
            symbols=args.symbols or DEFAULT_SYMBOLS,
            timeframe=args.timeframe,
            days=args.days,
            threshold=args.threshold,
            signals_json=args.signals_json,
        )
