"""
CLI orchestrator for support_bounce_long strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breakout_momentum.data_loader import fetch_ohlcv
from backtest import run_backtest
from config import (
    DAYS,
    DEFAULT_SYMBOLS,
    FEE_PCT,
    INITIAL_CAPITAL,
    MATRIX_MAX_CONFIGS_PER_SYMBOL,
    MATRIX_OVERRIDES_PATH,
    MATRIX_RUNS_DIR,
    MAX_POSITIONS,
    POSITION_PCT,
    ROBUST_DAYS_LIST,
    ROBUST_DAYS_WEIGHTS,
    TIMEFRAME,
    VALID_CONFIRMATION_MODES,
    WF_MIN_TRADES_FOR_PF,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
    get_symbol_params,
)
from matrix_test import run_matrix
from strategy import build_current_signal_payload, generate_signals
from walk_forward import run_multi_symbol

STRATEGY_NAME = Path(__file__).resolve().parent.name


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def _write_signals_json(path: str, signals: list[dict[str, Any]], strategy: str) -> None:
    payload = {
        "strategy": strategy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)


def _build_runtime_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    mapping = {
        "tolerance_pct": args.tolerance_pct,
        "min_touches": args.min_touches,
        "pivot_window": args.pivot_window,
        "support_proximity_pct": args.support_proximity_pct,
        "rejection_threshold": args.rejection_threshold,
        "min_rr": args.min_rr,
        "stop_buffer_pct": args.stop_buffer_pct,
        "pullback_lookback": args.pullback_lookback,
        "hold_timeout_bars": args.hold_timeout_bars,
    }

    for key, value in mapping.items():
        if value is not None:
            overrides[key] = value

    if args.confirmation_mode is not None:
        mode = str(args.confirmation_mode).strip().lower()
        if mode in VALID_CONFIRMATION_MODES:
            overrides["confirmation_mode"] = mode

    if args.require_pullback:
        overrides["require_pullback"] = True
    if args.no_require_pullback:
        overrides["require_pullback"] = False

    if args.allow_candidates:
        overrides["require_confirmed"] = False

    return overrides


def _build_signal_payload(last_row: pd.Series, symbol: str) -> dict[str, Any] | None:
    current = build_current_signal_payload(last_row)
    if current["signal"] != "LONG":
        return None

    setup = current.get("setup") or {}
    rr = float(setup.get("rr", 0.0)) if isinstance(setup, dict) else 0.0

    return {
        "symbol": symbol,
        "signal": "LONG",
        "prob": 0.5,
        "extra": (
            f"Entry={float(current.get('entry_price') or 0):.6f} "
            f"TP={float(current.get('take_profit') or 0):.6f} "
            f"SL={float(current.get('stop_loss') or 0):.6f} "
            f"RR={rr:.2f}"
        ),
    }


def print_current_signal(row: pd.Series, symbol: str, params: dict[str, Any]) -> None:
    payload = build_current_signal_payload(row)

    if payload["signal"] == "LONG":
        setup = payload.get("setup") or {}
        print(f"  {symbol:<12} LONG SIGNAL")
        print(f"               Entry: {float(payload.get('entry_price') or 0):,.6f}")
        print(f"               TP:    {float(payload.get('take_profit') or 0):,.6f}")
        print(f"               SL:    {float(payload.get('stop_loss') or 0):,.6f}")
        print(
            f"               RR:    {float(setup.get('rr', 0.0)):.2f}x | "
            f"Score: {float(setup.get('total_score', 0.0)):.1f}"
        )
    else:
        print(f"  {symbol:<12} NO SIGNAL")

    print("               Conditions:")
    for cond in payload.get("conditions") or []:
        state = "PASS" if cond.get("passed") else "FAIL"
        print(f"                 - {str(cond.get('name')):<20} {state:<4} | {cond.get('detail')}")

    if payload["signal"] != "LONG":
        print(f"               Blocked by: {payload.get('blocked_by')}")

    print(f"               Summary: {payload.get('summary_text')}")
    print(
        "               Config: "
        f"tol={float(params['tolerance_pct']):.2f} touch={int(params['min_touches'])} "
        f"piv={int(params['pivot_window'])} prox={float(params['support_proximity_pct']):.2f} "
        f"rej={float(params['rejection_threshold']):.2f} rr>={float(params['min_rr']):.2f} "
        f"mode={params['confirmation_mode']} pull={'on' if params['require_pullback'] else 'off'} "
        f"hold={int(params['hold_timeout_bars'])}"
    )


def main_scan(
    symbols: list[str] | None = None,
    timeframe: str = TIMEFRAME,
    days: int = DAYS,
    runtime_overrides: dict[str, Any] | None = None,
    signals_json: str | None = None,
) -> dict[str, Any] | None:
    symbols = symbols or DEFAULT_SYMBOLS

    print(f"\n{'=' * 72}")
    print("  SUPPORT-BOUNCE-LONG — SCAN + BACKTEST")
    print(f"{'=' * 72}")
    print(f"  Symbols:   {len(symbols)}")
    print(f"  Timeframe: {timeframe}")
    print(f"  History:   {days} days")
    print(f"  Overrides: {runtime_overrides if runtime_overrides else 'none'}")

    results: dict[str, Any] = {}

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n{'-' * 64}")
        print(f"[{idx}/{len(symbols)}] {symbol}")
        print(f"{'-' * 64}")

        try:
            params = get_symbol_params(symbol, overrides=runtime_overrides)
            raw = fetch_ohlcv(symbol, timeframe=timeframe, days=days)
            with_signals = generate_signals(raw, params)

            bt = run_backtest(
                with_signals,
                initial_capital=INITIAL_CAPITAL,
                position_pct=POSITION_PCT,
                max_positions=MAX_POSITIONS,
                fee_pct=FEE_PCT,
                hold_timeout_bars=int(params["hold_timeout_bars"]),
            )
            m = bt["metrics"]

            n_signals = int((with_signals["signal"] == "LONG").sum())
            print(f"  Signals: {n_signals}")
            print(
                f"  Return: {float(m['total_return']) * 100:>7.2f}% | "
                f"Sharpe: {float(m['sharpe']):>6.2f} | "
                f"WR: {float(m['win_rate']) * 100:>6.1f}% | "
                f"PF: {float(m['profit_factor']):>5.2f} | "
                f"Trades: {int(m['total_trades'])}"
            )

            results[symbol] = {
                "df": with_signals,
                "metrics": m,
                "backtest": bt,
                "params": params,
            }
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not results:
        print("No symbols processed successfully.")
        return None

    print(f"\n{'=' * 72}")
    print("  CURRENT SIGNALS")
    print(f"{'=' * 72}")

    payload: list[dict[str, Any]] = []
    for symbol, data in results.items():
        last_row = data["df"].iloc[-1]
        print_current_signal(last_row, symbol, data["params"])

        signal = _build_signal_payload(last_row, symbol)
        if signal:
            payload.append(signal)

    print(f"\n{'=' * 72}")
    print("  BACKTEST SUMMARY")
    print(f"{'=' * 72}")
    print(f"\n  {'Symbol':<12} {'Return':>10} {'Sharpe':>8} {'WR':>8} {'PF':>8} {'Trades':>8}")
    print(f"  {'-' * 56}")

    total_return = 0.0
    total_sharpe = 0.0
    total_wr = 0.0

    for symbol, data in results.items():
        m = data["metrics"]
        ret = float(m["total_return"]) * 100
        sharpe = float(m["sharpe"])
        wr = float(m["win_rate"]) * 100
        pf = float(m["profit_factor"])
        trades = int(m["total_trades"])

        print(f"  {symbol:<12} {ret:>9.2f}% {sharpe:>8.2f} {wr:>7.1f}% {pf:>8.2f} {trades:>8}")

        total_return += ret
        total_sharpe += sharpe
        total_wr += wr

    count = len(results)
    if count:
        print(f"  {'-' * 56}")
        print(f"  {'AVERAGE':<12} {total_return/count:>9.2f}% {total_sharpe/count:>8.2f} {total_wr/count:>7.1f}%")

    if signals_json:
        _write_signals_json(signals_json, payload, STRATEGY_NAME)
        print(f"\nSaved signals JSON: {signals_json}")

    return results


def main_walk_forward(
    symbols: list[str] | None = None,
    timeframe: str = TIMEFRAME,
    days: int = DAYS,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    min_trades: int = WF_MIN_TRADES_FOR_PF,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    return run_multi_symbol(
        symbols=symbols or DEFAULT_SYMBOLS,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        total_days=days,
        min_trades_for_pf=min_trades,
        timeframe=timeframe,
        runtime_overrides=runtime_overrides,
        verbose=True,
    )


def main_matrix(
    symbols: list[str] | None = None,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    robust_days: list[int] | None = None,
    robust_weights: list[float] | None = None,
    max_configs: int = MATRIX_MAX_CONFIGS_PER_SYMBOL,
) -> dict[str, dict[str, Any]]:
    symbols = symbols or DEFAULT_SYMBOLS
    robust_days_list = robust_days or list(ROBUST_DAYS_LIST)
    robust_weights_list = robust_weights or list(ROBUST_DAYS_WEIGHTS)

    MATRIX_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    all_best: dict[str, dict[str, Any]] = {}

    for symbol in symbols:
        print(f"\n{'=' * 72}\nRunning matrix for {symbol}\n{'=' * 72}")
        df = run_matrix(
            symbol=symbol,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            robust_days=robust_days_list,
            robust_weights=robust_weights_list,
            max_configs=max_configs,
        )

        safe = _safe_symbol(symbol)
        csv_path = MATRIX_RUNS_DIR / f"{safe}_matrix.csv"
        json_path = MATRIX_RUNS_DIR / f"{safe}_matrix.json"

        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=True, indent=2)

        if len(df) > 0:
            best = df.iloc[0].to_dict()
            all_best[symbol] = {
                "tolerance_pct": float(best["tolerance_pct"]),
                "min_touches": int(best["min_touches"]),
                "pivot_window": int(best["pivot_window"]),
                "support_proximity_pct": float(best["support_proximity_pct"]),
                "rejection_threshold": float(best["rejection_threshold"]),
                "min_rr": float(best["min_rr"]),
                "stop_buffer_pct": float(best["stop_buffer_pct"]),
                "confirmation_mode": str(best["confirmation_mode"]),
                "require_pullback": bool(best["require_pullback"]),
                "pullback_lookback": int(best["pullback_lookback"]),
                "hold_timeout_bars": int(best["hold_timeout_bars"]),
                "require_confirmed": True,
                "score": float(best["score"]),
            }

        print(f"Saved matrix: {csv_path} and {json_path}")

    merged: dict[str, Any] = {}
    if MATRIX_OVERRIDES_PATH.exists():
        try:
            with open(MATRIX_OVERRIDES_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                merged.update(existing)
        except Exception:
            pass

    merged.update(all_best)
    with open(MATRIX_OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=True, indent=2)

    print(
        f"\nSaved overrides: {MATRIX_OVERRIDES_PATH} "
        f"(updated {len(all_best)} symbol(s), total {len(merged)})"
    )

    return all_best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="support_bounce_long strategy")

    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--matrix", action="store_true")

    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--signals-json", default=None)

    parser.add_argument("--train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument("--min-trades", type=int, default=WF_MIN_TRADES_FOR_PF)

    parser.add_argument("--robust-days", default=",".join(str(x) for x in ROBUST_DAYS_LIST))
    parser.add_argument("--robust-weights", default=",".join(str(x) for x in ROBUST_DAYS_WEIGHTS))
    parser.add_argument("--max-configs", type=int, default=MATRIX_MAX_CONFIGS_PER_SYMBOL)

    parser.add_argument("--tolerance-pct", type=float, default=None)
    parser.add_argument("--min-touches", type=int, default=None)
    parser.add_argument("--pivot-window", type=int, default=None)
    parser.add_argument("--support-proximity-pct", type=float, default=None)
    parser.add_argument("--rejection-threshold", type=float, default=None)
    parser.add_argument("--min-rr", type=float, default=None)
    parser.add_argument("--stop-buffer-pct", type=float, default=None)
    parser.add_argument("--confirmation-mode", choices=sorted(VALID_CONFIRMATION_MODES), default=None)
    parser.add_argument("--require-pullback", action="store_true")
    parser.add_argument("--no-require-pullback", action="store_true")
    parser.add_argument("--pullback-lookback", type=int, default=None)
    parser.add_argument("--hold-timeout-bars", type=int, default=None)
    parser.add_argument("--allow-candidates", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runtime_overrides = _build_runtime_overrides(args)
    symbols = args.symbols or DEFAULT_SYMBOLS

    if args.scan:
        main_scan(
            symbols=symbols,
            timeframe=args.timeframe,
            days=args.days,
            runtime_overrides=runtime_overrides,
            signals_json=args.signals_json,
        )
    elif args.walk_forward:
        main_walk_forward(
            symbols=symbols,
            timeframe=args.timeframe,
            days=args.days,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            min_trades=args.min_trades,
            runtime_overrides=runtime_overrides,
        )
    elif args.matrix:
        robust_days = _parse_int_list(args.robust_days)
        robust_weights = _parse_float_list(args.robust_weights)
        main_matrix(
            symbols=symbols,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            robust_days=robust_days,
            robust_weights=robust_weights,
            max_configs=args.max_configs,
        )
    else:
        main_scan(
            symbols=symbols,
            timeframe=args.timeframe,
            days=args.days,
            runtime_overrides=runtime_overrides,
            signals_json=args.signals_json,
        )


if __name__ == "__main__":
    main()
