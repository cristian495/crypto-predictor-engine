"""
Generate frontend-ready JSON artifacts for support_bounce_long.

Outputs:
- signals.latest.json
- backtest.latest.json
- walkforward.latest.json (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breakout_momentum.data_loader import fetch_ohlcv
from backtest import run_backtest
from config import (
    BASE_SYMBOL_PARAMS,
    DAYS,
    DEFAULT_SYMBOLS,
    FEE_PCT,
    INITIAL_CAPITAL,
    MATRIX_OVERRIDES_PATH,
    MAX_POSITIONS,
    POSITION_PCT,
    ROBUST_DAYS_LIST,
    ROBUST_DAYS_WEIGHTS,
    STATIC_SYMBOL_OVERRIDES,
    TIMEFRAME,
    WF_MIN_TRADES_FOR_PF,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
    get_symbol_params,
)
from strategy import build_current_signal_payload, generate_signals
from walk_forward import walk_forward_validation

SCHEMA_VERSION = "support_bounce_long.frontend.v1"
STRATEGY_NAME = "support_bounce_long"
SOURCE_NAME = "python"


def _to_iso(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC").isoformat() if value.tzinfo else value.tz_localize("UTC").isoformat()

    if isinstance(value, np.datetime64):
        return pd.to_datetime(value, utc=True).isoformat()

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()

    if isinstance(value, str):
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(parsed):
            return parsed.isoformat()
        return value

    if isinstance(value, (int, float)):
        try:
            if abs(value) > 9_999_999_999:
                return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except Exception:
            return ""

    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def _clean_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_obj(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_obj(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return _to_iso(value)
    if isinstance(value, np.datetime64):
        return _to_iso(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _quality_label(stability_score: float) -> str:
    if stability_score >= 70:
        return "EXCELLENT"
    if stability_score >= 55:
        return "GOOD"
    if stability_score >= 40:
        return "MODERATE"
    return "POOR"


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
        overrides["confirmation_mode"] = str(args.confirmation_mode)

    if args.require_pullback:
        overrides["require_pullback"] = True
    if args.no_require_pullback:
        overrides["require_pullback"] = False

    if args.allow_candidates:
        overrides["require_confirmed"] = False

    return overrides


def _build_config_snapshot(runtime_overrides: dict[str, Any]) -> dict[str, Any]:
    return {
        "TIMEFRAME": TIMEFRAME,
        "DAYS": DAYS,
        "INITIAL_CAPITAL": INITIAL_CAPITAL,
        "POSITION_PCT": POSITION_PCT,
        "MAX_POSITIONS": MAX_POSITIONS,
        "FEE_PCT": FEE_PCT,
        "BASE_SYMBOL_PARAMS": BASE_SYMBOL_PARAMS,
        "STATIC_SYMBOL_OVERRIDES": STATIC_SYMBOL_OVERRIDES,
        "MATRIX_OVERRIDES_PATH": str(MATRIX_OVERRIDES_PATH),
        "WF_TRAIN_MONTHS": WF_TRAIN_MONTHS,
        "WF_TEST_MONTHS": WF_TEST_MONTHS,
        "WF_STEP_MONTHS": WF_STEP_MONTHS,
        "WF_MIN_TRADES_FOR_PF": WF_MIN_TRADES_FOR_PF,
        "DEFAULT_SYMBOLS": DEFAULT_SYMBOLS,
        "ROBUST_DAYS_LIST": ROBUST_DAYS_LIST,
        "ROBUST_DAYS_WEIGHTS": ROBUST_DAYS_WEIGHTS,
        "runtimeOverrides": runtime_overrides,
    }


def _build_historical_signals(df: pd.DataFrame) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        signal = "LONG" if str(row.get("signal", "NO_SIGNAL")) == "LONG" else "NO_SIGNAL"
        setup = row.get("setup_data") if isinstance(row.get("setup_data"), dict) else None

        history.append(
            {
                "timestamp": _to_iso(row.get("timestamp")),
                "signal": signal,
                "close": _safe_float(row.get("close")),
                "setup_status": row.get("setup_status"),
                "setup_id": row.get("setup_id"),
                "rr": _safe_optional_float(setup.get("rr")) if setup else None,
                "total_score": _safe_optional_float(setup.get("total_score")) if setup else None,
                "blocked_by": row.get("blocked_by"),
                "summary_text": row.get("summary_text"),
            }
        )

    return history


def _build_signals_artifact(
    run_id: str,
    generated_at: str,
    signal_items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "strategy": STRATEGY_NAME,
        "source": SOURCE_NAME,
        "runId": run_id,
        "generatedAt": generated_at,
        "symbols": signal_items,
    }


def _build_backtest_artifact(
    run_id: str,
    generated_at: str,
    symbol_results: list[dict[str, Any]],
    candle_cap: int,
    runtime_overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "strategy": STRATEGY_NAME,
        "source": SOURCE_NAME,
        "runId": run_id,
        "generatedAt": generated_at,
        "candleCap": candle_cap,
        "configSnapshot": _build_config_snapshot(runtime_overrides),
        "symbolResults": symbol_results,
    }


def _build_walkforward_artifact(
    run_id: str,
    generated_at: str,
    symbol_results: list[dict[str, Any]],
    train_months: int,
    test_months: int,
    step_months: int,
    days: int,
) -> dict[str, Any]:
    statuses = [item.get("status") for item in symbol_results]
    if statuses and all(status == "available" for status in statuses):
        global_status = "available"
    elif statuses and all(status == "not_available" for status in statuses):
        global_status = "not_available"
    else:
        global_status = "partial"

    return {
        "schemaVersion": SCHEMA_VERSION,
        "strategy": STRATEGY_NAME,
        "source": SOURCE_NAME,
        "runId": run_id,
        "generatedAt": generated_at,
        "status": global_status,
        "config": {
            "trainMonths": train_months,
            "testMonths": test_months,
            "stepMonths": step_months,
            "days": days,
        },
        "symbolResults": symbol_results,
    }


def _process_symbol(
    symbol: str,
    timeframe: str,
    days: int,
    candle_cap: int,
    runtime_overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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

    last = with_signals.iloc[-1]
    current_signal = build_current_signal_payload(last)

    signals_item = {
        "symbol": symbol,
        **_clean_obj(current_signal),
    }

    capped = with_signals.tail(max(20, int(candle_cap))).copy()

    trades_df = bt.get("trades")
    trades: list[dict[str, Any]] = []
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        for _, row in trades_df.iterrows():
            trades.append(
                {
                    "entry_time": _to_iso(row.get("entry_time")),
                    "exit_time": _to_iso(row.get("exit_time")),
                    "entry_price": _safe_float(row.get("entry_price")),
                    "exit_price": _safe_float(row.get("exit_price")),
                    "pnl_pct": _safe_float(row.get("pnl_pct")),
                    "pnl_after_fees": _safe_float(row.get("pnl_after_fees")),
                    "exit_reason": row.get("exit_reason") or "timeout",
                    "duration_bars": int(_safe_float(row.get("duration_bars"), default=0)),
                    "setup_id": row.get("setup_id"),
                    "entry_setup": _clean_obj(row.get("entry_setup")),
                }
            )

    equity_curve_values = bt.get("equity_curve")
    equity_timestamps = bt.get("equity_timestamps")
    equity_curve: list[dict[str, Any]] = []

    if equity_curve_values is not None and equity_timestamps is not None:
        for idx, equity in enumerate(equity_curve_values):
            ts_value = ""
            if idx < len(equity_timestamps):
                ts_value = _to_iso(equity_timestamps[idx])
            elif len(equity_timestamps) > 0:
                ts_value = _to_iso(equity_timestamps[-1])

            equity_curve.append(
                {
                    "timestamp": ts_value,
                    "equity": _safe_float(equity),
                }
            )

    supports = with_signals.attrs.get("supports") or []
    resistances = with_signals.attrs.get("resistances") or []

    symbol_backtest = {
        "symbol": symbol,
        "signalParams": _clean_obj(params),
        "metrics": _clean_obj(bt.get("metrics", {})),
        "candles": [
            {
                "timestamp": _to_iso(row.get("timestamp")),
                "open": _safe_float(row.get("open")),
                "high": _safe_float(row.get("high")),
                "low": _safe_float(row.get("low")),
                "close": _safe_float(row.get("close")),
                "volume": _safe_optional_float(row.get("volume")),
            }
            for _, row in capped.iterrows()
        ],
        "historicalSignals": _build_historical_signals(capped),
        "equityCurve": equity_curve,
        "trades": trades,
        "currentSignal": _clean_obj(current_signal),
        "supports": _clean_obj(supports[:20]),
        "resistances": _clean_obj(resistances[:20]),
    }

    return signals_item, symbol_backtest


def _build_walkforward_symbol_result(
    symbol: str,
    train_months: int,
    test_months: int,
    step_months: int,
    days: int,
    min_trades_for_pf: int,
    runtime_overrides: dict[str, Any],
) -> dict[str, Any]:
    params = get_symbol_params(symbol, overrides=runtime_overrides)

    rows = walk_forward_validation(
        symbol=symbol,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        total_days=days,
        symbol_params=params,
        timeframe=TIMEFRAME,
        verbose=False,
    )

    if not rows:
        return {
            "symbol": symbol,
            "status": "not_available",
            "summary": "Walkforward no disponible para este símbolo (sin ventanas válidas).",
            "segments": [],
        }

    segments = [
        {
            "segmentId": f"window-{int(row.get('window', idx + 1))}",
            "window": int(row.get("window", idx + 1)),
            "trainStart": _to_iso(row.get("train_start")),
            "trainEnd": _to_iso(row.get("train_end")),
            "testStart": _to_iso(row.get("test_start")),
            "testEnd": _to_iso(row.get("test_end")),
            "totalReturnPct": _safe_float(row.get("total_return")),
            "sharpe": _safe_float(row.get("sharpe")),
            "maxDrawdownPct": _safe_float(row.get("max_dd")),
            "winRatePct": _safe_float(row.get("win_rate")),
            "profitFactor": _safe_float(row.get("profit_factor")),
            "trades": int(_safe_float(row.get("num_trades"), default=0)),
            "expectancyPct": _safe_float(row.get("expectancy")),
            "signals": int(_safe_float(row.get("signals"), default=0)),
        }
        for idx, row in enumerate(rows)
    ]

    windows_tested = len(segments)
    positive_windows = sum(1 for row in segments if row["totalReturnPct"] > 0)
    sharpe_positive = sum(1 for row in segments if row["sharpe"] > 0)
    pf_positive = sum(1 for row in segments if row["profitFactor"] > 1 and row["trades"] >= min_trades_for_pf)

    positive_windows_pct = (positive_windows / windows_tested) * 100 if windows_tested else 0
    sharpe_pct = (sharpe_positive / windows_tested) * 100 if windows_tested else 0
    pf_pct = (pf_positive / windows_tested) * 100 if windows_tested else 0

    stability_score = positive_windows_pct * 0.4 + sharpe_pct * 0.3 + pf_pct * 0.3
    quality = _quality_label(stability_score)

    return {
        "symbol": symbol,
        "status": "available",
        "stabilityScore": round(stability_score, 2),
        "positiveWindowsPct": round(positive_windows_pct, 2),
        "summary": f"{quality}: {positive_windows}/{windows_tested} ventanas con retorno positivo.",
        "segments": segments,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export support_bounce_long JSON artifacts for frontend")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--candle-cap", type=int, default=1200)
    parser.add_argument("--include-walkforward", action="store_true")
    parser.add_argument("--train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument("--min-trades", type=int, default=WF_MIN_TRADES_FOR_PF)

    parser.add_argument("--tolerance-pct", type=float, default=None)
    parser.add_argument("--min-touches", type=int, default=None)
    parser.add_argument("--pivot-window", type=int, default=None)
    parser.add_argument("--support-proximity-pct", type=float, default=None)
    parser.add_argument("--rejection-threshold", type=float, default=None)
    parser.add_argument("--min-rr", type=float, default=None)
    parser.add_argument("--stop-buffer-pct", type=float, default=None)
    parser.add_argument("--confirmation-mode", default=None)
    parser.add_argument("--require-pullback", action="store_true")
    parser.add_argument("--no-require-pullback", action="store_true")
    parser.add_argument("--pullback-lookback", type=int, default=None)
    parser.add_argument("--hold-timeout-bars", type=int, default=None)
    parser.add_argument("--allow-candidates", action="store_true")

    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "exports" / "support_bounce_long"),
    )

    args = parser.parse_args()

    runtime_overrides = _build_runtime_overrides(args)
    symbols = args.symbols or DEFAULT_SYMBOLS

    run_id = f"sbl-{uuid.uuid4().hex[:12]}"
    generated_at = datetime.now(timezone.utc).isoformat()

    print(f"\n{'=' * 72}")
    print("  SUPPORT_BOUNCE_LONG FRONTEND EXPORT")
    print(f"{'=' * 72}")
    print(f"  Symbols:     {len(symbols)}")
    print(f"  Timeframe:   {args.timeframe}")
    print(f"  Days:        {args.days}")
    print(f"  Candle cap:  {args.candle_cap}")
    print(f"  Walkforward: {'on' if args.include_walkforward else 'off'}")
    print(f"  Overrides:   {runtime_overrides if runtime_overrides else 'none'}")
    print(f"  Out dir:     {args.out_dir}")

    signal_items: list[dict[str, Any]] = []
    backtest_results: list[dict[str, Any]] = []

    for symbol in symbols:
        try:
            print(f"Processing {symbol}...")
            signal_item, symbol_backtest = _process_symbol(
                symbol=symbol,
                timeframe=args.timeframe,
                days=args.days,
                candle_cap=args.candle_cap,
                runtime_overrides=runtime_overrides,
            )
            signal_items.append(signal_item)
            backtest_results.append(symbol_backtest)
            print(
                f"  done: signal={signal_item['signal']} "
                f"trades={symbol_backtest.get('metrics', {}).get('total_trades', 0)}"
            )
        except Exception as exc:
            print(f"  error on {symbol}: {exc}")

    if not backtest_results:
        raise RuntimeError("No symbol data exported. Aborting.")

    out_dir = Path(args.out_dir)
    signals_payload = _build_signals_artifact(run_id=run_id, generated_at=generated_at, signal_items=signal_items)
    backtest_payload = _build_backtest_artifact(
        run_id=run_id,
        generated_at=generated_at,
        symbol_results=backtest_results,
        candle_cap=args.candle_cap,
        runtime_overrides=runtime_overrides,
    )

    signals_path = out_dir / "signals.latest.json"
    backtest_path = out_dir / "backtest.latest.json"

    _write_json(signals_path, signals_payload)
    _write_json(backtest_path, backtest_payload)

    print(f"\nWrote {signals_path}")
    print(f"Wrote {backtest_path}")

    if args.include_walkforward:
        wf_results: list[dict[str, Any]] = []
        for symbol in symbols:
            try:
                print(f"Walkforward {symbol}...")
                wf_results.append(
                    _build_walkforward_symbol_result(
                        symbol=symbol,
                        train_months=args.train_months,
                        test_months=args.test_months,
                        step_months=args.step_months,
                        days=args.days,
                        min_trades_for_pf=args.min_trades,
                        runtime_overrides=runtime_overrides,
                    )
                )
            except Exception as exc:
                print(f"  walkforward error on {symbol}: {exc}")
                wf_results.append(
                    {
                        "symbol": symbol,
                        "status": "not_available",
                        "summary": f"Walkforward error: {exc}",
                        "segments": [],
                    }
                )

        wf_payload = _build_walkforward_artifact(
            run_id=run_id,
            generated_at=generated_at,
            symbol_results=wf_results,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            days=args.days,
        )

        wf_path = out_dir / "walkforward.latest.json"
        _write_json(wf_path, wf_payload)
        print(f"Wrote {wf_path}")

    print("\nExport complete.")


if __name__ == "__main__":
    main()
