"""
export_frontend_data.py
Generate frontend-ready JSON artifacts for buy_the_dip strategy.

Outputs (latest fixed files):
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

# Ensure sibling strategy modules and project packages can be imported
# regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breakout_momentum.data_loader import fetch_ohlcv, fetch_funding_rates
from backtest import run_backtest, liquidation_price_long
from portfolio_backtest import run_portfolio_backtest
from config import (
    DAYS,
    DEFAULT_SYMBOLS,
    DIP_LOOKBACK_HOURS,
    DIP_THRESHOLD_PCT,
    DIP_THRESHOLD_STRONG,
    DIP_EVENT_RESET_PCT,
    FEE_PCT,
    FAIL_FAST_BARS,
    FAIL_FAST_MIN_REBOUND_PCT,
    HOLD_TIMEOUT_HOURS,
    INITIAL_CAPITAL,
    LONG_ONLY,
    SLIPPAGE_PCT,
    USE_FUNDING,
    FUNDING_INTERVAL_HOURS,
    FUNDING_RATE_FALLBACK,
    LEVERAGE,
    MAINTENANCE_MARGIN_RATE,
    MAX_POSITIONS,
    MIN_VOLUME_MULT,
    ONE_ENTRY_PER_DIP_EVENT,
    POSITION_PCT,
    RSI_MAX,
    RSI_MIN,
    STOP_LOSS_PCT,
    SYMBOL_SIGNAL_OVERRIDES,
    TAKE_PROFIT_PCT,
    TIMEFRAME,
    USE_RSI_FILTER,
    USE_STRONG_THRESHOLD,
    USE_VOLUME_FILTER,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
    get_symbol_signal_params,
)
from main import _evaluate_buy_filters
from strategy import (
    add_dip_features, compute_exit_levels, generate_signals,
    load_regime_if_enabled, apply_regime_filter,
)
from walk_forward import walk_forward_validation

SCHEMA_VERSION = "buy_the_dip.frontend.v2"
STRATEGY_NAME = "buy_the_dip"
SOURCE_NAME = "python"


def _to_iso(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC").isoformat() if value.tzinfo else value.tz_localize("UTC").isoformat()

    if isinstance(value, np.datetime64):
        ts = pd.to_datetime(value, utc=True)
        return ts.isoformat()

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()

    if isinstance(value, str):
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(parsed):
            return parsed.isoformat()
        return value

    if isinstance(value, (int, float)):
        try:
            # Heuristic: milliseconds if too large for seconds.
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


def _quality_label(stability_score: float) -> str:
    if stability_score >= 70:
        return "EXCELLENT"
    if stability_score >= 55:
        return "GOOD"
    if stability_score >= 40:
        return "MODERATE"
    return "POOR"


def _build_config_snapshot() -> dict[str, Any]:
    return {
        "TIMEFRAME": TIMEFRAME,
        "DAYS": DAYS,
        "DIP_THRESHOLD_PCT": DIP_THRESHOLD_PCT,
        "DIP_THRESHOLD_STRONG": DIP_THRESHOLD_STRONG,
        "USE_STRONG_THRESHOLD": USE_STRONG_THRESHOLD,
        "DIP_LOOKBACK_HOURS": DIP_LOOKBACK_HOURS,
        "TAKE_PROFIT_PCT": TAKE_PROFIT_PCT,
        "STOP_LOSS_PCT": STOP_LOSS_PCT,
        "HOLD_TIMEOUT_HOURS": HOLD_TIMEOUT_HOURS,
        "POSITION_PCT": POSITION_PCT,
        "MAX_POSITIONS": MAX_POSITIONS,
        "INITIAL_CAPITAL": INITIAL_CAPITAL,
        "FEE_PCT": FEE_PCT,
        "SLIPPAGE_PCT": SLIPPAGE_PCT,
        "USE_FUNDING": USE_FUNDING,
        "FUNDING_INTERVAL_HOURS": FUNDING_INTERVAL_HOURS,
        "FUNDING_RATE_FALLBACK": FUNDING_RATE_FALLBACK,
        "LEVERAGE": LEVERAGE,
        "MAINTENANCE_MARGIN_RATE": MAINTENANCE_MARGIN_RATE,
        "MIN_VOLUME_MULT": MIN_VOLUME_MULT,
        "RSI_MIN": RSI_MIN,
        "RSI_MAX": RSI_MAX,
        "USE_VOLUME_FILTER": USE_VOLUME_FILTER,
        "USE_RSI_FILTER": USE_RSI_FILTER,
        "LONG_ONLY": LONG_ONLY,
        "ONE_ENTRY_PER_DIP_EVENT": ONE_ENTRY_PER_DIP_EVENT,
        "DIP_EVENT_RESET_PCT": DIP_EVENT_RESET_PCT,
        "FAIL_FAST_BARS": FAIL_FAST_BARS,
        "FAIL_FAST_MIN_REBOUND_PCT": FAIL_FAST_MIN_REBOUND_PCT,
        "WF_TRAIN_MONTHS": WF_TRAIN_MONTHS,
        "WF_TEST_MONTHS": WF_TEST_MONTHS,
        "WF_STEP_MONTHS": WF_STEP_MONTHS,
        "DEFAULT_SYMBOLS": DEFAULT_SYMBOLS,
        "SYMBOL_SIGNAL_OVERRIDES": SYMBOL_SIGNAL_OVERRIDES,
    }


def _build_futures_reality(
    metrics: dict[str, Any],
    last_close: float,
    stop_loss_pct: float,
    leverage: float,
    sl_below_liquidation: bool,
) -> dict[str, Any]:
    """
    Resumen de la realidad de futuros para la UI: retorno bruto vs neto,
    desglose de costes (funding/slippage+fees), apalancamiento, precio de
    liquidacion vs SL y alerta si el SL cae por debajo de liquidacion.

    net_return es el total_return del backtest (ya neto de todos los costes).
    gross_return_sum / net_return_sum son sumas de % por trade (desglose),
    no el retorno compuesto; sirven para ver a donde se fue el dinero.
    """
    lev = leverage if leverage and leverage > 0 else 1.0
    liq_price = liquidation_price_long(last_close, lev) if last_close else 0.0
    sl_price = last_close * (1.0 - stop_loss_pct) if last_close else 0.0

    return {
        "leverage": lev,
        "netReturnPct": _safe_float(metrics.get("total_return")) * 100,
        "grossReturnSumPct": _safe_float(metrics.get("gross_return")) * 100,
        "netReturnSumPct": _safe_float(metrics.get("net_return_sum")) * 100,
        "fundingPaidPct": _safe_float(metrics.get("funding_paid_pct")) * 100,
        "slippageFeesPct": _safe_float(metrics.get("slippage_paid_pct")) * 100,
        "nLiquidations": int(_safe_float(metrics.get("n_liquidations"))),
        "liquidationPrice": liq_price,
        "stopLossPrice": sl_price,
        "slBelowLiquidation": bool(sl_below_liquidation),
        "lastClose": last_close,
    }


def _build_historical_signals(df: pd.DataFrame, dip_threshold: float) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        signal = row.get("signal", "NO TRADE")
        mapped_signal = "LONG" if signal == "LONG" else "NO_SIGNAL"
        past_ret = _safe_float(row.get("past_return"), default=0.0)

        history.append(
            {
                "timestamp": _to_iso(row.get("timestamp")),
                "signal": mapped_signal,
                "close": _safe_float(row.get("close")),
                "past_return": past_ret,
                "dip_threshold": dip_threshold,
                "dip_size": _safe_optional_float(row.get("dip_size")),
                "rsi_14": _safe_optional_float(row.get("rsi_14")),
                "relative_volume": _safe_optional_float(row.get("relative_volume")),
                "reason": "dip_threshold passed" if mapped_signal == "LONG" else "conditions not met",
            }
        )

    return history


def _build_current_signal(last: pd.Series, signal_params: dict[str, Any]) -> dict[str, Any]:
    checks, blocking = _evaluate_buy_filters(last, signal_params)
    is_long = last.get("signal", "NO TRADE") == "LONG"

    close = _safe_float(last.get("close"))
    past_return_pct = _safe_float(last.get("past_return")) * 100
    dip_threshold_pct = _safe_float(signal_params.get("dip_threshold")) * 100

    entry_price = close if is_long else None
    take_profit = None
    stop_loss = None

    if is_long:
        levels = compute_exit_levels(close, _safe_float(signal_params.get("take_profit_pct")), _safe_float(signal_params.get("stop_loss_pct"), STOP_LOSS_PCT))
        take_profit = _safe_float(levels.get("take_profit"))
        stop_loss = _safe_float(levels.get("stop_loss"))

    if is_long:
        summary = "BUY THE DIP candidate detected on last bar"
    else:
        failed_name = (blocking.split("(")[0].strip() if blocking and blocking != "none" else "conditions")
        summary = f"No trade: {failed_name} not satisfied"

    return {
        "signal": "LONG" if is_long else "NO_SIGNAL",
        "timestamp": _to_iso(last.get("timestamp")),
        "close": close,
        "past_return": past_return_pct,
        "dip_threshold": dip_threshold_pct,
        "rsi_14": _safe_optional_float(last.get("rsi_14")),
        "relative_volume": _safe_optional_float(last.get("relative_volume")),
        "summary_text": summary,
        "blocked_by": None if is_long else (blocking if blocking != "none" else "conditions"),
        "entry_price": entry_price,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "conditions": [
            {
                "name": name,
                "passed": bool(passed),
                "detail": detail,
            }
            for name, passed, detail in checks
        ],
    }


def _build_trade_entry_signal(entry_row: pd.Series | None, signal_params: dict[str, Any]) -> dict[str, Any]:
    dip_threshold_pct = _safe_float(signal_params.get("dip_threshold")) * 100

    if entry_row is None:
        return {
            "summary_text": "Entry context not found in source candles",
            "blocked_by": "entry_row_not_found",
            "past_return": 0.0,
            "dip_threshold": dip_threshold_pct,
            "rsi_14": None,
            "relative_volume": None,
            "dip_size": None,
            "conditions": [],
        }

    checks, blocking = _evaluate_buy_filters(entry_row, signal_params)
    blocked_by = None if blocking == "none" else blocking

    return {
        "summary_text": (
            "Entry accepted: all buy filters passed"
            if blocked_by is None
            else f"Entry warning: {blocking}"
        ),
        "blocked_by": blocked_by,
        "past_return": _safe_float(entry_row.get("past_return")) * 100,
        "dip_threshold": dip_threshold_pct,
        "rsi_14": _safe_optional_float(entry_row.get("rsi_14")),
        "relative_volume": _safe_optional_float(entry_row.get("relative_volume")),
        "dip_size": _safe_optional_float(entry_row.get("dip_size")),
        "conditions": [
            {
                "name": name,
                "passed": bool(passed),
                "detail": detail,
            }
            for name, passed, detail in checks
        ],
    }


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
) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "strategy": STRATEGY_NAME,
        "source": SOURCE_NAME,
        "runId": run_id,
        "generatedAt": generated_at,
        "candleCap": candle_cap,
        "configSnapshot": _build_config_snapshot(),
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
    dip_threshold_override: float | None,
    regime=None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    signal_params = get_symbol_signal_params(symbol, dip_threshold_override=dip_threshold_override)

    df = fetch_ohlcv(symbol, timeframe=timeframe, days=days)
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
    df = apply_regime_filter(df, regime)

    # Funding real de Binance futures (opcional; degrada a fallback si no hay datos)
    funding_series = None
    if USE_FUNDING:
        try:
            funding_df = fetch_funding_rates(symbol, days=days)
            if funding_df is not None and len(funding_df) > 0:
                funding_series = funding_df.set_index("timestamp")["funding_rate"].sort_index()
        except Exception as exc:
            print(f"  funding unavailable for {symbol}: {exc}")

    leverage = _safe_float(signal_params.get("leverage"), default=LEVERAGE) or LEVERAGE

    backtest = run_backtest(
        df,
        initial_capital=INITIAL_CAPITAL,
        stop_loss_pct=signal_params["stop_loss_pct"],
        take_profit_pct=signal_params["take_profit_pct"],
        hold_timeout=signal_params["hold_timeout_hours"],
        position_pct=POSITION_PCT,
        max_positions=signal_params["max_positions"],
        fee_pct=FEE_PCT,
        fail_fast_bars=signal_params["fail_fast_bars"],
        fail_fast_min_rebound_pct=signal_params["fail_fast_min_rebound_pct"],
        slippage_pct=SLIPPAGE_PCT,
        leverage=leverage,
        funding_series=funding_series,
    )

    last = df.iloc[-1]
    current_signal = _build_current_signal(last, signal_params)

    signals_item = {
        "symbol": symbol,
        **current_signal,
    }

    capped_df = df.tail(max(20, int(candle_cap))).copy()
    row_by_timestamp = {
        _to_iso(row.get("timestamp")): row
        for _, row in df.iterrows()
    }

    trades_df = backtest.get("trades")
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        trades = []
        for _, row in trades_df.iterrows():
            entry_time_iso = _to_iso(row.get("entry_time"))
            entry_row = row_by_timestamp.get(entry_time_iso)

            trades.append(
                {
                    "entry_time": entry_time_iso,
                    "exit_time": _to_iso(row.get("exit_time")),
                    "entry_price": _safe_float(row.get("entry_price")),
                    "exit_price": _safe_float(row.get("exit_price")),
                    "pnl_pct": _safe_float(row.get("pnl_pct")),
                    "pnl_after_fees": _safe_float(row.get("pnl_after_fees")),
                    "exit_reason": row.get("exit_reason") or "timeout",
                    "duration_bars": int(_safe_float(row.get("duration_bars"), default=0)),
                    "dip_size": _safe_optional_float(row.get("dip_size")),
                    "max_rebound_pct": _safe_optional_float(row.get("max_rebound_pct")),
                    "entry_signal": _build_trade_entry_signal(entry_row, signal_params),
                }
            )
    else:
        trades = []

    equity_curve_values = backtest.get("equity_curve")
    equity_timestamps = backtest.get("equity_timestamps")
    equity_curve = []

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

    symbol_backtest = {
        "symbol": symbol,
        "signalParams": {
            "dip_threshold": _safe_float(signal_params["dip_threshold"]),
            "use_volume_filter": bool(signal_params["use_volume_filter"]),
            "use_rsi_filter": bool(signal_params["use_rsi_filter"]),
            "hold_timeout_hours": int(_safe_float(signal_params["hold_timeout_hours"])),
            "max_positions": int(_safe_float(signal_params["max_positions"])),
            "take_profit_pct": _safe_float(signal_params["take_profit_pct"]),
            "stop_loss_pct": _safe_float(signal_params["stop_loss_pct"]),
            "min_volume_mult": _safe_float(signal_params["min_volume_mult"]),
            "rsi_min": _safe_float(signal_params["rsi_min"]),
            "rsi_max": _safe_float(signal_params["rsi_max"]),
            "one_entry_per_dip_event": bool(signal_params["one_entry_per_dip_event"]),
            "dip_event_reset_pct": _safe_float(signal_params["dip_event_reset_pct"]),
            "fail_fast_bars": int(_safe_float(signal_params["fail_fast_bars"])) if signal_params["fail_fast_bars"] is not None else None,
            "fail_fast_min_rebound_pct": _safe_float(signal_params["fail_fast_min_rebound_pct"]),
            "leverage": _safe_float(leverage, default=LEVERAGE),
        },
        "metrics": backtest.get("metrics", {}),
        "futuresReality": _build_futures_reality(
            backtest.get("metrics", {}),
            last_close=_safe_float(last.get("close")),
            stop_loss_pct=_safe_float(signal_params["stop_loss_pct"]),
            leverage=_safe_float(leverage, default=LEVERAGE),
            sl_below_liquidation=bool(backtest.get("sl_below_liquidation", False)),
        ),
        "candles": [
            {
                "timestamp": _to_iso(row.get("timestamp")),
                "open": _safe_float(row.get("open")),
                "high": _safe_float(row.get("high")),
                "low": _safe_float(row.get("low")),
                "close": _safe_float(row.get("close")),
                "volume": _safe_optional_float(row.get("volume")),
            }
            for _, row in capped_df.iterrows()
        ],
        "historicalSignals": _build_historical_signals(
            capped_df,
            dip_threshold=_safe_float(signal_params["dip_threshold"]),
        ),
        "equityCurve": equity_curve,
        "trades": trades,
        "currentSignal": current_signal,
    }

    # Contexto para el backtest de cartera (capital compartido): el df con señales,
    # la serie de funding y los parametros resueltos del simbolo.
    portfolio_ctx = {
        "symbol": symbol,
        "df": df[["timestamp", "high", "low", "close", "signal", "dip_size"]].copy(),
        "funding_series": funding_series,
        "signal_params": signal_params,
    }

    return signals_item, symbol_backtest, portfolio_ctx


def _build_walkforward_symbol_result(
    symbol: str,
    train_months: int,
    test_months: int,
    step_months: int,
    days: int,
    dip_threshold_override: float | None,
    min_trades_for_pf: int,
) -> dict[str, Any]:
    wf_rows = walk_forward_validation(
        symbol,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        total_days=days,
        dip_threshold=dip_threshold_override,
    )

    if not wf_rows:
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
            "trainStart": "",
            "trainEnd": "",
            "testStart": _to_iso(row.get("test_start")),
            "testEnd": _to_iso(row.get("test_end")),
            "totalReturnPct": _safe_float(row.get("total_return")),
            "sharpe": _safe_float(row.get("sharpe")),
            "maxDrawdownPct": _safe_float(row.get("max_dd")),
            "winRatePct": _safe_float(row.get("win_rate")),
            "profitFactor": _safe_float(row.get("profit_factor")),
            "trades": int(_safe_float(row.get("num_trades"), default=0)),
            "expectancyPct": _safe_float(row.get("expectancy")),
        }
        for idx, row in enumerate(wf_rows)
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


def _build_portfolio_artifact(
    run_id: str,
    generated_at: str,
    portfolio_contexts: list[dict[str, Any]],
    equity_cap: int = 2000,
) -> dict[str, Any]:
    """
    Corre el backtest de cartera (capital compartido) y arma el artefacto JSON.
    Devuelve status not_available si hay menos de 2 simbolos validos.
    """
    valid = [c for c in portfolio_contexts if c and c.get("df") is not None and len(c["df"]) > 0]
    if len(valid) < 2:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "strategy": STRATEGY_NAME,
            "source": SOURCE_NAME,
            "runId": run_id,
            "generatedAt": generated_at,
            "status": "not_available",
            "summary": "Cartera no disponible: se necesitan al menos 2 simbolos.",
        }

    symbol_frames = {c["symbol"]: c["df"] for c in valid}
    symbol_params = {c["symbol"]: c["signal_params"] for c in valid}
    funding_by_symbol = {
        c["symbol"]: c["funding_series"] for c in valid if c.get("funding_series") is not None
    }

    result = run_portfolio_backtest(
        symbol_frames=symbol_frames,
        symbol_params=symbol_params,
        funding_by_symbol=funding_by_symbol,
        initial_capital=INITIAL_CAPITAL,
        position_pct=POSITION_PCT,
        max_positions=MAX_POSITIONS,
        fee_pct=FEE_PCT,
        slippage_pct=SLIPPAGE_PCT,
    )

    # Equity curve diezmada para no inflar el JSON (serie larga -> ~equity_cap puntos)
    eq = result["equity_curve"]
    eq_ts = result["equity_timestamps"]
    n = len(eq)
    stride = max(1, n // max(1, equity_cap))
    equity_curve = [
        {"timestamp": _to_iso(eq_ts[i]), "equity": _safe_float(eq[i])}
        for i in range(0, n, stride)
    ]
    if n and (n - 1) % stride != 0:
        equity_curve.append({"timestamp": _to_iso(eq_ts[-1]), "equity": _safe_float(eq[-1])})

    m = result["metrics"]

    # Comparativa: suma de retornos por-simbolo (modelo viejo, capital dedicado)
    # vs retorno de cartera (capital compartido). Ilustra la "inflacion" del viejo.
    sum_isolated = 0.0
    for c in valid:
        bt = run_backtest(
            c["df"].assign(open=c["df"]["close"]),  # open no se usa en salidas, placeholder
            initial_capital=INITIAL_CAPITAL,
            stop_loss_pct=c["signal_params"]["stop_loss_pct"],
            take_profit_pct=c["signal_params"]["take_profit_pct"],
            hold_timeout=c["signal_params"]["hold_timeout_hours"],
            max_positions=c["signal_params"]["max_positions"],
            fee_pct=FEE_PCT,
            slippage_pct=SLIPPAGE_PCT,
            leverage=c["signal_params"].get("leverage"),
            funding_series=c.get("funding_series"),
        )
        sum_isolated += _safe_float(bt["metrics"].get("total_return")) * 100

    return {
        "schemaVersion": SCHEMA_VERSION,
        "strategy": STRATEGY_NAME,
        "source": SOURCE_NAME,
        "runId": run_id,
        "generatedAt": generated_at,
        "status": "available",
        "config": {
            "initialCapital": INITIAL_CAPITAL,
            "positionPct": POSITION_PCT * 100,
            "maxPositions": MAX_POSITIONS,
            "symbols": list(symbol_frames.keys()),
        },
        "metrics": {
            "totalReturnPct": _safe_float(m.get("total_return")) * 100,
            "sharpe": _safe_float(m.get("sharpe")),
            "maxDrawdownPct": _safe_float(m.get("max_drawdown")) * 100,
            "winRatePct": _safe_float(m.get("win_rate")) * 100,
            "profitFactor": _safe_float(m.get("profit_factor")),
            "totalTrades": int(_safe_float(m.get("total_trades"))),
            "finalCapital": _safe_float(m.get("final_capital")),
            "nLiquidations": int(_safe_float(m.get("n_liquidations"))),
            "missedSignals": int(_safe_float(m.get("missed_signals"))),
        },
        "comparison": {
            "sumIsolatedReturnPct": round(sum_isolated, 2),
            "portfolioReturnPct": round(_safe_float(m.get("total_return")) * 100, 2),
        },
        "perSymbol": result["per_symbol"],
        "equityCurve": equity_curve,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export buy_the_dip JSON artifacts for frontend")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to process")
    parser.add_argument("--timeframe", default=TIMEFRAME, help=f"Candle timeframe (default: {TIMEFRAME})")
    parser.add_argument("--days", type=int, default=DAYS, help=f"Days of history (default: {DAYS})")
    parser.add_argument("--threshold", type=float, default=None, help="Global dip threshold override")
    parser.add_argument("--candle-cap", type=int, default=8760, help="Max candles per symbol for artifact")
    parser.add_argument("--include-walkforward", action="store_true", help="Generate walkforward.latest.json")
    parser.add_argument("--train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument("--min-trades", type=int, default=5, help="Min trades for PF consistency in walkforward")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "exports" / "buy_the_dip"),
        help="Output directory for latest JSON artifacts",
    )

    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS
    run_id = f"btd-{uuid.uuid4().hex[:12]}"
    generated_at = datetime.now(timezone.utc).isoformat()

    print(f"\n{'=' * 70}")
    print("  BUY_THE_DIP FRONTEND EXPORT")
    print(f"{'=' * 70}")
    print(f"  Symbols:     {len(symbols)}")
    print(f"  Timeframe:   {args.timeframe}")
    print(f"  Days:        {args.days}")
    print(f"  Candle cap:  {args.candle_cap}")
    print(f"  Threshold:   {args.threshold if args.threshold is not None else 'per-symbol override'}")
    print(f"  Walkforward: {'on' if args.include_walkforward else 'off'}")
    print(f"  Out dir:     {args.out_dir}")
    print(f"{'=' * 70}\n")

    signal_items: list[dict[str, Any]] = []
    backtest_results: list[dict[str, Any]] = []
    portfolio_contexts: list[dict[str, Any]] = []

    # Regimen de mercado (BTC vs SMA), calculado una vez; vacio si el filtro esta off.
    regime = load_regime_if_enabled(fetch_ohlcv, args.days)

    for symbol in symbols:
        try:
            print(f"Processing {symbol}...")
            signal_item, symbol_backtest, portfolio_ctx = _process_symbol(
                symbol=symbol,
                timeframe=args.timeframe,
                days=args.days,
                candle_cap=args.candle_cap,
                dip_threshold_override=args.threshold,
                regime=regime,
            )
            signal_items.append(signal_item)
            backtest_results.append(symbol_backtest)
            portfolio_contexts.append(portfolio_ctx)
            print(
                f"  done: signal={signal_item['signal']} trades={symbol_backtest.get('metrics', {}).get('total_trades', 0)}"
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
    )

    signals_path = out_dir / "signals.latest.json"
    backtest_path = out_dir / "backtest.latest.json"

    _write_json(signals_path, signals_payload)
    _write_json(backtest_path, backtest_payload)

    print(f"\nWrote {signals_path}")
    print(f"Wrote {backtest_path}")

    # ── Backtest de cartera (capital compartido) ──
    try:
        print("\nBuilding portfolio backtest (shared capital)...")
        portfolio_payload = _build_portfolio_artifact(
            run_id=run_id,
            generated_at=generated_at,
            portfolio_contexts=portfolio_contexts,
        )
        portfolio_path = out_dir / "portfolio.latest.json"
        _write_json(portfolio_path, portfolio_payload)
        print(f"Wrote {portfolio_path}")
        if portfolio_payload.get("status") == "available":
            pm = portfolio_payload["metrics"]
            cmp = portfolio_payload["comparison"]
            print(
                f"  portfolio: ret={pm['totalReturnPct']:.1f}% "
                f"maxDD={pm['maxDrawdownPct']:.1f}% missed={pm['missedSignals']} "
                f"| sum-isolated={cmp['sumIsolatedReturnPct']:.1f}%"
            )
    except Exception as exc:
        print(f"  portfolio error: {exc}")

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
                        dip_threshold_override=args.threshold,
                        min_trades_for_pf=args.min_trades,
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
