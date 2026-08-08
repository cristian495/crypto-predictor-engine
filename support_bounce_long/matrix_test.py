"""
Matrix optimization for support_bounce_long with robust multi-horizon scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breakout_momentum.data_loader import fetch_ohlcv
from backtest import run_backtest
from config import (
    DAYS,
    FEE_PCT,
    INITIAL_CAPITAL,
    MATRIX_CONFIRMATION_MODE_GRID,
    MATRIX_HOLD_TIMEOUT_GRID,
    MATRIX_MAX_CONFIGS_PER_SYMBOL,
    MATRIX_MIN_RR_GRID,
    MATRIX_MIN_TOUCHES_GRID,
    MATRIX_PIVOT_WINDOW_GRID,
    MATRIX_PULLBACK_LOOKBACK_GRID,
    MATRIX_REJECTION_THRESHOLD_GRID,
    MATRIX_REQUIRE_PULLBACK_GRID,
    MATRIX_STOP_BUFFER_GRID,
    MATRIX_SUPPORT_PROXIMITY_GRID,
    MATRIX_TOLERANCE_GRID,
    MAX_POSITIONS,
    POSITION_PCT,
    ROBUST_DAYS_LIST,
    ROBUST_DAYS_WEIGHTS,
    TIMEFRAME,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
)
from strategy import generate_signals


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _normalize_weights(weights: list[float]) -> list[float]:
    total = sum(float(w) for w in weights)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [float(w) / total for w in weights]


def _build_param_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    for (
        tolerance_pct,
        min_touches,
        pivot_window,
        support_proximity_pct,
        rejection_threshold,
        min_rr,
        stop_buffer_pct,
        confirmation_mode,
        require_pullback,
        hold_timeout_bars,
    ) in product(
        MATRIX_TOLERANCE_GRID,
        MATRIX_MIN_TOUCHES_GRID,
        MATRIX_PIVOT_WINDOW_GRID,
        MATRIX_SUPPORT_PROXIMITY_GRID,
        MATRIX_REJECTION_THRESHOLD_GRID,
        MATRIX_MIN_RR_GRID,
        MATRIX_STOP_BUFFER_GRID,
        MATRIX_CONFIRMATION_MODE_GRID,
        MATRIX_REQUIRE_PULLBACK_GRID,
        MATRIX_HOLD_TIMEOUT_GRID,
    ):
        pullback_candidates = (
            MATRIX_PULLBACK_LOOKBACK_GRID if require_pullback else [MATRIX_PULLBACK_LOOKBACK_GRID[0]]
        )

        for pullback_lookback in pullback_candidates:
            configs.append(
                {
                    "tolerance_pct": float(tolerance_pct),
                    "min_touches": int(min_touches),
                    "pivot_window": int(pivot_window),
                    "support_proximity_pct": float(support_proximity_pct),
                    "rejection_threshold": float(rejection_threshold),
                    "min_rr": float(min_rr),
                    "stop_buffer_pct": float(stop_buffer_pct),
                    "confirmation_mode": str(confirmation_mode),
                    "require_pullback": bool(require_pullback),
                    "pullback_lookback": int(pullback_lookback),
                    "hold_timeout_bars": int(hold_timeout_bars),
                    "require_confirmed": True,
                }
            )

    return configs


def _downsample_grid(configs: list[dict[str, Any]], max_configs: int) -> list[dict[str, Any]]:
    if max_configs <= 0 or len(configs) <= max_configs:
        return configs

    step = len(configs) / max_configs
    selected: list[dict[str, Any]] = []
    idx = 0.0

    while int(idx) < len(configs) and len(selected) < max_configs:
        selected.append(configs[int(idx)])
        idx += step

    return selected


def _enrich_with_backtest_metrics(raw_df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    with_signals = generate_signals(raw_df, params)

    bt = run_backtest(
        with_signals,
        initial_capital=INITIAL_CAPITAL,
        position_pct=POSITION_PCT,
        max_positions=MAX_POSITIONS,
        fee_pct=FEE_PCT,
        hold_timeout_bars=int(params["hold_timeout_bars"]),
    )
    m = bt["metrics"]

    return {
        "total_return_pct": round(float(m.get("total_return", 0.0)) * 100, 3),
        "sharpe": float(m.get("sharpe", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "max_drawdown_pct": round(float(m.get("max_drawdown", 0.0)) * 100, 3),
        "win_rate_pct": round(float(m.get("win_rate", 0.0)) * 100, 3),
        "profit_factor": float(m.get("profit_factor", 0.0)),
        "total_trades": int(m.get("total_trades", 0)),
        "expectancy_pct": round(float(m.get("expectancy", 0.0)) * 100, 4),
        "signals": int((with_signals["signal"] == "LONG").sum()),
    }


def _walk_forward_for_config(
    raw_df: pd.DataFrame,
    params: dict[str, Any],
    train_months: int,
    test_months: int,
    step_months: int,
) -> dict[str, Any]:
    warmup_size = int(train_months * 30 * 24)
    test_size = int(test_months * 30 * 24)
    step_size = int(step_months * 30 * 24)

    start_idx = warmup_size
    windows_total = 0
    windows_with_trades = 0
    positive_return_windows = 0
    sharpe_positive_windows = 0
    pf_above_1_windows = 0

    returns: list[float] = []
    sharpes: list[float] = []
    pfs: list[float] = []
    mdds: list[float] = []
    trades_total = 0

    while True:
        test_end_idx = start_idx + test_size
        if test_end_idx > len(raw_df):
            break

        windows_total += 1
        test_data = raw_df.iloc[start_idx:test_end_idx].copy()
        if len(test_data) < 100:
            start_idx += step_size
            continue

        with_signals = generate_signals(test_data, params)
        bt = run_backtest(
            with_signals,
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
            continue

        windows_with_trades += 1
        trades_total += trades

        total_return = float(m.get("total_return", 0.0))
        sharpe = float(m.get("sharpe", 0.0))
        pf = float(m.get("profit_factor", 0.0))
        max_dd = float(m.get("max_drawdown", 0.0))

        returns.append(total_return)
        sharpes.append(sharpe)
        pfs.append(pf)
        mdds.append(max_dd)

        if total_return > 0:
            positive_return_windows += 1
        if sharpe > 0:
            sharpe_positive_windows += 1
        if pf > 1:
            pf_above_1_windows += 1

        start_idx += step_size

    if windows_with_trades == 0:
        return {
            "wf_windows_total": windows_total,
            "wf_windows_with_trades": 0,
            "wf_positive_pct": 0.0,
            "wf_sharpe_positive_pct": 0.0,
            "wf_pf_above_1_pct": 0.0,
            "wf_avg_return_pct": 0.0,
            "wf_avg_sharpe": 0.0,
            "wf_avg_pf": 0.0,
            "wf_avg_max_dd_pct": 0.0,
            "wf_total_trades": 0,
        }

    denom = windows_with_trades
    return {
        "wf_windows_total": windows_total,
        "wf_windows_with_trades": windows_with_trades,
        "wf_positive_pct": round(positive_return_windows / denom * 100, 2),
        "wf_sharpe_positive_pct": round(sharpe_positive_windows / denom * 100, 2),
        "wf_pf_above_1_pct": round(pf_above_1_windows / denom * 100, 2),
        "wf_avg_return_pct": round(sum(returns) / denom * 100, 3),
        "wf_avg_sharpe": round(sum(sharpes) / denom, 3),
        "wf_avg_pf": round(sum(pfs) / denom, 3),
        "wf_avg_max_dd_pct": round(sum(mdds) / denom * 100, 3),
        "wf_total_trades": trades_total,
    }


def _score_row_base(row: dict[str, Any]) -> float:
    if int(row["total_trades"]) < 30 or int(row["wf_windows_with_trades"]) < 3:
        return -1.0

    wf_pos = float(row["wf_positive_pct"]) / 100.0
    wf_pf = min(max((float(row["wf_avg_pf"]) - 1.0) / 1.0, 0.0), 1.0)
    sharpe = min(max(float(row["sharpe"]) / 2.0, 0.0), 1.0)
    drawdown = max(0.0, 1.0 - abs(float(row["max_drawdown"])) / 0.50)
    trades = min(float(row["total_trades"]) / 600.0, 1.0)

    score = wf_pos * 0.38 + wf_pf * 0.20 + sharpe * 0.18 + drawdown * 0.14 + trades * 0.10
    return round(score * 100, 2)


def _robust_score_for_candidate(
    symbol: str,
    params: dict[str, Any],
    days_list: list[int],
    weights: list[float],
    train_months: int,
    test_months: int,
    step_months: int,
    raw_cache: dict[int, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    horizon_scores: list[float] = []
    horizon_details: dict[str, Any] = {}

    reference_days = max(days_list)
    reference_row: dict[str, Any] | None = None

    for days, weight in zip(days_list, weights):
        if raw_cache is not None and int(days) in raw_cache:
            raw_df = raw_cache[int(days)]
        else:
            raw_df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=int(days))
            if raw_cache is not None:
                raw_cache[int(days)] = raw_df

        base = _enrich_with_backtest_metrics(raw_df, params)
        wf = _walk_forward_for_config(
            raw_df=raw_df,
            params=params,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
        )

        row = dict(base)
        row.update(wf)
        score_base = _score_row_base(row)

        horizon_scores.append(score_base * float(weight))
        horizon_details[str(days)] = {
            "score": score_base,
            "total_return_pct": row["total_return_pct"],
            "sharpe": row["sharpe"],
            "profit_factor": row["profit_factor"],
            "max_drawdown_pct": row["max_drawdown_pct"],
            "total_trades": row["total_trades"],
            "wf_positive_pct": row["wf_positive_pct"],
            "wf_windows_with_trades": row["wf_windows_with_trades"],
        }

        if int(days) == int(reference_days):
            reference_row = row

    robust_score = round(sum(horizon_scores), 2)
    reference = reference_row or {
        "total_return_pct": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "total_trades": 0,
        "expectancy_pct": 0.0,
        "signals": 0,
        "wf_windows_total": 0,
        "wf_windows_with_trades": 0,
        "wf_positive_pct": 0.0,
        "wf_sharpe_positive_pct": 0.0,
        "wf_pf_above_1_pct": 0.0,
        "wf_avg_return_pct": 0.0,
        "wf_avg_sharpe": 0.0,
        "wf_avg_pf": 0.0,
        "wf_avg_max_dd_pct": 0.0,
        "wf_total_trades": 0,
    }

    out = dict(reference)
    out["score"] = robust_score
    out["score_base"] = _score_row_base(reference)
    out["robust_horizon_details"] = horizon_details
    out["robust_horizon_details_json"] = json.dumps(horizon_details, ensure_ascii=True)
    return out


def run_matrix(
    symbol: str,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
    robust_days: list[int] | None = None,
    robust_weights: list[float] | None = None,
    max_configs: int = MATRIX_MAX_CONFIGS_PER_SYMBOL,
) -> pd.DataFrame:
    robust_days_list = robust_days or list(ROBUST_DAYS_LIST)
    robust_weights_list = robust_weights or list(ROBUST_DAYS_WEIGHTS)

    if len(robust_days_list) != len(robust_weights_list):
        robust_weights_list = [1.0] * len(robust_days_list)
    robust_weights_list = _normalize_weights(robust_weights_list)

    full_configs = _build_param_grid()
    configs = _downsample_grid(full_configs, int(max_configs))

    print(f"\n{'=' * 72}")
    print("  SUPPORT-BOUNCE-LONG MATRIX")
    print(f"{'=' * 72}")
    print(f"  Symbol:      {symbol}")
    print(f"  Timeframe:   {TIMEFRAME}")
    print(f"  Horizons:    {robust_days_list}")
    print(f"  Weights:     {robust_weights_list}")
    print(f"  Grid full:   {len(full_configs)}")
    print(f"  Grid used:   {len(configs)}")

    rows: list[dict[str, Any]] = []
    raw_cache: dict[int, pd.DataFrame] = {}

    for idx, params in enumerate(configs, 1):
        print(
            f"[{idx:>4}/{len(configs)}] tol={params['tolerance_pct']:.2f} "
            f"touch={params['min_touches']} piv={params['pivot_window']} "
            f"prox={params['support_proximity_pct']:.2f} rej={params['rejection_threshold']:.2f} "
            f"rr={params['min_rr']:.2f} mode={params['confirmation_mode']} "
            f"pull={'on' if params['require_pullback'] else 'off'} hold={params['hold_timeout_bars']}"
        )

        robust = _robust_score_for_candidate(
            symbol=symbol,
            params=params,
            days_list=robust_days_list,
            weights=robust_weights_list,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            raw_cache=raw_cache,
        )

        row = {
            "symbol": symbol,
            **params,
            **robust,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["score", "wf_positive_pct", "sharpe", "profit_factor", "total_trades"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run support_bounce_long matrix test")
    parser.add_argument("--symbol", default="BNB/USDT")
    parser.add_argument("--train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument(
        "--robust-days",
        default=",".join(str(x) for x in ROBUST_DAYS_LIST),
        help="Comma-separated horizons in days",
    )
    parser.add_argument(
        "--robust-weights",
        default=",".join(str(x) for x in ROBUST_DAYS_WEIGHTS),
        help="Comma-separated weights aligned with --robust-days",
    )
    parser.add_argument("--max-configs", type=int, default=MATRIX_MAX_CONFIGS_PER_SYMBOL)
    parser.add_argument("--out-csv", default="support_bounce_long/matrix_results.csv")
    parser.add_argument("--out-json", default="support_bounce_long/matrix_results.json")
    args = parser.parse_args()

    robust_days = _parse_int_list(args.robust_days)
    robust_weights = [float(x.strip()) for x in args.robust_weights.split(",") if x.strip()]

    if not robust_days:
        robust_days = list(ROBUST_DAYS_LIST)
    if not robust_weights:
        robust_weights = list(ROBUST_DAYS_WEIGHTS)

    df = run_matrix(
        symbol=args.symbol,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        robust_days=robust_days,
        robust_weights=robust_weights,
        max_configs=args.max_configs,
    )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=True, indent=2)

    cols = [
        "score",
        "tolerance_pct",
        "min_touches",
        "pivot_window",
        "support_proximity_pct",
        "rejection_threshold",
        "min_rr",
        "confirmation_mode",
        "require_pullback",
        "hold_timeout_bars",
        "total_return_pct",
        "sharpe",
        "profit_factor",
        "total_trades",
        "wf_positive_pct",
    ]

    print(f"\n{'=' * 72}")
    print("  TOP 5 CONFIGS")
    print(f"{'=' * 72}")
    print(df.head(5)[cols].to_string(index=False))
    print(f"\nSaved:\n  - {out_csv}\n  - {out_json}")


if __name__ == "__main__":
    main()
