"""
Matrix test for Sell-the-Rip with robust multi-horizon scoring.
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, "..")

from breakout_momentum.data_loader import fetch_ohlcv
from backtest import run_backtest
from config import (
    ALLOW_COUNTERTREND_REVERSAL,
    COUNTERTREND_MIN_DROP_1H,
    COUNTERTREND_MIN_PULLBACK_PCT,
    COUNTERTREND_RIP_LOOKBACK_HOURS,
    DAYS,
    FEE_PCT,
    INITIAL_CAPITAL,
    MATRIX_HOLD_GRID,
    MATRIX_MAX_CONFIGS_PER_SYMBOL,
    MATRIX_MIN_VOLUME_MULT_GRID,
    MATRIX_RSI_MAX_GRID,
    MATRIX_RSI_MIN_GRID,
    MATRIX_SL_GRID,
    MATRIX_THRESHOLDS,
    MATRIX_TP_GRID,
    MATRIX_USE_RSI_GRID,
    MATRIX_USE_TREND_GRID,
    MATRIX_USE_VOLUME_GRID,
    ROBUST_DAYS_LIST,
    ROBUST_DAYS_WEIGHTS,
    TIMEFRAME,
    WF_STEP_MONTHS,
    WF_TEST_MONTHS,
    WF_TRAIN_MONTHS,
)
from strategy import add_rip_features, generate_signals


def _parse_thresholds(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _build_param_grid(thresholds: list[float]) -> list[dict]:
    """Build compact but expressive grid with conditional dimensions."""
    # Keep RSI as coherent pairs to avoid exploding combinations.
    rsi_pairs = list(zip(MATRIX_RSI_MIN_GRID, MATRIX_RSI_MAX_GRID))

    configs = []
    for threshold, use_rsi, use_vol, use_trend, tp, sl, hold in product(
        thresholds,
        MATRIX_USE_RSI_GRID,
        MATRIX_USE_VOLUME_GRID,
        MATRIX_USE_TREND_GRID,
        MATRIX_TP_GRID,
        MATRIX_SL_GRID,
        MATRIX_HOLD_GRID,
    ):
        if tp >= sl:
            # Avoid weak R:R on shorts.
            continue

        if use_rsi:
            rsi_candidates = rsi_pairs
        else:
            rsi_candidates = [(MATRIX_RSI_MIN_GRID[0], MATRIX_RSI_MAX_GRID[-1])]

        if use_vol:
            vol_candidates = MATRIX_MIN_VOLUME_MULT_GRID
        else:
            vol_candidates = [MATRIX_MIN_VOLUME_MULT_GRID[0]]

        for rsi_min, rsi_max in rsi_candidates:
            for min_vol in vol_candidates:
                configs.append(
                    {
                        "threshold": float(threshold),
                        "use_rsi_filter": bool(use_rsi),
                        "use_volume_filter": bool(use_vol),
                        "use_trend_filter": bool(use_trend),
                        "rsi_min": float(rsi_min),
                        "rsi_max": float(rsi_max),
                        "min_volume_mult": float(min_vol),
                        "take_profit_pct": float(tp),
                        "stop_loss_pct": float(sl),
                        "hold_timeout_hours": int(hold),
                    }
                )
    return configs


def _downsample_grid(configs: list[dict], max_configs: int) -> list[dict]:
    if max_configs <= 0 or len(configs) <= max_configs:
        return configs
    step = len(configs) / max_configs
    picked = []
    i = 0.0
    while int(i) < len(configs) and len(picked) < max_configs:
        picked.append(configs[int(i)])
        i += step
    return picked


def _walk_forward_for_config(
    df_features: pd.DataFrame,
    params: dict,
    train_months: int,
    test_months: int,
    step_months: int,
) -> dict:
    warmup_size = int(train_months * 30 * 24)
    test_size = int(test_months * 30 * 24)
    step_size = int(step_months * 30 * 24)

    start_idx = warmup_size
    windows_total = 0
    windows_with_trades = 0
    positive_return_windows = 0
    sharpe_positive_windows = 0
    pf_above_1_windows = 0

    returns = []
    sharpes = []
    pfs = []
    mdds = []
    trades_total = 0

    while True:
        test_end_idx = start_idx + test_size
        if test_end_idx > len(df_features):
            break

        windows_total += 1
        test_data = df_features.iloc[start_idx:test_end_idx].copy()
        if len(test_data) < 100:
            start_idx += step_size
            continue

        with_signals = generate_signals(
            test_data,
            rip_threshold=params["threshold"],
            use_volume_filter=params["use_volume_filter"],
            use_rsi_filter=params["use_rsi_filter"],
            use_trend_filter=params["use_trend_filter"],
            allow_countertrend_reversal=ALLOW_COUNTERTREND_REVERSAL,
            countertrend_rip_lookback_hours=COUNTERTREND_RIP_LOOKBACK_HOURS,
            countertrend_min_drop_1h=COUNTERTREND_MIN_DROP_1H,
            countertrend_min_pullback_pct=COUNTERTREND_MIN_PULLBACK_PCT,
            min_volume_mult=params["min_volume_mult"],
            rsi_min=params["rsi_min"],
            rsi_max=params["rsi_max"],
        )

        bt = run_backtest(
            with_signals,
            initial_capital=INITIAL_CAPITAL,
            stop_loss_pct=params["stop_loss_pct"],
            take_profit_pct=params["take_profit_pct"],
            hold_timeout=params["hold_timeout_hours"],
            fee_pct=FEE_PCT,
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


def _score_row_base(row: dict) -> float:
    """Score on one horizon; prioritizes robustness over raw return."""
    if row["total_trades"] < 40 or row["wf_windows_with_trades"] < 3:
        return -1.0

    wf_pos = row["wf_positive_pct"] / 100.0
    wf_pf = min(max((row["wf_avg_pf"] - 1.0) / 1.0, 0.0), 1.0)
    sharpe = min(max(row["sharpe"] / 2.0, 0.0), 1.0)
    drawdown = max(0.0, 1.0 - abs(row["max_drawdown"]) / 0.50)
    trades = min(row["total_trades"] / 800.0, 1.0)

    score = wf_pos * 0.38 + wf_pf * 0.20 + sharpe * 0.18 + drawdown * 0.14 + trades * 0.10
    return round(score * 100, 2)


def _enrich_with_backtest_metrics(df_features: pd.DataFrame, params: dict) -> dict:
    with_signals = generate_signals(
        df_features,
        rip_threshold=params["threshold"],
        use_volume_filter=params["use_volume_filter"],
        use_rsi_filter=params["use_rsi_filter"],
        use_trend_filter=params["use_trend_filter"],
        allow_countertrend_reversal=ALLOW_COUNTERTREND_REVERSAL,
        countertrend_rip_lookback_hours=COUNTERTREND_RIP_LOOKBACK_HOURS,
        countertrend_min_drop_1h=COUNTERTREND_MIN_DROP_1H,
        countertrend_min_pullback_pct=COUNTERTREND_MIN_PULLBACK_PCT,
        min_volume_mult=params["min_volume_mult"],
        rsi_min=params["rsi_min"],
        rsi_max=params["rsi_max"],
    )

    bt = run_backtest(
        with_signals,
        initial_capital=INITIAL_CAPITAL,
        stop_loss_pct=params["stop_loss_pct"],
        take_profit_pct=params["take_profit_pct"],
        hold_timeout=params["hold_timeout_hours"],
        fee_pct=FEE_PCT,
    )
    m = bt["metrics"]

    return {
        "total_return_pct": round(float(m["total_return"]) * 100, 3),
        "sharpe": float(m["sharpe"]),
        "max_drawdown": float(m["max_drawdown"]),
        "max_drawdown_pct": round(float(m["max_drawdown"]) * 100, 3),
        "win_rate_pct": round(float(m["win_rate"]) * 100, 3),
        "profit_factor": float(m["profit_factor"]),
        "total_trades": int(m["total_trades"]),
        "expectancy_pct": round(float(m["expectancy"]) * 100, 4),
    }


def _robust_score_for_candidate(
    symbol: str,
    params: dict,
    days_list: list[int],
    weights: list[float],
    train_months: int,
    test_months: int,
    step_months: int,
    features_cache: dict[int, pd.DataFrame] | None = None,
) -> dict:
    """Evaluate same params across multiple horizons and return weighted robust score."""
    horizon_scores = []
    horizon_details = {}

    for days, w in zip(days_list, weights):
        if features_cache is not None and int(days) in features_cache:
            df_features = features_cache[int(days)]
        else:
            df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=int(days))
            df_features = add_rip_features(df)
            if features_cache is not None:
                features_cache[int(days)] = df_features

        base = _enrich_with_backtest_metrics(df_features, params)
        wf = _walk_forward_for_config(
            df_features=df_features,
            params=params,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
        )
        row = dict(base)
        row.update(wf)
        score = _score_row_base(row)
        horizon_scores.append(score * float(w))
        horizon_details[str(days)] = {
            "score": score,
            "wf_positive_pct": row["wf_positive_pct"],
            "wf_sharpe_positive_pct": row["wf_sharpe_positive_pct"],
            "wf_pf_above_1_pct": row["wf_pf_above_1_pct"],
            "total_trades": row["total_trades"],
            "total_return_pct": row["total_return_pct"],
            "max_drawdown_pct": row["max_drawdown_pct"],
        }

    robust_score = round(sum(horizon_scores), 2)
    return {
        "robust_score": robust_score,
        "robust_horizon_details": json.dumps(horizon_details, ensure_ascii=True),
    }


def run_matrix(
    symbol: str,
    days: int,
    thresholds: list[float],
    train_months: int,
    test_months: int,
    step_months: int,
    robust_days: list[int] | None = None,
    shortlist_top_k: int = 40,
    max_configs: int = MATRIX_MAX_CONFIGS_PER_SYMBOL,
) -> pd.DataFrame:
    robust_days = robust_days or ROBUST_DAYS_LIST

    print(f"\n{'='*72}")
    print("  SELL-THE-RIP MATRIX TEST")
    print(f"{'='*72}")
    print(f"  Symbol:      {symbol}")
    print(f"  Timeframe:   {TIMEFRAME}")
    print(f"  Base days:   {days}")
    print(f"  Robust days: {robust_days}")
    print(f"  Thresholds:  {thresholds}")

    grid = _build_param_grid(thresholds)
    grid = _downsample_grid(grid, max_configs=max_configs)
    print(f"  Matrix size: {len(grid)} configs")
    print(f"{'='*72}\n")

    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=days)
    df_features = add_rip_features(df)

    rows = []
    for idx, params in enumerate(grid, 1):
        print(
            f"[{idx:>4}/{len(grid)}] "
            f"th={params['threshold']:.2f} rsi={'on' if params['use_rsi_filter'] else 'off'} "
            f"vol={'on' if params['use_volume_filter'] else 'off'} trend={'on' if params['use_trend_filter'] else 'off'} "
            f"tp={params['take_profit_pct']:.2f} sl={params['stop_loss_pct']:.2f} hold={params['hold_timeout_hours']}"
        )

        base = _enrich_with_backtest_metrics(df_features, params)
        wf = _walk_forward_for_config(
            df_features=df_features,
            params=params,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
        )

        row = {"symbol": symbol, **params, **base, **wf}
        row["score_base"] = _score_row_base(row)
        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by=["score_base", "wf_positive_pct", "sharpe", "profit_factor", "total_trades"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    # Stage 2: robust multi-horizon rescoring on top candidates.
    top_n = min(shortlist_top_k, len(results))
    if top_n > 0:
        w = ROBUST_DAYS_WEIGHTS
        if len(w) != len(robust_days) or sum(w) <= 0:
            w = [1 / len(robust_days)] * len(robust_days)
        robust_features_cache: dict[int, pd.DataFrame] = {}
        for d in robust_days:
            d_int = int(d)
            print(f"  Preloading robust horizon: {d_int} days")
            robust_df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=d_int)
            robust_features_cache[d_int] = add_rip_features(robust_df)

        for i in range(top_n):
            params = {
                "threshold": float(results.loc[i, "threshold"]),
                "use_rsi_filter": bool(results.loc[i, "use_rsi_filter"]),
                "use_volume_filter": bool(results.loc[i, "use_volume_filter"]),
                "use_trend_filter": bool(results.loc[i, "use_trend_filter"]),
                "rsi_min": float(results.loc[i, "rsi_min"]),
                "rsi_max": float(results.loc[i, "rsi_max"]),
                "min_volume_mult": float(results.loc[i, "min_volume_mult"]),
                "take_profit_pct": float(results.loc[i, "take_profit_pct"]),
                "stop_loss_pct": float(results.loc[i, "stop_loss_pct"]),
                "hold_timeout_hours": int(results.loc[i, "hold_timeout_hours"]),
            }
            robust = _robust_score_for_candidate(
                symbol=symbol,
                params=params,
                days_list=robust_days,
                weights=w,
                train_months=train_months,
                test_months=test_months,
                step_months=step_months,
                features_cache=robust_features_cache,
            )
            results.loc[i, "robust_score"] = robust["robust_score"]
            results.loc[i, "robust_horizon_details"] = robust["robust_horizon_details"]

    results["robust_score"] = results["robust_score"].fillna(-1.0)
    results = results.sort_values(
        by=["robust_score", "score_base", "wf_positive_pct", "max_drawdown"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    results["score"] = results["robust_score"]
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Sell-the-Rip matrix test")
    parser.add_argument("--symbol", default="LINK/USDT")
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--thresholds", default=",".join(str(x) for x in MATRIX_THRESHOLDS))
    parser.add_argument("--wf-train-months", type=int, default=WF_TRAIN_MONTHS)
    parser.add_argument("--wf-test-months", type=int, default=WF_TEST_MONTHS)
    parser.add_argument("--wf-step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument("--robust-days", default=",".join(str(x) for x in ROBUST_DAYS_LIST))
    parser.add_argument("--shortlist-top-k", type=int, default=40)
    parser.add_argument("--max-configs", type=int, default=MATRIX_MAX_CONFIGS_PER_SYMBOL)
    parser.add_argument("--out-csv", default="sell_the_rip/matrix_results.csv")
    parser.add_argument("--out-json", default="sell_the_rip/matrix_results.json")
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.thresholds)
    if not thresholds:
        raise ValueError("No thresholds provided")

    robust_days = [int(x.strip()) for x in args.robust_days.split(",") if x.strip()]

    results = run_matrix(
        symbol=args.symbol,
        days=args.days,
        thresholds=thresholds,
        train_months=args.wf_train_months,
        test_months=args.wf_test_months,
        step_months=args.wf_step_months,
        robust_days=robust_days,
        shortlist_top_k=args.shortlist_top_k,
        max_configs=args.max_configs,
    )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, ensure_ascii=True, indent=2)

    print(f"\n{'='*72}")
    print("  TOP 5 CONFIGS")
    print(f"{'='*72}")
    print(
        results.head(5)[
            [
                "threshold",
                "use_rsi_filter",
                "use_volume_filter",
                "use_trend_filter",
                "rsi_min",
                "rsi_max",
                "min_volume_mult",
                "take_profit_pct",
                "stop_loss_pct",
                "hold_timeout_hours",
                "total_return_pct",
                "sharpe",
                "profit_factor",
                "wf_positive_pct",
                "total_trades",
                "score_base",
                "robust_score",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved CSV:  {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
