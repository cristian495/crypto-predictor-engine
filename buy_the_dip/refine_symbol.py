"""
refine_symbol.py
Second-pass search that ranks by ROBUSTNESS, not in-sample return.

Why this exists: optimize_symbol.py's IS score rewarded in-sample return and
consistency, and on INJ it picked a config with IS PF 1.35 that degraded to
OOS PF 1.01 with a -31% drawdown. Classic curve-fit.

Here every config is scored on three independent views:
  - IS  / OOS   (chronological split, same as optimize_symbol.py)
  - Walk-forward stability over the FULL history (27 rolling windows)
and ranked by a robustness score that penalises IS->OOS degradation.

Usage:
    ../venv/bin/python refine_symbol.py --symbol INJ/USDT
"""

import argparse
import itertools
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, "..")

from optimize_symbol import (
    load_features, split_is_oos, evaluate, CACHE_DIR, PARAM_KEYS,
)
from strategy import generate_signals
from backtest import run_backtest
from config import INITIAL_CAPITAL, FEE_PCT, TIMEFRAME

_DF = None
_DF_IS = None
_DF_OOS = None


def _bt(sig: pd.DataFrame, params: dict) -> dict:
    """Backtest pre-computed signals and flatten the metrics."""
    bt = run_backtest(
        sig,
        initial_capital=INITIAL_CAPITAL,
        stop_loss_pct=params["stop_loss_pct"],
        take_profit_pct=params["take_profit_pct"],
        hold_timeout=params["hold_timeout_hours"],
        max_positions=params["max_positions"],
        fee_pct=FEE_PCT,
        fail_fast_bars=params["fail_fast_bars"],
        fail_fast_min_rebound_pct=params["fail_fast_min_rebound_pct"],
    )
    m = bt["metrics"]
    return {
        "total_return_pct": float(m.get("total_return", 0)) * 100,
        "sharpe": float(m.get("sharpe", 0)),
        "max_dd_pct": float(m.get("max_drawdown", 0)) * 100,
        "win_rate_pct": float(m.get("win_rate", 0)) * 100,
        "profit_factor": float(m.get("profit_factor", 0)),
        "trades": int(m.get("total_trades", 0)),
        "expectancy_pct": float(m.get("expectancy", 0)) * 100,
    }


def walk_forward_stability(sig: pd.DataFrame, params: dict,
                           train_months=6, test_months=2, step_months=2) -> dict:
    """
    Same window scheme and stability formula as walk_forward.py.

    Takes PRE-COMPUTED signals: they depend only on entry params, so the caller
    generates them once and reuses them across every TP/SL/hold variant.
    """
    warmup = int(train_months * 30 * 24)
    test_size = int(test_months * 30 * 24)
    step = int(step_months * 30 * 24)

    start = warmup
    n_win = n_pos = n_sharpe = n_pf = 0
    while start + test_size <= len(sig):
        chunk = sig.iloc[start:start + test_size]
        bt = run_backtest(
            chunk,
            initial_capital=INITIAL_CAPITAL,
            stop_loss_pct=params["stop_loss_pct"],
            take_profit_pct=params["take_profit_pct"],
            hold_timeout=params["hold_timeout_hours"],
            max_positions=params["max_positions"],
            fee_pct=FEE_PCT,
            fail_fast_bars=params["fail_fast_bars"],
            fail_fast_min_rebound_pct=params["fail_fast_min_rebound_pct"],
        )
        m = bt["metrics"]
        if int(m.get("total_trades", 0)) > 0:
            n_win += 1
            if float(m.get("total_return", 0)) > 0:
                n_pos += 1
            if float(m.get("sharpe", 0)) > 0:
                n_sharpe += 1
            if float(m.get("profit_factor", 0)) > 1:
                n_pf += 1
        start += step

    if n_win == 0:
        return {"wf_windows": 0, "wf_positive_pct": 0.0, "wf_stability": 0.0}

    pos_pct = n_pos / n_win * 100
    sh_pct = n_sharpe / n_win * 100
    pf_pct = n_pf / n_win * 100
    return {
        "wf_windows": n_win,
        "wf_positive_pct": round(pos_pct, 1),
        "wf_stability": round(pos_pct * 0.4 + sh_pct * 0.3 + pf_pct * 0.3, 1),
    }


def robustness_score(r: dict, min_trades: int) -> float:
    """
    Rank by durability. An edge that only exists in-sample scores near zero.
    """
    if r["oos_trades"] < min_trades or r["is_trades"] < min_trades:
        return -1.0
    # Both halves must be independently profitable. No averaging away a loss.
    if r["oos_pf"] <= 1.0 or r["is_pf"] <= 1.0:
        return -1.0

    retention = min(r["oos_pf"] / r["is_pf"], 1.0)          # degradation penalty
    oos_edge = min(max((r["oos_pf"] - 1.0) / 0.5, 0.0), 1.0)
    is_edge = min(max((r["is_pf"] - 1.0) / 0.5, 0.0), 1.0)
    stability = r["wf_stability"] / 100.0
    dd = max(0.0, 1.0 - abs(r["oos_dd_pct"]) / 25.0)
    sample = min(r["full_trades"] / 300.0, 1.0)

    score = (
        retention * 0.25
        + oos_edge * 0.25
        + stability * 0.20
        + is_edge * 0.10
        + dd * 0.10
        + sample * 0.10
    )
    return round(score * 100, 2)


def _init_worker(cache_path: str, oos_frac: float):
    global _DF, _DF_IS, _DF_OOS
    _DF = pd.read_pickle(cache_path)
    _DF_IS, _DF_OOS = split_is_oos(_DF, oos_frac)


ENTRY_KEYS = ["dip_threshold", "use_volume_filter", "use_rsi_filter",
              "min_volume_mult", "rsi_min", "rsi_max",
              "one_entry_per_dip_event", "dip_event_reset_pct"]


def _group_task(args):
    """
    One entry-config -> signals generated ONCE -> every execution variant
    backtested against them. This is the whole speedup: 960 configs collapse
    to 30 signal generations instead of 3,840.
    """
    entry_params, variants, min_trades, oos_cut = args
    try:
        sig = generate_signals(
            _DF,
            dip_threshold=entry_params["dip_threshold"],
            use_volume_filter=entry_params["use_volume_filter"],
            use_rsi_filter=entry_params["use_rsi_filter"],
            min_volume_mult=entry_params["min_volume_mult"],
            rsi_min=entry_params["rsi_min"],
            rsi_max=entry_params["rsi_max"],
            one_entry_per_dip_event=entry_params["one_entry_per_dip_event"],
            dip_event_reset_pct=entry_params["dip_event_reset_pct"],
        )
    except Exception as exc:
        return [{**entry_params, "robust_score": -1.0, "error": str(exc)}]

    sig_is = sig.iloc[:oos_cut]
    sig_oos = sig.iloc[oos_cut:]

    rows = []
    for variant in variants:
        params = {**entry_params, **variant}
        try:
            rows.append(_row_from_signals(sig, sig_is, sig_oos, params, min_trades))
        except Exception as exc:
            rows.append({**params, "robust_score": -1.0, "error": str(exc)})
    return rows


def _row_from_signals(sig, sig_is, sig_oos, params, min_trades):
    i = _bt(sig_is, params)
    s = _bt(sig_oos, params)
    f = _bt(sig, params)
    wf = walk_forward_stability(sig, params)

    row = {
        **params,
        "is_return": i["total_return_pct"], "is_pf": i["profit_factor"],
        "is_sharpe": i["sharpe"], "is_trades": i["trades"],
        "oos_return": s["total_return_pct"], "oos_pf": s["profit_factor"],
        "oos_sharpe": s["sharpe"], "oos_wr": s["win_rate_pct"],
        "oos_dd_pct": s["max_dd_pct"], "oos_trades": s["trades"],
        "full_return": f["total_return_pct"], "full_pf": f["profit_factor"],
        "full_sharpe": f["sharpe"], "full_wr": f["win_rate_pct"],
        "full_dd_pct": f["max_dd_pct"], "full_trades": f["trades"],
        **wf,
    }
    row["pf_retention"] = round(row["oos_pf"] / row["is_pf"], 3) if row["is_pf"] else 0.0
    row["robust_score"] = robustness_score(row, min_trades)
    return row


def main():
    p = argparse.ArgumentParser(description="Robustness-ranked refinement search")
    p.add_argument("--symbol", default="INJ/USDT")
    p.add_argument("--days", type=int, default=1825)
    p.add_argument("--oos-frac", type=float, default=0.30)
    p.add_argument("--min-trades", type=int, default=40,
                   help="Minimum trades required in BOTH halves")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--thresholds", default="-0.07,-0.08,-0.09,-0.10,-0.12")
    p.add_argument("--tps", default="0.03,0.04,0.05,0.06")
    p.add_argument("--sls", default="0.14,0.20")
    p.add_argument("--holds", default="24,36")
    p.add_argument("--maxpos", default="2,3")
    p.add_argument("--rsi-only", action="store_true",
                   help="Drop the no-RSI variant from the sweep")
    p.add_argument("--tag", default="refine", help="Output filename suffix")
    p.add_argument("--quick", action="store_true",
                   help="Tiny grid, for verifying the pool works")
    args = p.parse_args()

    slug = args.symbol.replace("/", "_")
    cache_path = CACHE_DIR / f"{slug}_{TIMEFRAME}_{args.days}d.pkl"
    df = load_features(args.symbol, args.days)
    dis, doos = split_is_oos(df, args.oos_frac)
    oos_cut = len(dis)  # workers slice pre-computed signals at this index

    print(f"\n{'='*78}")
    print(f"  ROBUSTNESS REFINEMENT: {args.symbol}")
    print(f"{'='*78}")
    print(f"  IS : {dis['timestamp'].min().date()} -> {dis['timestamp'].max().date()}")
    print(f"  OOS: {doos['timestamp'].min().date()} -> {doos['timestamp'].max().date()}")
    print(f"  Ranking: PF retention + OOS edge + walk-forward stability")
    print(f"  Hard gate: PF > 1 in BOTH halves, >={args.min_trades} trades each")
    print(f"{'='*78}")

    base = {
        "dip_threshold": -0.08, "use_rsi_filter": False, "use_volume_filter": True,
        "hold_timeout_hours": 24, "max_positions": 2,
        "take_profit_pct": 0.04, "stop_loss_pct": 0.20,
        "min_volume_mult": 0.5, "rsi_min": 10, "rsi_max": 40,
        "one_entry_per_dip_event": False, "dip_event_reset_pct": -0.04,
        "fail_fast_bars": None, "fail_fast_min_rebound_pct": 0.01,
    }

    # Conservative region: deeper dips, modest targets, RSI gating.
    rsi_variants = [
        (False, 10, 40),
        (True, 25, 50), (True, 25, 60),
        (True, 30, 50), (True, 30, 60),
        (True, 35, 55),
    ]
    def _nums(raw, cast=float):
        return [cast(x) for x in raw.split(",") if x.strip()]

    grid = {
        "dip_threshold": _nums(args.thresholds),
        "take_profit_pct": _nums(args.tps),
        "stop_loss_pct": _nums(args.sls),
        "hold_timeout_hours": _nums(args.holds, int),
        "max_positions": _nums(args.maxpos, int),
    }
    if args.rsi_only:
        rsi_variants = [v for v in rsi_variants if v[0]]
    if args.quick:
        grid = {"dip_threshold": [-0.08, -0.10], "take_profit_pct": [0.04],
                "stop_loss_pct": [0.20], "hold_timeout_hours": [24],
                "max_positions": [2]}
        rsi_variants = rsi_variants[:2]
    # Execution knobs: cheap to vary, they reuse the same signals.
    exec_keys = ["take_profit_pct", "stop_loss_pct", "hold_timeout_hours", "max_positions"]
    variants = [dict(zip(exec_keys, v))
                for v in itertools.product(*(grid[k] for k in exec_keys))]

    # Entry knobs: each combination costs one generate_signals() call.
    tasks = []
    for threshold in grid["dip_threshold"]:
        for use_rsi, rmin, rmax in rsi_variants:
            entry = dict(base)
            entry["dip_threshold"] = threshold
            entry["use_rsi_filter"] = use_rsi
            entry["rsi_min"] = rmin
            entry["rsi_max"] = rmax
            tasks.append((entry, variants, args.min_trades, oos_cut))

    n_configs = len(tasks) * len(variants)
    print(f"\n  {n_configs} configs = {len(tasks)} signal sets x {len(variants)} exec variants")
    print(f"  Each: IS + OOS + FULL + {26} walk-forward windows\n", flush=True)
    t0 = time.time()
    rows = []
    # fork: children inherit the loaded module state instead of re-importing
    # ccxt/pandas per worker, which is what stalled the spawn-based pool here.
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=ctx,
        initializer=_init_worker, initargs=(str(cache_path), args.oos_frac),
    ) as pool:
        for done, group in enumerate(pool.map(_group_task, tasks, chunksize=1), 1):
            rows.extend(group)
            el = time.time() - t0
            eta = (len(tasks) - done) / (done / el) if el else 0
            print(f"    signal set {done}/{len(tasks)}  "
                  f"({len(rows)}/{n_configs} configs)  "
                  f"{el:5.0f}s  ~{eta:4.0f}s left", flush=True)

    res = pd.DataFrame(rows).sort_values("robust_score", ascending=False).reset_index(drop=True)
    out = Path(__file__).parent / "matrix_runs" / f"{slug}_{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    survivors = res[res["robust_score"] > 0]
    print(f"\n{'='*78}")
    print(f"  {len(survivors)}/{len(res)} configs passed the both-halves-profitable gate")
    print(f"{'='*78}")

    cols = ["dip_threshold", "use_rsi_filter", "rsi_min", "rsi_max",
            "take_profit_pct", "stop_loss_pct", "hold_timeout_hours", "max_positions",
            "robust_score", "is_pf", "oos_pf", "pf_retention", "wf_stability",
            "oos_return", "oos_dd_pct", "full_return", "full_pf", "full_trades"]
    show = res.head(args.top)[cols].copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)
    print(f"\n  TOP {args.top} BY ROBUSTNESS\n")
    print(show.to_string(index=False))
    print(f"\n  Saved: matrix_runs/{slug}_refine.csv\n")


if __name__ == "__main__":
    main()
