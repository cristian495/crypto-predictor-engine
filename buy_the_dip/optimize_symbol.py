"""
optimize_symbol.py
Parameter search for Buy-the-Dip over the full parameter set.

Unlike matrix_test.py (which sweeps only threshold x rsi x volume and leaves
TP/SL/hold/max_positions at their global config values), this searches every
knob get_symbol_signal_params() can return.

Anti-overfitting design:
- History is split IS / OOS. The search NEVER sees OOS.
- IS score rewards consistency across IS sub-periods, not raw return.
- Finalists are ranked by IS score, then reported with their OOS result.
  A config that wins IS and collapses OOS is curve-fit, and it shows.

Usage:
    ../venv/bin/python optimize_symbol.py --symbol INJ/USDT
"""

import argparse
import itertools
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "..")

from breakout_momentum.data_loader import fetch_ohlcv
from strategy import add_dip_features, generate_signals
from backtest import run_backtest
from config import INITIAL_CAPITAL, FEE_PCT, TIMEFRAME

CACHE_DIR = Path(__file__).parent / "opt_cache"

# Worker-global, populated by _init_worker to avoid pickling the frame per task
_DF_IS = None
_DF_OOS = None


# ── Data ────────────────────────────────────────────────────────

def load_features(symbol: str, days: int, refresh: bool = False) -> pd.DataFrame:
    """Fetch OHLCV + dip features, cached to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = symbol.replace("/", "_")
    cache = CACHE_DIR / f"{slug}_{TIMEFRAME}_{days}d.pkl"

    if cache.exists() and not refresh:
        return pd.read_pickle(cache)

    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=days)
    df = add_dip_features(df)
    df.to_pickle(cache)
    return df


def split_is_oos(df: pd.DataFrame, oos_frac: float):
    """Chronological split. OOS is the most recent tail."""
    cut = int(len(df) * (1 - oos_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ── Evaluation ──────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, params: dict, n_chunks: int = 0) -> dict:
    """Run signals + backtest for one config. Returns flat metrics dict."""
    sig = generate_signals(
        df,
        dip_threshold=params["dip_threshold"],
        use_volume_filter=params["use_volume_filter"],
        use_rsi_filter=params["use_rsi_filter"],
        min_volume_mult=params["min_volume_mult"],
        rsi_min=params["rsi_min"],
        rsi_max=params["rsi_max"],
        one_entry_per_dip_event=params["one_entry_per_dip_event"],
        dip_event_reset_pct=params["dip_event_reset_pct"],
    )

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
    out = {
        "total_return_pct": float(m.get("total_return", 0)) * 100,
        "sharpe": float(m.get("sharpe", 0)),
        "max_dd_pct": float(m.get("max_drawdown", 0)) * 100,
        "win_rate_pct": float(m.get("win_rate", 0)) * 100,
        "profit_factor": float(m.get("profit_factor", 0)),
        "trades": int(m.get("total_trades", 0)),
        "expectancy_pct": float(m.get("expectancy", 0)) * 100,
    }

    # Sub-period consistency, computed from the trade log (cheap - no re-run).
    if n_chunks > 0:
        trades = bt["trades"]
        out["chunks_positive"] = 0
        out["chunks_total"] = n_chunks
        out["worst_chunk_pct"] = 0.0

        if isinstance(trades, pd.DataFrame) and not trades.empty:
            edges = pd.date_range(
                df["timestamp"].iloc[0], df["timestamp"].iloc[-1],
                periods=n_chunks + 1,
            )
            bucket = pd.cut(trades["entry_time"], bins=edges, include_lowest=True)
            by_chunk = trades.groupby(bucket, observed=False)["pnl_after_fees"].sum()
            out["chunks_positive"] = int((by_chunk > 0).sum())
            out["worst_chunk_pct"] = float(by_chunk.min()) * 100 if len(by_chunk) else 0.0

    return out


def score_is(r: dict, min_trades: int) -> float:
    """
    Composite in-sample score, 0-100. Consistency-weighted on purpose:
    a config that only works in one regime should not win.
    """
    if r["trades"] < min_trades or r["expectancy_pct"] <= 0:
        return -1.0

    consistency = r.get("chunks_positive", 0) / max(r.get("chunks_total", 1), 1)
    pf = min(max((r["profit_factor"] - 1.0) / 0.8, 0.0), 1.0)
    sharpe = min(max(r["sharpe"] / 1.5, 0.0), 1.0)
    dd = max(0.0, 1.0 - abs(r["max_dd_pct"]) / 35.0)
    sample = min(r["trades"] / float(min_trades * 3), 1.0)

    score = (
        consistency * 0.35
        + pf * 0.25
        + sharpe * 0.20
        + dd * 0.10
        + sample * 0.10
    )
    return round(score * 100, 2)


# ── Parallel grid ───────────────────────────────────────────────

def _init_worker(cache_path: str, oos_frac: float):
    global _DF_IS, _DF_OOS
    df = pd.read_pickle(cache_path)
    _DF_IS, _DF_OOS = split_is_oos(df, oos_frac)


def _eval_task(args):
    params, n_chunks, min_trades = args
    try:
        r = evaluate(_DF_IS, params, n_chunks=n_chunks)
    except Exception as exc:  # keep the sweep alive on a bad combo
        return {**params, "score": -1.0, "error": str(exc)}
    r["score"] = score_is(r, min_trades)
    return {**params, **r}


def run_grid(grid: dict, base: dict, cache_path: Path, oos_frac: float,
             n_chunks: int, min_trades: int, label: str) -> pd.DataFrame:
    """Cartesian product over `grid`, other keys taken from `base`."""
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    tasks = []
    for values in combos:
        params = dict(base)
        params.update(dict(zip(keys, values)))
        tasks.append((params, n_chunks, min_trades))

    print(f"\n  {label}: {len(tasks)} configs across {len(keys)} dims")
    print(f"    varying: {', '.join(keys)}")

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(
        initializer=_init_worker, initargs=(str(cache_path), oos_frac)
    ) as pool:
        for i, row in enumerate(pool.map(_eval_task, tasks, chunksize=4), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(tasks):
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (len(tasks) - i) / rate if rate else 0
                print(f"    {i}/{len(tasks)}  {el:5.0f}s elapsed  ~{eta:4.0f}s left")

    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ── Reporting ───────────────────────────────────────────────────

PARAM_KEYS = [
    "dip_threshold", "use_rsi_filter", "use_volume_filter",
    "hold_timeout_hours", "max_positions", "take_profit_pct", "stop_loss_pct",
    "min_volume_mult", "rsi_min", "rsi_max",
    "one_entry_per_dip_event", "dip_event_reset_pct",
    "fail_fast_bars", "fail_fast_min_rebound_pct",
]


def print_table(df: pd.DataFrame, cols: list, n: int, title: str):
    print(f"\n  {title}")
    show = df.head(n)[cols].copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)
    print(show.to_string(index=False))


def emit_config_block(symbol: str, params: dict, notes: list) -> str:
    """Render a paste-ready SYMBOL_SIGNAL_OVERRIDES entry."""
    lines = [f'    "{symbol}": {{']
    for k in PARAM_KEYS:
        v = params[k]
        if isinstance(v, float):
            v = f"{v:g}"
        lines.append(f'        "{k}": {v},')
    lines.append("    },")
    for note in notes:
        lines.append(f"    # {note}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Full-parameter search for Buy-the-Dip")
    p.add_argument("--symbol", default="INJ/USDT")
    p.add_argument("--days", type=int, default=1825)
    p.add_argument("--oos-frac", type=float, default=0.30,
                   help="Fraction of history held out from the search")
    p.add_argument("--chunks", type=int, default=6,
                   help="IS sub-periods used for the consistency term")
    p.add_argument("--min-trades", type=int, default=60,
                   help="Minimum IS trades for a config to be scorable")
    p.add_argument("--finalists", type=int, default=15)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    slug = args.symbol.replace("/", "_")
    cache_path = CACHE_DIR / f"{slug}_{TIMEFRAME}_{args.days}d.pkl"

    print(f"\n{'='*74}")
    print(f"  PARAMETER SEARCH: {args.symbol}")
    print(f"{'='*74}")

    df = load_features(args.symbol, args.days, refresh=args.refresh)
    df_is, df_oos = split_is_oos(df, args.oos_frac)

    print(f"  Rows:  {len(df)}  ({df['timestamp'].min().date()} -> {df['timestamp'].max().date()})")
    print(f"  IS:    {len(df_is):>6} rows  {df_is['timestamp'].min().date()} -> {df_is['timestamp'].max().date()}")
    print(f"  OOS:   {len(df_oos):>6} rows  {df_oos['timestamp'].min().date()} -> {df_oos['timestamp'].max().date()}  (held out)")
    print(f"{'='*74}")

    base = {
        "dip_threshold": -0.08,
        "use_rsi_filter": False,
        "use_volume_filter": True,
        "hold_timeout_hours": 24,
        "max_positions": 2,
        "take_profit_pct": 0.04,
        "stop_loss_pct": 0.20,
        "min_volume_mult": 0.5,
        "rsi_min": 10,
        "rsi_max": 40,
        "one_entry_per_dip_event": False,
        "dip_event_reset_pct": -0.04,
        "fail_fast_bars": None,
        "fail_fast_min_rebound_pct": 0.01,
    }

    kw = dict(cache_path=cache_path, oos_frac=args.oos_frac,
              n_chunks=args.chunks, min_trades=args.min_trades)

    # ── Stage 1: entry/exit core ────────────────────────────────
    stage1 = run_grid(
        grid={
            "dip_threshold": [-0.05, -0.06, -0.07, -0.08, -0.10, -0.12],
            "take_profit_pct": [0.03, 0.04, 0.06, 0.08],
            "stop_loss_pct": [0.10, 0.14, 0.20],
            "hold_timeout_hours": [12, 24, 36, 48],
            "max_positions": [2, 4],
        },
        base=base, label="STAGE 1 (entry/exit core)", **kw,
    )
    print_table(stage1, ["dip_threshold", "take_profit_pct", "stop_loss_pct",
                         "hold_timeout_hours", "max_positions", "score",
                         "total_return_pct", "sharpe", "profit_factor",
                         "trades", "chunks_positive"], 10, "STAGE 1 TOP 10")

    best1 = {k: stage1.iloc[0][k] for k in
             ["dip_threshold", "take_profit_pct", "stop_loss_pct",
              "hold_timeout_hours", "max_positions"]}
    base2 = {**base, **best1}
    base2["hold_timeout_hours"] = int(base2["hold_timeout_hours"])
    base2["max_positions"] = int(base2["max_positions"])

    # ── Stage 2: filters ────────────────────────────────────────
    stage2 = run_grid(
        grid={
            "use_rsi_filter": [False, True],
            "rsi_min": [10, 20, 25, 30],
            "rsi_max": [40, 50, 60, 70],
            "use_volume_filter": [False, True],
            "min_volume_mult": [0.5, 0.8, 1.2],
        },
        base=base2, label="STAGE 2 (filters)", **kw,
    )
    print_table(stage2, ["use_rsi_filter", "rsi_min", "rsi_max",
                         "use_volume_filter", "min_volume_mult", "score",
                         "total_return_pct", "sharpe", "profit_factor",
                         "trades", "chunks_positive"], 10, "STAGE 2 TOP 10")

    best2 = {k: stage2.iloc[0][k] for k in
             ["use_rsi_filter", "rsi_min", "rsi_max",
              "use_volume_filter", "min_volume_mult"]}
    base3 = {**base2, **best2}
    base3["use_rsi_filter"] = bool(base3["use_rsi_filter"])
    base3["use_volume_filter"] = bool(base3["use_volume_filter"])

    # ── Stage 3: event control + fail-fast, refining the core ───
    stage3 = run_grid(
        grid={
            "one_entry_per_dip_event": [False, True],
            "dip_event_reset_pct": [-0.04, -0.02],
            "fail_fast_bars": [None, 6, 10, 14],
            "fail_fast_min_rebound_pct": [0.01, 0.02],
            "take_profit_pct": sorted({base3["take_profit_pct"],
                                       base3["take_profit_pct"] - 0.01,
                                       base3["take_profit_pct"] + 0.01}),
            "hold_timeout_hours": sorted({base3["hold_timeout_hours"],
                                          base3["hold_timeout_hours"] - 4,
                                          base3["hold_timeout_hours"] + 4}),
        },
        base=base3, label="STAGE 3 (event control + fail-fast)", **kw,
    )
    print_table(stage3, ["one_entry_per_dip_event", "fail_fast_bars",
                         "fail_fast_min_rebound_pct", "take_profit_pct",
                         "hold_timeout_hours", "score", "total_return_pct",
                         "sharpe", "profit_factor", "trades",
                         "chunks_positive"], 10, "STAGE 3 TOP 10")

    # ── Finalists: score on the held-out OOS ────────────────────
    pool = pd.concat([stage1, stage2, stage3], ignore_index=True)
    pool = pool[pool["score"] > 0].drop_duplicates(subset=PARAM_KEYS)
    pool = pool.sort_values("score", ascending=False).head(args.finalists)

    print(f"\n{'='*74}")
    print(f"  OUT-OF-SAMPLE VALIDATION ({len(pool)} finalists, never seen by search)")
    print(f"{'='*74}")

    final_rows = []
    for _, row in pool.iterrows():
        params = {k: row[k] for k in PARAM_KEYS}
        params["hold_timeout_hours"] = int(params["hold_timeout_hours"])
        params["max_positions"] = int(params["max_positions"])
        params["use_rsi_filter"] = bool(params["use_rsi_filter"])
        params["use_volume_filter"] = bool(params["use_volume_filter"])
        params["one_entry_per_dip_event"] = bool(params["one_entry_per_dip_event"])
        if pd.isna(params["fail_fast_bars"]):
            params["fail_fast_bars"] = None
        else:
            params["fail_fast_bars"] = int(params["fail_fast_bars"])

        oos = evaluate(df_oos, params, n_chunks=3)
        full = evaluate(df, params, n_chunks=0)

        final_rows.append({
            **params,
            "is_score": row["score"],
            "is_return": row["total_return_pct"],
            "is_pf": row["profit_factor"],
            "is_sharpe": row["sharpe"],
            "is_trades": row["trades"],
            "oos_return": oos["total_return_pct"],
            "oos_pf": oos["profit_factor"],
            "oos_sharpe": oos["sharpe"],
            "oos_wr": oos["win_rate_pct"],
            "oos_dd": oos["max_dd_pct"],
            "oos_trades": oos["trades"],
            "oos_chunks_pos": oos["chunks_positive"],
            "full_return": full["total_return_pct"],
            "full_pf": full["profit_factor"],
            "full_sharpe": full["sharpe"],
            "full_trades": full["trades"],
        })

    fin = pd.DataFrame(final_rows)
    # Degradation: how much of the IS profit factor survives OOS
    fin["pf_retention"] = (fin["oos_pf"] / fin["is_pf"]).round(3)
    fin = fin.sort_values(["oos_pf", "is_score"], ascending=False).reset_index(drop=True)

    print_table(fin, ["dip_threshold", "take_profit_pct", "stop_loss_pct",
                      "hold_timeout_hours", "max_positions", "use_rsi_filter",
                      "use_volume_filter", "is_score", "is_pf", "oos_pf",
                      "pf_retention", "oos_return", "oos_sharpe", "oos_trades",
                      "oos_chunks_pos"], len(fin),
                "FINALISTS (sorted by OOS profit factor)")

    out_dir = Path(__file__).parent / "matrix_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fin.to_csv(out_dir / f"{slug}_optimize.csv", index=False)
    pool.to_csv(out_dir / f"{slug}_optimize_is.csv", index=False)

    # Winner: best OOS PF among configs that also held up in-sample
    robust = fin[(fin["oos_pf"] > 1.0) & (fin["pf_retention"] > 0.75)]
    winner = (robust if not robust.empty else fin).iloc[0]
    wp = {k: winner[k] for k in PARAM_KEYS}
    wp["hold_timeout_hours"] = int(wp["hold_timeout_hours"])
    wp["max_positions"] = int(wp["max_positions"])
    wp["fail_fast_bars"] = None if pd.isna(wp["fail_fast_bars"]) else int(wp["fail_fast_bars"])

    print(f"\n{'='*74}")
    print("  RECOMMENDED CONFIG")
    print(f"{'='*74}\n")
    print(emit_config_block(args.symbol, wp, [
        f"IS  ({int(winner['is_trades'])} trades): "
        f"return {winner['is_return']:.2f}%  PF {winner['is_pf']:.2f}  "
        f"Sharpe {winner['is_sharpe']:.2f}",
        f"OOS ({int(winner['oos_trades'])} trades): "
        f"return {winner['oos_return']:.2f}%  PF {winner['oos_pf']:.2f}  "
        f"Sharpe {winner['oos_sharpe']:.2f}  WR {winner['oos_wr']:.1f}%  "
        f"DD {winner['oos_dd']:.2f}%",
        f"FULL ({int(winner['full_trades'])} trades): "
        f"return {winner['full_return']:.2f}%  PF {winner['full_pf']:.2f}  "
        f"Sharpe {winner['full_sharpe']:.2f}",
        f"PF retention IS->OOS: {winner['pf_retention']:.2f}",
    ]))
    print(f"\n  Saved: matrix_runs/{slug}_optimize.csv")
    print(f"{'='*74}\n")


if __name__ == "__main__":
    main()
