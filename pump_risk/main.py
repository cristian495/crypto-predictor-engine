"""
main.py
CLI for Pump Risk detector.

Usage:
    python main.py --scan
    python main.py --scan --target-date 2026-03-16 --symbols XRP/USDT
    python main.py --walk-forward --symbols XRP/USDT --show-dates
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, "..")

from breakout_momentum.data_loader import fetch_ohlcv, fetch_funding_rates, fetch_open_interest
from config import (
    TIMEFRAME,
    DAYS,
    DEFAULT_SYMBOLS,
    HIGH_RISK_MIN_SCORE,
    MEDIUM_RISK_MIN_SCORE,
    PUMP_LABEL_MODE,
    SCAN_MODE,
    SCAN_SIGNAL_LEVEL,
    SCAN_USE_INTRADAY_TRIGGER,
    TRIGGER_TIMEFRAME,
    TRIGGER_BREAKOUT_BUFFER_PCT,
    TRIGGER_MIN_REL_VOLUME,
    TRIGGER_MIN_CLOSE_POS,
    TRIGGER_REQUIRE_GREEN,
    TRIGGER_VOL_LOOKBACK_BARS,
)
from strategy import (
    add_pump_risk_features,
    build_intraday_trigger_table,
    build_signal_mask,
    label_columns,
    explain_row_conditions,
    run_walk_forward_evaluation,
    signal_threshold,
)


def _build_signal_payload(
    symbol: str,
    stage: str,
    signal_date,
    target_date,
    score: float,
    risk_bucket: str,
    base_signal: bool,
    confirm_signal: bool,
    trigger_row: dict | None,
) -> dict:
    rel_vol = float(trigger_row.get("trigger_rel_vol", float("nan"))) if trigger_row else float("nan")
    close_pos = float(trigger_row.get("trigger_close_pos", float("nan"))) if trigger_row else float("nan")
    trigger_ts = trigger_row.get("trigger_ts", pd.NaT) if trigger_row else pd.NaT
    trigger_ts_str = "-" if pd.isna(trigger_ts) else str(trigger_ts)
    prob = min(0.99, max(0.0, float(score) / max(HIGH_RISK_MIN_SCORE, 1.0)))

    return {
        "symbol": symbol,
        "signal": "LONG",
        "prob": prob,
        "extra": (
            f"pump_risk stage={stage} score={score:.1f} risk={risk_bucket} "
            f"early={base_signal} confirm={confirm_signal} "
            f"signal_date={signal_date} target_date={target_date} "
            f"trigger_ts={trigger_ts_str} rel_vol={rel_vol:.2f} close_pos={close_pos:.2f}"
        ),
    }


def _write_signals_json(path: str, signals: list[dict]) -> None:
    payload = {
        "strategy": "pump_risk",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
    }
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=True)


def _select_signal_row_for_target(df: pd.DataFrame, target_date) -> pd.Series | None:
    """
    Select daily row used as pre-signal for target_date (target_date = signal_date + 1 day).
    Prefer exact target_date-1, then fallback to latest row before target_date.
    """
    preferred_signal_date = target_date - timedelta(days=1)
    exact = df[df["date"] == preferred_signal_date]
    if not exact.empty:
        return exact.iloc[-1]

    prior = df[df["date"] < target_date]
    if prior.empty:
        return None
    return prior.iloc[-1]


def _normalize_scan_mode(scan_mode: str | None) -> str:
    mode = (scan_mode or SCAN_MODE).strip().lower()
    if mode not in {"early", "confirm", "both"}:
        raise ValueError(f"unsupported scan mode: {scan_mode!r}")
    return mode


def _binary_metrics(signal: pd.Series, target: pd.Series) -> dict:
    s = signal.astype(bool)
    t = target.astype(bool)
    tp = int((s & t).sum())
    fp = int((s & ~t).sum())
    fn = int((~s & t).sum())
    signals = int(s.sum())
    pumps = int(t.sum())
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    return {
        "signals": signals,
        "pumps": pumps,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def _compute_active_eval(
    df: pd.DataFrame,
    signal_level: str,
    label_mode: str,
    scan_mode: str,
    use_intraday_trigger: bool,
    trigger_table: pd.DataFrame | None = None,
) -> dict:
    mode = _normalize_scan_mode(scan_mode)
    target_col, _ = label_columns(label_mode)
    ev = df.dropna(subset=[target_col]).copy().reset_index(drop=True)
    if ev.empty:
        return {
            "signals": 0,
            "pumps": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "early_signals": 0,
            "confirm_signals": 0,
        }

    early = build_signal_mask(ev, signal_level=signal_level, label_mode=label_mode).astype(bool)
    confirm = pd.Series(False, index=ev.index, dtype=bool)

    if mode in {"confirm", "both"}:
        if use_intraday_trigger:
            hit_map: dict = {}
            if trigger_table is not None and not trigger_table.empty:
                hit_map = {
                    row["target_date"]: bool(row["trigger_hit"])
                    for _, row in trigger_table.iterrows()
                }
            target_dates = (pd.to_datetime(ev["date"]) + pd.Timedelta(days=1)).dt.date
            trig = target_dates.map(hit_map).fillna(False).astype(bool)
            confirm = early & trig
        else:
            # Bypass mode: treat confirm as early when trigger is disabled.
            confirm = early.copy()

    if mode == "early":
        active_signal = early
    elif mode == "confirm":
        active_signal = confirm
    else:
        active_signal = early | confirm

    target = ev[target_col].astype(bool)
    out = _binary_metrics(active_signal, target)
    out["early_signals"] = int(early.sum())
    out["confirm_signals"] = int(confirm.sum())
    return out


def main_scan(
    symbols: list[str] | None = None,
    timeframe: str = TIMEFRAME,
    days: int = DAYS,
    signals_json: str | None = None,
    scan_mode: str = SCAN_MODE,
    label_mode: str = PUMP_LABEL_MODE,
    signal_level: str = SCAN_SIGNAL_LEVEL,
    use_intraday_trigger: bool = SCAN_USE_INTRADAY_TRIGGER,
    trigger_timeframe: str = TRIGGER_TIMEFRAME,
    trigger_breakout_buffer_pct: float = TRIGGER_BREAKOUT_BUFFER_PCT,
    trigger_min_rel_volume: float = TRIGGER_MIN_REL_VOLUME,
    trigger_min_close_pos: float = TRIGGER_MIN_CLOSE_POS,
    trigger_require_green: bool = TRIGGER_REQUIRE_GREEN,
    trigger_vol_lookback_bars: int = TRIGGER_VOL_LOOKBACK_BARS,
    target_date: str | None = None,
):
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    mode = _normalize_scan_mode(scan_mode)
    scan_target_date = (
        pd.to_datetime(target_date).date()
        if target_date
        else datetime.now(timezone.utc).date()
    )
    needs_trigger_eval = mode in {"confirm", "both"}

    print("=" * 70)
    print("  PUMP-RISK DETECTOR — MULTI-CRYPTO ENTRY SCANNER")
    print("=" * 70)
    print("  Mode:          Scanner")
    print(f"  Scan mode:     {mode}")
    print(f"  Symbols:       {len(symbols)}")
    print(f"  Timeframe:     {timeframe}")
    print(f"  History:       {days} days")
    print(f"  Signal level:  {signal_level.upper()} (score >= {signal_threshold(signal_level):.1f})")
    print(f"  Risk buckets:  HIGH >= {HIGH_RISK_MIN_SCORE:.1f}, MEDIUM >= {MEDIUM_RISK_MIN_SCORE:.1f}")
    print(f"  Pump label:    {label_mode} (threshold +3.0%)")
    print(f"  Target date:   {scan_target_date} UTC")
    if needs_trigger_eval and use_intraday_trigger:
        print(
            "  Trigger:       "
            f"on ({trigger_timeframe}, break>{trigger_breakout_buffer_pct*100:.2f}%, "
            f"rel_vol>={trigger_min_rel_volume:.2f}, close_pos>={trigger_min_close_pos:.2f}, "
            f"green={trigger_require_green}, vol_lb={trigger_vol_lookback_bars})"
        )
    elif needs_trigger_eval and (not use_intraday_trigger):
        print("  Trigger:       bypassed (--no-intraday-trigger)")
    else:
        print("  Trigger:       not required for this scan mode")
    print("=" * 70)

    print("\n[1/3] Processing symbols...")
    results: dict[str, dict] = {}
    signals: list[dict] = []

    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(symbols)}] {symbol}")
        print("─" * 60)
        try:
            ohlcv = fetch_ohlcv(symbol=symbol, timeframe=timeframe, days=days)
            funding = fetch_funding_rates(symbol=symbol, days=days)
            oi = fetch_open_interest(symbol=symbol, timeframe=timeframe, days=days)
            df = add_pump_risk_features(ohlcv, funding_df=funding, oi_df=oi)

            if df.empty:
                raise RuntimeError("empty feature dataset")

            signal_row = _select_signal_row_for_target(df, target_date=scan_target_date)
            if signal_row is None:
                raise RuntimeError("no valid signal row before target date")
            signal_date = signal_row["date"]
            signal_df = pd.DataFrame([signal_row])

            base_signal = bool(build_signal_mask(signal_df, signal_level=signal_level, label_mode=label_mode).iloc[0])
            early_long = base_signal
            score = float(signal_row["pump_risk_score"])
            risk_bucket = str(signal_row["pump_risk"])

            trigger_hit = True
            trigger_row: dict | None = None
            trigger_table: pd.DataFrame | None = None
            if needs_trigger_eval and use_intraday_trigger:
                intraday = fetch_ohlcv(symbol=symbol, timeframe=trigger_timeframe, days=days)
                trigger_table = build_intraday_trigger_table(
                    intraday_df=intraday,
                    daily_df=df,
                    breakout_buffer_pct=trigger_breakout_buffer_pct,
                    min_rel_volume=trigger_min_rel_volume,
                    min_close_pos=trigger_min_close_pos,
                    require_green=trigger_require_green,
                    vol_lookback_bars=trigger_vol_lookback_bars,
                )
                trow = trigger_table[trigger_table["target_date"] == scan_target_date]
                if trow.empty:
                    trigger_hit = False
                else:
                    trigger_row = trow.iloc[0].to_dict()
                    trigger_hit = bool(trigger_row["trigger_hit"])

            if needs_trigger_eval and use_intraday_trigger:
                confirm_long = base_signal and trigger_hit
            elif needs_trigger_eval and (not use_intraday_trigger):
                confirm_long = base_signal
            else:
                confirm_long = False

            if mode == "early":
                entry_long = early_long
                entry_stage = "EARLY"
            elif mode == "confirm":
                entry_long = confirm_long
                entry_stage = "CONFIRM"
            else:
                entry_long = early_long or confirm_long
                if early_long and confirm_long:
                    entry_stage = "BOTH"
                elif confirm_long:
                    entry_stage = "CONFIRM"
                else:
                    entry_stage = "EARLY"

            eval_active = _compute_active_eval(
                df=df,
                signal_level=signal_level,
                label_mode=label_mode,
                scan_mode=mode,
                use_intraday_trigger=use_intraday_trigger,
                trigger_table=trigger_table,
            )
            last = df.iloc[-1]

            print(
                "  Current: "
                f"Risk={last['pump_risk']:<6} "
                f"Score={last['pump_risk_score']:.1f}  "
                f"Close={last['close']:.4f}  "
                f"Ret1={last['ret1']*100:+.2f}%"
            )
            print(
                "  Entry:   "
                f"{'LONG SIGNAL!' if entry_long else 'NO SIGNAL'} "
                f"(stage={entry_stage}, signal_date={signal_date}, target_date={scan_target_date}, score={score:.1f})"
            )
            trigger_ts = trigger_row.get("trigger_ts", pd.NaT) if trigger_row else pd.NaT
            trigger_rel_vol = trigger_row.get("trigger_rel_vol", float("nan")) if trigger_row else float("nan")
            trigger_close_pos = trigger_row.get("trigger_close_pos", float("nan")) if trigger_row else float("nan")
            if needs_trigger_eval and use_intraday_trigger:
                trigger_state = "PASS" if trigger_hit else "FAIL"
                trigger_detail = (
                    f"hit={trigger_hit}, ts={trigger_ts}, rel_vol={float(trigger_rel_vol):.2f}, "
                    f"close_pos={float(trigger_close_pos):.2f}"
                )
            elif needs_trigger_eval and (not use_intraday_trigger):
                trigger_state = "BYPASS"
                trigger_detail = "not evaluated (--no-intraday-trigger)"
            else:
                trigger_state = "N/A"
                trigger_detail = "not required for scan_mode=early"
            print(
                "               Conditions:\n"
                f"                 - base_signal     {'PASS' if base_signal else 'FAIL'} | "
                f"signal_level={signal_level}, score={score:.1f}, risk={risk_bucket}\n"
                f"                 - intraday_trigger {trigger_state:<6} | {trigger_detail}\n"
                f"                 - entry_mode      {'PASS' if entry_long else 'FAIL'} | "
                f"mode={mode}, early={early_long}, confirm={confirm_long}, chosen={entry_stage}"
            )
            print(
                "  Eval(active): "
                f"Precision={eval_active['precision']:.1%}  "
                f"Recall={eval_active['recall']:.1%}  "
                f"Signals={eval_active['signals']}  "
                f"Pumps={eval_active['pumps']}  "
                f"TP={eval_active['tp']} FP={eval_active['fp']} FN={eval_active['fn']}  "
                f"(early={eval_active['early_signals']}, confirm={eval_active['confirm_signals']})"
            )

            results[symbol] = {
                "df": df,
                "last": last,
                "signal_row": signal_row,
                "eval_active": eval_active,
                "entry_long": entry_long,
                "entry_stage": entry_stage,
                "signal_date": signal_date,
                "target_date": scan_target_date,
                "score": score,
                "early_long": early_long,
                "confirm_long": confirm_long,
                "trigger_hit": trigger_hit,
                "trigger_state": trigger_state,
            }

            if entry_long:
                signals.append(
                    _build_signal_payload(
                        symbol=symbol,
                        stage=entry_stage,
                        signal_date=signal_date,
                        target_date=scan_target_date,
                        score=score,
                        risk_bucket=risk_bucket,
                        base_signal=early_long,
                        confirm_signal=confirm_long,
                        trigger_row=trigger_row,
                    )
                )

        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not results:
        print("No symbols processed successfully.")
        return None

    print("\n[2/3] Signal-date and current conditions...")
    print("\n" + "=" * 70)
    print("  CONDITIONS — Entry Row vs Current Row")
    print("=" * 70)
    for symbol, item in results.items():
        signal_row = item["signal_row"]
        last = item["last"]
        print(f"  {symbol}")
        print(
            "               Signal row (used for Entry): "
            f"date={item['signal_date']} risk={signal_row['pump_risk']} "
            f"score={signal_row['pump_risk_score']:.1f} ret1={signal_row['ret1']*100:+.2f}%"
        )
        for name, ok, detail in explain_row_conditions(signal_row):
            state = "PASS" if ok else "FAIL"
            print(f"                 - {name:<15} {state:<4} | {detail}")
        print(
            "               Current row (latest): "
            f"date={last['date']} risk={last['pump_risk']} "
            f"score={last['pump_risk_score']:.1f} ret1={last['ret1']*100:+.2f}%"
        )
        for name, ok, detail in explain_row_conditions(last):
            state = "PASS" if ok else "FAIL"
            print(f"                 - {name:<15} {state:<4} | {detail}")
    print("=" * 70)

    print("\n[3/3] Summary...")
    print("\n" + "=" * 70)
    print("  SUMMARY — Pump Risk + Entry Gate (Active Eval)")
    print("=" * 70)
    print(
        f"{'Symbol':<12}{'Risk':>8}{'Score':>8}{'Entry':>8}{'Stage':>9}{'Early':>8}{'Cfm':>8}{'Trig':>8}"
        f"{'Prec(A)':>9}{'Rec(A)':>9}{'Sig(A)':>8}{'Pumps':>8}"
    )
    print("-" * 70)
    for symbol, item in results.items():
        last = item["last"]
        a = item["eval_active"]
        print(
            f"{symbol:<12}"
            f"{str(last['pump_risk']):>8}"
            f"{item['score']:>8.1f}"
            f"{('LONG' if item['entry_long'] else 'NONE'):>8}"
            f"{item['entry_stage']:>9}"
            f"{str(item['early_long']):>8}"
            f"{str(item['confirm_long']):>8}"
            f"{item['trigger_state']:>8}"
            f"{a['precision']:>9.1%}"
            f"{a['recall']:>9.1%}"
            f"{a['signals']:>8}"
            f"{a['pumps']:>8}"
        )
    print("=" * 70)

    if signals_json:
        _write_signals_json(signals_json, signals)
        print(f"Signals JSON written to: {signals_json}")

    return results


def _print_wf_summary(symbol: str, wf: dict, signal_level: str) -> None:
    agg = wf["aggregate"]
    avg = wf["avg_metrics"]
    med = wf["median_metrics"]
    windows = wf["window_metrics"]

    print(
        f"  Aggregate ({signal_level.upper()}): "
        f"signals={agg['signals']} pumps={agg['pumps']} "
        f"tp={agg['tp']} fp={agg['fp']} fn={agg['fn']} "
        f"prec={agg['precision']:.1%} rec={agg['recall']:.1%} "
        f"f1={agg['f1']:.1%} base={agg['base_rate']:.1%} lift={agg['lift']:.2f}"
    )
    print(
        f"  Label mode: {wf.get('label_mode', PUMP_LABEL_MODE)} "
        f"(target={wf.get('target_column', 'pump_next')})"
    )
    default_rule = f"score>={wf.get('threshold', 0):.1f}"
    print(f"  Signal rule: {wf.get('signal_rule', default_rule)}")
    if wf.get("use_intraday_trigger", False):
        p = wf.get("trigger_params", {})
        print(
            "  Trigger: "
            f"enabled (break>{p.get('breakout_buffer_pct', 0.0)*100:.2f}%, "
            f"rel_vol>={p.get('min_rel_volume', 0.0):.2f}, "
            f"close_pos>={p.get('min_close_pos', 0.0):.2f}, "
            f"green={p.get('require_green', True)}, "
            f"vol_lb={int(p.get('vol_lookback_bars', 24))})"
        )
    print(
        "  Window stats: "
        f"windows={len(windows)}  "
        f"avg_prec={avg.get('precision', 0.0):.1%}  "
        f"med_prec={med.get('precision', 0.0):.1%}  "
        f"avg_rec={avg.get('recall', 0.0):.1%}  "
        f"med_rec={med.get('recall', 0.0):.1%}"
    )


def _print_wf_windows_table(window_rows: list[dict]) -> None:
    if not window_rows:
        print("No walk-forward windows generated.")
        return
    wdf = pd.DataFrame(window_rows)
    cols = [
        "window_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "signals",
        "pumps",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "lift",
    ]
    print(
        wdf[cols]
        .to_string(
            index=False,
            formatters={
                "precision": "{:.1%}".format,
                "recall": "{:.1%}".format,
                "f1": "{:.1%}".format,
                "lift": "{:.2f}".format,
            },
        )
    )


def _print_detection_dates(detections: pd.DataFrame, signal_level: str, label_mode: str) -> None:
    print("\n" + "=" * 70)
    print(f"  DETECTION DATES ({signal_level.upper()})")
    print("=" * 70)
    if detections.empty:
        print("No detections for selected level.")
        return

    ddf = detections.copy()
    ddf = ddf.sort_values(["signal_date", "window_id"]).reset_index(drop=True)
    has_trigger_cols = "trigger_ts" in ddf.columns
    trigger_cols = ["trigger_ts", "trigger_rel_vol", "trigger_close_pos"] if has_trigger_cols else []
    if label_mode in {"either", "both"}:
        print(
            ddf[
                [
                    "window_id",
                    "signal_date",
                    "target_date",
                    "score",
                    "pump_next",
                    "next_intraday_pct",
                    "next_ret_pct",
                    *trigger_cols,
                ]
            ].to_string(
                index=False,
                formatters={
                    "score": "{:.1f}".format,
                    "next_intraday_pct": "{:+.2f}%".format,
                    "next_ret_pct": "{:+.2f}%".format,
                    "trigger_rel_vol": "{:.2f}".format,
                    "trigger_close_pos": "{:.2f}".format,
                },
            )
        )
    else:
        move_col = "next_intraday_pct" if label_mode == "intraday" else "next_ret_pct"
        print(
            ddf[["window_id", "signal_date", "target_date", "score", "pump_next", move_col, *trigger_cols]]
            .to_string(
                index=False,
                formatters={
                    "score": "{:.1f}".format,
                    move_col: "{:+.2f}%".format,
                    "trigger_rel_vol": "{:.2f}".format,
                    "trigger_close_pos": "{:.2f}".format,
                },
            )
        )
    if ddf["signal_date"].duplicated().any():
        print("\n  UNIQUE DATES (for chart validation)")
        if label_mode in {"either", "both"}:
            uniq = (
                ddf.groupby(["signal_date", "target_date"], as_index=False)
                .agg(
                    score=("score", "max"),
                    pump_next=("pump_next", "max"),
                    next_intraday_pct=("next_intraday_pct", "mean"),
                    next_ret_pct=("next_ret_pct", "mean"),
                    windows=("window_id", "nunique"),
                )
                .sort_values("signal_date")
            )
            print(
                uniq[
                    [
                        "signal_date",
                        "target_date",
                        "score",
                        "pump_next",
                        "next_intraday_pct",
                        "next_ret_pct",
                        "windows",
                    ]
                ].to_string(
                    index=False,
                    formatters={
                        "score": "{:.1f}".format,
                        "next_intraday_pct": "{:+.2f}%".format,
                        "next_ret_pct": "{:+.2f}%".format,
                    },
                )
            )
        else:
            move_col = "next_intraday_pct" if label_mode == "intraday" else "next_ret_pct"
            uniq = (
                ddf.groupby(["signal_date", "target_date"], as_index=False)
                .agg(
                    score=("score", "max"),
                    pump_next=("pump_next", "max"),
                    next_move_pct=(move_col, "mean"),
                    windows=("window_id", "nunique"),
                )
                .sort_values("signal_date")
            )
            print(
                uniq[["signal_date", "target_date", "score", "pump_next", "next_move_pct", "windows"]]
                .to_string(
                    index=False,
                    formatters={
                        "score": "{:.1f}".format,
                        "next_move_pct": "{:+.2f}%".format,
                    },
                )
            )
    total = len(ddf)
    hits = int(ddf["pump_next"].sum())
    hit_rate = (hits / total) if total else 0.0
    print("-" * 70)
    print(
        f"Total detections: {total} | Hits: {hits} | Hit rate: {hit_rate:.1%} | "
        f"Range: {ddf['signal_date'].iloc[0]} -> {ddf['signal_date'].iloc[-1]}"
    )


def main_walk_forward(
    symbols: list[str] | None = None,
    timeframe: str = TIMEFRAME,
    days: int = DAYS,
    signal_level: str = "high",
    label_mode: str = PUMP_LABEL_MODE,
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 30,
    use_intraday_trigger: bool = False,
    trigger_timeframe: str = TRIGGER_TIMEFRAME,
    trigger_breakout_buffer_pct: float = TRIGGER_BREAKOUT_BUFFER_PCT,
    trigger_min_rel_volume: float = TRIGGER_MIN_REL_VOLUME,
    trigger_min_close_pos: float = TRIGGER_MIN_CLOSE_POS,
    trigger_require_green: bool = TRIGGER_REQUIRE_GREEN,
    trigger_vol_lookback_bars: int = TRIGGER_VOL_LOOKBACK_BARS,
    show_dates: bool = False,
):
    if symbols is None:
        symbols = ["XRP/USDT"]
    threshold = signal_threshold(signal_level)

    print("=" * 70)
    print("  PUMP-RISK — INDEPENDENT WALK-FORWARD EVALUATION")
    print("=" * 70)
    print("  Mode:          Walk-Forward")
    print(f"  Symbols:       {len(symbols)}")
    print(f"  Timeframe:     {timeframe}")
    print(f"  History:       {days} days")
    print(f"  Signal level:  {signal_level.upper()} (score >= {threshold:.1f})")
    print(f"  Pump label:    {label_mode} (threshold +3.0%)")
    print(f"  WF windows:    train={train_days}d test={test_days}d step={step_days}d")
    if use_intraday_trigger:
        print(
            "  Trigger:       "
            f"on ({trigger_timeframe}, break>{trigger_breakout_buffer_pct*100:.2f}%, "
            f"rel_vol>={trigger_min_rel_volume:.2f}, close_pos>={trigger_min_close_pos:.2f}, "
            f"green={trigger_require_green}, vol_lb={trigger_vol_lookback_bars})"
        )
    else:
        print("  Trigger:       off")
    print("=" * 70)

    print("\n[1/3] Processing symbols...")
    results: dict[str, dict] = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(symbols)}] {symbol}")
        print("─" * 60)
        try:
            ohlcv = fetch_ohlcv(symbol=symbol, timeframe=timeframe, days=days)
            funding = fetch_funding_rates(symbol=symbol, days=days)
            oi = fetch_open_interest(symbol=symbol, timeframe=timeframe, days=days)
            df = add_pump_risk_features(ohlcv, funding_df=funding, oi_df=oi)
            if df.empty:
                raise RuntimeError("empty feature dataset")

            intraday = None
            if use_intraday_trigger:
                intraday = fetch_ohlcv(symbol=symbol, timeframe=trigger_timeframe, days=days)

            wf = run_walk_forward_evaluation(
                df=df,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                signal_level=signal_level,
                label_mode=label_mode,
                use_intraday_trigger=use_intraday_trigger,
                intraday_df=intraday,
                trigger_breakout_buffer_pct=trigger_breakout_buffer_pct,
                trigger_min_rel_volume=trigger_min_rel_volume,
                trigger_min_close_pos=trigger_min_close_pos,
                trigger_require_green=trigger_require_green,
                trigger_vol_lookback_bars=trigger_vol_lookback_bars,
            )
            _print_wf_summary(symbol, wf, signal_level)
            results[symbol] = {"df": df, "wf": wf}

        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not results:
        print("No symbols processed successfully.")
        return None

    print("\n[2/3] Window metrics...")
    print("\n" + "=" * 70)
    print("  WALK-FORWARD WINDOWS")
    print("=" * 70)
    for symbol, item in results.items():
        print(f"\n  {symbol}")
        _print_wf_windows_table(item["wf"]["window_metrics"])
    print("=" * 70)

    print("\n[3/3] Detection dates...")
    if show_dates:
        for symbol, item in results.items():
            print(f"\n  {symbol}")
            _print_detection_dates(item["wf"]["detections"], signal_level, label_mode=label_mode)
    else:
        print("Use --show-dates to print detection dates.")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pump-risk detector")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--timeframe", default=TIMEFRAME)
    p.add_argument("--days", type=int, default=DAYS)
    p.add_argument("--target-date", default=None, help="UTC target date for scan (YYYY-MM-DD)")
    p.add_argument("--signals-json", default=None)
    p.add_argument("--scan-mode", choices=["early", "confirm", "both"], default=SCAN_MODE)
    p.add_argument("--signal-level", choices=["high", "medium"], default=SCAN_SIGNAL_LEVEL)
    p.add_argument("--label-mode", choices=["close", "intraday", "either", "both"], default=PUMP_LABEL_MODE)
    p.add_argument("--train-days", type=int, default=180)
    p.add_argument("--test-days", type=int, default=60)
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--use-intraday-trigger", action="store_true")
    p.add_argument("--no-intraday-trigger", action="store_true")
    p.add_argument("--trigger-timeframe", default=TRIGGER_TIMEFRAME)
    p.add_argument("--trigger-breakout-buffer-pct", type=float, default=TRIGGER_BREAKOUT_BUFFER_PCT)
    p.add_argument("--trigger-min-rel-volume", type=float, default=TRIGGER_MIN_REL_VOLUME)
    p.add_argument("--trigger-min-close-pos", type=float, default=TRIGGER_MIN_CLOSE_POS)
    p.add_argument("--trigger-require-green", action="store_true")
    p.add_argument("--trigger-no-require-green", action="store_true")
    p.add_argument("--trigger-vol-lookback-bars", type=int, default=TRIGGER_VOL_LOOKBACK_BARS)
    p.add_argument("--show-dates", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_intraday_trigger = SCAN_USE_INTRADAY_TRIGGER
    if args.use_intraday_trigger:
        use_intraday_trigger = True
    if args.no_intraday_trigger:
        use_intraday_trigger = False

    trigger_require_green = TRIGGER_REQUIRE_GREEN
    if args.trigger_require_green:
        trigger_require_green = True
    if args.trigger_no_require_green:
        trigger_require_green = False

    if args.walk_forward:
        main_walk_forward(
            symbols=args.symbols if args.symbols else ["XRP/USDT"],
            timeframe=args.timeframe,
            days=args.days,
            signal_level=args.signal_level,
            label_mode=args.label_mode,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            use_intraday_trigger=use_intraday_trigger,
            trigger_timeframe=args.trigger_timeframe,
            trigger_breakout_buffer_pct=args.trigger_breakout_buffer_pct,
            trigger_min_rel_volume=args.trigger_min_rel_volume,
            trigger_min_close_pos=args.trigger_min_close_pos,
            trigger_require_green=trigger_require_green,
            trigger_vol_lookback_bars=args.trigger_vol_lookback_bars,
            show_dates=args.show_dates,
        )
    elif args.scan:
        main_scan(
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days,
            signals_json=args.signals_json,
            scan_mode=args.scan_mode,
            label_mode=args.label_mode,
            signal_level=args.signal_level,
            use_intraday_trigger=use_intraday_trigger,
            trigger_timeframe=args.trigger_timeframe,
            trigger_breakout_buffer_pct=args.trigger_breakout_buffer_pct,
            trigger_min_rel_volume=args.trigger_min_rel_volume,
            trigger_min_close_pos=args.trigger_min_close_pos,
            trigger_require_green=trigger_require_green,
            trigger_vol_lookback_bars=args.trigger_vol_lookback_bars,
            target_date=args.target_date,
        )
    else:
        print("Use --scan or --walk-forward.")
