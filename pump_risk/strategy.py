"""
strategy.py
Feature engineering and scoring logic for Pump Risk detection.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from config import (
    PUMP_THRESHOLD_PCT,
    PUMP_LABEL_MODE,
    TRIGGER_BREAKOUT_BUFFER_PCT,
    TRIGGER_MIN_REL_VOLUME,
    TRIGGER_MIN_CLOSE_POS,
    TRIGGER_REQUIRE_GREEN,
    TRIGGER_VOL_LOOKBACK_BARS,
    RSI_MIN,
    RSI_MAX,
    MAX_VOL_RATIO_PRE,
    MAX_ABS_RET_PRE,
    MIN_ATR_PCT,
    MAX_FUNDING_FOR_SQUEEZE,
    MIN_FUNDING_DELTA,
    MIN_OI_CHG,
    WEIGHTS,
    HIGH_RISK_MIN_SCORE,
    MEDIUM_RISK_MIN_SCORE,
)


def _compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder RSI using EMA smoothing."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1 / length, adjust=False).mean() / down.ewm(alpha=1 / length, adjust=False).mean()
    return 100 - (100 / (1 + rs))


def _daily_funding_mean(funding_df: pd.DataFrame) -> pd.DataFrame:
    if funding_df is None or funding_df.empty:
        return pd.DataFrame(columns=["date", "funding_mean"])

    f = funding_df.copy()
    f["date"] = pd.to_datetime(f["timestamp"], utc=True).dt.date
    return f.groupby("date", as_index=False)["funding_rate"].mean().rename(
        columns={"funding_rate": "funding_mean"}
    )


def _daily_oi_last(oi_df: pd.DataFrame) -> pd.DataFrame:
    if oi_df is None or oi_df.empty:
        return pd.DataFrame(columns=["date", "open_interest", "oi_chg1"])

    o = oi_df.copy()
    o["date"] = pd.to_datetime(o["timestamp"], utc=True).dt.date
    o["open_interest"] = pd.to_numeric(o["open_interest"], errors="coerce")
    out = o.groupby("date", as_index=False)["open_interest"].last().sort_values("date")
    out["oi_chg1"] = out["open_interest"].pct_change()
    return out


def add_pump_risk_features(
    ohlcv_df: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
    oi_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build daily feature set and risk score.

    Score at date t estimates the risk of a pump on t+1.
    """
    df = ohlcv_df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.date

    # Core price/volume features
    df["ret1"] = df["close"].pct_change()
    df["ret3"] = df["close"].pct_change(3)
    df["ret5"] = df["close"].pct_change(5)
    df["vol_ratio20"] = df["volume"] / df["volume"].rolling(20).mean()
    df["rsi14"] = _compute_rsi(df["close"], 14)

    # Volatility regime
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["close"]

    # Confirmation-only features (not used in pre-score by default)
    df["prev_day_high"] = df["high"].shift(1)
    df["break_prev_high"] = df["close"] > df["prev_day_high"]

    # Futures context
    d_funding = _daily_funding_mean(funding_df)
    d_oi = _daily_oi_last(oi_df)
    df = df.merge(d_funding, on="date", how="left")
    df = df.merge(d_oi[["date", "open_interest", "oi_chg1"]], on="date", how="left")
    df["funding_mean"] = df["funding_mean"].fillna(0.0)
    df["funding_delta"] = df["funding_mean"] - df["funding_mean"].shift(1)

    # Binary conditions for score
    df["cond_rsi_zone"] = df["rsi14"].between(RSI_MIN, RSI_MAX, inclusive="both")
    df["cond_calm_volume"] = df["vol_ratio20"] <= MAX_VOL_RATIO_PRE
    df["cond_calm_day"] = df["ret1"].abs() <= MAX_ABS_RET_PRE
    df["cond_funding_neg"] = df["funding_mean"] <= MAX_FUNDING_FOR_SQUEEZE
    df["cond_funding_rising"] = df["funding_delta"] > MIN_FUNDING_DELTA
    df["cond_atr_regime"] = df["atr_pct"] >= MIN_ATR_PCT
    df["cond_oi_rising"] = df["oi_chg1"] > MIN_OI_CHG

    # Weighted score
    score = (
        WEIGHTS["rsi_zone"] * df["cond_rsi_zone"].astype(float)
        + WEIGHTS["calm_volume"] * df["cond_calm_volume"].astype(float)
        + WEIGHTS["calm_day"] * df["cond_calm_day"].astype(float)
        + WEIGHTS["funding_neg"] * df["cond_funding_neg"].astype(float)
        + WEIGHTS["funding_rising"] * df["cond_funding_rising"].astype(float)
        + WEIGHTS["atr_regime"] * df["cond_atr_regime"].astype(float)
        + WEIGHTS["oi_rising"] * df["cond_oi_rising"].astype(float)
    )
    df["pump_risk_score"] = score

    # Labels for historical evaluation (t predicts t+1 movement)
    df["next_ret1"] = df["ret1"].shift(-1)
    df["intraday_move"] = np.where(df["open"] > 0, (df["high"] / df["open"]) - 1.0, np.nan)
    df["next_intraday_move"] = df["intraday_move"].shift(-1)

    has_close = df["next_ret1"].notna()
    has_intraday = df["next_intraday_move"].notna()
    has_both = has_close & has_intraday

    close_hit = df["next_ret1"] >= PUMP_THRESHOLD_PCT
    intraday_hit = df["next_intraday_move"] >= PUMP_THRESHOLD_PCT

    df["pump_next_close"] = np.where(has_close, close_hit, np.nan)
    df["pump_next_intraday"] = np.where(has_intraday, intraday_hit, np.nan)
    df["pump_next_either"] = np.where(has_both, close_hit | intraday_hit, np.nan)
    df["pump_next_both"] = np.where(has_both, close_hit & intraday_hit, np.nan)

    # Backwards-compatible alias used by older reports
    df["pump_next"] = df["pump_next_close"]

    # Risk bucket
    df["pump_risk"] = "LOW"
    df.loc[df["pump_risk_score"] >= MEDIUM_RISK_MIN_SCORE, "pump_risk"] = "MEDIUM"
    df.loc[df["pump_risk_score"] >= HIGH_RISK_MIN_SCORE, "pump_risk"] = "HIGH"

    return df


def signal_threshold(signal_level: str) -> float:
    """Map signal level to numeric threshold."""
    level = (signal_level or "high").lower().strip()
    if level == "medium":
        return MEDIUM_RISK_MIN_SCORE
    return HIGH_RISK_MIN_SCORE


def _signal_rule_name(signal_level: str, label_mode: str) -> str:
    """
    Human-readable signal policy.

    For `label_mode=either` at MEDIUM, use a precision-oriented gate:
    at least 2 of [funding_neg, atr_regime, break_prev_high].
    """
    level = (signal_level or "high").lower().strip()
    mode = normalize_label_mode(label_mode)
    if mode == "either" and level == "medium":
        return "either_medium_gate(>=2 of funding_neg,atr_regime,break_prev_high)"
    threshold = signal_threshold(level)
    return f"score>={threshold:.1f}"


def build_signal_mask(df: pd.DataFrame, signal_level: str, label_mode: str) -> pd.Series:
    """
    Build binary signal mask for evaluation.

    Default: score threshold by level.
    Optimized path (XRP walk-forward objective):
    - label_mode=either + medium => 2-of-3 gate on high-value conditions.
    """
    level = (signal_level or "high").lower().strip()
    mode = normalize_label_mode(label_mode)

    if mode == "either" and level == "medium":
        needed = ["cond_funding_neg", "cond_atr_regime", "break_prev_high"]
        if all(c in df.columns for c in needed):
            gate_score = (
                df["cond_funding_neg"].astype(int)
                + df["cond_atr_regime"].astype(int)
                + df["break_prev_high"].astype(int)
            )
            return gate_score >= 2

    threshold = signal_threshold(level)
    return df["pump_risk_score"] >= threshold


def build_intraday_trigger_table(
    intraday_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    breakout_buffer_pct: float = TRIGGER_BREAKOUT_BUFFER_PCT,
    min_rel_volume: float = TRIGGER_MIN_REL_VOLUME,
    min_close_pos: float = TRIGGER_MIN_CLOSE_POS,
    require_green: bool = TRIGGER_REQUIRE_GREEN,
    vol_lookback_bars: int = TRIGGER_VOL_LOOKBACK_BARS,
) -> pd.DataFrame:
    """
    Build trigger outcomes by target date (t+1).

    Trigger candle requires:
    - close breaks previous daily high
    - relative volume threshold
    - strong close in upper range (and green candle if enabled)
    """
    if intraday_df is None or intraday_df.empty:
        return pd.DataFrame(
            columns=[
                "target_date",
                "trigger_hit",
                "trigger_ts",
                "trigger_price",
                "prev_day_high",
                "breakout_price",
                "trigger_rel_vol",
                "trigger_close_pos",
                "day_open",
                "day_high",
                "day_intraday_pct",
            ]
        )

    h = intraday_df.copy().sort_values("timestamp").reset_index(drop=True)
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    h["target_date"] = h["timestamp"].dt.date

    lookback = int(max(1, vol_lookback_bars))
    h["rel_vol"] = h["volume"] / h["volume"].rolling(lookback, min_periods=lookback).mean()

    candle_range = (h["high"] - h["low"]).replace(0, np.nan)
    h["close_pos"] = ((h["close"] - h["low"]) / candle_range).fillna(0.5)
    h["is_green"] = h["close"] > h["open"]

    d = daily_df[["date", "high"]].copy().dropna().sort_values("date").reset_index(drop=True)
    d["target_date"] = (pd.to_datetime(d["date"]) + pd.Timedelta(days=1)).dt.date
    prev_high_by_target = dict(zip(d["target_date"], d["high"]))
    h["prev_day_high"] = h["target_date"].map(prev_high_by_target)
    h["breakout_price"] = h["prev_day_high"] * (1.0 + float(breakout_buffer_pct))

    cond_breakout = h["close"] >= h["breakout_price"]
    cond_volume = h["rel_vol"] >= float(min_rel_volume)
    cond_close = h["close_pos"] >= float(min_close_pos)
    if require_green:
        cond_close = cond_close & h["is_green"]

    h["trigger_hit"] = h["prev_day_high"].notna() & cond_breakout & cond_volume & cond_close

    rows: list[dict] = []
    for target_date, g in h.groupby("target_date", sort=True):
        day_open = float(g["open"].iloc[0]) if len(g) else np.nan
        day_high = float(g["high"].max()) if len(g) else np.nan
        day_intraday_pct = ((day_high / day_open) - 1.0) if day_open and day_open > 0 else np.nan

        hit_rows = g[g["trigger_hit"]]
        if hit_rows.empty:
            rows.append(
                {
                    "target_date": target_date,
                    "trigger_hit": False,
                    "trigger_ts": pd.NaT,
                    "trigger_price": np.nan,
                    "prev_day_high": float(g["prev_day_high"].iloc[0]) if len(g) else np.nan,
                    "breakout_price": float(g["breakout_price"].iloc[0]) if len(g) else np.nan,
                    "trigger_rel_vol": np.nan,
                    "trigger_close_pos": np.nan,
                    "day_open": day_open,
                    "day_high": day_high,
                    "day_intraday_pct": day_intraday_pct,
                }
            )
            continue

        first = hit_rows.iloc[0]
        rows.append(
            {
                "target_date": target_date,
                "trigger_hit": True,
                "trigger_ts": pd.to_datetime(first["timestamp"], utc=True),
                "trigger_price": float(first["close"]),
                "prev_day_high": float(first["prev_day_high"]),
                "breakout_price": float(first["breakout_price"]),
                "trigger_rel_vol": float(first["rel_vol"]),
                "trigger_close_pos": float(first["close_pos"]),
                "day_open": day_open,
                "day_high": day_high,
                "day_intraday_pct": day_intraday_pct,
            }
        )

    return pd.DataFrame(rows)


def normalize_label_mode(label_mode: str | None) -> str:
    """Normalize label mode with safe fallback."""
    mode = (label_mode or PUMP_LABEL_MODE).lower().strip()
    if mode not in {"close", "intraday", "either", "both"}:
        raise ValueError(f"unsupported label mode: {label_mode!r}")
    return mode


def label_columns(label_mode: str | None) -> tuple[str, str]:
    """
    Return target/next-move column names for evaluation.
    - close: target=next close->close return >= threshold
    - intraday: target=next day open->high move >= threshold
    - either: target=close OR intraday
    - both: target=close AND intraday
    """
    mode = normalize_label_mode(label_mode)
    if mode == "intraday":
        return "pump_next_intraday", "next_intraday_move"
    if mode == "either":
        return "pump_next_either", "next_intraday_move"
    if mode == "both":
        return "pump_next_both", "next_intraday_move"
    return "pump_next_close", "next_ret1"


def _binary_metrics(signal: pd.Series, target: pd.Series) -> dict:
    """Compute confusion-matrix metrics for a binary signal."""
    s = signal.astype(bool)
    t = target.astype(bool)

    tp = int((s & t).sum())
    fp = int((s & ~t).sum())
    fn = int((~s & t).sum())
    tn = int((~s & ~t).sum())
    signals = int(s.sum())
    pumps = int(t.sum())
    total = int(len(t))

    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    base_rate = (pumps / total) if total else 0.0
    lift = (precision / base_rate) if base_rate > 0 else 0.0

    return {
        "total": total,
        "signals": signals,
        "pumps": pumps,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "base_rate": base_rate,
        "lift": lift,
    }


def run_walk_forward_evaluation(
    df: pd.DataFrame,
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 30,
    signal_level: str = "high",
    label_mode: str = PUMP_LABEL_MODE,
    use_intraday_trigger: bool = False,
    intraday_df: pd.DataFrame | None = None,
    trigger_breakout_buffer_pct: float = TRIGGER_BREAKOUT_BUFFER_PCT,
    trigger_min_rel_volume: float = TRIGGER_MIN_REL_VOLUME,
    trigger_min_close_pos: float = TRIGGER_MIN_CLOSE_POS,
    trigger_require_green: bool = TRIGGER_REQUIRE_GREEN,
    trigger_vol_lookback_bars: int = TRIGGER_VOL_LOOKBACK_BARS,
) -> dict:
    """
    Walk-forward evaluation for pump-risk as standalone detector.

    Note:
    - No model fitting is used.
    - Features are causal (rolling/ewm/shifted from past to present).
    - Signals at t are evaluated versus label at t+1.
    - Optional intraday trigger can filter daily signals before counting entries.
    """
    mode = normalize_label_mode(label_mode)
    target_col, next_move_col = label_columns(mode)

    ev = df.dropna(subset=[target_col, next_move_col]).copy().reset_index(drop=True)
    if ev.empty:
        return {
            "windows": [],
            "window_metrics": [],
            "detections": pd.DataFrame(),
            "aggregate": _binary_metrics(pd.Series(dtype=bool), pd.Series(dtype=bool)),
            "avg_metrics": {},
            "median_metrics": {},
            "signal_level": signal_level.lower(),
            "threshold": signal_threshold(signal_level),
            "signal_rule": _signal_rule_name(signal_level, mode),
            "label_mode": mode,
            "target_column": target_col,
            "next_move_column": next_move_col,
            "use_intraday_trigger": use_intraday_trigger,
            "trigger_params": {
                "breakout_buffer_pct": float(trigger_breakout_buffer_pct),
                "min_rel_volume": float(trigger_min_rel_volume),
                "min_close_pos": float(trigger_min_close_pos),
                "require_green": bool(trigger_require_green),
                "vol_lookback_bars": int(trigger_vol_lookback_bars),
            },
        }

    threshold = signal_threshold(signal_level)
    signal_rule = _signal_rule_name(signal_level, mode)
    trigger_params = {
        "breakout_buffer_pct": float(trigger_breakout_buffer_pct),
        "min_rel_volume": float(trigger_min_rel_volume),
        "min_close_pos": float(trigger_min_close_pos),
        "require_green": bool(trigger_require_green),
        "vol_lookback_bars": int(trigger_vol_lookback_bars),
    }

    trigger_table = pd.DataFrame()
    trigger_hit_by_date: dict = {}
    trigger_info_by_date: dict = {}
    if use_intraday_trigger:
        if intraday_df is None or intraday_df.empty:
            raise ValueError("intraday trigger enabled but intraday_df is empty")

        trigger_table = build_intraday_trigger_table(
            intraday_df=intraday_df,
            daily_df=df,
            breakout_buffer_pct=trigger_breakout_buffer_pct,
            min_rel_volume=trigger_min_rel_volume,
            min_close_pos=trigger_min_close_pos,
            require_green=trigger_require_green,
            vol_lookback_bars=trigger_vol_lookback_bars,
        )
        if not trigger_table.empty:
            trigger_hit_by_date = {
                row["target_date"]: bool(row["trigger_hit"])
                for _, row in trigger_table.iterrows()
            }
            trigger_info_by_date = {
                row["target_date"]: row.to_dict()
                for _, row in trigger_table.iterrows()
            }

        signal_rule = (
            f"{signal_rule} + intraday_trigger("
            f"break>{trigger_breakout_buffer_pct*100:.2f}%, "
            f"rel_vol>={trigger_min_rel_volume:.2f}, "
            f"close_pos>={trigger_min_close_pos:.2f}, "
            f"green={bool(trigger_require_green)})"
        )
    windows: list[dict] = []
    detections: list[dict] = []

    start = int(train_days)
    window_id = 1

    while (start + test_days) <= len(ev):
        train = ev.iloc[start - train_days:start]
        test = ev.iloc[start:start + test_days]

        sig_base = build_signal_mask(test, signal_level=signal_level, label_mode=mode)
        if use_intraday_trigger:
            target_dates = (pd.to_datetime(test["date"]) + pd.Timedelta(days=1)).dt.date
            sig_trigger = target_dates.map(trigger_hit_by_date).fillna(False).astype(bool)
            sig = sig_base & sig_trigger
        else:
            sig = sig_base
        tgt = test[target_col].astype(bool)
        m = _binary_metrics(sig, tgt)

        # Consistency checks requested by user
        if (m["tp"] + m["fp"]) != m["signals"]:
            raise RuntimeError(f"metric consistency failed in window {window_id}: tp+fp != signals")
        if (m["tp"] + m["fn"]) != m["pumps"]:
            raise RuntimeError(f"metric consistency failed in window {window_id}: tp+fn != pumps")
        if m["base_rate"] > 0:
            expected_lift = m["precision"] / m["base_rate"]
            if abs(expected_lift - m["lift"]) > 1e-12:
                raise RuntimeError(f"metric consistency failed in window {window_id}: lift mismatch")

        window_row = {
            "window_id": window_id,
            "train_start": train["date"].iloc[0],
            "train_end": train["date"].iloc[-1],
            "test_start": test["date"].iloc[0],
            "test_end": test["date"].iloc[-1],
            **m,
        }
        windows.append(window_row)

        sig_rows = test[sig]
        for _, row in sig_rows.iterrows():
            signal_date = pd.to_datetime(row["date"])
            target_date = (signal_date + timedelta(days=1)).date()
            trigger_row = trigger_info_by_date.get(target_date, {})
            detections.append(
                {
                    "window_id": window_id,
                    "signal_date": row["date"],
                    "target_date": target_date,
                    "score": float(row["pump_risk_score"]),
                    "pump_next": bool(row[target_col]),
                    "next_move_pct": float(row.get(next_move_col, np.nan)) * 100.0,
                    "next_ret_pct": float(row.get("next_ret1", np.nan)) * 100.0,
                    "next_intraday_pct": float(row.get("next_intraday_move", np.nan)) * 100.0,
                    "trigger_ts": trigger_row.get("trigger_ts", pd.NaT),
                    "trigger_price": trigger_row.get("trigger_price", np.nan),
                    "trigger_rel_vol": trigger_row.get("trigger_rel_vol", np.nan),
                    "trigger_close_pos": trigger_row.get("trigger_close_pos", np.nan),
                }
            )

        start += int(step_days)
        window_id += 1

    wdf = pd.DataFrame(windows)
    ddf = pd.DataFrame(detections)

    # Aggregate from all test rows in all windows
    # This intentionally counts overlapping rows if step < test_days,
    # mirroring strategy-style rolling evaluation.
    all_signal = pd.Series(dtype=bool)
    all_target = pd.Series(dtype=bool)
    signal_parts = []
    target_parts = []
    start = int(train_days)
    while (start + test_days) <= len(ev):
        test = ev.iloc[start:start + test_days]
        sig_base = build_signal_mask(test, signal_level=signal_level, label_mode=mode).astype(bool)
        if use_intraday_trigger:
            target_dates = (pd.to_datetime(test["date"]) + pd.Timedelta(days=1)).dt.date
            sig_trigger = target_dates.map(trigger_hit_by_date).fillna(False).astype(bool)
            sig_final = sig_base & sig_trigger
        else:
            sig_final = sig_base
        signal_parts.append(sig_final)
        target_parts.append(test[target_col].astype(bool))
        start += int(step_days)
    if signal_parts:
        all_signal = pd.concat(signal_parts, axis=0, ignore_index=True)
        all_target = pd.concat(target_parts, axis=0, ignore_index=True)
    aggregate = _binary_metrics(all_signal, all_target)

    metric_cols = ["precision", "recall", "f1", "base_rate", "lift", "signals", "pumps", "tp", "fp", "fn"]
    avg_metrics = wdf[metric_cols].mean(numeric_only=True).to_dict() if not wdf.empty else {}
    median_metrics = wdf[metric_cols].median(numeric_only=True).to_dict() if not wdf.empty else {}

    return {
        "windows": windows,
        "window_metrics": windows,
        "detections": ddf,
        "aggregate": aggregate,
        "avg_metrics": avg_metrics,
        "median_metrics": median_metrics,
        "signal_level": signal_level.lower(),
        "threshold": threshold,
        "signal_rule": signal_rule,
        "label_mode": mode,
        "target_column": target_col,
        "next_move_column": next_move_col,
        "use_intraday_trigger": use_intraday_trigger,
        "trigger_params": trigger_params,
    }


def evaluate_pump_risk(df: pd.DataFrame, label_mode: str = PUMP_LABEL_MODE) -> dict:
    """
    Evaluate binary prediction quality for HIGH and MEDIUM thresholds.
    """
    mode = normalize_label_mode(label_mode)
    target_col, _ = label_columns(mode)
    ev = df.dropna(subset=[target_col]).copy()
    if ev.empty:
        return {
            "signals_high": 0,
            "signals_medium": 0,
            "pumps": 0,
            "precision_high": 0.0,
            "recall_high": 0.0,
            "precision_medium": 0.0,
            "recall_medium": 0.0,
            "base_rate": 0.0,
            "label_mode": mode,
        }

    target = ev[target_col].astype(bool)

    def _metrics_for(level: str) -> tuple[int, float, float]:
        signal = build_signal_mask(ev, signal_level=level, label_mode=mode)
        tp = int((signal & target).sum())
        fp = int((signal & ~target).sum())
        fn = int((~signal & target).sum())
        signals = int(signal.sum())
        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        return signals, precision, recall

    signals_high, precision_high, recall_high = _metrics_for("high")
    signals_medium, precision_medium, recall_medium = _metrics_for("medium")
    pumps = int(target.sum())
    base_rate = float(target.mean()) if len(target) else 0.0

    return {
        "signals_high": signals_high,
        "signals_medium": signals_medium,
        "pumps": pumps,
        "precision_high": precision_high,
        "recall_high": recall_high,
        "precision_medium": precision_medium,
        "recall_medium": recall_medium,
        "base_rate": base_rate,
        "label_mode": mode,
    }


def explain_row_conditions(row: pd.Series) -> list[tuple[str, bool, str]]:
    """Human-readable condition report for current row."""
    return [
        (
            "rsi_zone",
            bool(row.get("cond_rsi_zone", False)),
            f"rsi14={row.get('rsi14', float('nan')):.2f} in [{RSI_MIN:.0f},{RSI_MAX:.0f}]",
        ),
        (
            "calm_volume",
            bool(row.get("cond_calm_volume", False)),
            f"vol_ratio20={row.get('vol_ratio20', float('nan')):.2f} <= {MAX_VOL_RATIO_PRE:.2f}",
        ),
        (
            "calm_day",
            bool(row.get("cond_calm_day", False)),
            f"|ret1|={abs(row.get('ret1', float('nan')))*100:.2f}% <= {MAX_ABS_RET_PRE*100:.2f}%",
        ),
        (
            "funding_neg",
            bool(row.get("cond_funding_neg", False)),
            f"funding_mean={row.get('funding_mean', float('nan')):+.6f} <= {MAX_FUNDING_FOR_SQUEEZE:+.6f}",
        ),
        (
            "funding_rising",
            bool(row.get("cond_funding_rising", False)),
            f"funding_delta={row.get('funding_delta', float('nan')):+.6f} > {MIN_FUNDING_DELTA:+.6f}",
        ),
        (
            "atr_regime",
            bool(row.get("cond_atr_regime", False)),
            f"atr_pct={row.get('atr_pct', float('nan'))*100:.2f}% >= {MIN_ATR_PCT*100:.2f}%",
        ),
        (
            "oi_rising",
            bool(row.get("cond_oi_rising", False)),
            f"oi_chg1={row.get('oi_chg1', float('nan'))*100:.2f}% > {MIN_OI_CHG*100:.2f}%",
        ),
        (
            "break_prev_high",
            bool(row.get("break_prev_high", False)),
            (
                f"close={row.get('close', float('nan')):.4f} "
                f"> prev_high={row.get('prev_day_high', float('nan')):.4f}"
            ),
        ),
    ]
