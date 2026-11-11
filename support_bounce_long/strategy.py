"""
Core strategy logic for support_bounce_long.

This module ports the long setup detector from predictor-ui/lib to Python,
keeping the same criteria and scoring behavior.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _clamp01(value: float) -> float:
    if pd.isna(value):
        return 0.0
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    return float(value)


def _to_time_str(value: Any) -> str:
    if value is None:
        return ""

    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts.isoformat()
    except Exception:
        pass

    return str(value)


def detect_pivot_extremes(df: pd.DataFrame, pivot_window: int = 1, mode: str = "low") -> list[dict[str, Any]]:
    pivots: list[dict[str, Any]] = []

    n = len(df)
    if n < pivot_window * 2 + 1:
        return pivots

    for i in range(pivot_window, n - pivot_window):
        current = float(df.iloc[i]["low"] if mode == "low" else df.iloc[i]["high"])
        is_extreme = True

        for offset in range(1, pivot_window + 1):
            left = float(df.iloc[i - offset]["low"] if mode == "low" else df.iloc[i - offset]["high"])
            right = float(df.iloc[i + offset]["low"] if mode == "low" else df.iloc[i + offset]["high"])

            if mode == "low":
                if not (current < left and current < right):
                    is_extreme = False
                    break
            else:
                if not (current > left and current > right):
                    is_extreme = False
                    break

        if not is_extreme:
            continue

        pivots.append(
            {
                "index": i,
                "time": _to_time_str(df.iloc[i].get("timestamp")),
                "price": current,
            }
        )

    return pivots


def _pivot_within_tolerance(pivot_price: float, mid_price: float, tolerance_pct: float) -> bool:
    tolerance = (mid_price * tolerance_pct) / 100.0
    return abs(pivot_price - mid_price) <= tolerance


def cluster_price_zones(
    pivots: list[dict[str, Any]],
    tolerance_pct: float,
    kind: str,
) -> list[dict[str, Any]]:
    if not pivots:
        return []

    safe_tolerance_pct = max(float(tolerance_pct), 0.01)
    sorted_by_price = sorted(pivots, key=lambda x: float(x["price"]))
    groups: list[dict[str, Any]] = []

    def recalc_group(group: dict[str, Any]) -> dict[str, Any]:
        prices = [float(item["price"]) for item in group["pivots"]]
        group["min"] = min(prices)
        group["max"] = max(prices)
        group["mid"] = (group["min"] + group["max"]) / 2.0
        return group

    for pivot in sorted_by_price:
        pivot_price = float(pivot["price"])

        best_group_idx = -1
        best_distance = float("inf")

        for idx, group in enumerate(groups):
            if not _pivot_within_tolerance(pivot_price, float(group["mid"]), safe_tolerance_pct):
                continue

            distance = abs(pivot_price - float(group["mid"]))
            if distance < best_distance:
                best_distance = distance
                best_group_idx = idx

        if best_group_idx == -1:
            groups.append(
                {
                    "pivots": [pivot],
                    "min": pivot_price,
                    "max": pivot_price,
                    "mid": pivot_price,
                }
            )
            continue

        groups[best_group_idx]["pivots"].append(pivot)
        groups[best_group_idx] = recalc_group(groups[best_group_idx])

    provisional_zones: list[dict[str, Any]] = []

    for idx, group in enumerate(groups):
        unique_dates = sorted({str(item["time"]) for item in group["pivots"]})
        touch_indexes = sorted({int(item["index"]) for item in group["pivots"]})

        adaptive_half_band = max(
            (float(group["mid"]) * safe_tolerance_pct) / 100.0 / 2.0,
            (float(group["max"]) - float(group["min"])) / 2.0,
            0.0000001,
        )

        min_price = float(group["mid"]) - adaptive_half_band
        max_price = float(group["mid"]) + adaptive_half_band
        last_touch_date = unique_dates[-1] if unique_dates else "1970-01-01T00:00:00+00:00"

        last_touch = pd.to_datetime(last_touch_date, utc=True, errors="coerce")
        if pd.isna(last_touch):
            last_touch_ts = 0.0
        else:
            last_touch_ts = float(last_touch.value)  # nanoseconds epoch

        provisional_zones.append(
            {
                "id": f"{kind}-{idx + 1}",
                "kind": kind,
                "min_price": min_price,
                "max_price": max_price,
                "mid_price": float(group["mid"]),
                "touches": len(group["pivots"]),
                "touch_dates": unique_dates,
                "touch_indexes": touch_indexes,
                "last_touch_ts": last_touch_ts,
            }
        )

    min_ts = min(float(z["last_touch_ts"]) for z in provisional_zones)
    max_ts = max(float(z["last_touch_ts"]) for z in provisional_zones)
    ts_range = max(max_ts - min_ts, 1.0)

    zones = []
    for zone in provisional_zones:
        recency_bonus = ((float(zone["last_touch_ts"]) - min_ts) / ts_range) * 10.0
        zones.append(
            {
                "id": zone["id"],
                "kind": zone["kind"],
                "min_price": zone["min_price"],
                "max_price": zone["max_price"],
                "mid_price": zone["mid_price"],
                "touches": zone["touches"],
                "touch_dates": zone["touch_dates"],
                "touch_indexes": zone["touch_indexes"],
                "score": float(zone["touches"]) * 10.0 + recency_bonus,
            }
        )

    zones.sort(
        key=lambda z: (
            -float(z["score"]),
            -int(z["touches"]),
            -float(z["mid_price"]),
        )
    )
    return zones


def detect_supports(df: pd.DataFrame, tolerance_pct: float, min_touches: int, pivot_window: int) -> list[dict[str, Any]]:
    if df.empty:
        return []

    pivots = detect_pivot_extremes(df, pivot_window=max(1, int(pivot_window)), mode="low")
    zones = cluster_price_zones(pivots, max(0.01, float(tolerance_pct)), "support")
    zones = [z for z in zones if int(z["touches"]) >= max(1, int(min_touches))]
    zones.sort(key=lambda z: (-float(z["score"]), -int(z["touches"])))
    return zones


def detect_resistances(df: pd.DataFrame, tolerance_pct: float, min_touches: int, pivot_window: int) -> list[dict[str, Any]]:
    if df.empty:
        return []

    pivots = detect_pivot_extremes(df, pivot_window=max(1, int(pivot_window)), mode="high")
    zones = cluster_price_zones(pivots, max(0.01, float(tolerance_pct)), "resistance")
    zones = [z for z in zones if int(z["touches"]) >= max(1, int(min_touches))]
    zones.sort(key=lambda z: (-float(z["score"]), -int(z["touches"])))
    return zones


def is_near_support(candle: dict[str, Any], support: dict[str, Any], support_proximity_pct: float) -> bool:
    safe_pct = max(0.0, float(support_proximity_pct))
    zone_buffer = (float(support["mid_price"]) * safe_pct) / 100.0
    half_band = (float(support["max_price"]) - float(support["min_price"])) / 2.0
    expanded_min = float(support["min_price"]) - zone_buffer
    expanded_max = float(support["max_price"]) + zone_buffer

    low_inside = float(candle["low"]) >= expanded_min and float(candle["low"]) <= expanded_max
    close_inside = float(candle["close"]) >= expanded_min and float(candle["close"]) <= expanded_max
    range_intersects = float(candle["low"]) <= expanded_max and float(candle["high"]) >= expanded_min
    close_near_mid = abs(float(candle["close"]) - float(support["mid_price"])) <= zone_buffer + half_band

    return bool(low_inside or close_inside or range_intersects or close_near_mid)


def detect_bullish_rejection(candle: dict[str, Any], rejection_threshold: float) -> dict[str, Any]:
    safe_threshold = max(0.05, float(rejection_threshold))

    lower_wick = max(0.0, min(float(candle["open"]), float(candle["close"])) - float(candle["low"]))
    body = abs(float(candle["close"]) - float(candle["open"]))
    full_range = max(0.0, float(candle["high"]) - float(candle["low"]))

    wick_ratio = lower_wick / full_range if full_range > 0 else 0.0
    wick_vs_body = lower_wick / body if body > 0 else (1.0 if lower_wick > 0 else 0.0)

    wick_ratio_pass = wick_ratio >= safe_threshold
    wick_body_pass = lower_wick > body
    passed = full_range > 0 and (wick_ratio_pass or wick_body_pass)

    wick_ratio_strength = _clamp01(wick_ratio / safe_threshold)
    wick_body_strength = _clamp01(wick_vs_body / 1.5)
    score = max(wick_ratio_strength, wick_body_strength) if passed else 0.0

    tags: list[str] = []
    if wick_ratio_pass:
        tags.append(f"Bullish rejection wick {wick_ratio * 100:.1f}%")
    elif wick_body_pass:
        tags.append("Lower wick > body")

    return {
        "passed": bool(passed),
        "lower_wick": lower_wick,
        "body": body,
        "full_range": full_range,
        "wick_ratio": wick_ratio,
        "score": score,
        "tags": tags,
    }


def detect_bullish_confirmation(candles: list[dict[str, Any]], index: int, mode: str) -> dict[str, Any]:
    candle = candles[index]
    previous = candles[index - 1] if index > 0 else None

    bullish_body = float(candle["close"]) > float(candle["open"])
    close_above_previous = float(candle["close"]) > float(previous["close"]) if previous else False

    passed = False
    score = 0.0
    tags: list[str] = []

    if mode == "aggressive":
        passed = True
        score = 1.0 if bullish_body or close_above_previous else 0.65
        tags.append("Aggressive mode")
    elif mode == "balanced":
        passed = bullish_body
        score = 1.0 if passed else (0.35 if close_above_previous else 0.0)
        if passed:
            tags.append("Bullish close > open")
    else:
        passed = bullish_body and close_above_previous
        score = 1.0 if passed else (0.4 if bullish_body or close_above_previous else 0.0)
        if passed:
            tags.append("Bullish close + close > prev close")

    return {
        "passed": bool(passed),
        "bullish_body": bool(bullish_body),
        "close_above_previous": bool(close_above_previous),
        "score": _clamp01(score),
        "tags": tags,
    }


def detect_pullback_context(candles: list[dict[str, Any]], index: int, lookback: int) -> dict[str, Any]:
    safe_lookback = max(2, int(lookback))
    start = max(0, index - safe_lookback)
    sample = candles[start:index]

    if len(sample) < 2:
        return {
            "passed": True,
            "score": 0.4,
            "tags": ["Limited pullback history"],
        }

    avg_close = sum(float(item["close"]) for item in sample) / len(sample)
    current_close = float(candles[index]["close"])
    close_below_avg = current_close < avg_close

    down_steps = 0
    for i in range(1, len(sample)):
        if float(sample[i]["close"]) < float(sample[i - 1]["close"]):
            down_steps += 1

    bearish_ratio = down_steps / max(len(sample) - 1, 1)
    has_bearish_sequence = bearish_ratio >= 0.5

    passed = close_below_avg or has_bearish_sequence
    score = (
        _clamp01((0.55 if close_below_avg else 0.0) + (0.65 if has_bearish_sequence else 0.0))
        if passed
        else _clamp01(bearish_ratio * 0.5)
    )

    tags: list[str] = []
    if close_below_avg:
        tags.append(f"Close below {safe_lookback}c avg")
    if has_bearish_sequence:
        tags.append("Recent bearish pullback")

    return {
        "passed": bool(passed),
        "score": score,
        "tags": tags,
    }


def score_long_setup(
    rejection_factor: float,
    confirmation_factor: float,
    context_factor: float,
    rr: float,
    min_rr: float,
) -> dict[str, float]:
    rejection_score = 20.0 * _clamp01(rejection_factor)
    confirmation_score = 20.0 * _clamp01(confirmation_factor)
    context_score = 10.0 * _clamp01(context_factor)
    rr_score = 20.0 * _clamp01(rr / max(min_rr, 0.1))

    return {
        "rejection_score": rejection_score,
        "confirmation_score": confirmation_score,
        "context_score": context_score,
        "rr_score": rr_score,
        "total_score": min(100.0, 30.0 + rejection_score + confirmation_score + context_score + rr_score),
    }


def find_nearest_resistance_above(price: float, resistances: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [item for item in resistances if float(item["mid_price"]) > float(price)]
    if not candidates:
        return None
    candidates.sort(key=lambda x: float(x["mid_price"]))
    return candidates[0]


def _build_long_setup(
    candle: dict[str, Any],
    candle_index: int,
    support: dict[str, Any],
    resistances: list[dict[str, Any]],
    rejection_score_factor: float,
    rejection_tags: list[str],
    confirmation_passed: bool,
    confirmation_score_factor: float,
    confirmation_tags: list[str],
    context_passed: bool,
    context_score_factor: float,
    context_tags: list[str],
    params: dict[str, Any],
) -> dict[str, Any] | None:
    entry_price = float(candle["close"])

    stop_buffer_value = (float(support["min_price"]) * float(params["stop_buffer_pct"])) / 100.0
    stop_price = float(support["min_price"]) - stop_buffer_value
    risk = entry_price - stop_price
    if not (risk > 0):
        return None

    target_zone = find_nearest_resistance_above(entry_price, resistances)
    if not target_zone:
        return None

    target_price = float(target_zone["mid_price"])
    reward = target_price - entry_price
    if not (reward > 0):
        return None

    rr = reward / risk
    if rr < float(params["min_rr"]):
        return None

    scores = score_long_setup(
        rejection_factor=rejection_score_factor,
        confirmation_factor=confirmation_score_factor,
        context_factor=context_score_factor,
        rr=rr,
        min_rr=float(params["min_rr"]),
    )

    tags = [
        f"Near support {float(support['min_price']):.3f}-{float(support['max_price']):.3f}",
        *rejection_tags,
        *(confirmation_tags if confirmation_passed else ["Awaiting bullish confirmation"]),
        *(
            context_tags
            if bool(params["require_pullback"]) and context_passed
            else (["Pullback context missing"] if bool(params["require_pullback"]) else ["Pullback filter disabled"])
        ),
        f"RR {rr:.2f}x",
    ]

    candle_time = _to_time_str(candle.get("timestamp"))

    return {
        "id": f"long-{candle_time}-{candle_index}-{support['id']}",
        "candle_index": int(candle_index),
        "time": candle_time,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "rr": rr,
        "support_zone_id": support["id"],
        "resistance_zone_id": target_zone["id"],
        "rejection_score": scores["rejection_score"],
        "confirmation_score": scores["confirmation_score"],
        "context_score": scores["context_score"],
        "rr_score": scores["rr_score"],
        "total_score": scores["total_score"],
        "tags": tags,
        "status": "confirmed" if confirmation_passed else "candidate",
    }


def _choose_best_setup(
    candle: dict[str, Any],
    candle_index: int,
    near_supports: list[dict[str, Any]],
    resistances: list[dict[str, Any]],
    rejection: dict[str, Any],
    confirmation: dict[str, Any],
    context: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any] | None:
    best_setup: dict[str, Any] | None = None

    for support in near_supports:
        setup = _build_long_setup(
            candle=candle,
            candle_index=candle_index,
            support=support,
            resistances=resistances,
            rejection_score_factor=float(rejection["score"]),
            rejection_tags=list(rejection["tags"]),
            confirmation_passed=bool(confirmation["passed"]),
            confirmation_score_factor=float(confirmation["score"]),
            confirmation_tags=list(confirmation["tags"]),
            context_passed=bool(context["passed"]),
            context_score_factor=float(context["score"]),
            context_tags=list(context["tags"]),
            params=params,
        )

        if not setup:
            continue

        if (
            best_setup is None
            or float(setup["total_score"]) > float(best_setup["total_score"])
            or (
                float(setup["total_score"]) == float(best_setup["total_score"])
                and setup["status"] == "confirmed"
            )
        ):
            best_setup = setup

    return best_setup


def evaluate_candle(
    candles: list[dict[str, Any]],
    index: int,
    supports: list[dict[str, Any]],
    resistances: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    candle = candles[index]

    near_supports = [
        support
        for support in supports
        if is_near_support(candle, support, float(params["support_proximity_pct"]))
    ]
    near_supports.sort(key=lambda s: abs(float(candle["low"]) - float(s["mid_price"])))

    rejection = detect_bullish_rejection(candle, float(params["rejection_threshold"]))
    confirmation = detect_bullish_confirmation(candles, index, str(params["confirmation_mode"]))
    context = detect_pullback_context(candles, index, int(params["pullback_lookback"]))

    best_setup: dict[str, Any] | None = None

    if near_supports and bool(rejection["passed"]):
        if (not bool(params["require_pullback"])) or bool(context["passed"]):
            best_setup = _choose_best_setup(
                candle=candle,
                candle_index=index,
                near_supports=near_supports,
                resistances=resistances,
                rejection=rejection,
                confirmation=confirmation,
                context=context,
                params=params,
            )

    support_detail = (
        f"nearest={near_supports[0]['id']} low={float(candle['low']):.6f} mid={float(near_supports[0]['mid_price']):.6f}"
        if near_supports
        else "no support in proximity"
    )

    rejection_detail = (
        f"wick_ratio={float(rejection['wick_ratio']) * 100:.2f}% th={float(params['rejection_threshold']) * 100:.2f}%"
    )

    confirmation_required = bool(params.get("require_confirmed", True))
    confirmation_passed_for_signal = bool(confirmation["passed"]) if confirmation_required else True
    confirmation_detail = (
        f"mode={params['confirmation_mode']} pass={bool(confirmation['passed'])}"
        if confirmation_required
        else "confirmation disabled for signal"
    )

    pullback_required = bool(params["require_pullback"])
    pullback_passed = bool(context["passed"]) if pullback_required else True
    pullback_detail = (
        f"lookback={int(params['pullback_lookback'])} pass={bool(context['passed'])}"
        if pullback_required
        else "pullback filter disabled"
    )

    rr_passed = best_setup is not None
    rr_detail = (
        f"rr={float(best_setup['rr']):.2f} min_rr={float(params['min_rr']):.2f}"
        if best_setup is not None
        else f"no valid setup with rr>={float(params['min_rr']):.2f}"
    )

    conditions = [
        {
            "name": "support_proximity",
            "passed": bool(near_supports),
            "detail": support_detail,
        },
        {
            "name": "bullish_rejection",
            "passed": bool(rejection["passed"]),
            "detail": rejection_detail,
        },
        {
            "name": "confirmation",
            "passed": confirmation_passed_for_signal,
            "detail": confirmation_detail,
        },
        {
            "name": "pullback_context",
            "passed": pullback_passed,
            "detail": pullback_detail,
        },
        {
            "name": "rr_filter",
            "passed": rr_passed,
            "detail": rr_detail,
        },
    ]

    signal_is_long = bool(best_setup is not None and confirmation_passed_for_signal)

    blocked_by = "none"
    for condition in conditions:
        if not bool(condition["passed"]):
            blocked_by = f"{condition['name']} ({condition['detail']})"
            break

    if signal_is_long and best_setup is not None:
        summary_text = (
            f"LONG {best_setup['status']} near {best_setup['support_zone_id']} "
            f"rr={float(best_setup['rr']):.2f} score={float(best_setup['total_score']):.1f}"
        )
    else:
        failed_name = blocked_by.split("(")[0].strip() if blocked_by != "none" else "conditions"
        summary_text = f"No signal: {failed_name} not satisfied"

    return {
        "signal": "LONG" if signal_is_long else "NO_SIGNAL",
        "setup": best_setup,
        "conditions": conditions,
        "blocked_by": None if signal_is_long else blocked_by,
        "summary_text": summary_text,
        "rejection": rejection,
        "confirmation": confirmation,
        "context": context,
    }


def generate_signals(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """
    Generate LONG/NO_SIGNAL signals and attach setup details per candle.

    Output columns:
      - signal: LONG | NO_SIGNAL
      - signal_direction: 1 | 0
      - setup_data: dict | None
      - setup_id: str | None
      - setup_status: confirmed|candidate|None
      - blocked_by: str | None
      - summary_text: str
      - conditions: list[dict]
    """
    out = df.copy()

    supports = detect_supports(
        out,
        tolerance_pct=float(params["tolerance_pct"]),
        min_touches=int(params["min_touches"]),
        pivot_window=int(params["pivot_window"]),
    )
    resistances = detect_resistances(
        out,
        tolerance_pct=float(params["tolerance_pct"]),
        min_touches=int(params["min_touches"]),
        pivot_window=int(params["pivot_window"]),
    )

    out["signal"] = "NO_SIGNAL"
    out["signal_direction"] = 0
    out["setup_data"] = None
    out["setup_id"] = None
    out["setup_status"] = None
    out["blocked_by"] = None
    out["summary_text"] = "No signal"
    out["conditions"] = None

    candles = [
        {
            "timestamp": row.get("timestamp"),
            "open": float(row.get("open", 0.0)),
            "high": float(row.get("high", 0.0)),
            "low": float(row.get("low", 0.0)),
            "close": float(row.get("close", 0.0)),
            "volume": float(row.get("volume", 0.0)) if row.get("volume") is not None else 0.0,
        }
        for _, row in out.iterrows()
    ]

    setups: list[dict[str, Any]] = []

    for i in range(len(candles)):
        evaluation = evaluate_candle(candles, i, supports, resistances, params)

        out.at[out.index[i], "conditions"] = evaluation["conditions"]
        out.at[out.index[i], "blocked_by"] = evaluation["blocked_by"]
        out.at[out.index[i], "summary_text"] = evaluation["summary_text"]

        setup = evaluation["setup"]
        if setup is not None:
            setups.append(setup)
            out.at[out.index[i], "setup_data"] = setup
            out.at[out.index[i], "setup_id"] = setup["id"]
            out.at[out.index[i], "setup_status"] = setup["status"]

        if evaluation["signal"] == "LONG":
            out.at[out.index[i], "signal"] = "LONG"
            out.at[out.index[i], "signal_direction"] = 1

    setups.sort(key=lambda s: (-float(s["total_score"]), -int(s["candle_index"])))

    out.attrs["supports"] = supports
    out.attrs["resistances"] = resistances
    out.attrs["setups"] = setups
    return out


def detect_long_setups(df: pd.DataFrame, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Compatibility helper: return detected setups sorted by score/time."""
    with_signals = generate_signals(df, params)
    return list(with_signals.attrs.get("setups", []))


def build_current_signal_payload(last_row: pd.Series) -> dict[str, Any]:
    """Build standard current-signal payload from the latest row."""
    raw_signal = str(last_row.get("signal", "NO_SIGNAL"))
    is_long = raw_signal == "LONG"
    setup = last_row.get("setup_data") if isinstance(last_row.get("setup_data"), dict) else None

    return {
        "signal": "LONG" if is_long else "NO_SIGNAL",
        "timestamp": _to_time_str(last_row.get("timestamp")),
        "close": float(last_row.get("close", 0.0)),
        "summary_text": str(last_row.get("summary_text", "")),
        "blocked_by": None if is_long else (last_row.get("blocked_by") or "conditions"),
        "entry_price": float(setup["entry_price"]) if is_long and setup else None,
        "take_profit": float(setup["target_price"]) if is_long and setup else None,
        "stop_loss": float(setup["stop_price"]) if is_long and setup else None,
        "setup": setup,
        "conditions": list(last_row.get("conditions") or []),
    }
