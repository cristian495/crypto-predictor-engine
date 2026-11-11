"""
Sell-the-Rip strategy logic (simple, deterministic, short-only).
"""

import numpy as np
import pandas as pd

from config import (
    RIP_LOOKBACK_HOURS,
    RIP_THRESHOLD_PCT,
    RIP_THRESHOLD_STRONG,
    USE_STRONG_THRESHOLD,
    MIN_VOLUME_MULT,
    USE_VOLUME_FILTER,
    RSI_MIN,
    RSI_MAX,
    USE_RSI_FILTER,
    USE_TREND_FILTER,
    EMA_TREND_LEN,
    EMA_SLOPE_HOURS,
    ALLOW_COUNTERTREND_REVERSAL,
    COUNTERTREND_RIP_LOOKBACK_HOURS,
    COUNTERTREND_MIN_DROP_1H,
    COUNTERTREND_MIN_PULLBACK_PCT,
)


def add_rip_features(df: pd.DataFrame, lookback: int = RIP_LOOKBACK_HOURS) -> pd.DataFrame:
    df = df.copy()

    # 24h momentum used for rip detection
    df["past_return"] = df["close"].pct_change(lookback)
    df["ret_1h"] = df["close"].pct_change(1)

    # Relative volume
    vol_ma = df["volume"].rolling(20).mean()
    df["relative_volume"] = df["volume"] / vol_ma.replace(0, np.nan)

    # RSI(14)
    if "rsi_14" not in df.columns:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

    # Simple trend filter: bearish regime
    df["ema200"] = df["close"].ewm(span=EMA_TREND_LEN, adjust=False).mean()
    df["ema200_slope_24h"] = df["ema200"].pct_change(EMA_SLOPE_HOURS)
    df["trend_bearish"] = (df["close"] < df["ema200"]) & (df["ema200_slope_24h"] <= 0)

    return df


def generate_signals(
    df: pd.DataFrame,
    rip_threshold: float = None,
    use_volume_filter: bool = USE_VOLUME_FILTER,
    use_rsi_filter: bool = USE_RSI_FILTER,
    use_trend_filter: bool = USE_TREND_FILTER,
    allow_countertrend_reversal: bool = ALLOW_COUNTERTREND_REVERSAL,
    countertrend_rip_lookback_hours: int = COUNTERTREND_RIP_LOOKBACK_HOURS,
    countertrend_min_drop_1h: float = COUNTERTREND_MIN_DROP_1H,
    countertrend_min_pullback_pct: float = COUNTERTREND_MIN_PULLBACK_PCT,
    min_volume_mult: float = MIN_VOLUME_MULT,
    rsi_min: float = RSI_MIN,
    rsi_max: float = RSI_MAX,
) -> pd.DataFrame:
    """Generate SHORT signals after strong 24h pumps."""
    out = df.copy()

    if rip_threshold is None:
        rip_threshold = RIP_THRESHOLD_STRONG if USE_STRONG_THRESHOLD else RIP_THRESHOLD_PCT

    if "past_return" not in out.columns:
        out = add_rip_features(out)

    if "ret_1h" not in out.columns:
        out["ret_1h"] = out["close"].pct_change(1)

    lookback = max(2, int(countertrend_rip_lookback_hours))
    out["past_return_recent_max"] = out["past_return"].rolling(lookback, min_periods=1).max()
    out["recent_high_for_pullback"] = out["high"].rolling(lookback, min_periods=2).max().shift(1)
    out["pullback_from_recent_high"] = out["close"] / out["recent_high_for_pullback"] - 1.0

    out["signal"] = "NO TRADE"
    out["signal_direction"] = 0
    out["rip_size"] = 0.0

    for idx in out.index:
        row = out.loc[idx]
        past_ret = row.get("past_return", np.nan)
        past_ret_recent_max = row.get("past_return_recent_max", np.nan)
        ret_1h = row.get("ret_1h", np.nan)
        pullback = row.get("pullback_from_recent_high", np.nan)
        rel_vol = row.get("relative_volume", np.nan)
        rsi = row.get("rsi_14", np.nan)
        trend_bearish = bool(row.get("trend_bearish", False))

        if pd.isna(past_ret):
            continue

        rip_now = float(past_ret) > float(rip_threshold)
        rip_recent = pd.notna(past_ret_recent_max) and float(past_ret_recent_max) > float(rip_threshold)
        reversal_ok = (
            pd.notna(ret_1h)
            and pd.notna(pullback)
            and float(ret_1h) <= -float(countertrend_min_drop_1h)
            and float(pullback) <= -float(countertrend_min_pullback_pct)
        )

        # Legacy behavior in bearish regime: short immediate rip.
        trend_path_ok = rip_now

        # New behavior in non-bearish regime: require recent rip + reversal confirmation.
        countertrend_path_ok = (
            use_trend_filter
            and (not trend_bearish)
            and allow_countertrend_reversal
            and rip_recent
            and reversal_ok
        )

        if use_trend_filter:
            if trend_bearish:
                entry_ok = trend_path_ok
            else:
                entry_ok = countertrend_path_ok
        else:
            entry_ok = trend_path_ok

        if entry_ok:
            if use_volume_filter and (pd.isna(rel_vol) or float(rel_vol) < float(min_volume_mult)):
                continue

            if use_rsi_filter and (pd.isna(rsi) or float(rsi) < float(rsi_min) or float(rsi) > float(rsi_max)):
                continue

            out.loc[idx, "signal"] = "SHORT"
            out.loc[idx, "signal_direction"] = -1
            out.loc[idx, "rip_size"] = float(max(float(past_ret), float(past_ret_recent_max) if pd.notna(past_ret_recent_max) else float(past_ret)))

    return out


def compute_exit_levels(
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> dict:
    """Compute short exit levels."""
    return {
        "stop_loss": entry_price * (1 + stop_loss_pct),
        "take_profit": entry_price * (1 - take_profit_pct),
    }
