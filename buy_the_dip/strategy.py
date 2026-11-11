"""
strategy.py
Buy-the-Dip strategy - simple, proven edge.

Core Logic:
- Buy when price drops significantly (>5-7% in 24h)
- Crypto tends to bounce after big drops
- No ML, no overfitting, just statistics

Edge Validation:
- BNB: 1158 events, +1.97% return, 62.5% win rate
- DOGE: 2585 events, +1.75% return, 60.1% win rate
- ETH: consistent positive returns after dips
"""

import pandas as pd
import numpy as np
from config import (
    DIP_THRESHOLD_PCT,
    DIP_THRESHOLD_STRONG,
    USE_STRONG_THRESHOLD,
    DIP_LOOKBACK_HOURS,
    TAKE_PROFIT_PCT,
    STOP_LOSS_PCT,
    MIN_VOLUME_MULT,
    USE_VOLUME_FILTER,
    RSI_MIN,
    RSI_MAX,
    USE_RSI_FILTER,
    ONE_ENTRY_PER_DIP_EVENT,
    DIP_EVENT_RESET_PCT,
)


def add_dip_features(df: pd.DataFrame, lookback: int = DIP_LOOKBACK_HOURS) -> pd.DataFrame:
    """
    Add features needed for dip detection.

    Args:
        df: DataFrame with OHLCV data
        lookback: Hours to look back for calculating drop

    Returns:
        DataFrame with dip features
    """
    df = df.copy()

    # Calculate past return (how much has price dropped)
    df["past_return"] = df["close"].pct_change(lookback)

    # Rolling high for reference
    df["rolling_high"] = df["high"].rolling(lookback).max()

    # Drop from rolling high
    df["drop_from_high"] = (df["close"] - df["rolling_high"]) / df["rolling_high"]

    # Relative volume
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(20).mean()
        df["relative_volume"] = df["volume"] / vol_ma
    else:
        df["relative_volume"] = 1.0

    # RSI if not present
    if "rsi_14" not in df.columns:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

    return df


def generate_signals(
    df: pd.DataFrame,
    dip_threshold: float = None,
    use_volume_filter: bool = USE_VOLUME_FILTER,
    use_rsi_filter: bool = USE_RSI_FILTER,
    min_volume_mult: float = MIN_VOLUME_MULT,
    rsi_min: float = RSI_MIN,
    rsi_max: float = RSI_MAX,
    one_entry_per_dip_event: bool = ONE_ENTRY_PER_DIP_EVENT,
    dip_event_reset_pct: float = DIP_EVENT_RESET_PCT,
) -> pd.DataFrame:
    """
    Generate buy signals when dip conditions are met.

    Entry Rule (simple):
    - past_return < dip_threshold (e.g., < -5%)

    Optional Filters:
    - Volume > MIN_VOLUME_MULT (liquidity check)
    - RSI between RSI_MIN and RSI_MAX

    Args:
        df: DataFrame with dip features
        dip_threshold: Override threshold (default from config)
        use_volume_filter: Apply volume filter
        use_rsi_filter: Apply RSI filter
        min_volume_mult: Minimum relative volume when volume filter is active
        rsi_min: Lower RSI bound when RSI filter is active
        rsi_max: Upper RSI bound when RSI filter is active
        one_entry_per_dip_event: If True, only one LONG per dip event
        dip_event_reset_pct: Reset threshold to allow a new dip event

    Returns:
        DataFrame with signals
    """
    df = df.copy()

    # Determine threshold
    if dip_threshold is None:
        dip_threshold = DIP_THRESHOLD_STRONG if USE_STRONG_THRESHOLD else DIP_THRESHOLD_PCT

    # Add features if not present
    if "past_return" not in df.columns:
        df = add_dip_features(df)

    # Initialize signals
    df["signal"] = "NO TRADE"
    df["signal_direction"] = 0
    df["dip_size"] = 0.0

    # Stateful control to avoid stacking multiple entries in the same dip event
    dip_event_consumed = False

    for idx in df.index:
        row = df.loc[idx]

        past_ret = row.get("past_return", 0)
        rel_vol = row.get("relative_volume", 1.0)
        rsi = row.get("rsi_14", 50)

        # Skip NaN
        if pd.isna(past_ret):
            continue

        if one_entry_per_dip_event and past_ret > dip_event_reset_pct:
            dip_event_consumed = False

        # ── Core Rule: Big Dip ──────────────────────────────
        if past_ret < dip_threshold:
            if one_entry_per_dip_event and dip_event_consumed:
                continue

            # Optional: Volume filter
            if use_volume_filter:
                if pd.isna(rel_vol) or rel_vol < min_volume_mult:
                    continue

            # Optional: RSI filter
            if use_rsi_filter:
                if pd.isna(rsi) or rsi < rsi_min or rsi > rsi_max:
                    continue

            # Signal: BUY THE DIP
            df.loc[idx, "signal"] = "LONG"
            df.loc[idx, "signal_direction"] = 1
            df.loc[idx, "dip_size"] = past_ret

            if one_entry_per_dip_event:
                dip_event_consumed = True

    return df


def compute_exit_levels(
    entry_price: float,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    stop_loss_pct: float = STOP_LOSS_PCT,
) -> dict:
    """
    Compute exit levels for a LONG position.

    Args:
        entry_price: Entry price
        take_profit_pct: Take profit percentage
        stop_loss_pct: Stop loss percentage

    Returns:
        Dict with stop_loss and take_profit prices
    """
    return {
        "stop_loss": entry_price * (1 - stop_loss_pct),
        "take_profit": entry_price * (1 + take_profit_pct),
    }


def print_signal_status(df: pd.DataFrame, symbol: str = ""):
    """Print current signal status."""
    last = df.iloc[-1]
    prefix = f"  {symbol:<12}" if symbol else "  "

    past_ret = last.get("past_return", 0) * 100
    rsi = last.get("rsi_14", 50)

    if last["signal"] == "LONG":
        print(f"{prefix}🔥 BUY THE DIP! Drop={past_ret:.1f}% RSI={rsi:.1f}")
    else:
        threshold = DIP_THRESHOLD_STRONG if USE_STRONG_THRESHOLD else DIP_THRESHOLD_PCT
        threshold_pct = threshold * 100
        print(f"{prefix}NO SIGNAL  24h={past_ret:+.1f}% (need <{threshold_pct:.0f}%)")
