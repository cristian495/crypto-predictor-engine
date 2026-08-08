"""
config.py
Configuration for the Pump Risk detector.
"""

# Data
TIMEFRAME = "1d"
DAYS = 730
DEFAULT_SYMBOLS = [
    "BNB/USDT",
    # "XRP/USDT",
    # "DOGE/USDT",
    # "TRX/USDT",
    # "ADA/USDT",
    # "SOL/USDT",
    # "LINK/USDT",
]

# Labeling (used for historical evaluation only)
PUMP_THRESHOLD_PCT = 0.03  # +3% daily move
PUMP_LABEL_MODE = "either"  # close | intraday | either | both

# Scan defaults (for live multi-symbol monitoring)
SCAN_MODE = "confirm"  # early | confirm | both
SCAN_SIGNAL_LEVEL = "medium"  # high | medium
SCAN_USE_INTRADAY_TRIGGER = True

# Intraday trigger layer (strict profile)
TRIGGER_TIMEFRAME = "1h"
TRIGGER_BREAKOUT_BUFFER_PCT = 0.001  # require close >= prev_day_high * (1 + buffer)
TRIGGER_MIN_REL_VOLUME = 1.10        # 1h volume vs rolling mean
TRIGGER_MIN_CLOSE_POS = 0.60         # close location in candle range [0,1]
TRIGGER_REQUIRE_GREEN = True        # close > open
TRIGGER_VOL_LOOKBACK_BARS = 12      # for relative volume on 1h candles

# Feature thresholds (computed on day t to estimate risk for day t+1)
RSI_MIN = 30.0
RSI_MAX = 55.0
MAX_VOL_RATIO_PRE = 0.90
MAX_ABS_RET_PRE = 0.03
MIN_ATR_PCT = 0.04
MAX_FUNDING_FOR_SQUEEZE = 0.0
MIN_FUNDING_DELTA = 0.0
MIN_OI_CHG = 0.0

# Score weights
WEIGHTS = {
    "rsi_zone": 1.0,
    "calm_volume": 1.0,
    "calm_day": 1.0,
    "funding_neg": 1.0,
    "funding_rising": 1.0,
    "atr_regime": 1.0,
    "oi_rising": 0.0,  # kept at 0 by default (inconsistent across symbols)
}

# Risk buckets
HIGH_RISK_MIN_SCORE = 6.0
MEDIUM_RISK_MIN_SCORE = 4.0
