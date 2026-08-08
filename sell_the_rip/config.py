"""
Configuration for Sell-the-Rip strategy (simple short-only mirror of buy_the_dip).
"""

from pathlib import Path
import json

# Data
TIMEFRAME = "1h"
# DAYS = 365
DAYS = 1825

# Universe (same as downtrend_breakout)
DEFAULT_SYMBOLS = [
    # "SUI/USDT",
    # "DOGE/USDT",
    # "LINK/USDT",
    # "SOL/USDT",
    # "ADA/USDT",

    # optimizados
    "BNB/USDT",
    "XRP/USDT"
]

# Entry: short after strong 24h pump
RIP_THRESHOLD_PCT = 0.05
RIP_THRESHOLD_STRONG = 0.06
USE_STRONG_THRESHOLD = True
RIP_LOOKBACK_HOURS = 24

# Exits (short)
TAKE_PROFIT_PCT = 0.03
STOP_LOSS_PCT = 0.05
HOLD_TIMEOUT_HOURS = 24

# Portfolio
POSITION_PCT = 0.15
MAX_POSITIONS = 3
INITIAL_CAPITAL = 10_000.0
FEE_PCT = 0.001

# Optional filters
MIN_VOLUME_MULT = 0.8
USE_VOLUME_FILTER = False

RSI_MIN = 60
RSI_MAX = 95
USE_RSI_FILTER = True

USE_TREND_FILTER = True
EMA_TREND_LEN = 200
EMA_SLOPE_HOURS = 24

# Countertrend reversal entry (captures pump->rejection even if EMA200 trend is not bearish)
ALLOW_COUNTERTREND_REVERSAL = False
COUNTERTREND_RIP_LOOKBACK_HOURS = 8
COUNTERTREND_MIN_DROP_1H = 0.01
COUNTERTREND_MIN_PULLBACK_PCT = 0.012

# Per-symbol defaults (matrix can refine)
DEFAULT_RIP_THRESHOLD = RIP_THRESHOLD_STRONG if USE_STRONG_THRESHOLD else RIP_THRESHOLD_PCT

BASE_SYMBOL_PARAMS = {
    "rip_threshold": DEFAULT_RIP_THRESHOLD,
    "use_volume_filter": USE_VOLUME_FILTER,
    "use_rsi_filter": USE_RSI_FILTER,
    "use_trend_filter": USE_TREND_FILTER,
    "allow_countertrend_reversal": ALLOW_COUNTERTREND_REVERSAL,
    "countertrend_rip_lookback_hours": COUNTERTREND_RIP_LOOKBACK_HOURS,
    "countertrend_min_drop_1h": COUNTERTREND_MIN_DROP_1H,
    "countertrend_min_pullback_pct": COUNTERTREND_MIN_PULLBACK_PCT,
    "min_volume_mult": MIN_VOLUME_MULT,
    "rsi_min": RSI_MIN,
    "rsi_max": RSI_MAX,
    "take_profit_pct": TAKE_PROFIT_PCT,
    "stop_loss_pct": STOP_LOSS_PCT,
    "hold_timeout_hours": HOLD_TIMEOUT_HOURS,
}

SYMBOL_SIGNAL_OVERRIDES = {
    "SUI/USDT": {"rip_threshold": 0.06, "use_rsi_filter": True, "use_volume_filter": False, "use_trend_filter": True},
    "DOGE/USDT": {"rip_threshold": 0.06, "use_rsi_filter": True, "use_volume_filter": False, "use_trend_filter": True},
    "LINK/USDT": {"rip_threshold": 0.05, "use_rsi_filter": True, "use_volume_filter": False, "use_trend_filter": True},
    "BNB/USDT": {"rip_threshold": 0.05, "use_rsi_filter": True, "use_volume_filter": False, "use_trend_filter": True},
    "SOL/USDT": {"rip_threshold": 0.06, "use_rsi_filter": True, "use_volume_filter": False, "use_trend_filter": True},
    "ADA/USDT": {"rip_threshold": 0.06, "use_rsi_filter": True, "use_volume_filter": False, "use_trend_filter": True},
}

MATRIX_OVERRIDES_PATH = Path(__file__).resolve().parent / "matrix_overrides.json"


def _load_matrix_overrides() -> dict:
    if not MATRIX_OVERRIDES_PATH.exists():
        return {}
    try:
        with open(MATRIX_OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def get_symbol_signal_params(symbol: str, rip_threshold_override: float = None) -> dict:
    params = dict(BASE_SYMBOL_PARAMS)
    params.update(SYMBOL_SIGNAL_OVERRIDES.get(symbol, {}))
    params.update(_load_matrix_overrides().get(symbol, {}))
    if rip_threshold_override is not None:
        params["rip_threshold"] = rip_threshold_override
    return params


# Walk-forward
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS = 2
WF_STEP_MONTHS = 2

# Matrix grids (expanded)
MATRIX_THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
MATRIX_USE_RSI_GRID = [False, True]
MATRIX_USE_VOLUME_GRID = [False, True]
MATRIX_USE_TREND_GRID = [False, True]
MATRIX_RSI_MIN_GRID = [55, 60, 65]
MATRIX_RSI_MAX_GRID = [85, 90, 95]
MATRIX_MIN_VOLUME_MULT_GRID = [0.8, 1.0]
MATRIX_TP_GRID = [0.02, 0.03, 0.04]
MATRIX_SL_GRID = [0.04, 0.05, 0.06, 0.07]
MATRIX_HOLD_GRID = [12, 24, 36]
MATRIX_MAX_CONFIGS_PER_SYMBOL = 900

# Multi-horizon robust scoring (optimize on all these horizons)
ROBUST_DAYS_LIST = [365, 1095, 1825]
# Prioritize long-horizon robustness to reduce 365-day overfitting.
ROBUST_DAYS_WEIGHTS = [0.2, 0.3, 0.5]
