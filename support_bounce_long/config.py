"""
Configuration for support_bounce_long.

This strategy migrates the UI long detector logic (supports + bullish rejection)
into the engine for reproducible backtest, walk-forward, and matrix optimization.
"""

from __future__ import annotations

from pathlib import Path
import json

# Data
TIMEFRAME = "1h"
DAYS = 365

DEFAULT_SYMBOLS = [
    # "BNB/USDT",
    # "DOGE/USDT",
    "XRP/USDT",
    # "ADA/USDT",
    # "LINK/USDT",
    # "SOL/USDT",
]

# Portfolio / execution
INITIAL_CAPITAL = 10_000.0
POSITION_PCT = 0.15
MAX_POSITIONS = 3
FEE_PCT = 0.001

# Signal / setup defaults (ported from predictor-ui/lib defaults)
BASE_SYMBOL_PARAMS = {
    # Zone detector
    "tolerance_pct": 0.6,
    "min_touches": 2,
    "pivot_window": 1,
    # Long setup
    "support_proximity_pct": 0.25,
    "rejection_threshold": 0.35,
    "min_rr": 1.4,
    "stop_buffer_pct": 0.15,
    "confirmation_mode": "balanced",  # aggressive|balanced|conservative
    "require_pullback": True,
    "pullback_lookback": 6,
    # Exit policy
    "hold_timeout_bars": 24,
    # Signal policy
    "require_confirmed": True,
}

# Optional hand-curated symbol tweaks (kept empty by default)
STATIC_SYMBOL_OVERRIDES: dict[str, dict] = {}

MATRIX_OVERRIDES_PATH = Path(__file__).resolve().parent / "matrix_overrides.json"
MATRIX_RUNS_DIR = Path(__file__).resolve().parent / "matrix_runs"

VALID_CONFIRMATION_MODES = {"aggressive", "balanced", "conservative"}

# Walk-forward
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS = 2
WF_STEP_MONTHS = 2
WF_MIN_TRADES_FOR_PF = 5

# Matrix tuning grid
MATRIX_TOLERANCE_GRID = [0.4, 0.6, 0.8]
MATRIX_MIN_TOUCHES_GRID = [2, 3]
MATRIX_PIVOT_WINDOW_GRID = [1, 2]
MATRIX_SUPPORT_PROXIMITY_GRID = [0.2, 0.25, 0.35]
MATRIX_REJECTION_THRESHOLD_GRID = [0.25, 0.35, 0.45]
MATRIX_MIN_RR_GRID = [1.2, 1.4, 1.6]
MATRIX_STOP_BUFFER_GRID = [0.10, 0.15, 0.25]
MATRIX_CONFIRMATION_MODE_GRID = ["aggressive", "balanced", "conservative"]
MATRIX_REQUIRE_PULLBACK_GRID = [False, True]
MATRIX_PULLBACK_LOOKBACK_GRID = [4, 6, 8]
MATRIX_HOLD_TIMEOUT_GRID = [12, 24, 36]
MATRIX_MAX_CONFIGS_PER_SYMBOL = 800

# Multi-horizon robust scoring
ROBUST_DAYS_LIST = [365, 1095, 1825]
ROBUST_DAYS_WEIGHTS = [0.2, 0.3, 0.5]


def _load_matrix_overrides() -> dict[str, dict]:
    if not MATRIX_OVERRIDES_PATH.exists():
        return {}

    try:
        with open(MATRIX_OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def get_symbol_params(symbol: str, overrides: dict | None = None) -> dict:
    """Resolve params from base -> static -> matrix -> runtime overrides."""
    params = dict(BASE_SYMBOL_PARAMS)
    params.update(STATIC_SYMBOL_OVERRIDES.get(symbol, {}))
    params.update(_load_matrix_overrides().get(symbol, {}))
    if overrides:
        params.update(overrides)

    mode = str(params.get("confirmation_mode", "balanced")).strip().lower()
    if mode not in VALID_CONFIRMATION_MODES:
        mode = "balanced"
    params["confirmation_mode"] = mode

    params["min_touches"] = max(1, int(params.get("min_touches", BASE_SYMBOL_PARAMS["min_touches"])))
    params["pivot_window"] = max(1, int(params.get("pivot_window", BASE_SYMBOL_PARAMS["pivot_window"])))
    params["pullback_lookback"] = max(2, int(params.get("pullback_lookback", BASE_SYMBOL_PARAMS["pullback_lookback"])))
    params["hold_timeout_bars"] = max(1, int(params.get("hold_timeout_bars", BASE_SYMBOL_PARAMS["hold_timeout_bars"])))

    params["tolerance_pct"] = max(0.01, float(params.get("tolerance_pct", BASE_SYMBOL_PARAMS["tolerance_pct"])))
    params["support_proximity_pct"] = max(0.0, float(params.get("support_proximity_pct", BASE_SYMBOL_PARAMS["support_proximity_pct"])))
    params["rejection_threshold"] = max(0.05, float(params.get("rejection_threshold", BASE_SYMBOL_PARAMS["rejection_threshold"])))
    params["min_rr"] = max(0.1, float(params.get("min_rr", BASE_SYMBOL_PARAMS["min_rr"])))
    params["stop_buffer_pct"] = max(0.0, float(params.get("stop_buffer_pct", BASE_SYMBOL_PARAMS["stop_buffer_pct"])))

    params["require_pullback"] = bool(params.get("require_pullback", BASE_SYMBOL_PARAMS["require_pullback"]))
    params["require_confirmed"] = bool(params.get("require_confirmed", BASE_SYMBOL_PARAMS["require_confirmed"]))

    return params
