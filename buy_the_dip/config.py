"""
config.py
Configuration for Buy-the-Dip strategy.

VALIDATED EDGE (Walk-Forward Tested):
- Buy after 7%+ drops in 24h
- BNB: Stability 76.4/100 (EXCELLENT), +25% backtest return
- DOGE: Stability 62.7/100 (GOOD), +7.6% backtest return
- Average Win Rate: 70%
- Time-based exit (24h hold)

This strategy passed walk-forward validation with 82% of
windows showing positive returns for BNB.
"""

### Sobreescritura ideal config -> overrides estáticos -> matrix_overrides.json -> CLI override

# ── Data ─────────────────────────────────────────────────────
TIMEFRAME = "1h"
DAYS = 365  # 1 years of data
# DAYS = 1825  # 5 years of data

# VALIDATED symbols (passed walk-forward)
DEFAULT_SYMBOLS = [
    # Excelentes
    # "SUI/USDT",    # Stability 73.zz6/100 - EXCELLENT
    # "UNI/USDT",    # Stability 73.zz6/100 - EXCELLENT
    # "BNB/USDT",    # Stability 73.zz6/100 - EXCELLENT
    # "ADA/USDT",    # Stability 63.8/100 - GOOD
    "XRP/USDT",    # Stability 56.4/100 - GOOD
    "INJ/USDT",    # Stability 60/100 - GOOD
    # "AGLD/USDT",    # Stability 60/100 - GOOD
    # "DASH/USDT",    # Stability 60/100 - GOOD

    # Buenos
    # "LINK/USDT",   # Stability 74.1/100 - EXCELLENT
    # "BNB/USDT",    # Stability 76.4/100 - EXCELLENT
    # "POL/USDT",    # Stability 66.7/100 - GOOD

    # Neutrales
    # "DOGE/USDT",   # Stability 62.7/100 - GOOD
    # "BTC/USDT",    # Stability 43.8/100 - MODERATE
    # "AVAX/USDT",   # Stability 55.6/100 - GOOD
    # "SOL/USDT",   # Stability 51.5/100 - MODERATE

    # Eliminar
    # "APT/USDT",    # Stability 54.7/100 - GOOD
    # "ARB/USDT",    # Stability 48/100 - GOOD
    # "ETH/USDT",   # Stability 45/100 - MODERATE
    # "PAXG/USDT",    # Stability 35/100 - POOR
]

# ── Entry Rules ─────────────────────────────────────────────
# The core edge: buy after big drops
DIP_THRESHOLD_PCT = -0.05       # Enter when 24h return < -5%
DIP_THRESHOLD_STRONG = -0.07    # Stronger edge at -7%
USE_STRONG_THRESHOLD = True     # Use -7% for higher quality signals

# Lookback for calculating drop
DIP_LOOKBACK_HOURS = 24

# ── Exit Rules ──────────────────────────────────────────────
# Based on observed mean reversion behavior
# Key insight: After 7% drops, price bounces ~1.5-2% in 24h
# CRITICAL: Avg intra-period drawdown is -6%, so NO tight SL
# Using TIME-BASED exit to capture the natural bounce
TAKE_PROFIT_PCT = 0.04        # 4% TP (captures ~36% of trades)
STOP_LOSS_PCT = 0.20            # 20% SL (catastrophic only - should never hit)
# STOP_LOSS_PCT = 0.16            # 20% SL (catastrophic only - should never hit)
# STOP_LOSS_PCT = 0.13            # 20% SL (catastrophic only - should never hit)
#   XRP/USDT         28.89%     1.59    68.7%     1.80      198
# STOP_LOSS_PCT = 0.14            # 20% SL (catastrophic only - should never hit)
#   XRP/USDT         29.85%     1.63    69.2%     1.85      198
HOLD_TIMEOUT_HOURS = 24         # Exit after 24h regardless (optimal timing)

# ── Position Sizing ─────────────────────────────────────────
POSITION_PCT = 0.20             # 20% of capital per trade (higher conviction)
MAX_POSITIONS = 2               # Max simultaneous positions
INITIAL_CAPITAL = 10_000.0
FEE_PCT = 0.001                 # 0.1% per side

# ── Filters (optional, can disable) ─────────────────────────
# Volume filter - ensure there's liquidity
MIN_VOLUME_MULT = 0.5           # Volume should be at least 50% of average
USE_VOLUME_FILTER = False       # Disabled by default (edge works without it)

# RSI filter - don't buy extreme oversold (may be dead cat)
RSI_MIN = 10                    # Avoid RSI < 10 (extreme panic)
RSI_MAX = 40                    # Only buy when RSI < 40 (oversold)
USE_RSI_FILTER = False          # Disabled by default

# ── Entry/Event Control (optional) ────────────────────────
# If enabled: allow only one entry per dip event until price reset
ONE_ENTRY_PER_DIP_EVENT = False
DIP_EVENT_RESET_PCT = -0.04      # Reset event when 24h return rises above -4%

# ── Fail-Fast Exit (optional) ─────────────────────────────
# If enabled: close early if trade doesn't rebound enough in N bars
FAIL_FAST_BARS = None            # e.g., 10 to enable, None to disable
FAIL_FAST_MIN_REBOUND_PCT = 0.01 # 1% minimum rebound required by fail_fast_bars

# BTC alignment - optional
USE_BTC_FILTER = False          # Don't require BTC alignment

# ── Trading Mode ────────────────────────────────────────────
LONG_ONLY = True                # Buy the dip = LONG only

# ── Per-Symbol Overrides ────────────────────────────────────
# These values come from matrix tests (matrix_runs/*_matrix.csv).
# Optional extra keys per symbol:
# - hold_timeout_hours (int)
# - max_positions (int)
# - take_profit_pct (float)
# - stop_loss_pct (float)
# - min_volume_mult (float)
# - rsi_min (float)
# - rsi_max (float)
# - one_entry_per_dip_event (bool)
# - dip_event_reset_pct (float)
# - fail_fast_bars (int)
# - fail_fast_min_rebound_pct (float)
DEFAULT_DIP_THRESHOLD = DIP_THRESHOLD_STRONG if USE_STRONG_THRESHOLD else DIP_THRESHOLD_PCT

SYMBOL_SIGNAL_OVERRIDES = {
    "SUI/USDT": {"dip_threshold": -0.07, "use_rsi_filter": True, "use_volume_filter": False},
    "BNB/USDT": {"dip_threshold": -0.05, "use_rsi_filter": False, "use_volume_filter": True},
    
    "DOGE/USDT": {"dip_threshold": -0.05, "use_rsi_filter": True, "use_volume_filter": True},
    "ADA/USDT": {"dip_threshold": -0.06, "use_rsi_filter": False, "use_volume_filter": True},
    # "XRP/USDT": {"dip_threshold": -0.06, "use_rsi_filter": False, "use_volume_filter": True},
    #   XRP/USDT         14.85%     1.31    67.7%     1.77       96

    # "XRP/USDT": {"dip_threshold": -0.045, "use_rsi_filter": False, "use_volume_filter": True, "hold_timeout_hours": 24, "max_positions": 2},
    #   XRP/USDT         16.69%     0.90    61.3%     1.45      271

    # "XRP/USDT": {"dip_threshold": -0.045, "use_rsi_filter": False, "use_volume_filter": True, "hold_timeout_hours": 24, "max_positions": 3},
    #   XRP/USDT         15.08%     0.93    61.6%     1.46      216

    # "XRP/USDT": {"dip_threshold": -0.045, "use_rsi_filter": False, "use_volume_filter": True, "hold_timeout_hours": 36, "max_positions": 3},
    #   XRP/USDT         16.16%     0.92    65.4%     1.42      211

    # "XRP/USDT": {"dip_threshold": -0.05, "use_rsi_filter": False, "use_volume_filter": True, "hold_timeout_hours": 36, "max_positions": 3},
    #   XRP/USDT         16.20%     0.96    66.7%     1.45      177

    # "XRP/USDT": {"dip_threshold": -0.05, "use_rsi_filter": False, "use_volume_filter": True, "hold_timeout_hours": 36, "max_positions": 4},
    #   XRP/USDT         18.11%     0.94    66.1%     1.46      218
    
    # "XRP/USDT": {"dip_threshold": -0.05, "use_rsi_filter": False, "use_volume_filter": True, "hold_timeout_hours": 24, "max_positions": 4},
    #   XRP/USDT         16.42%     0.93    61.7%     1.52      222

    # "XRP/USDT": {"dip_threshold": -0.055, "use_rsi_filter": False, "use_volume_filter": True, "hold_timeout_hours": 36, "max_positions": 4},
    #   XRP/USDT         16.96%     0.94    66.8%     1.52      184
    
    # "XRP/USDT": {"dip_threshold": -0.055, "use_rsi_filter": False, "use_volume_filter": True, "hold_timeout_hours": 24, "max_positions": 4},
    #   XRP/USDT         22.94%     1.30    67.7%     1.78      189

    # "XRP/USDT": {
    #     "dip_threshold": -0.055, 
    #     "use_rsi_filter": False, 
    #     "use_volume_filter": True, 
    #     "hold_timeout_hours": 27, 
    #     "max_positions": 4,
    #     "take_profit_pct": 0.04
    # },
    #  XRP/USDT         24.70%     1.35    67.5%     1.80      188

    # "XRP/USDT": { <-- BEST OVERALL
    #     "dip_threshold": -0.055,
    #     "use_rsi_filter": False,
    #     "use_volume_filter": True,
    #     "hold_timeout_hours": 28,
    #     "max_positions": 4,
    #     "take_profit_pct": 0.04,
    #     "stop_loss_pct": 0.20,
    # },
    # 365 days
    #   XRP/USDT         26.12%     1.41    67.9%     1.82      187
    #   XRP/USDT          1.32%     0.56    61.1%     1.45       35    -7.03%      66.7 (GOOD)        2/3 (66.7%)
    # 1825 days
    #   XRP/USDT         82.05%     0.61    70.4%     1.29     1162
    #   XRP/USDT          1.45%     1.05    67.4%     5.86       31    -7.80% 74.1 (EXCELLENT)      20/27 (74.1%)


    # "XRP/USDT": {
    #     "dip_threshold": -0.055, 
    #     "use_rsi_filter": False, 
    #     "use_volume_filter": True, 
    #     "hold_timeout_hours": 28, 
    #     "max_positions": 4,
    #     "take_profit_pct": 0.04,
    #     "stop_loss_pct": 0.14
    # },
    # 365 days
    #   XRP/USDT         29.85%     1.63    69.2%     1.85      198
    #   XRP/USDT          2.90%     1.03    61.4%     1.61       36    -6.77% 100.0 (EXCELLENT)       3/3 (100.0%)
    # 1825 days
    #   XRP/USDT         72.69%     0.59    69.7%     1.26     1227
    #   XRP/USDT          1.33%     1.04    67.3%     5.90       33    -7.83%      65.2 (GOOD)      17/27 (63.0%)


    # "XRP/USDT": {
    #     "dip_threshold": -0.055,
    #     "use_rsi_filter": True,
    #     "use_volume_filter": True,
    #     "hold_timeout_hours": 28,
    #     "max_positions": 4,
    #     "take_profit_pct": 0.04,
    #     "stop_loss_pct": 0.20,
    #     "min_volume_mult": 0.60,
    #     "rsi_min": 10,
    #     "rsi_max": 60
    # },
    # 365 days
    #   XRP/USDT         25.33%     1.45    67.8%     1.87      171 
    #   XRP/USDT          1.52%     0.56    61.3%     1.59       32    -6.87%      66.7 (GOOD)        2/3 (66.7%)
    # 1825 days
    #   XRP/USDT         86.58%     0.65    70.5%     1.32     1090
    #   XRP/USDT          1.70%     1.14    67.6%     5.82       29    -7.58% 72.6 (EXCELLENT)      19/27 (70.4%)


    "XRP/USDT": {
        "dip_threshold": -0.055,
        "use_rsi_filter": True,
        "use_volume_filter": True,
        "hold_timeout_hours": 28,
        "max_positions": 4,
        "take_profit_pct": 0.04,
        "stop_loss_pct": 0.20,
        "min_volume_mult": 0.60,
        "rsi_min": 25,
        "rsi_max": 60
    },
    # 365 days
    #   XRP/USDT         25.32%     1.98    66.4%     2.58      110
    #   XRP/USDT          2.46%     1.36    53.2%     1.74       18    -3.50%      66.7 (GOOD)        2/3 (66.7%)
    # 1825 days
    #   XRP/USDT         96.64%     0.77    70.8%     1.37      898
    #   XRP/USDT          1.68%     1.13    66.4%    13.88       23    -6.10% 78.5 (EXCELLENT)      20/27 (74.1%)

    # "XRP/USDT": {
    #     "dip_threshold": -0.055,
    #     "use_rsi_filter": True,
    #     "use_volume_filter": True,
    #     "hold_timeout_hours": 28,
    #     "max_positions": 4,
    #     "take_profit_pct": 0.06,
    #     "stop_loss_pct": 0.20,
    #     "min_volume_mult": 0.80,
    #     "rsi_min": 25,
    #     "rsi_max": 58,
    #     "one_entry_per_dip_event": False,
    #     "dip_event_reset_pct": -0.04,
    #     "fail_fast_bars": 10,
    #     "fail_fast_min_rebound_pct": 0.01
    # },
    # 365 days
    #   XRP/USDT         24.82%     2.12    53.7%     2.68       95
    #   XRP/USDT          2.53%     2.10    50.2%     2.57       16    -3.47% 76.7 (EXCELLENT)        2/3 (66.7%)
    # 1825 days
    #   XRP/USDT        124.18%     0.91    61.6%     1.51      721
    #   XRP/USDT          2.28%     1.50    59.5%     3.53       20    -5.28% 81.1 (EXCELLENT)      21/27 (77.8%)

    "LINK/USDT": {"dip_threshold": -0.07, "use_rsi_filter": False, "use_volume_filter": False},
    "POL/USDT": {"dip_threshold": -0.08, "use_rsi_filter": False, "use_volume_filter": True},
    "AVAX/USDT": {"dip_threshold": -0.08, "use_rsi_filter": False, "use_volume_filter": True},
    "SOL/USDT": {"dip_threshold": -0.08, "use_rsi_filter": False, "use_volume_filter": True},
    # "INJ/USDT": {"dip_threshold": -0.08, "use_rsi_filter": False, "use_volume_filter": True},
    # 1825 days - config anterior
    #   IS  PF 1.20  |  OOS PF 1.24  |  WF stability 74.2  |  FULL 39.52%  PF 1.21   676 trades

    "INJ/USDT": {
        "dip_threshold": -0.08,
        "use_rsi_filter": True,
        "use_volume_filter": True,
        "hold_timeout_hours": 24,
        "max_positions": 3,
        "take_profit_pct": 0.03,
        "stop_loss_pct": 0.14,
        "min_volume_mult": 0.5,
        "rsi_min": 30,
        "rsi_max": 60,
    },
    # 1825 days - optimize_symbol.py + refine_symbol.py (1440 configs evaluadas)
    # Split IS 2021-08→2025-02 / OOS 2025-02→2026-08 (OOS oculto durante la busqueda)
    #   IS  PF 1.28  |  OOS PF 1.73  (retencion 1.35)  |  WF stability 68.1 / 26 ventanas
    #   FULL 55.23%  PF 1.35   635 trades
    # Meseta: las 24 configs vecinas pasan todas (OOS PF 1.29-1.93). No es un pico aislado.
    # Cambios clave vs anterior: RSI filter ON (30-60) y TP 4%→3%.
    "BTC/USDT": {"dip_threshold": -0.07, "use_rsi_filter": True, "use_volume_filter": False},
}


def get_symbol_signal_params(symbol: str, dip_threshold_override: float = None) -> dict:
    """
    Resolve signal parameters for a symbol with fallback to global defaults.

    Args:
        symbol: Trading pair (e.g., BTC/USDT)
        dip_threshold_override: Optional CLI override applied to all symbols

    Returns:
        Dict with dip_threshold, use_volume_filter, use_rsi_filter,
        hold_timeout_hours, max_positions, take_profit_pct, stop_loss_pct,
        min_volume_mult, rsi_min, rsi_max, one_entry_per_dip_event,
        dip_event_reset_pct, fail_fast_bars, fail_fast_min_rebound_pct
    """
    params = {
        "dip_threshold": DEFAULT_DIP_THRESHOLD,
        "use_volume_filter": USE_VOLUME_FILTER,
        "use_rsi_filter": USE_RSI_FILTER,
        "hold_timeout_hours": HOLD_TIMEOUT_HOURS,
        "max_positions": MAX_POSITIONS,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "min_volume_mult": MIN_VOLUME_MULT,
        "rsi_min": RSI_MIN,
        "rsi_max": RSI_MAX,
        "one_entry_per_dip_event": ONE_ENTRY_PER_DIP_EVENT,
        "dip_event_reset_pct": DIP_EVENT_RESET_PCT,
        "fail_fast_bars": FAIL_FAST_BARS,
        "fail_fast_min_rebound_pct": FAIL_FAST_MIN_REBOUND_PCT,
    }
    params.update(SYMBOL_SIGNAL_OVERRIDES.get(symbol, {}))

    if dip_threshold_override is not None:
        params["dip_threshold"] = dip_threshold_override

    return params

# ── Walk-Forward Validation ─────────────────────────────────
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS = 2
WF_STEP_MONTHS = 2
