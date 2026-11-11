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
#
# Revisión 2026-08-10: screening de los 70 pares USDT spot de mayor volumen en Binance
# con parámetros GLOBALES (sin tuning por moneda), 5 años, exigiendo que las dos mitades
# temporales fueran rentables. 31 tenían historia suficiente; 6 pasaron. Sobre esos se
# corrió un backtest de cartera con capital compartido (los scripts por símbolo usan
# capital dedicado y no capturan el solapamiento de dips entre monedas).
DEFAULT_SYMBOLS = [
    # Añadidos 2026-08-10 — screening en 2 mitades disjuntas + walk_forward.py --days 1825
    "ICP/USDT",    # WF 63.0 GOOD, consistencia 17/27. Params GLOBALES a propósito (ver abajo).
                   #   El más rentable de la cesta pero el menos estable: vigilar.
    "XLM/USDT",    # WF 76.3 EXCELLENT, consistencia 23/27. Override optimizado abajo.
    "SUI/USDT",    # WF 68.8 GOOD, consistencia 11/16. Solo 3.3 años de historia.
                   #   OJO: últimos 365d en NEGATIVO (-7.05%, PF 0.86). Verificado 2026-08-10
                   #   que NO es problema de parámetros: th-7/RSIon, th-7/RSIoff y th-8/RSIon
                   #   dan -7.1%, -6.1% y -5.6%. Es régimen del símbolo. A 5 años PF ~1.45 con
                   #   las tres. No re-tunear para arreglar el último año: eso es curve-fitting.

    # Incumbentes, revalidados en la misma corrida
    "XRP/USDT",    # WF 76.5 EXCELLENT, consistencia 22/26.
    # Media de la cesta: WF 71.0 EXCELLENT, Sharpe 1.46, PF 6.18, MaxDD -4.50%

    # Retirado 2026-11-10: en backtest de CARTERA (capital compartido, x3, 365d) BNB
    # dio -61.5% de PnL con WR 0% (solo 2 trades tomados, ambos perdedores). Aislado
    # tambien perdia neto tras costes de futuros (-9.9%, PF 0.74). Probadas 7 configs
    # (th/tp/sl/hold/rsi): ninguna lo vuelve rentable de forma fiable; las positivas
    # eran ruido (4 trades/año) o breakeven. Su edge de buy-the-dip es genuinamente debil.
    # "BNB/USDT",    # WF 70.5 EXCELLENT, consistencia 17/22. Aporta MUY poco PnL con th=-7%.

    # Retirado 2026-08-10: en backtest de cartera (capital compartido, ventana común
    # 2023-05 -> 2026-08) INJ resta incluso con sus params tuneados. Quitarlo de la cesta
    # BNB+XRP+INJ sube Sharpe 0.83 -> 1.06 y baja DD -16.5% -> -11.7%, con MÁS retorno.
    # En la cesta de 6 el efecto se repite (Sharpe 1.03 -> 1.12). Su override sigue abajo.
    # "INJ/USDT",    # Stability 60/100 - GOOD
    # "UNI/USDT",    # Stability 73.zz6/100 - EXCELLENT
    # "ADA/USDT",    # Stability 63.8/100 - GOOD
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
# MAX_POSITIONS: óptimo de CARTERA elegido 2026-11-10 (cesta sin BNB, x3, 365d).
# Barrido 2→8: con 4 el Sharpe de cartera es máximo (0.98) y el ratio Return/DD
# también (1.22, +50.1% ret / -41.0% DD). Subir a 5 añade más DD que retorno; de 6
# en adelante todo empeora. Con 2 se perdían ~690 señales por falta de slots.
MAX_POSITIONS = 4               # Max simultaneous positions (óptimo de cartera)
INITIAL_CAPITAL = 10_000.0

# ── Futures Realism (Binance USDⓈ-M) ────────────────────────
# El edge es pequeño (TP 3-4%), asi que los costes reales de futuros
# deciden si un simbolo es rentable NETO. El backtest antiguo usaba fee
# spot 0.1% y nada de slippage/funding/liquidacion -> inflaba el retorno.
FEE_PCT = 0.0004                # 0.04% taker por lado (Binance futures), NO 0.1% spot

# Slippage: entras justo tras una caida del 7% (peor liquidez). Conservador fijo.
SLIPPAGE_PCT = 0.0005           # 0.05% por lado, aplicado en entrada y salida

# Funding: en perpetuos se paga/cobra cada 8h. LONG-only paga cuando el
# funding es positivo (mercado alcista) y cobra cuando es negativo (tras caidas).
# Se descuenta con datos REALES de Binance si estan disponibles.
USE_FUNDING = True              # Descontar funding real durante la posicion
FUNDING_INTERVAL_HOURS = 8      # Binance liquida funding cada 8h
FUNDING_RATE_FALLBACK = 0.0001  # 0.01%/8h si no hay datos reales (conservador)

# Apalancamiento: con x2-x3 un -20% de precio NO liquida (margen aguanta),
# pero se modela para alertar si el SL configurado cae por debajo de liquidacion.
LEVERAGE = 3                    # Apalancamiento por defecto (override por simbolo)
# Margen de mantenimiento aprox. Binance para notional bajo (~0.5%). Conservador.
MAINTENANCE_MARGIN_RATE = 0.005

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
# - leverage (int|float)   # apalancamiento del simbolo; default global LEVERAGE
DEFAULT_DIP_THRESHOLD = DIP_THRESHOLD_STRONG if USE_STRONG_THRESHOLD else DIP_THRESHOLD_PCT

SYMBOL_SIGNAL_OVERRIDES = {
    # ── XLM/USDT ── optimize_symbol.py 2026-08-10, 1825 days, OOS = 30% más reciente.
    # Único de los tres candidatos nuevos cuya optimización sobrevivió fuera de muestra:
    #   IS  (238 trades): 64.37%  PF 1.80  Sharpe 1.23
    #   OOS ( 97 trades): 25.01%  PF 2.34  Sharpe 1.46  WR 71.1%  DD -5.29%
    #   retención PF IS->OOS: 1.30  (el PF MEJORA fuera de muestra)
    # Los 15 finalistas convergen al mismo núcleo con retención >= 0.98 => es meseta, no pico.
    # El óptimo caía en el borde del grid en 4 ejes, así que se extendió cada uno por separado:
    #   take_profit_pct  meseta plana 0.06-0.15 (OOS ret 23.9-25.5%) -> 0.08 queda dentro
    #   stop_loss_pct    pico interior real: 0.17->2.11  0.20->2.34  0.25->2.23
    #   rsi_max          saturado: 70/80/85/100 dan resultados IDÉNTICOS, el límite no ata
    #   min_volume_mult  subir solo encoge la muestra (2.5 deja 46 trades OOS) -> 1.2
    "XLM/USDT": {
        "dip_threshold": -0.06,
        "use_rsi_filter": True,
        "use_volume_filter": True,
        "hold_timeout_hours": 24,
        "max_positions": 2,
        "take_profit_pct": 0.08,
        "stop_loss_pct": 0.20,
        "min_volume_mult": 1.2,
        "rsi_min": 10,
        "rsi_max": 70,
        "one_entry_per_dip_event": False,
        "dip_event_reset_pct": -0.04,
        "fail_fast_bars": None,
        "fail_fast_min_rebound_pct": 0.01,
    },

    # ── ICP/USDT ── SIN override a propósito: usa los parámetros globales.
    # optimize_symbol.py 2026-08-10 produjo th=-10% TP=6% SL=14% hold=12h, pero los 15
    # finalistas tenían retención PF IS->OOS de 0.55-0.60 y retorno OOS de solo +1.34%
    # (Sharpe 0.14, WR 51.2%, 3/6 sub-periodos positivos). Curve-fitting.
    # Comparado en el MISMO periodo fuera de muestra, los globales ganan:
    #   config "optimizada": PF OOS 1.18, +1.3%
    #   globales:            PF 1.25 en H2 completo, +16.8%
    # No pegar aquí el bloque RECOMMENDED sin mirar antes la retención IS->OOS.

    # ── SUI/USDT ── se mantiene el override previo. La optimización de 2026-08-10 fue
    # descartada: los 15 finalistas daban retorno OOS NEGATIVO (-3.7% a -11.8%),
    # retención 0.30-0.51 y 0-2 de 6 sub-periodos positivos. Ni un candidato salvable.
    "SUI/USDT": {"dip_threshold": -0.07, "use_rsi_filter": True, "use_volume_filter": False},
    # "BNB/USDT": {"dip_threshold": -0.05, "use_rsi_filter": False, "use_volume_filter": True},
    # 1825 days (medido 2026-08, hereda TP 4% / SL 20% / hold 24h / maxpos 2 globales)
    #   IS PF 1.43  |  OOS PF 1.40  |  WF stability 63.0  |  FULL 37.29%  PF 1.42  405 trades

    "BNB/USDT": {
        "dip_threshold": -0.07,
        "use_rsi_filter": False,
        "use_volume_filter": True,
        "hold_timeout_hours": 28,
        "max_positions": 4,
        "take_profit_pct": 0.03,
        "stop_loss_pct": 0.20,
        "min_volume_mult": 0.5,
    },
    # 1825 days - refine_symbol.py (1944 configs evaluadas)
    # Split IS 2021-08→2025-02 / OOS 2025-02→2026-08 (OOS oculto durante la busqueda)
    #   IS PF 1.75  |  OOS PF 1.78  |  WF stability 73.2 / 26 ventanas
    #   FULL 46.63%  PF 1.75  WR 77.1%  DD -12.8%  354 trades
    # OJO: el OOS de BNB fue lateral (+2.8%), no un desplome como INJ/XRP.
    # Evidencia mas debil que en esos dos: peso mas el WF stability que el OOS.
    # max_positions=4 es critico: casi todas las vecinas con maxpos=2 fallan la puerta.
    # RSI filter OFF gana aqui, al reves que en INJ y XRP. mvm 0.6 no mejora (si en XRP).
    
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


    # "XRP/USDT": {
    #     "dip_threshold": -0.055,
    #     "use_rsi_filter": True,
    #     "use_volume_filter": True,
    #     "hold_timeout_hours": 28,
    #     "max_positions": 4,
    #     "take_profit_pct": 0.04,
    #     "stop_loss_pct": 0.20,
    #     "min_volume_mult": 0.60,
    #     "rsi_min": 25,
    #     "rsi_max": 60
    # },
    # 365 days
    #   XRP/USDT         25.32%     1.98    66.4%     2.58      110
    #   XRP/USDT          2.46%     1.36    53.2%     1.74       18    -3.50%      66.7 (GOOD)        2/3 (66.7%)
    # 1825 days (medido 2026-08, ventana 2021-08 -> 2026-08)
    #   IS PF 1.22  |  OOS PF 2.05  |  WF stability 63.8  |  FULL 55.36%  PF 1.35  678 trades

    "XRP/USDT": {
        "dip_threshold": -0.07,
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
    # 1825 days - refine_symbol.py (1800 configs evaluadas)
    # Split IS 2021-08→2025-02 / OOS 2025-02→2026-08 (OOS oculto durante la busqueda)
    #   IS PF 1.56  |  OOS PF 3.07 (retencion 1.97)  |  WF stability 81.5 / 26 ventanas
    #   FULL 86.45%  PF 1.75  WR 73.1%  DD -15.0%  405 trades
    # Meseta: las 36 configs vecinas pasan todas (OOS PF 1.48-3.18). Muy estable.
    # UNICO cambio vs config anterior: dip_threshold -0.055 -> -0.07.
    # El resto (hold 28h, maxpos 4, TP 4%, SL 20%, RSI 25-60, mvm 0.60) lo confirmo la busqueda.

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
        dip_event_reset_pct, fail_fast_bars, fail_fast_min_rebound_pct,
        leverage
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
        "leverage": LEVERAGE,
    }
    params.update(SYMBOL_SIGNAL_OVERRIDES.get(symbol, {}))

    if dip_threshold_override is not None:
        params["dip_threshold"] = dip_threshold_override

    return params

# ── Walk-Forward Validation ─────────────────────────────────
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS = 2
WF_STEP_MONTHS = 2
