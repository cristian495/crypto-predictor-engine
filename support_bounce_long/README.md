# Support Bounce Long Strategy

## Overview

`support_bounce_long` migra la lógica de detección LONG del frontend (`predictor-ui/lib`) al engine para tener:

- señales reproducibles (`LONG` / `NO_SIGNAL`)
- backtest de portafolio
- walk-forward validation
- optimización por moneda (matrix)
- artifacts JSON para frontend

## Core Logic (Parity with UI)

1. Detecta pivots y zonas de soporte/resistencia.
2. Evalúa cercanía a soporte.
3. Exige bullish rejection.
4. Evalúa confirmación (`aggressive|balanced|conservative`).
5. Evalúa contexto de pullback (opcional).
6. Calcula setup con stop bajo soporte, target en resistencia y filtro de `min_rr`.

## Setup Environment

```bash
# From predictor-engine root
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

### Scan + Backtest

```bash
# From predictor-engine root
python support_bounce_long/main.py --scan --symbols BNB/USDT XRP/USDT --days 365
```

### Walk-forward

```bash
python support_bounce_long/main.py --walk-forward --symbols BNB/USDT XRP/USDT --days 1095
```

### Matrix Optimization (per symbol)

```bash
python support_bounce_long/main.py --matrix --symbols BNB/USDT XRP/USDT --max-configs 800
```

Outputs:

- `support_bounce_long/matrix_runs/<SYMBOL>_matrix.csv`
- `support_bounce_long/matrix_runs/<SYMBOL>_matrix.json`
- `support_bounce_long/matrix_overrides.json`

### Export Data for Frontend

```bash
python support_bounce_long/export_frontend_data.py \
  --include-walkforward \
  --out-dir exports/support_bounce_long
```

Generates:

- `exports/support_bounce_long/signals.latest.json`
- `exports/support_bounce_long/backtest.latest.json`
- `exports/support_bounce_long/walkforward.latest.json` (optional)

## Useful Tuning Flags

```bash
--tolerance-pct
--min-touches
--pivot-window
--support-proximity-pct
--rejection-threshold
--min-rr
--stop-buffer-pct
--confirmation-mode
--require-pullback / --no-require-pullback
--pullback-lookback
--hold-timeout-bars
--allow-candidates
```
