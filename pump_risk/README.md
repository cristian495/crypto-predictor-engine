# Pump Risk Detector

Detects **next-day pump risk** (LOW / MEDIUM / HIGH) from daily market context:

- RSI zone (30-55)
- Calm prior volume
- Calm prior day return
- Negative funding (squeeze fuel)
- Funding rising
- ATR volatility regime
- Optional OI rising condition

## Run

Scanner de entrada (perfil estricto por defecto desde `config.py`):

```bash
cd pump_risk
../venv/bin/python main.py --scan
```

Scanner para símbolos específicos + JSON para monitor:

```bash
cd pump_risk
../venv/bin/python main.py --scan \
  --symbols XRP/USDT DOGE/USDT TRX/USDT ADA/USDT SOL/USDT LINK/USDT \
  --signals-json /tmp/pump_risk_signals.json
```

Scanner en doble fase (alerta temprana + confirmación en una sola corrida):

```bash
cd pump_risk
../venv/bin/python main.py --scan \
  --scan-mode both \
  --symbols XRP/USDT DOGE/USDT \
  --signals-json /tmp/pump_risk_signals_both.json
```

Validación histórica de un día objetivo (UTC):

```bash
cd pump_risk
../venv/bin/python main.py --scan \
  --symbols XRP/USDT \
  --target-date 2026-03-16
```

Walk-forward independiente (tipo estrategia) + fechas detectadas:

```bash
cd pump_risk
../venv/bin/python main.py --walk-forward \
  --symbols XRP/USDT \
  --label-mode intraday \
  --signal-level high \
  --train-days 180 \
  --test-days 60 \
  --step-days 30 \
  --show-dates
```

Walk-forward con trigger intradía (doble confirmación de entrada):

```bash
cd pump_risk
../venv/bin/python main.py --walk-forward \
  --symbols XRP/USDT \
  --label-mode either \
  --signal-level medium \
  --train-days 180 \
  --test-days 60 \
  --step-days 30 \
  --use-intraday-trigger \
  --trigger-timeframe 1h \
  --trigger-breakout-buffer-pct 0.001 \
  --trigger-min-rel-volume 1.10 \
  --trigger-min-close-pos 0.60 \
  --trigger-vol-lookback-bars 12 \
  --show-dates
```

## Notes

- `HIGH` risk means "higher probability than baseline", not certainty.
- Label modes:
  - `close`: next-day close->close return `>= +3%`
  - `intraday`: next-day open->high move `>= +3%`
  - `either`: `close OR intraday`
  - `both`: `close AND intraday`
- En `--scan`, el resumen usa métricas activas `Prec(A)` / `Rec(A)` del modo elegido
  (`early|confirm|both`) con el `label_mode` seleccionado.
- Optimization applied:
  - For `--label-mode either --signal-level medium`, signal uses
    `>=2` of `{funding_neg, atr_regime, break_prev_high}` to reduce false positives.
- `--scan-mode`:
  - `confirm` (default): requiere gate base + trigger intradía
  - `early`: solo gate base diario (más temprano, más ruido)
  - `both`: evalúa ambos y emite una sola señal con `stage=EARLY|CONFIRM|BOTH`
- `--scan` emite señal `LONG` en JSON cuando el modo elegido pasa.
  El JSON en `--signals-json` queda listo para consumo por `monitor.py`.
- Intraday trigger (when enabled):
  - `close` rompe máximo del día previo
  - volumen relativo mínimo (`rel_vol`)
  - cierre fuerte en la vela (posición de cierre en rango y vela verde)
