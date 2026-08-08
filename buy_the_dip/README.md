# Buy-the-Dip Strategy

## Overview

Simple, rule-based strategy that buys after significant price drops.
**Validated with walk-forward testing** - no overfitting.

## Edge (Validated)

| Symbol | Stability Score | Backtest Return | Win Rate |
|--------|----------------|-----------------|----------|
| BNB/USDT | **76.4/100** (EXCELLENT) | +25.5% | 70% |
| DOGE/USDT | **62.7/100** (GOOD) | +7.6% | 69% |

## How It Works

1. **Entry**: Buy when price drops >7% in 24 hours
2. **Exit**: Time-based (24 hours) OR take profit at +4%
3. **Stop Loss**: 20% (catastrophic protection only)

## Why It Works

After a 7%+ drop in crypto:
- Average bounce: +1.5-2% in next 24h
- Win rate: 60-70%
- Intra-period drawdown averages -6%, so tight SL doesn't work

## Files

- `config.py` - Configuration parameters
- `strategy.py` - Signal generation
- `backtest.py` - Backtest engine
- `run_backtest.py` - Run backtests
- `walk_forward.py` - Walk-forward validation
- `main.py` - Scanner CLI + walk-forward entrypoint

## Usage

### Run Backtest
```bash
cd buy_the_dip
python run_backtest.py
```

### Validate Strategy
```bash
python walk_forward.py
```

### Scan for Live Signals
```bash
python main.py --scan
```

### Setup Environment (First Time)
```bash
# From predictor-engine root
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Activate Existing Environment
```bash
# From predictor-engine root
source venv/bin/activate
```

### Export Data for Frontend
```bash
# From predictor-engine root
python buy_the_dip/export_frontend_data.py \
  --include-walkforward \
  --out-dir exports/buy_the_dip
```

Generates fixed artifacts:
- `exports/buy_the_dip/backtest.latest.json`
- `exports/buy_the_dip/signals.latest.json`
- `exports/buy_the_dip/walkforward.latest.json` (optional if `--include-walkforward`)

If you only want backtest + signals:
```bash
python buy_the_dip/export_frontend_data.py --out-dir exports/buy_the_dip
```

## Key Parameters

- `DIP_THRESHOLD_STRONG = -0.07` (7% drop required)
- `TAKE_PROFIT_PCT = 0.04` (4% take profit)
- `HOLD_TIMEOUT_HOURS = 24` (exit after 24h)
- `POSITION_PCT = 0.20` (20% per trade)

## Per-Symbol TP Override

You can now set `take_profit_pct` inside `SYMBOL_SIGNAL_OVERRIDES` in `config.py`:

```python
"XRP/USDT": {
    "dip_threshold": -0.055,
    "use_rsi_filter": False,
    "use_volume_filter": True,
    "hold_timeout_hours": 27,
    "max_positions": 4,
    "take_profit_pct": 0.045,
},
```

If `take_profit_pct` is omitted, the strategy uses global `TAKE_PROFIT_PCT`.

## Walk-Forward Results (BNB)

- 22 out-of-sample windows tested
- **81.8% had positive returns**
- **81.8% had positive Sharpe**
- Average return per window: +1.21%
- Average win rate: 70.5%

## Important Notes

1. **Time-based exit is critical** - Don't use tight stop losses
2. **-7% threshold is optimal** - Looser thresholds reduce edge
3. **BNB > DOGE > ETH** - Not all symbols have the same edge
4. **No ML, no overfitting** - Pure statistical edge
