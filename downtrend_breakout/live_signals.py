"""
live_signals.py
Paper-trading signal logger + Discord alerts for downtrend_breakout (SHORT).

Complementaria a buy_the_dip: esta opera en tendencia BAJISTA (BTC bajo su regimen),
justo cuando buy_the_dip se apaga. Escanea la ultima vela 1H de cada simbolo, detecta
señales SHORT y:
  1. Las AGREGA (append) a live_signals.jsonl (una linea por señal, dedupe por simbolo+vela).
  2. Alerta por Discord si la señal es NUEVA.

NO ejecuta ordenes: paper trading. Pensado para correr cada hora (cron del VPS via worker).

Uso:
    python live_signals.py                 # scan + log + alerta
    python live_signals.py --dry-run       # no escribe ni alerta; solo imprime
    python live_signals.py --no-alert      # escribe log pero no alerta
    python live_signals.py --test-alert    # mensaje de prueba a Discord
    python live_signals.py --log-path X

Config (env):
    DISCORD_WEBHOOK_URL    # webhook de Discord. Si falta, no alerta.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TIMEFRAME, DEFAULT_SYMBOLS, MODE_DEFAULT
from main import _resolve_symbol_params, _evaluate_downtrend_conditions
from breakout_momentum.data_loader import fetch_ohlcv
from strategy import add_features, generate_signals
from backtest import compute_exit_levels

# Downtrend necesita historia para EMA200 + features; margen amplio.
LOOKBACK_DAYS = 90


def _fetch_btc(days: int):
    try:
        return fetch_ohlcv("BTC/USDT", timeframe=TIMEFRAME, days=days)
    except Exception:
        return None


def _detect_signal(symbol: str, btc_df=None) -> dict | None:
    """
    Evalua la ULTIMA vela de un simbolo. Devuelve dict de señal SHORT si la hay,
    o signal=NO_SIGNAL con el motivo de bloqueo (para registro).
    """
    params = _resolve_symbol_params(symbol)

    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=LOOKBACK_DAYS)
    df = add_features(df, btc_df=btc_df)
    df = generate_signals(df, symbol_params=params, mode=MODE_DEFAULT)

    last = df.iloc[-1]
    signal = last.get("signal", "NO TRADE")
    is_short = signal == "SHORT"
    checks, blocking = _evaluate_downtrend_conditions(df, params, MODE_DEFAULT)

    close = float(last.get("close", 0.0))
    atr = float(last.get("atr14", 0.0))
    direction = -1  # short_only
    levels = compute_exit_levels(
        close, direction, atr,
        stop_loss_atr_mult=float(params.get("stop_loss_atr_mult", 1.5)),
        take_profit_r=float(params.get("take_profit_r_mult", 1.6)),
    )

    bar_ts = last.get("timestamp")
    bar_iso = bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts)

    return {
        "symbol": symbol,
        "signal": "SHORT" if is_short else "NO_SIGNAL",
        "direction": -1,
        "bar_timestamp": bar_iso,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "close": close,
        "ret_24h_pct": round(float(last.get("ret_24h", 0.0)) * 100, 2),
        "adx14": round(float(last.get("adx14", 0.0)), 1),
        "atr14": round(atr, 6),
        "entry_price": close if is_short else None,
        "take_profit": round(levels["take_profit"], 6) if is_short else None,
        "stop_loss": round(levels["stop_loss"], 6) if is_short else None,
        "take_profit_r": float(params.get("take_profit_r_mult", 1.6)),
        "stop_loss_atr_mult": float(params.get("stop_loss_atr_mult", 1.5)),
        "hold_timeout_bars": int(params.get("hold_timeout_bars", 48)),
        "probability": round(float(last.get("probability", 0.0)), 3),
        "entry_reason": last.get("entry_reason", "") if is_short else None,
        "blocked_by": None if is_short else (blocking if blocking != "none" else None),
        "conditions": [{"name": n, "passed": bool(ok), "detail": d} for n, ok, d in checks],
    }


def _already_logged(log_path: Path, symbol: str, bar_ts: str) -> bool:
    if not log_path.exists():
        return False
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if (
                    rec.get("symbol") == symbol
                    and rec.get("bar_timestamp") == bar_ts
                    and rec.get("signal") == "SHORT"
                ):
                    return True
    except Exception:
        return False
    return False


def _append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _send_discord_raw(webhook_url: str, content: str) -> bool:
    payload = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "downtrend-breakout-live-signals/1.0 (+https://github.com)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        print(f"  Discord HTTPError {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"  Discord error: {e}")
        return False


def _send_discord(webhook_url: str, signal: dict) -> bool:
    s = signal
    content = (
        f"\U0001F4C9 **SELL / SHORT — {s['symbol']}**  _(tendencia bajista)_\n"
        f"ret 24h: **{s['ret_24h_pct']:+.2f}%**  ADX: {s['adx14']}  motivo: {s.get('entry_reason') or 'breakdown'}\n"
        f"Entrada (short): **${s['entry_price']:,.6g}**\n"
        f"\U0001F3AF TP: ${s['take_profit']:,.6g} ({s['take_profit_r']:.1f}R)\n"
        f"\U0001F6D1 SL: ${s['stop_loss']:,.6g} ({s['stop_loss_atr_mult']:.1f}x ATR)\n"
        f"⏱️ Timeout: {s['hold_timeout_bars']} velas\n"
        f"_paper trading — no se ejecuta ninguna orden_"
    )
    return _send_discord_raw(webhook_url, content)


def run(symbols: list[str], log_path: Path, dry_run: bool, no_alert: bool) -> int:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    alert_enabled = bool(webhook) and not no_alert and not dry_run

    if not webhook and not no_alert and not dry_run:
        print("  (aviso) DISCORD_WEBHOOK_URL no definido: no se enviaran alertas.")

    print(f"{'=' * 64}")
    print("  DOWNTREND BREAKOUT — LIVE SIGNALS (paper trading, SHORT)")
    print(f"{'=' * 64}")
    print(f"  Symbols: {len(symbols)}  |  log: {log_path}")
    print(f"  Mode:    {'DRY-RUN' if dry_run else 'LIVE'}  |  alerts: {'on' if alert_enabled else 'off'}")

    btc_df = _fetch_btc(LOOKBACK_DAYS)
    print(f"  BTC context: {'cargado' if btc_df is not None else 'NO disponible (filtro btc degradara)'}")
    print(f"{'=' * 64}")

    n_short = 0
    n_new = 0
    for symbol in symbols:
        try:
            sig = _detect_signal(symbol, btc_df=btc_df)
        except Exception as e:
            print(f"  {symbol:<10} ERROR: {e}")
            continue

        if sig["signal"] != "SHORT":
            blk = sig.get("blocked_by") or "conditions not met"
            print(f"  {symbol:<10} no signal ({blk})")
            continue

        n_short += 1
        is_new = not _already_logged(log_path, symbol, sig["bar_timestamp"])
        tag = "NEW" if is_new else "dup"
        print(
            f"  {symbol:<10} \U0001F4C9 SHORT [{tag}] ret24={sig['ret_24h_pct']:+.2f}% "
            f"entry=${sig['entry_price']:.6g} TP=${sig['take_profit']:.6g}"
        )

        if dry_run:
            continue

        if is_new:
            n_new += 1
            _append_log(log_path, sig)
            if alert_enabled:
                ok = _send_discord(webhook, sig)
                print(f"             discord: {'sent' if ok else 'FAILED'}")

    print(f"{'=' * 64}")
    print(f"  SHORT signals: {n_short}  |  new logged: {n_new}")
    print(f"{'=' * 64}")
    return n_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live signal logger + Discord alerts (downtrend, paper)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols (default: config cesta)")
    parser.add_argument("--dry-run", action="store_true", help="No escribe log ni alerta; solo imprime")
    parser.add_argument("--no-alert", action="store_true", help="Escribe log pero no alerta Discord")
    parser.add_argument("--test-alert", action="store_true", help="Envia un mensaje de prueba a Discord y sale")
    parser.add_argument(
        "--log-path",
        default=str(Path(__file__).resolve().parents[1] / "exports" / "downtrend_breakout" / "live_signals.jsonl"),
        help="Ruta del log JSONL acumulativo",
    )
    args = parser.parse_args()

    if args.test_alert:
        webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
        if not webhook:
            print("  ERROR: DISCORD_WEBHOOK_URL no definido. Pon el webhook en .env y vuelve a probar.")
            sys.exit(1)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ok = _send_discord_raw(
            webhook,
            f"✅ **Prueba de alerta — Downtrend Breakout (SHORT)**\n"
            f"Tu webhook funciona. Enviado {ts}.\n"
            f"_paper trading — no se ejecuta ninguna orden_",
        )
        print("  discord test:", "sent ✅" if ok else "FAILED ❌")
        sys.exit(0 if ok else 1)

    run(
        symbols=args.symbols or DEFAULT_SYMBOLS,
        log_path=Path(args.log_path),
        dry_run=args.dry_run,
        no_alert=args.no_alert,
    )
