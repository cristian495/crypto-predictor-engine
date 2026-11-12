"""
live_signals.py
Paper-trading signal logger + Discord alerts for Buy-the-Dip.

Corre el scan sobre la ultima vela 1H de cada simbolo de la cesta, detecta
señales nuevas (LONG) y:
  1. Las AGREGA (append) a un log inmutable live_signals.jsonl (una linea por señal).
  2. Alerta por Discord (webhook) solo si la señal es NUEVA (dedupe por simbolo+vela).

NO ejecuta ordenes: es paper trading. Pensado para correr cada hora (cron en VPS).

Uso:
    python live_signals.py                 # scan + log + alerta (si hay webhook)
    python live_signals.py --dry-run       # no escribe log ni alerta; solo imprime
    python live_signals.py --no-alert      # escribe log pero no alerta
    python live_signals.py --log-path X    # ruta del jsonl (default: exports/buy_the_dip/live_signals.jsonl)

Config (env):
    DISCORD_WEBHOOK_URL   # webhook de Discord. Si falta, no alerta (avisa).
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

from config import TIMEFRAME, DEFAULT_SYMBOLS, get_symbol_signal_params
from breakout_momentum.data_loader import fetch_ohlcv
from strategy import (
    add_dip_features, generate_signals, compute_exit_levels,
    load_regime_if_enabled, apply_regime_filter,
)
from main import _evaluate_buy_filters

# Suficientes velas para calcular features (rolling 24h + RSI 14 + vol 20) con margen.
LOOKBACK_DAYS = 10


def _detect_signal(symbol: str, regime=None) -> dict | None:
    """
    Evalua la ULTIMA vela de un simbolo. Devuelve un dict de señal si es LONG,
    o un dict con signal=NO_SIGNAL y el motivo de bloqueo (para registro).
    Aplica el filtro de regimen si `regime` viene dado.
    """
    signal_params = get_symbol_signal_params(symbol)

    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, days=LOOKBACK_DAYS)
    df = add_dip_features(df)
    df = generate_signals(
        df,
        dip_threshold=signal_params["dip_threshold"],
        use_volume_filter=signal_params["use_volume_filter"],
        use_rsi_filter=signal_params["use_rsi_filter"],
        min_volume_mult=signal_params["min_volume_mult"],
        rsi_min=signal_params["rsi_min"],
        rsi_max=signal_params["rsi_max"],
        one_entry_per_dip_event=signal_params["one_entry_per_dip_event"],
        dip_event_reset_pct=signal_params["dip_event_reset_pct"],
    )
    df = apply_regime_filter(df, regime)

    last = df.iloc[-1]
    is_long = last.get("signal", "NO TRADE") == "LONG"
    checks, blocking = _evaluate_buy_filters(last, signal_params)

    close = float(last.get("close", 0.0))
    tp_pct = float(signal_params["take_profit_pct"])
    sl_pct = float(signal_params["stop_loss_pct"])
    levels = compute_exit_levels(close, tp_pct, sl_pct)

    bar_ts = last.get("timestamp")
    bar_iso = bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts)

    return {
        "symbol": symbol,
        "signal": "LONG" if is_long else "NO_SIGNAL",
        "bar_timestamp": bar_iso,              # identifica la vela (clave de dedupe)
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "close": close,
        "past_return_pct": round(float(last.get("past_return", 0.0)) * 100, 2),
        "dip_threshold_pct": round(float(signal_params["dip_threshold"]) * 100, 2),
        "rsi_14": None if last.get("rsi_14") is None else round(float(last.get("rsi_14", 0.0)), 1),
        "entry_price": close if is_long else None,
        "take_profit": round(levels["take_profit"], 6) if is_long else None,
        "stop_loss": round(levels["stop_loss"], 6) if is_long else None,
        "take_profit_pct": round(tp_pct * 100, 2),
        "stop_loss_pct": round(sl_pct * 100, 2),
        "hold_timeout_hours": int(signal_params["hold_timeout_hours"]),
        "leverage": float(signal_params.get("leverage", 1.0) or 1.0),
        "blocked_by": None if is_long else (blocking if blocking != "none" else None),
        "conditions": [{"name": n, "passed": bool(ok), "detail": d} for n, ok, d in checks],
    }


def _already_logged(log_path: Path, symbol: str, bar_ts: str) -> bool:
    """True si ya existe una señal LONG registrada para este simbolo+vela (dedupe)."""
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
                    and rec.get("signal") == "LONG"
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
    """Envia un mensaje crudo a Discord. Devuelve True si la entrega fue OK."""
    payload = json.dumps({"content": content}).encode("utf-8")
    # Discord rechaza (403) el User-Agent por defecto de urllib ("Python-urllib/x").
    # Hay que enviar uno explicito, como hacen curl/navegadores.
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "buy-the-dip-live-signals/1.0 (+https://github.com)",
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
    """Envia una alerta a Discord. Devuelve True si la entrega fue OK."""
    s = signal
    lev = s.get("leverage", 1.0)
    content = (
        f"🔻 **BUY THE DIP — {s['symbol']}**\n"
        f"Caida 24h: **{s['past_return_pct']:+.2f}%** (umbral {s['dip_threshold_pct']:.1f}%)\n"
        f"Entrada: **${s['entry_price']:,.6g}**  ·  apalancamiento x{lev:g}\n"
        f"🎯 TP: ${s['take_profit']:,.6g} (+{s['take_profit_pct']:.1f}%)\n"
        f"🛑 SL: ${s['stop_loss']:,.6g} (-{s['stop_loss_pct']:.1f}%)\n"
        f"⏱️ Timeout: {s['hold_timeout_hours']}h"
        + (f"  ·  RSI: {s['rsi_14']}" if s.get("rsi_14") is not None else "")
        + "\n_paper trading — no se ejecuta ninguna orden_"
    )
    return _send_discord_raw(webhook_url, content)


def run(symbols: list[str], log_path: Path, dry_run: bool, no_alert: bool) -> int:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    alert_enabled = bool(webhook) and not no_alert and not dry_run

    if not webhook and not no_alert and not dry_run:
        print("  (aviso) DISCORD_WEBHOOK_URL no definido: no se enviaran alertas.")

    print(f"{'=' * 64}")
    print("  BUY-THE-DIP — LIVE SIGNALS (paper trading)")
    print(f"{'=' * 64}")
    print(f"  Symbols: {len(symbols)}  |  log: {log_path}")
    print(f"  Mode:    {'DRY-RUN' if dry_run else 'LIVE'}  |  alerts: {'on' if alert_enabled else 'off'}")

    # Regimen de mercado (BTC vs SMA): una sola descarga para todos los simbolos.
    regime = load_regime_if_enabled(fetch_ohlcv, LOOKBACK_DAYS)
    if len(regime):
        bullish_now = bool(regime.iloc[-1]) if len(regime) else True
        print(f"  Regimen: {'ALCISTA/lateral (opera)' if bullish_now else 'BAJISTA (bloquea LONG)'}")
    print(f"{'=' * 64}")

    n_long = 0
    n_new = 0
    for symbol in symbols:
        try:
            sig = _detect_signal(symbol, regime=regime)
        except Exception as e:
            print(f"  {symbol:<10} ERROR: {e}")
            continue

        if sig["signal"] != "LONG":
            blk = sig.get("blocked_by") or "conditions not met"
            print(f"  {symbol:<10} no signal ({blk})")
            continue

        n_long += 1
        is_new = not _already_logged(log_path, symbol, sig["bar_timestamp"])
        tag = "NEW" if is_new else "dup"
        print(
            f"  {symbol:<10} 🔻 LONG [{tag}] drop={sig['past_return_pct']:+.2f}% "
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
    print(f"  LONG signals: {n_long}  |  new logged: {n_new}")
    print(f"{'=' * 64}")
    return n_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live signal logger + Discord alerts (paper)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols (default: config cesta)")
    parser.add_argument("--dry-run", action="store_true", help="No escribe log ni alerta; solo imprime")
    parser.add_argument("--no-alert", action="store_true", help="Escribe log pero no alerta Discord")
    parser.add_argument("--test-alert", action="store_true", help="Envia un mensaje de prueba a Discord y sale")
    parser.add_argument(
        "--log-path",
        default=str(Path(__file__).resolve().parents[1] / "exports" / "buy_the_dip" / "live_signals.jsonl"),
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
            f"✅ **Prueba de alerta — Buy-the-Dip**\n"
            f"Tu webhook de Discord funciona. Enviado {ts}.\n"
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
