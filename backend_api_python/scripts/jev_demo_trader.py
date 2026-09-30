#!/usr/bin/env python3
"""
JEV demo trader: TypeSafe Jev decides the position, a Bybit DEMO account executes it.

Once per closed candle the script sends a compact market snapshot to Jev and asks one
typed question: which position should be held (long, flat or short). Everything that
protects the account stays in code and cannot be overridden by Jev:

  - demo account only: the Bybit client is built for the demo environment and checked;
  - position size is a fixed share of equity, leverage is capped at 3x;
  - every position has an exchange-side stop-loss (if it cannot be set, the position is closed);
  - minimum Jev confidence, an opening cool-down and a daily cap on new positions;
  - any Jev failure means "do nothing" (an open position keeps its stop-loss).

The default is a dry run: decisions are logged, no orders are sent. Add --execute to trade
on the demo account. Every cycle is appended to a JSON-lines log for later evaluation.

Use a dedicated demo account and symbol. This script runs outside QuantDinger, so do not run
a QuantDinger strategy on the same symbol at the same time. The account must be in Bybit
one-way position mode.

Environment:
  JEV_API_KEY, JEV_BASE_URL, JEV_MODEL, JEV_TIMEOUT_SECONDS   same names as System Settings
  BYBIT_DEMO_API_KEY, BYBIT_DEMO_API_SECRET                   keys created in Bybit Demo Trading

Examples:
  python scripts/jev_demo_trader.py --once                     # one dry-run decision
  python scripts/jev_demo_trader.py --execute --interval 60    # trade BTC/USDT on demo, 1h candles
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO_BASE_URL = "https://api-demo.bybit.com"
INTERVALS_MIN = (15, 30, 60, 120, 240)
HIGHER_INTERVAL_MIN = {15: 60, 30: 120, 60: 240, 120: 720, 240: 1440}
CANDLES_PER_FRAME = 120
MIN_CANDLES = 55
MAX_LEVERAGE = 3
MAX_CONSECUTIVE_ERRORS = 5

CRITERIA = {
    "long": "Trend, momentum and volatility across the supplied timeframes support holding a long position.",
    "flat": "Evidence is mixed, weak or stale, or the risk is unclear; hold no position.",
    "short": "Trend, momentum and volatility across the supplied timeframes support holding a short position.",
}


class JevError(Exception):
    """Jev returned an unusable answer."""


@dataclass(frozen=True)
class Config:
    symbol: str = "BTC/USDT"
    interval_min: int = 60
    position_pct: float = 0.10
    leverage: int = 1
    stop_loss_pct: float = 0.02
    min_confidence: float = 0.60
    allow_short: bool = False
    cooldown_bars: int = 2
    max_opens_per_day: int = 6
    execute: bool = False

    @property
    def options(self) -> Tuple[str, ...]:
        return ("long", "flat", "short") if self.allow_short else ("long", "flat")

    def validate(self) -> "Config":
        if self.interval_min not in INTERVALS_MIN:
            raise ValueError(f"interval must be one of {INTERVALS_MIN} minutes")
        if not 0 < self.position_pct <= 0.5:
            raise ValueError("position-pct must be in (0, 0.5]")
        if not 1 <= self.leverage <= MAX_LEVERAGE:
            raise ValueError(f"leverage must be between 1 and {MAX_LEVERAGE}")
        if not 0.002 <= self.stop_loss_pct <= 0.2:
            raise ValueError("stop-loss-pct must be between 0.002 and 0.2")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min-confidence must be between 0 and 1")
        if self.cooldown_bars < 0 or self.max_opens_per_day < 1:
            raise ValueError("cooldown-bars must be >= 0 and max-opens-per-day >= 1")
        return self


@dataclass(frozen=True)
class Candle:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Position:
    side: str = "flat"  # long | short | flat
    size: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    position_idx: int = 0


@dataclass(frozen=True)
class Decision:
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: int = 0


@dataclass(frozen=True)
class Plan:
    action: str  # hold | long | flat | short
    reason: str


# --------------------------------------------------------------------------- market snapshot


def interval_label(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}m"
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    return f"{minutes // 60}h"


def bybit_interval(minutes: int) -> str:
    return "D" if minutes == 1440 else str(minutes)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_klines(rows: Sequence[Sequence[Any]], *, interval_min: int, now_ms: int) -> List[Candle]:
    """Bybit lists candles newest first as [start, open, high, low, close, volume, turnover].

    Returns closed candles only, oldest first.
    """
    span = interval_min * 60_000
    candles: List[Candle] = []
    for row in rows or []:
        try:
            start = int(row[0])
            o, h, low, c, v = (float(x) for x in row[1:6])
        except (TypeError, ValueError, IndexError):
            continue
        if start + span > now_ms:
            continue
        candles.append(Candle(start, o, h, low, c, v))
    candles.sort(key=lambda candle: candle.ts_ms)
    return candles


def _ema(values: Sequence[float], period: int) -> Optional[float]:
    if period <= 0 or len(values) < period:
        return None
    k = 2.0 / (period + 1)
    value = sum(values[:period]) / period
    for price in values[period:]:
        value = price * k + value * (1.0 - k)
    return value


def _rsi(closes: Sequence[float], period: int = 14) -> Optional[float]:
    if len(closes) <= period:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains += max(diff, 0.0)
        losses += max(-diff, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(diff, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-diff, 0.0)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _atr(candles: Sequence[Candle], period: int = 14) -> Optional[float]:
    if len(candles) <= period:
        return None
    ranges = [
        max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))
        for prev, cur in zip(candles, candles[1:])
    ]
    atr = sum(ranges[:period]) / period
    for value in ranges[period:]:
        atr = (atr * (period - 1) + value) / period
    return atr


def _pct(value: Optional[float], base: Optional[float]) -> Optional[float]:
    if value is None or not base:
        return None
    return (value / base - 1.0) * 100.0


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(value, digits)


def frame_features(candles: Sequence[Candle]) -> Dict[str, Any]:
    """Compact, point-in-time features of one timeframe (closed candles only)."""
    if len(candles) < MIN_CANDLES:
        raise ValueError(f"not_enough_candles:{len(candles)}<{MIN_CANDLES}")
    closes = [c.close for c in candles]
    last = closes[-1]
    window = candles[-24:]
    volume_avg = sum(c.volume for c in window) / len(window)
    atr = _atr(candles)
    return {
        "last_close": last,
        "last_candle_close_time": _iso(candles[-1].ts_ms),
        "chg_1_bar_pct": _round(_pct(last, closes[-2])),
        "chg_6_bars_pct": _round(_pct(last, closes[-7])),
        "chg_24_bars_pct": _round(_pct(last, closes[-25])),
        "ema20_vs_ema50_pct": _round(_pct(_ema(closes, 20), _ema(closes, 50))),
        "close_vs_ema50_pct": _round(_pct(last, _ema(closes, 50))),
        "rsi14": _round(_rsi(closes), 1),
        "atr14_pct": _round(atr / last * 100.0 if atr is not None else None),
        "vs_24_bar_high_pct": _round(_pct(last, max(c.high for c in window))),
        "vs_24_bar_low_pct": _round(_pct(last, min(c.low for c in window))),
        "volume_vs_24_bar_avg": _round(candles[-1].volume / volume_avg if volume_avg else None, 2),
    }


def build_state(
    cfg: Config,
    frames: Mapping[str, Mapping[str, Any]],
    position: Position,
    equity: float,
    now_ms: int,
) -> Dict[str, Any]:
    main_close = float(frames[interval_label(cfg.interval_min)]["last_close"])
    pnl_pct = None
    if position.side != "flat" and position.entry_price > 0:
        direction = 1.0 if position.side == "long" else -1.0
        pnl_pct = round((main_close / position.entry_price - 1.0) * direction * 100.0, 3)
    return {
        "symbol": cfg.symbol,
        "market_type": "swap",
        "as_of": _iso(now_ms),
        "timeframes": dict(frames),
        "position": {
            "side": position.side,
            "size": position.size,
            "entry_price": position.entry_price or None,
            "unrealized_pnl_pct": pnl_pct,
        },
        "account": {"equity_usdt": round(equity, 2)},
        "limits": {
            "position_pct_of_equity": cfg.position_pct,
            "leverage": cfg.leverage,
            "exchange_stop_loss_pct": round(cfg.stop_loss_pct * 100.0, 3),
            "shorting_allowed": cfg.allow_short,
        },
    }


# --------------------------------------------------------------------------- Jev


def build_questions(options: Sequence[str]) -> Dict[str, Any]:
    return {
        "position": {
            "type": "choice",
            "instructions": (
                "Using only the supplied point-in-time market evidence and the current position, choose the "
                "position to hold over the next few candles. Choose flat when the evidence is mixed, weak or "
                "the risk is unclear; a flat answer is never a failure."
            ),
            "criteria": {name: CRITERIA[name] for name in options},
        }
    }


def parse_jev_answer(
    payload: Any, options: Sequence[str], question: str = "position"
) -> Tuple[str, Dict[str, float], float]:
    """Validate a Jev choice answer with the same rules as the platform's AI decision filter."""
    if not isinstance(payload, dict):
        raise JevError("response_not_an_object")
    answers = payload.get("answers") or payload.get("result") or payload.get("data") or {}
    answer = answers.get(question) if isinstance(answers, dict) else None
    if not isinstance(answer, dict) and isinstance(answers, dict):
        nested = answers.get("answers")
        answer = nested.get(question) if isinstance(nested, dict) else None
    if not isinstance(answer, dict):
        raise JevError(f"{question}_answer_missing")
    allowed = set(options)
    choice = str(answer.get("choice") or answer.get("selected") or "").strip().lower()
    if choice not in allowed:
        raise JevError(f"invalid_choice:{choice or 'empty'}")
    raw = answer.get("probabilities") or answer.get("probability") or {}
    if not isinstance(raw, dict) or set(raw) != allowed:
        raise JevError("probabilities_incomplete")
    try:
        probabilities = {str(k): float(v) for k, v in raw.items()}
        confidence = float(answer.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise JevError("probability_or_confidence_invalid") from exc
    if any(p < 0 or p > 1 for p in probabilities.values()) or not 0 <= confidence <= 1:
        raise JevError("value_out_of_range")
    if abs(sum(probabilities.values()) - 1.0) > 0.001:
        raise JevError("probabilities_do_not_sum_to_one")
    if probabilities[choice] != max(probabilities.values()):
        raise JevError("choice_not_highest_probability")
    return choice, probabilities, confidence


def ask_jev(
    state: Mapping[str, Any],
    *,
    options: Sequence[str],
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    post: Callable[..., Any] = requests.post,
    questions: Optional[Mapping[str, Any]] = None,
    question: str = "position",
) -> Decision:
    url = base_url.strip().rstrip("/")
    if not url.endswith("/systemone"):
        url += "/systemone"
    started = time.perf_counter()
    response = post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "state": state, "questions": questions or build_questions(options)},
        timeout=max(1.0, min(float(timeout), 30.0)),
    )
    response.raise_for_status()
    choice, probabilities, confidence = parse_jev_answer(response.json(), options, question)
    return Decision(choice, confidence, probabilities, int((time.perf_counter() - started) * 1000))


def jev_settings(env: Mapping[str, str]) -> Dict[str, Any]:
    api_key = str(env.get("JEV_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("JEV_API_KEY is not set (get a key at https://console.typesafe.ai/)")
    return {
        "api_key": api_key,
        "base_url": str(env.get("JEV_BASE_URL") or "https://api.typesafe.ai/v1").strip(),
        "model": str(env.get("JEV_MODEL") or "jev-latest").strip(),
        "timeout": float(env.get("JEV_TIMEOUT_SECONDS") or 8),
    }


# --------------------------------------------------------------------------- decision rules


def plan_position(
    decision: Optional[Decision],
    *,
    current_side: str,
    allow_short: bool,
    min_confidence: float,
) -> Plan:
    if decision is None:
        return Plan("hold", "jev_unavailable")
    if decision.confidence < min_confidence:
        return Plan("hold", "low_confidence")
    target = decision.choice
    if target == "short" and not allow_short:
        target = "flat"
    if target == current_side:
        return Plan("hold", f"already_{target}")
    return Plan(target, f"jev_{decision.choice}")


def position_qty(equity: float, price: float, *, position_pct: float, leverage: float) -> float:
    if equity <= 0 or price <= 0:
        return 0.0
    return equity * position_pct * leverage / price


def protective_stop_price(entry_price: float, side: str, stop_loss_pct: float) -> float:
    return entry_price * (1.0 - stop_loss_pct) if side == "long" else entry_price * (1.0 + stop_loss_pct)


# --------------------------------------------------------------------------- Bybit demo broker


class BybitDemoBroker:
    """Thin wrapper over the platform's BybitClient that refuses anything but the demo host."""

    def __init__(self, client: Any, to_symbol: Optional[Callable[[str], str]] = None):
        base_url = str(getattr(client, "base_url", "") or "").rstrip("/")
        if base_url != DEMO_BASE_URL:
            raise RuntimeError(f"refusing to trade: client is not on the Bybit demo host ({base_url!r})")
        if to_symbol is None:
            from app.services.live_trading.symbols import to_bybit_symbol as to_symbol
        self.client = client
        self._symbol = to_symbol

    def get_klines(self, symbol: str, minutes: int, limit: int) -> List[Sequence[Any]]:
        raw = self.client._public_request(
            "GET",
            "/v5/market/kline",
            params={
                "category": "linear",
                "symbol": self._symbol(symbol),
                "interval": bybit_interval(minutes),
                "limit": limit,
            },
        )
        return ((raw.get("result") or {}).get("list")) or []

    def get_equity(self) -> float:
        rows = ((self.client.get_wallet_balance().get("result") or {}).get("list")) or []
        for row in rows:
            try:
                equity = float(row.get("totalEquity") or 0)
            except (TypeError, ValueError, AttributeError):
                continue
            if equity > 0:
                return equity
        raise RuntimeError("could not read demo account equity")

    def last_price(self, symbol: str) -> float:
        price = float((self.client.get_ticker(symbol=symbol) or {}).get("last") or 0)
        if price <= 0:
            raise RuntimeError(f"could not read price for {symbol}")
        return price

    def get_position(self, symbol: str) -> Position:
        rows = ((self.client.get_positions(symbol=symbol).get("result") or {}).get("list")) or []
        if any(int(row.get("positionIdx") or 0) != 0 for row in rows):
            raise RuntimeError("Bybit hedge mode is not supported: switch the demo account to one-way mode")
        for row in rows:
            size = float(row.get("size") or 0)
            if size <= 0:
                continue
            return Position(
                side="long" if str(row.get("side")).lower() == "buy" else "short",
                size=size,
                entry_price=float(row.get("avgPrice") or 0),
                stop_loss=float(row.get("stopLoss") or 0),
                position_idx=0,
            )
        return Position()

    def set_leverage(self, symbol: str, leverage: int) -> None:
        if not self.client.set_leverage(symbol=symbol, leverage=leverage):
            raise RuntimeError(f"could not set leverage {leverage}x for {symbol}")

    def market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> None:
        self.client.place_market_order(
            symbol=symbol,
            side=side,
            qty=qty,
            reduce_only=reduce_only,
            client_order_id=f"jev{int(time.time() * 1000)}",
        )

    def set_stop_loss(self, symbol: str, position: Position, stop_price: float) -> None:
        price, precision = self.client._normalize_price(symbol=symbol, price=stop_price)
        if price <= 0:
            raise RuntimeError("invalid stop price")
        self.client._signed_request(
            "POST",
            "/v5/position/trading-stop",
            json_body={
                "category": "linear",
                "symbol": self._symbol(symbol),
                "positionIdx": position.position_idx,
                "tpslMode": "Full",
                "stopLoss": self.client._dec_str(price, strict_precision=precision),
                "slTriggerBy": "MarkPrice",
            },
        )


def make_broker(env: Mapping[str, str]) -> BybitDemoBroker:
    api_key = str(env.get("BYBIT_DEMO_API_KEY") or "").strip()
    secret = str(env.get("BYBIT_DEMO_API_SECRET") or "").strip()
    if not api_key or not secret:
        raise SystemExit("Set BYBIT_DEMO_API_KEY and BYBIT_DEMO_API_SECRET (keys from Bybit Demo Trading)")
    from app.services.live_trading.factory import create_client

    client = create_client(
        {
            "exchange_id": "bybit",
            "api_key": api_key,
            "secret_key": secret,
            "environment": "demo",
            "market_scope": "swap",
        },
        market_type="swap",
    )
    return BybitDemoBroker(client)


# --------------------------------------------------------------------------- trader


class Trader:
    def __init__(
        self,
        broker: Any,
        cfg: Config,
        ask: Callable[[Mapping[str, Any]], Decision],
        *,
        log_path: Optional[Path] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.broker = broker
        self.cfg = cfg
        self.ask = ask
        self.log_path = log_path
        self.clock = clock
        self.sleep = sleep
        self.last_open_ms = 0
        self.opens_today = 0
        self.day = ""
        self._leverage_applied = False

    def run_cycle(self) -> Dict[str, Any]:
        cfg = self.cfg
        now_ms = int(self.clock() * 1000)
        position = self.broker.get_position(cfg.symbol)
        actions: List[str] = []
        if cfg.execute and self._ensure_stop(position):
            actions.append("stop_loss_restored")
            position = self.broker.get_position(cfg.symbol)
        equity = self.broker.get_equity()
        frames: Dict[str, Dict[str, Any]] = {}
        for minutes in (cfg.interval_min, HIGHER_INTERVAL_MIN[cfg.interval_min]):
            rows = self.broker.get_klines(cfg.symbol, minutes, CANDLES_PER_FRAME)
            candles = parse_klines(rows, interval_min=minutes, now_ms=now_ms)
            frames[interval_label(minutes)] = frame_features(candles)
        state = build_state(cfg, frames, position, equity, now_ms)

        decision: Optional[Decision] = None
        error = ""
        try:
            decision = self.ask(state)
        except (JevError, requests.RequestException, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
        plan = plan_position(
            decision,
            current_side=position.side,
            allow_short=cfg.allow_short,
            min_confidence=cfg.min_confidence,
        )
        cycle_error = ""
        try:
            actions.extend(self._act(plan, position, equity, now_ms))
        except Exception as exc:
            cycle_error = f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            self._log(
                {
                    "ts": _iso(now_ms),
                    "symbol": cfg.symbol,
                    "interval_min": cfg.interval_min,
                    "execute": cfg.execute,
                    "position_before": asdict(position),
                    "equity_usdt": equity,
                    "state": state,
                    "jev": asdict(decision) if decision else None,
                    "jev_error": error,
                    "plan": asdict(plan),
                    "actions": actions,
                    "cycle_error": cycle_error,
                }
            )
        return {"position": position.side, "decision": decision, "error": error, "plan": plan, "actions": actions}

    def _act(self, plan: Plan, position: Position, equity: float, now_ms: int) -> List[str]:
        if plan.action == "hold":
            return []
        actions: List[str] = []
        opening = plan.action in ("long", "short")
        block = self._open_block(now_ms) if opening else ""
        if position.side != "flat" and plan.action != position.side:
            actions.append(self._close(position))
        if opening:
            actions.append(f"open_skipped:{block}" if block else self._open(plan.action, equity, now_ms))
        return actions

    def _open_block(self, now_ms: int) -> str:
        cfg = self.cfg
        day = _iso(now_ms)[:10]
        if day != self.day:
            self.day, self.opens_today = day, 0
        if self.opens_today >= cfg.max_opens_per_day:
            return "daily_limit"
        if self.last_open_ms and now_ms - self.last_open_ms < cfg.cooldown_bars * cfg.interval_min * 60_000:
            return "cooldown"
        return ""

    def _close(self, position: Position) -> str:
        if not self.cfg.execute:
            return f"would_close_{position.side}"
        self.broker.market_order(
            self.cfg.symbol, "sell" if position.side == "long" else "buy", position.size, reduce_only=True
        )
        self._wait_for(lambda pos: pos.side == "flat", "position_not_closed")
        return f"closed_{position.side}"

    def _open(self, side: str, equity: float, now_ms: int) -> str:
        cfg = self.cfg
        if not cfg.execute:
            return f"would_open_{side}"
        price = self.broker.last_price(cfg.symbol)
        qty = position_qty(equity, price, position_pct=cfg.position_pct, leverage=cfg.leverage)
        if qty <= 0:
            return "open_skipped:zero_qty"
        if not self._leverage_applied:
            self.broker.set_leverage(cfg.symbol, cfg.leverage)
            self._leverage_applied = True
        self.broker.market_order(cfg.symbol, "buy" if side == "long" else "sell", qty)
        opened = self._wait_for(lambda pos: pos.side == side, "position_not_visible_after_order")
        self.last_open_ms, self.opens_today = now_ms, self.opens_today + 1
        self._ensure_stop(opened)
        return f"opened_{side}"

    def _ensure_stop(self, position: Position) -> bool:
        """Every position must have a stop-loss; if one cannot be set, close the position."""
        if position.side == "flat" or position.stop_loss > 0:
            return False
        entry = position.entry_price or self.broker.last_price(self.cfg.symbol)
        stop = protective_stop_price(entry, position.side, self.cfg.stop_loss_pct)
        try:
            self.broker.set_stop_loss(self.cfg.symbol, position, stop)
        except Exception as exc:
            self._close(position)
            raise RuntimeError(f"stop_loss_failed_position_closed: {exc}") from exc
        return True

    def _wait_for(self, done: Callable[[Position], bool], error: str) -> Position:
        for _ in range(6):
            position = self.broker.get_position(self.cfg.symbol)
            if done(position):
                return position
            self.sleep(1.0)
        raise RuntimeError(error)

    def _log(self, record: Mapping[str, Any]) -> None:
        if self.log_path is None:
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# --------------------------------------------------------------------------- CLI


def seconds_until_next_close(now_s: float, interval_min: int, delay_s: float = 5.0) -> float:
    span = interval_min * 60
    return (int(now_s // span) + 1) * span + delay_s - now_s


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    parser.add_argument("--symbol", default=defaults.symbol, help="e.g. BTC/USDT (Bybit USDT perpetual)")
    parser.add_argument("--interval", type=int, default=defaults.interval_min, help=f"candle minutes {INTERVALS_MIN}")
    parser.add_argument("--position-pct", type=float, default=defaults.position_pct, help="share of equity per trade")
    parser.add_argument("--leverage", type=int, default=defaults.leverage, help=f"1..{MAX_LEVERAGE}")
    parser.add_argument("--stop-loss-pct", type=float, default=defaults.stop_loss_pct, help="0.02 = 2%%")
    parser.add_argument("--min-confidence", type=float, default=defaults.min_confidence)
    parser.add_argument("--allow-short", action="store_true", help="let Jev choose short (default: long/flat only)")
    parser.add_argument("--cooldown-bars", type=int, default=defaults.cooldown_bars)
    parser.add_argument("--max-opens-per-day", type=int, default=defaults.max_opens_per_day)
    parser.add_argument("--execute", action="store_true", help="place orders on the demo account (default: dry run)")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--log-file", default="jev_demo_trader.jsonl")
    return parser.parse_args(argv)


def summarize(result: Mapping[str, Any], cfg: Config) -> str:
    decision = result["decision"]
    jev = f"jev={decision.choice} conf={decision.confidence:.2f}" if decision else f"jev=none ({result['error']})"
    mode = "EXECUTE" if cfg.execute else "dry-run"
    return f"[{mode}] {cfg.symbol} {result['position']} | {jev} | {result['plan'].reason} | actions={result['actions']}"


def main(argv: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None) -> int:
    args = parse_args(argv)
    env = os.environ if env is None else env
    try:
        cfg = Config(
            symbol=args.symbol,
            interval_min=args.interval,
            position_pct=args.position_pct,
            leverage=args.leverage,
            stop_loss_pct=args.stop_loss_pct,
            min_confidence=args.min_confidence,
            allow_short=args.allow_short,
            cooldown_bars=args.cooldown_bars,
            max_opens_per_day=args.max_opens_per_day,
            execute=args.execute,
        ).validate()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    jev = jev_settings(env)
    trader = Trader(
        make_broker(env),
        cfg,
        lambda state: ask_jev(state, options=cfg.options, **jev),
        log_path=Path(args.log_file),
    )
    mode = "EXECUTE on Bybit DEMO" if cfg.execute else "dry-run"
    print(f"{mode}: {cfg.symbol} every {cfg.interval_min}m, log={args.log_file}")

    failures = 0
    try:
        while True:
            try:
                print(summarize(trader.run_cycle(), cfg), flush=True)
                failures = 0
            except Exception as exc:
                failures += 1
                print(
                    f"cycle failed ({failures}/{MAX_CONSECUTIVE_ERRORS}): {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                if failures >= MAX_CONSECUTIVE_ERRORS:
                    return 1
            if args.once:
                return 1 if failures else 0
            time.sleep(seconds_until_next_close(time.time(), cfg.interval_min))
    except KeyboardInterrupt:
        print("stopped; an open position keeps its exchange stop-loss")
        return 0


if __name__ == "__main__":
    sys.exit(main())
