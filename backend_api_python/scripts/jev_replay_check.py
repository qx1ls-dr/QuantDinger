#!/usr/bin/env python3
"""
JEV replay check: does Jev's market judgement beat simple rules on past candles?

For every closed candle Jev gets the same question as scripts/jev_demo_trader.py (hold long,
flat or short) with an anonymised snapshot: relative indicators only, no symbol, dates or
prices, so the model cannot recall what happened next. Jev sees the market only, not a
position, so every candle is an independent judgement.

On the same candles, after fees and slippage (full-account position, no leverage, no stops,
no funding) it compares:
  - buy and hold;
  - trend: the platform's Dual Moving Average template (SMA 20 above SMA 60);
  - jev: Jev decides the position (low-confidence answers keep the previous position);
  - trend + jev: the trend gives the direction, Jev decides when to enter, exits follow the trend;
  - random timing: Jev's own positions shifted in time (same trades and time in market).

With --question regime Jev is asked a different question instead: is the market trending or
ranging? The trend rule then trades only while Jev says "trend", and is compared with the plain
trend rule and with the same filter switched on at random times.

In the default mode the report also shows what the candles looked like when Jev chose long,
for example whether it mostly bought after drops.

Market data comes from Bybit's public API (no keys needed). Jev answers are cached, so a re-run
with other fees or thresholds costs nothing. Without JEV_API_KEY only the simple rules are shown.

Examples:
  python scripts/jev_replay_check.py                           # BTC and ETH, 4h, 1500 candles each
  python scripts/jev_replay_check.py --interval 60 --bars 3000
  python scripts/jev_replay_check.py --question regime          # Jev as a trend/range filter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import jev_demo_trader as jt  # noqa: E402

SIDE = {"long": 1, "flat": 0, "short": -1}
# The live bot requests CANDLES_PER_FRAME candles and one of them is still forming.
WINDOW = jt.CANDLES_PER_FRAME - 1
TREND_FAST, TREND_SLOW = 20, 60
KLINE_PAGE = 1000
RANDOM_SHIFTS = 1000
RANDOM_SEED = 7
PASS_PERCENTILE = 0.95
MIN_DIRECTIONAL_ANSWERS = 300
MAX_FAILURES_WITHOUT_ANSWER = 10
CONFIDENCE_BUCKETS = ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0001))

BUY_HOLD = "buy and hold"
TREND = "trend (SMA 20/60)"
JEV = "jev decides"
TREND_JEV = "trend + jev entry"
RANDOM = "random timing (median)"
TREND_REGIME = "trend + jev regime"
RANDOM_GATE = "random regime (median)"

REGIME_OPTIONS = ("trend", "range")
REGIME_QUESTIONS = {
    "regime": {
        "type": "choice",
        "instructions": (
            "Using only the supplied point-in-time market evidence, judge whether price is likely to keep "
            "moving persistently in one direction over the next several candles, or to chop sideways."
        ),
        "criteria": {
            "trend": "Price is likely to keep moving in one direction, so trend-following entries should work.",
            "range": "Price is likely to chop sideways or reverse, so trend-following entries would be whipsawed.",
        },
    }
}
# Main-timeframe features shown when profiling Jev's answers: (key, column title, suffix).
PROFILE_FEATURES = (
    ("rsi14", "rsi14", ""),
    ("chg_6_bars_pct", "6-candle chg", "%"),
    ("close_vs_ema50_pct", "vs EMA50", "%"),
    ("ema20_vs_ema50_pct", "EMA20 vs 50", "%"),
)


@dataclass(frozen=True)
class Snapshot:
    ts_ms: int  # decision time: close of the candle
    ret: float  # close-to-close return of the next candle
    trend: int  # Dual Moving Average target: 1 long, 0 flat, -1 short
    state: Dict[str, Any]  # what Jev sees


@dataclass(frozen=True)
class Result:
    net_return: float
    max_drop: float
    trades: int
    fees: float
    in_market: float


@dataclass
class AskStats:
    asked: int = 0
    cached: int = 0
    invalid: int = 0
    failed: int = 0
    last_error: str = ""


@dataclass(frozen=True)
class HitStats:
    answers: int  # long or short answers
    hits: int
    baseline: float  # expected hit rate of the same answers placed at random candles
    buckets: Tuple[Tuple[str, int, int], ...]  # (confidence range, hits, answers)


# --------------------------------------------------------------------------- market data


def bybit_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "").upper()


def fetch_candles(
    get: Callable[..., Any],
    host: str,
    symbol: str,
    minutes: int,
    count: int,
    *,
    now_ms: int,
) -> List[jt.Candle]:
    """The last ``count`` closed USDT-perpetual candles, oldest first (public endpoint, no keys)."""
    rows_by_ts: Dict[int, Sequence[Any]] = {}
    end = now_ms
    while len(rows_by_ts) < count + 1:  # +1: the newest row is usually still forming
        response = get(
            f"{host.rstrip('/')}/v5/market/kline",
            params={
                "category": "linear",
                "symbol": bybit_symbol(symbol),
                "interval": jt.bybit_interval(minutes),
                "end": end,
                "limit": KLINE_PAGE,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("retCode") not in (0, "0", None):
            raise RuntimeError(f"Bybit kline error for {symbol}: {payload.get('retMsg')}")
        rows = (payload.get("result") or {}).get("list") or []
        for row in rows:
            rows_by_ts[int(row[0])] = row
        if len(rows) < KLINE_PAGE:
            break
        oldest = min(int(row[0]) for row in rows)
        if oldest - 1 >= end:
            break
        end = oldest - 1
    return jt.parse_klines(list(rows_by_ts.values()), interval_min=minutes, now_ms=now_ms)[-count:]


def trend_target(closes: Sequence[float], allow_short: bool) -> int:
    fast = sum(closes[-TREND_FAST:]) / TREND_FAST
    slow = sum(closes[-TREND_SLOW:]) / TREND_SLOW
    if fast > slow:
        return 1
    return -1 if allow_short else 0


def anonymise(features: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in features.items() if key not in ("last_close", "last_candle_close_time")}


def build_snapshots(
    main: Sequence[jt.Candle],
    higher: Sequence[jt.Candle],
    *,
    interval_min: int,
    higher_min: int,
    bars: int,
    allow_short: bool,
) -> List[Snapshot]:
    span, higher_span = interval_min * 60_000, higher_min * 60_000
    higher_ends = [candle.ts_ms + higher_span for candle in higher]
    label, higher_label = jt.interval_label(interval_min), jt.interval_label(higher_min)
    snapshots: List[Snapshot] = []
    first = max(max(TREND_SLOW, jt.MIN_CANDLES) - 1, len(main) - 1 - bars)
    for i in range(first, len(main) - 1):
        if main[i + 1].ts_ms != main[i].ts_ms + span:
            continue  # gap in the data
        decision_ms = main[i].ts_ms + span
        closed_higher = bisect_right(higher_ends, decision_ms)
        higher_window = higher[max(0, closed_higher - WINDOW):closed_higher]
        if len(higher_window) < jt.MIN_CANDLES:
            continue
        main_window = main[max(0, i + 1 - WINDOW):i + 1]
        state = {
            "market_type": "swap",
            "timeframes": {
                label: anonymise(jt.frame_features(main_window)),
                higher_label: anonymise(jt.frame_features(higher_window)),
            },
            "limits": {"shorting_allowed": allow_short},
        }
        snapshots.append(
            Snapshot(
                ts_ms=decision_ms,
                ret=main[i + 1].close / main[i].close - 1.0,
                trend=trend_target([candle.close for candle in main_window], allow_short),
                state=state,
            )
        )
    return snapshots[-bars:]


# --------------------------------------------------------------------------- asking Jev


def jev_question(kind: str, allow_short: bool) -> Tuple[str, Tuple[str, ...], Dict[str, Any]]:
    """(question name, answer options, question definition) for --question."""
    if kind == "regime":
        return "regime", REGIME_OPTIONS, REGIME_QUESTIONS
    options = ("long", "flat", "short") if allow_short else ("long", "flat")
    return "position", options, jt.build_questions(options)


def cache_key(
    symbol: str, interval_min: int, snapshot: Snapshot, questions: Mapping[str, Any], model: str
) -> str:
    request = {"state": snapshot.state, "questions": questions, "model": model}
    digest = hashlib.sha1(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{bybit_symbol(symbol)}|{interval_min}|{snapshot.ts_ms}|{digest}"


def load_cache(path: Path) -> Dict[str, jt.Decision]:
    cache: Dict[str, jt.Decision] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            cache[item["key"]] = jt.Decision(
                str(item["choice"]),
                float(item["confidence"]),
                {str(k): float(v) for k, v in dict(item["probabilities"]).items()},
            )
        except (ValueError, KeyError, TypeError):
            continue
    return cache


def _retryable(exc: requests.RequestException) -> bool:
    response = getattr(exc, "response", None)
    return response is None or response.status_code == 429 or response.status_code >= 500


def ask_with_retry(
    ask: Callable[[Mapping[str, Any]], jt.Decision],
    state: Mapping[str, Any],
    *,
    attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> jt.Decision:
    for attempt in range(attempts):
        try:
            return ask(state)
        except requests.RequestException as exc:
            if attempt == attempts - 1 or not _retryable(exc):
                raise
            sleep(2.0**attempt)
    raise RuntimeError("unreachable")


def collect_answers(
    jobs: Sequence[Tuple[str, Mapping[str, Any]]],
    ask: Callable[[Mapping[str, Any]], jt.Decision],
    *,
    cache: Dict[str, jt.Decision],
    cache_path: Optional[Path],
    workers: int,
    sleep: Callable[[float], None] = time.sleep,
) -> AskStats:
    """Ask Jev about every uncached job; answers go into ``cache`` and are appended to ``cache_path``."""
    stats = AskStats()
    pending: Dict[str, Mapping[str, Any]] = {}
    for key, state in jobs:
        if key in cache:
            stats.cached += 1
        else:
            pending.setdefault(key, state)
    if not pending:
        return stats
    progress_step = max(100, len(pending) // 10)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ask_with_retry, ask, state, sleep=sleep): key for key, state in pending.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                decision = future.result()
            except (jt.JevError, ValueError) as exc:
                stats.invalid += 1
                stats.last_error = f"invalid answer: {exc}"[:200]
                continue
            except requests.RequestException as exc:
                stats.failed += 1
                stats.last_error = f"{type(exc).__name__}: {exc}"[:200]
                if stats.asked == 0 and stats.failed >= MAX_FAILURES_WITHOUT_ANSWER:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(f"Jev is not answering ({stats.last_error}). Check JEV_API_KEY and JEV_BASE_URL.")
                continue
            stats.asked += 1
            cache[key] = decision
            if cache_path is not None:
                record = {
                    "key": key,
                    "choice": decision.choice,
                    "confidence": decision.confidence,
                    "probabilities": decision.probabilities,
                }
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            if stats.asked % progress_step == 0:
                print(f"  Jev answered {stats.asked}/{len(pending)}", file=sys.stderr, flush=True)
    return stats


# --------------------------------------------------------------------------- strategies and scoring


def jev_positions(decisions: Sequence[Optional[jt.Decision]], min_confidence: float) -> List[int]:
    position, out = 0, []
    for decision in decisions:
        if decision is not None and decision.confidence >= min_confidence:
            position = SIDE[decision.choice]
        out.append(position)
    return out


def trend_jev_positions(
    trend: Sequence[int],
    decisions: Sequence[Optional[jt.Decision]],
    min_confidence: float,
) -> List[int]:
    """The trend gives the direction; an entry happens only on a candle where Jev agrees."""
    position, out = 0, []
    for target, decision in zip(trend, decisions):
        if target == 0:
            position = 0
        elif position != target:
            agrees = (
                decision is not None
                and decision.confidence >= min_confidence
                and SIDE[decision.choice] == target
            )
            position = target if agrees else 0
        out.append(position)
    return out


def evaluate(positions: Sequence[int], returns: Sequence[float], cost: float) -> Result:
    equity = peak = 1.0
    max_drop = fees = 0.0
    trades = held = previous = 0
    for position, ret in zip(positions, returns):
        if position != previous:
            fee = abs(position - previous) * cost
            equity *= 1.0 - fee
            fees += fee
            trades += 1 if position else 0
        equity *= 1.0 + position * ret
        held += 1 if position else 0
        peak = max(peak, equity)
        max_drop = min(max_drop, equity / peak - 1.0)
        previous = position
    if previous:
        fee = abs(previous) * cost
        equity *= 1.0 - fee
        fees += fee
    return Result(equity - 1.0, max_drop, trades, fees, held / len(returns) if returns else 0.0)


def regime_gates(decisions: Sequence[Optional[jt.Decision]], min_confidence: float) -> List[int]:
    """1 while Jev's last confident answer was "trend", else 0."""
    gate, out = 0, []
    for decision in decisions:
        if decision is not None and decision.confidence >= min_confidence:
            gate = 1 if decision.choice == "trend" else 0
        out.append(gate)
    return out


def gated(trend: Sequence[int], gates: Sequence[int]) -> List[int]:
    return [target * gate for target, gate in zip(trend, gates)]


def random_gate_timing(
    trend: Sequence[int],
    gates: Sequence[int],
    returns: Sequence[float],
    cost: float,
    rng: random.Random,
    draws: Optional[int] = None,
) -> List[Result]:
    """The same regime filter switched on at random times (same share of time on)."""
    draws = RANDOM_SHIFTS if draws is None else draws
    gates = list(gates)
    n = len(gates)
    if n < 2 or len(set(gates)) < 2:
        return []
    results = []
    for _ in range(draws):
        k = rng.randrange(1, n)
        results.append(evaluate(gated(trend, gates[k:] + gates[:k]), returns, cost))
    return results


def random_timing(
    positions: Sequence[int],
    returns: Sequence[float],
    cost: float,
    rng: random.Random,
    draws: Optional[int] = None,
) -> List[Result]:
    """Jev's own positions shifted in time: same trades and time in market, random timing."""
    draws = RANDOM_SHIFTS if draws is None else draws
    positions = list(positions)
    n = len(positions)
    if n < 2 or len(set(positions)) < 2:
        return []
    results = []
    for _ in range(draws):
        k = rng.randrange(1, n)
        results.append(evaluate(positions[k:] + positions[:k], returns, cost))
    return results


def median_result(results: Sequence[Result]) -> Result:
    return Result(
        statistics.median(r.net_return for r in results),
        statistics.median(r.max_drop for r in results),
        round(statistics.median(r.trades for r in results)),
        statistics.median(r.fees for r in results),
        statistics.median(r.in_market for r in results),
    )


def timing_percentile(actual: Sequence[float], runs: Sequence[Optional[Sequence[float]]]) -> Optional[float]:
    """Share of random timings (averaged over symbols) that did worse than Jev's actual timing."""
    if not actual or all(r is None for r in runs):
        return None
    draws = min(len(r) for r in runs if r is not None)
    filled = [list(r[:draws]) if r is not None else [a] * draws for a, r in zip(actual, runs)]
    target = statistics.fmean(actual)
    means = [statistics.fmean(column) for column in zip(*filled)]
    return sum(1 for mean in means if mean < target) / len(means)


def bucket_label(low: float, high: float) -> str:
    if low == 0.0:
        return f"<{high:g}"
    if high > 1.0:
        return f"{low:g}+"
    return f"{low:g}-{high:g}"


def hit_stats(pairs: Sequence[Tuple[Optional[jt.Decision], float]]) -> HitStats:
    moves = [ret for _, ret in pairs]
    up = sum(1 for ret in moves if ret > 0) / len(moves) if moves else 0.0
    down = sum(1 for ret in moves if ret < 0) / len(moves) if moves else 0.0
    directional = [(d, ret) for d, ret in pairs if d is not None and d.choice != "flat"]
    n = len(directional)
    longs = sum(1 for d, _ in directional if d.choice == "long")
    buckets = []
    for low, high in CONFIDENCE_BUCKETS:
        subset = [(d, ret) for d, ret in directional if low <= d.confidence < high]
        wins = sum(1 for d, ret in subset if SIDE[d.choice] * ret > 0)
        buckets.append((bucket_label(low, high), wins, len(subset)))
    return HitStats(
        answers=n,
        hits=sum(1 for d, ret in directional if SIDE[d.choice] * ret > 0),
        baseline=(longs * up + (n - longs) * down) / n if n else 0.0,
        buckets=tuple(buckets),
    )


def profile_lines(
    rows: Sequence[Tuple[jt.Decision, Mapping[str, Any], float]],
    options: Sequence[str],
    label: str,
) -> List[str]:
    """Average main-timeframe features per answer, and whether long answers were dip buying."""
    lines = [f"What the {label} candle looked like when Jev answered (averages):"]
    header = f"  {'answer':<8}{'count':>7}" + "".join(f"{title:>15}" for _, title, _ in PROFILE_FEATURES)
    lines.append(header)
    for option in options:
        group = [features for decision, features, _ in rows if decision.choice == option]
        if not group:
            continue
        cells = []
        for key, _, suffix in PROFILE_FEATURES:
            values = [float(f[key]) for f in group if f.get(key) is not None]
            mean = statistics.fmean(values) if values else None
            text = "-" if mean is None else (f"{mean:+.2f}{suffix}" if suffix else f"{mean:.1f}")
            cells.append(f"{text:>15}")
        lines.append(f"  {option:<8}{len(group):>7}" + "".join(cells))
    longs = [(features, ret) for decision, features, ret in rows if decision.choice == "long"]
    below = [ret for features, ret in longs if (features.get("close_vs_ema50_pct") or 0.0) < 0]
    above = [ret for features, ret in longs if (features.get("close_vs_ema50_pct") or 0.0) >= 0]
    if longs:
        def right(rets: Sequence[float]) -> str:
            return f"right {sum(1 for r in rets if r > 0) / len(rets):.0%}" if rets else "none"

        lines.append(
            f"  Long answers with price below its 50-candle average: {len(below)} of {len(longs)} "
            f"({len(below) / len(longs):.0%}), {right(below)}; above it: {len(above)}, {right(above)}."
        )
        if len(longs) >= 20 and len(below) / len(longs) >= 0.6:
            lines.append("  -> Jev mostly buys dips: it goes long after prices fell below their average.")
        elif len(longs) >= 20 and len(below) / len(longs) <= 0.4:
            lines.append("  -> Jev mostly buys strength: it goes long when prices are above their average.")
    return lines


def regime_verdict_lines(
    net: Mapping[str, Mapping[str, float]],
    percentile: Optional[float],
    answers: int,
) -> List[str]:
    symbols = list(net)
    wins = sum(1 for s in symbols if net[s][TREND_REGIME] > net[s][TREND])
    enough = answers >= MIN_DIRECTIONAL_ANSWERS
    passed = enough and wins == len(symbols) and percentile is not None and percentile >= PASS_PERCENTILE
    lines = [
        "Verdict",
        f"  Jev as a trend/range filter: {'PASSED' if passed else 'not proven'} "
        f"(better than plain trend on {wins} of {len(symbols)} symbols; "
        f"needs all of them and {PASS_PERCENTILE:.0%} of random filters)",
    ]
    if not enough:
        lines.append(f"  Too few Jev answers ({answers}) for a firm answer: add --bars or --symbols.")
    if passed:
        step = "test the trend strategy with the Jev regime filter on the demo account"
    elif statistics.fmean(net[s][TREND] for s in symbols) > 0:
        step = "trade the plain trend strategy (Dual Moving Average template); stop using Jev for trade decisions"
    else:
        step = "do not trade any of these yet: even the trend rule lost money on this data"
    lines.append(f"  Next step: {step}.")
    return lines


# --------------------------------------------------------------------------- report


def day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def format_table(results: Mapping[str, Result]) -> List[str]:
    lines = [f"  {'strategy':<24}{'return':>9}{'max drop':>10}{'trades':>8}{'fees':>7}{'in market':>11}"]
    for name, r in results.items():
        lines.append(
            f"  {name:<24}{r.net_return * 100:>+8.1f}%{r.max_drop * 100:>+9.1f}%"
            f"{r.trades:>8}{r.fees * 100:>6.1f}%{r.in_market * 100:>10.0f}%"
        )
    return lines


def verdict_lines(
    net: Mapping[str, Mapping[str, float]],
    percentile: Optional[float],
    directional_answers: int,
    interval_min: int,
) -> List[str]:
    symbols = list(net)
    jev_wins = sum(1 for s in symbols if net[s][JEV] > net[s][TREND])
    filter_wins = sum(1 for s in symbols if net[s][TREND_JEV] > net[s][TREND])
    enough = directional_answers >= MIN_DIRECTIONAL_ANSWERS
    alone_ok = (
        enough
        and jev_wins == len(symbols)
        and percentile is not None
        and percentile >= PASS_PERCENTILE
    )
    filter_ok = enough and filter_wins == len(symbols)
    lines = [
        "Verdict",
        f"  Jev deciding alone: {'PASSED' if alone_ok else 'not proven'} "
        f"(better than trend on {jev_wins} of {len(symbols)} symbols; "
        f"needs all of them and {PASS_PERCENTILE:.0%} of random timings)",
        f"  Jev deciding entries for the trend: {'helps' if filter_ok else 'not proven'} "
        f"(better than plain trend on {filter_wins} of {len(symbols)} symbols)",
    ]
    if not enough:
        lines.append(
            f"  Too few long/short answers ({directional_answers}) for a firm answer: add --bars or --symbols."
        )
    if alone_ok:
        step = f"run scripts/jev_demo_trader.py on the demo account with --interval {interval_min}"
    elif filter_ok and statistics.fmean(net[s][TREND_JEV] for s in symbols) > 0:
        step = "trade the trend strategy and let Jev decide the entries"
    elif statistics.fmean(net[s][TREND] for s in symbols) > 0:
        step = "trade the plain trend strategy (Dual Moving Average template); Jev adds nothing measurable here"
    else:
        step = "do not trade any of these yet: even the trend rule lost money on this data"
    lines.append(f"  Next step: {step}.")
    return lines


def report_lines(
    snapshots: Mapping[str, Sequence[Snapshot]],
    decisions: Optional[Mapping[str, Sequence[Optional[jt.Decision]]]],
    stats: Optional[AskStats],
    *,
    interval_min: int,
    min_confidence: float,
    allow_short: bool,
    cost: float,
    rng: random.Random,
    question: str = "position",
) -> List[str]:
    regime = question == "regime"
    sides = "long, flat or short" if allow_short else "long or flat only"
    asked = "Jev asked: trend or range" if regime else "Jev asked: which position"
    label = jt.interval_label(interval_min)
    lines = [
        f"JEV replay check | {label} candles | {sides} | {asked} | cost {cost * 100:.3f}% per side "
        "(fee + slippage) | full-account position, no leverage, stops or funding",
        "",
    ]
    net: Dict[str, Dict[str, float]] = {}
    actual: List[float] = []
    random_runs: List[Optional[List[float]]] = []
    pairs: List[Tuple[Optional[jt.Decision], float]] = []
    profile_rows: List[Tuple[jt.Decision, Mapping[str, Any], float]] = []
    trend_by_answer: Dict[str, List[float]] = {}
    for symbol, snaps in snapshots.items():
        returns = [snap.ret for snap in snaps]
        trend = [snap.trend for snap in snaps]
        results = {BUY_HOLD: evaluate([1] * len(snaps), returns, cost), TREND: evaluate(trend, returns, cost)}
        if decisions is not None:
            answers = decisions[symbol]
            if regime:
                gates = regime_gates(answers, min_confidence)
                results[TREND_REGIME] = evaluate(gated(trend, gates), returns, cost)
                shifted = random_gate_timing(trend, gates, returns, cost, rng)
                if shifted:
                    results[RANDOM_GATE] = median_result(shifted)
                actual.append(results[TREND_REGIME].net_return)
                for answer, target, ret in zip(answers, trend, returns):
                    if answer is not None:
                        trend_by_answer.setdefault(answer.choice, []).append(target * ret)
            else:
                positions = jev_positions(answers, min_confidence)
                results[JEV] = evaluate(positions, returns, cost)
                results[TREND_JEV] = evaluate(trend_jev_positions(trend, answers, min_confidence), returns, cost)
                shifted = random_timing(positions, returns, cost, rng)
                if shifted:
                    results[RANDOM] = median_result(shifted)
                actual.append(results[JEV].net_return)
                pairs.extend(zip(answers, returns))
                profile_rows.extend(
                    (answer, snap.state["timeframes"][label], snap.ret)
                    for answer, snap in zip(answers, snaps)
                    if answer is not None
                )
            random_runs.append([r.net_return for r in shifted] if shifted else None)
        net[symbol] = {name: result.net_return for name, result in results.items()}
        lines.append(f"{symbol}: {len(snaps)} candles, {day(snaps[0].ts_ms)} .. {day(snaps[-1].ts_ms)}")
        lines.extend(format_table(results))
        lines.append("")

    if decisions is None or stats is None:
        lines.append("JEV_API_KEY is not set: only the simple rules are shown.")
        return lines

    stats_line = (
        f"Jev answers: {stats.asked} new, {stats.cached} from cache, "
        f"{stats.invalid} invalid, {stats.failed} failed"
    )
    lines.append(stats_line + (f" (last error: {stats.last_error})" if stats.last_error else ""))
    percentile = timing_percentile(actual, random_runs)

    if regime:
        for choice in REGIME_OPTIONS:
            values = trend_by_answer.get(choice, [])
            if values:
                lines.append(
                    f'When Jev said "{choice}", the trend rule made {statistics.fmean(values) * 100:+.3f}% '
                    f"per candle before fees (n={len(values)})."
                )
        if percentile is None:
            lines.append("Jev never changed its answer, so the filter cannot be tested.")
        else:
            lines.append(
                f"Jev's filter beats {percentile:.0%} of the same filter switched on at random times "
                f"(needs {PASS_PERCENTILE:.0%})."
            )
        lines.append("")
        answered = sum(len(values) for values in trend_by_answer.values())
        lines.extend(regime_verdict_lines(net, percentile, answered))
    else:
        hits = hit_stats(pairs)
        if hits.answers:
            rate = hits.hits / hits.answers
            margin = math.sqrt(rate * (1.0 - rate) / hits.answers)
            lines.append(
                f"Jev direction right: {rate:.1%} ± {margin:.1%} of {hits.answers} long/short answers "
                f"(the same answers at random candles: {hits.baseline:.1%})"
            )
            by_confidence = ", ".join(f"{name} {wins / n:.0%} (n={n})" for name, wins, n in hits.buckets if n)
            lines.append(f"  by confidence: {by_confidence}")
        else:
            lines.append("Jev never chose long or short.")
        if percentile is None:
            lines.append("Jev never changed position, so its timing cannot be tested.")
        else:
            lines.append(
                f"Jev timing beats {percentile:.0%} of random timings with the same trades "
                f"(needs {PASS_PERCENTILE:.0%})."
            )
        if profile_rows:
            lines.append("")
            options = ("long", "flat", "short") if allow_short else ("long", "flat")
            lines.extend(profile_lines(profile_rows, options, label))
        lines.append("")
        lines.extend(verdict_lines(net, percentile, hits.answers, interval_min))
    lines.append("")
    lines.append("Past data only; Jev saw no symbol, dates or prices. Funding, stops and cool-downs are not modelled.")
    return lines


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT", help="comma-separated Bybit USDT perpetuals")
    parser.add_argument("--interval", type=int, default=240, help=f"candle minutes {jt.INTERVALS_MIN}")
    parser.add_argument("--bars", type=int, default=1500, help="candles per symbol to test")
    parser.add_argument(
        "--question",
        choices=("position", "regime"),
        default="position",
        help="position: Jev picks the position; regime: Jev says trend or range and filters the trend rule",
    )
    parser.add_argument("--allow-short", action="store_true", help="let Jev and the trend rule go short")
    parser.add_argument("--min-confidence", type=float, default=jt.Config().min_confidence)
    parser.add_argument("--fee", type=float, default=0.00055, help="per side (Bybit taker 0.055%%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="per side (platform backtest default)")
    parser.add_argument("--workers", type=int, default=4, help="parallel Jev requests")
    parser.add_argument("--cache", default="jev_replay_cache.jsonl", help="file with cached Jev answers")
    parser.add_argument("--bybit-host", default="https://api.bybit.com", help="Bybit REST host for public candles")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> List[str]:
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    if not symbols:
        raise ValueError("give at least one symbol")
    if args.interval not in jt.INTERVALS_MIN:
        raise ValueError(f"interval must be one of {jt.INTERVALS_MIN} minutes")
    if not 200 <= args.bars <= 20000:
        raise ValueError("bars must be between 200 and 20000")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("min-confidence must be between 0 and 1")
    if not (0 <= args.fee <= 0.01 and 0 <= args.slippage <= 0.01):
        raise ValueError("fee and slippage must be between 0 and 0.01")
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    return symbols


def load_snapshots(
    symbols: Sequence[str],
    args: argparse.Namespace,
    *,
    get: Callable[..., Any],
    now_ms: int,
) -> Dict[str, List[Snapshot]]:
    higher_min = jt.HIGHER_INTERVAL_MIN[args.interval]
    main_count = args.bars + WINDOW + 1
    higher_count = math.ceil(main_count * args.interval / higher_min) + WINDOW + 1
    snapshots: Dict[str, List[Snapshot]] = {}
    for symbol in symbols:
        print(f"Loading {symbol} candles...", file=sys.stderr, flush=True)
        main_candles = fetch_candles(get, args.bybit_host, symbol, args.interval, main_count, now_ms=now_ms)
        higher_candles = fetch_candles(get, args.bybit_host, symbol, higher_min, higher_count, now_ms=now_ms)
        snaps = build_snapshots(
            main_candles,
            higher_candles,
            interval_min=args.interval,
            higher_min=higher_min,
            bars=args.bars,
            allow_short=args.allow_short,
        )
        if not snaps:
            raise RuntimeError(f"not enough history for {symbol}")
        snapshots[symbol] = snaps
    return snapshots


def ask_jev_about(
    snapshots: Mapping[str, Sequence[Snapshot]],
    args: argparse.Namespace,
    env: Mapping[str, str],
    *,
    post: Callable[..., Any],
) -> Tuple[Dict[str, List[Optional[jt.Decision]]], AskStats]:
    name, options, questions = jev_question(args.question, args.allow_short)
    settings = jt.jev_settings(env)
    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    keys = {
        symbol: [cache_key(symbol, args.interval, snap, questions, settings["model"]) for snap in snaps]
        for symbol, snaps in snapshots.items()
    }
    jobs = [(key, snap.state) for symbol, snaps in snapshots.items() for key, snap in zip(keys[symbol], snaps)]
    cached = sum(1 for key, _ in jobs if key in cache)
    print(f"Asking Jev about {len(jobs)} candles ({cached} already cached)...", file=sys.stderr, flush=True)
    stats = collect_answers(
        jobs,
        lambda state: jt.ask_jev(state, options=options, post=post, questions=questions, question=name, **settings),
        cache=cache,
        cache_path=cache_path,
        workers=args.workers,
    )
    return {symbol: [cache.get(key) for key in keys[symbol]] for symbol in snapshots}, stats


def main(
    argv: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    *,
    get: Callable[..., Any] = requests.get,
    post: Callable[..., Any] = requests.post,
    now: Callable[[], float] = time.time,
) -> int:
    args = parse_args(argv)
    env = os.environ if env is None else env
    try:
        symbols = validate_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        snapshots = load_snapshots(symbols, args, get=get, now_ms=int(now() * 1000))
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"error: could not load candles from {args.bybit_host}: {exc}", file=sys.stderr)
        print("hint: if Bybit is blocked where you are, try --bybit-host https://api.bytick.com", file=sys.stderr)
        return 1
    decisions: Optional[Dict[str, List[Optional[jt.Decision]]]] = None
    stats: Optional[AskStats] = None
    if str(env.get("JEV_API_KEY") or "").strip():
        decisions, stats = ask_jev_about(snapshots, args, env, post=post)
    lines = report_lines(
        snapshots,
        decisions,
        stats,
        interval_min=args.interval,
        min_confidence=args.min_confidence,
        allow_short=args.allow_short,
        cost=args.fee + args.slippage,
        rng=random.Random(RANDOM_SEED),
        question=args.question,
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
