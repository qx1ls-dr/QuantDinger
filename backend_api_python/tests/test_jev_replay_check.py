import math
import random
from dataclasses import replace

import pytest
import requests

from scripts import jev_demo_trader as jt
from scripts import jev_replay_check as rc

HOST = "https://api.bybit.com"
DAY_MS = 86_400_000
NOW_MS = 1_790_000_000_000
HISTORY_START = NOW_MS - 900 * DAY_MS


def price(t_ms):
    days = (t_ms - HISTORY_START) / DAY_MS
    return 100.0 * (1.0 + 0.0005 * days) * (1.0 + 0.08 * math.sin(days / 9.0))


class FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeKlines:
    """Bybit /v5/market/kline: newest first, the newest candle may still be forming."""

    def __init__(self, ret_code=0):
        self.ret_code = ret_code
        self.urls = []
        self.calls = []

    def __call__(self, url, params=None, timeout=None):
        self.urls.append(url)
        self.calls.append(dict(params))
        minutes = 1440 if params["interval"] == "D" else int(params["interval"])
        span = minutes * 60_000
        start = (min(int(params["end"]), NOW_MS) // span) * span
        rows = []
        while len(rows) < int(params["limit"]) and start >= HISTORY_START:
            o, c = price(start), price(start + span)
            rows.append([str(start), f"{o:.4f}", f"{max(o, c) * 1.002:.4f}", f"{min(o, c) * 0.998:.4f}",
                         f"{c:.4f}", "10", "1000"])
            start -= span
        return FakeHttpResponse({"retCode": self.ret_code, "retMsg": "OK", "result": {"list": rows}})


class FakeJev:
    """Answers "long" when the 4h fast average is above the slow one."""

    def __init__(self):
        self.urls = []
        self.requests = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.urls.append(url)
        self.requests.append(json)
        bullish = json["state"]["timeframes"]["4h"]["ema20_vs_ema50_pct"] > 0
        probabilities = {"long": 0.7, "flat": 0.3} if bullish else {"long": 0.3, "flat": 0.7}
        choice = "long" if bullish else "flat"
        return FakeHttpResponse({"answers": {"position": {"choice": choice, "probabilities": probabilities,
                                                          "confidence": 0.7}}})


def decision(choice="long", confidence=0.8):
    probabilities = {"long": 0.1, "flat": 0.1, "short": 0.1}
    probabilities[choice] = 0.8
    return jt.Decision(choice, confidence, probabilities)


def load(interval=240, bars=300, allow_short=False):
    fake = FakeKlines()
    higher = jt.HIGHER_INTERVAL_MIN[interval]
    main = rc.fetch_candles(fake, HOST, "BTC/USDT", interval, bars + rc.WINDOW + 1, now_ms=NOW_MS)
    hi = rc.fetch_candles(fake, HOST, "BTC/USDT", higher, 200, now_ms=NOW_MS)
    snaps = rc.build_snapshots(main, hi, interval_min=interval, higher_min=higher, bars=bars, allow_short=allow_short)
    return main, hi, snaps


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"HTTP {status}", response=response)


# ---------------------------------------------------------------- market data


def test_fetch_candles_pages_back_and_keeps_closed_candles_only():
    fake = FakeKlines()
    candles = rc.fetch_candles(fake, HOST + "/", "BTC/USDT", 240, 1500, now_ms=NOW_MS)
    span = 240 * 60_000
    assert len(candles) == 1500
    assert all(b.ts_ms - a.ts_ms == span for a, b in zip(candles, candles[1:]))
    assert candles[-1].ts_ms + span <= NOW_MS < candles[-1].ts_ms + 2 * span
    assert fake.urls[0] == "https://api.bybit.com/v5/market/kline"
    assert [call["end"] for call in fake.calls][0] == NOW_MS
    assert len(fake.calls) == 2 and fake.calls[1]["end"] < fake.calls[0]["end"]
    assert {(call["category"], call["symbol"], call["interval"]) for call in fake.calls} == {
        ("linear", "BTCUSDT", "240")
    }


def test_fetch_candles_reports_bybit_errors():
    with pytest.raises(RuntimeError, match="Bybit kline error"):
        rc.fetch_candles(FakeKlines(ret_code=10001), HOST, "BTC/USDT", 240, 100, now_ms=NOW_MS)


def test_snapshots_hold_the_next_candle_return_and_no_identifying_data():
    main, _, snaps = load(bars=300)
    span = 240 * 60_000
    assert len(snaps) == 300
    assert snaps[-1].ts_ms == main[-2].ts_ms + span
    assert snaps[-1].ret == pytest.approx(main[-1].close / main[-2].close - 1.0)
    state = snaps[-1].state
    assert set(state) == {"market_type", "timeframes", "limits"}
    assert set(state["timeframes"]) == {"4h", "1d"}
    for frame in state["timeframes"].values():
        assert "last_close" not in frame and "last_candle_close_time" not in frame
        assert "rsi14" in frame and "ema20_vs_ema50_pct" in frame


def test_main_timeframe_uses_only_candles_closed_at_decision_time():
    main, hi, snaps = load(bars=300)
    future_scrambled = main[:-1] + [replace(main[-1], close=main[-1].close * 10, high=main[-1].high * 10)]
    again = rc.build_snapshots(future_scrambled, hi, interval_min=240, higher_min=1440, bars=300, allow_short=False)
    assert [(s.state, s.trend) for s in again] == [(s.state, s.trend) for s in snaps]
    assert again[-1].ret == pytest.approx(10 * (1 + snaps[-1].ret) - 1)


def test_higher_timeframe_uses_only_candles_closed_at_decision_time():
    main, hi, _ = load(bars=300)
    main = main[:-60]
    snaps = rc.build_snapshots(main, hi, interval_min=240, higher_min=1440, bars=200, allow_short=False)
    last_decision = snaps[-1].ts_ms
    assert any(c.ts_ms + DAY_MS > last_decision for c in hi)
    future_scrambled = [
        c if c.ts_ms + DAY_MS <= last_decision else replace(c, close=c.close * 10, high=c.high * 10) for c in hi
    ]
    scrambled = rc.build_snapshots(main, future_scrambled, interval_min=240, higher_min=1440, bars=200,
                                   allow_short=False)
    assert [s.state for s in scrambled] == [s.state for s in snaps]


def test_trend_target_follows_the_dual_moving_average_template():
    rising = [float(i) for i in range(1, 61)]
    assert rc.trend_target(rising, allow_short=False) == 1
    assert rc.trend_target(rising[::-1], allow_short=False) == 0
    assert rc.trend_target(rising[::-1], allow_short=True) == -1


def test_cache_key_changes_with_question_model_and_state():
    _, _, snaps = load(bars=200)
    key = rc.cache_key("BTC/USDT", 240, snaps[0], ("long", "flat"), "jev-latest")
    assert key.startswith(f"BTCUSDT|240|{snaps[0].ts_ms}|")
    assert key != rc.cache_key("BTC/USDT", 240, snaps[0], ("long", "flat", "short"), "jev-latest")
    assert key != rc.cache_key("BTC/USDT", 240, snaps[0], ("long", "flat"), "jev-2")
    assert key != rc.cache_key("BTC/USDT", 240, snaps[1], ("long", "flat"), "jev-latest")


# ---------------------------------------------------------------- strategies and scoring


def test_jev_positions_keep_the_last_confident_answer():
    answers = [None, decision("long", 0.7), decision("flat", 0.5), decision("flat", 0.7)]
    assert rc.jev_positions(answers, 0.6) == [0, 1, 1, 0]


def test_trend_with_jev_entries_waits_for_agreement_and_exits_with_the_trend():
    trend = [1, 1, 1, 0, 1]
    answers = [decision("flat", 0.7), decision("long", 0.7), decision("flat", 0.7), decision("long", 0.7),
               decision("long", 0.5)]
    assert rc.trend_jev_positions(trend, answers, 0.6) == [0, 1, 1, 0, 0]


def test_evaluate_charges_fees_on_every_change_and_tracks_drawdown():
    result = rc.evaluate([1, 1, 0], [0.1, -0.05, 0.2], 0.001)
    assert result.net_return == pytest.approx(0.999 * 1.1 * 0.95 * 0.999 - 1.0)
    assert result.max_drop == pytest.approx(0.95 * 0.999 - 1.0)
    assert (result.trades, result.fees, result.in_market) == (1, pytest.approx(0.002), pytest.approx(2 / 3))

    flip = rc.evaluate([1, -1], [0.0, 0.0], 0.01)
    assert flip.trades == 2 and flip.fees == pytest.approx(0.04)
    assert rc.evaluate([-1], [0.1], 0.0).net_return == pytest.approx(-0.1)


def test_perfect_timing_beats_random_timing():
    rng = random.Random(3)
    returns = [rng.gauss(0.0, 0.01) for _ in range(400)]
    perfect = [1 if r > 0 else 0 for r in returns]
    actual = rc.evaluate(perfect, returns, 0.0).net_return
    shifted = rc.random_timing(perfect, returns, 0.0, random.Random(1), draws=200)
    assert len(shifted) == 200
    assert rc.timing_percentile([actual], [[r.net_return for r in shifted]]) > 0.99


def test_constant_positions_have_no_timing_to_test():
    assert rc.random_timing([1] * 50, [0.01] * 50, 0.0, random.Random(1)) == []
    assert rc.timing_percentile([0.1], [None]) is None
    assert rc.timing_percentile([0.1, 0.0], [None, [-0.1, 0.1]]) == 0.5


def test_hit_stats_compare_with_the_same_answers_at_random_candles():
    pairs = [
        (decision("long", 0.65), 0.01),
        (decision("long", 0.75), -0.01),
        (decision("flat", 0.9), 0.02),
        (None, 0.03),
        (decision("short", 0.55), -0.02),
    ]
    stats = rc.hit_stats(pairs)
    assert (stats.answers, stats.hits) == (3, 2)
    assert stats.baseline == pytest.approx((2 * 0.6 + 1 * 0.4) / 3)
    assert {label: (wins, n) for label, wins, n in stats.buckets} == {
        "<0.5": (0, 0),
        "0.5-0.6": (1, 1),
        "0.6-0.7": (1, 1),
        "0.7-0.8": (0, 1),
        "0.8+": (0, 0),
    }


def net(jev, trend, trend_jev):
    return {rc.JEV: jev, rc.TREND: trend, rc.TREND_JEV: trend_jev}


def test_verdict_passes_jev_only_with_a_clear_edge():
    good = {"BTC/USDT": net(0.2, 0.1, 0.05), "ETH/USDT": net(0.15, 0.05, 0.0)}
    lines = rc.verdict_lines(good, 0.97, 500, 240)
    assert "Jev deciding alone: PASSED" in lines[1]
    assert "jev_demo_trader.py" in lines[-1] and "--interval 240" in lines[-1]
    assert "not proven" in rc.verdict_lines(good, 0.90, 500, 240)[1]
    assert "Too few" in "\n".join(rc.verdict_lines(good, 0.97, 100, 240))


def test_verdict_falls_back_to_the_trend_or_to_nothing():
    filter_helps = {"BTC/USDT": net(0.0, 0.1, 0.12), "ETH/USDT": net(-0.1, 0.05, 0.06)}
    assert "let Jev decide the entries" in rc.verdict_lines(filter_helps, 0.5, 500, 240)[-1]
    trend_only = {"BTC/USDT": net(0.0, 0.1, 0.05), "ETH/USDT": net(-0.1, 0.05, 0.06)}
    assert "plain trend" in rc.verdict_lines(trend_only, 0.5, 500, 240)[-1]
    nothing = {"BTC/USDT": net(-0.1, -0.05, -0.06), "ETH/USDT": net(-0.1, -0.05, -0.07)}
    assert "do not trade" in rc.verdict_lines(nothing, 0.5, 500, 240)[-1]


# ---------------------------------------------------------------- asking Jev


def test_collect_answers_uses_the_cache_and_saves_new_answers(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    cache = {"a": decision("long")}
    asked = []

    def ask(state):
        asked.append(state["n"])
        return decision("flat", 0.7)

    stats = rc.collect_answers([("a", {"n": 1}), ("b", {"n": 2}), ("b", {"n": 2})], ask, cache=cache,
                               cache_path=cache_path, workers=2)
    assert asked == [2]
    assert (stats.asked, stats.cached) == (1, 1)
    assert cache["b"].choice == "flat"
    assert rc.load_cache(cache_path)["b"] == decision("flat", 0.7)


def test_collect_answers_retries_rate_limits_and_counts_bad_answers():
    attempts = []
    sleeps = []

    def ask(state):
        if state["n"] == 1:
            attempts.append(1)
            if len(attempts) == 1:
                raise http_error(429)
            return decision("long")
        raise jt.JevError("probabilities_incomplete")

    cache = {}
    stats = rc.collect_answers([("a", {"n": 1}), ("b", {"n": 2})], ask, cache=cache, cache_path=None, workers=1,
                               sleep=sleeps.append)
    assert len(attempts) == 2 and sleeps == [1.0]
    assert (stats.asked, stats.invalid, stats.failed) == (1, 1, 0)
    assert set(cache) == {"a"}


def test_collect_answers_stops_when_jev_never_answers():
    attempts = {}

    def ask(state):
        attempts[state["n"]] = attempts.get(state["n"], 0) + 1
        raise http_error(401)

    jobs = [(str(i), {"n": i}) for i in range(30)]
    with pytest.raises(SystemExit, match="Jev is not answering"):
        rc.collect_answers(jobs, ask, cache={}, cache_path=None, workers=1, sleep=lambda _: None)
    assert len(attempts) >= rc.MAX_FAILURES_WITHOUT_ANSWER
    assert set(attempts.values()) == {1}  # a rejected key is not retried


# ---------------------------------------------------------------- CLI


def test_main_compares_jev_with_simple_rules_and_reuses_the_cache(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(rc, "RANDOM_SHIFTS", 50)
    argv = ["--bars", "250", "--cache", str(tmp_path / "cache.jsonl"), "--workers", "2"]
    jev = FakeJev()
    assert rc.main(argv, env={"JEV_API_KEY": "k"}, get=FakeKlines(), post=jev, now=lambda: NOW_MS / 1000) == 0
    out = capsys.readouterr().out
    for text in ("long or flat only", "BTC/USDT: 250 candles", "ETH/USDT: 250 candles", rc.BUY_HOLD, rc.TREND,
                 rc.JEV, rc.TREND_JEV,
                 rc.RANDOM, "Jev answers: 500 new, 0 from cache", "Jev direction right", "Verdict", "Next step"):
        assert text in out
    assert len(jev.requests) == 500
    assert jev.urls[0] == "https://api.typesafe.ai/v1/systemone"
    sent = jev.requests[0]
    assert sent["model"] == "jev-latest"
    assert set(sent["state"]) == {"market_type", "timeframes", "limits"}
    for frame in sent["state"]["timeframes"].values():
        assert "last_close" not in frame and "last_candle_close_time" not in frame
    assert set(sent["questions"]["position"]["criteria"]) == {"long", "flat"}

    again = FakeJev()
    assert rc.main(argv, env={"JEV_API_KEY": "k"}, get=FakeKlines(), post=again, now=lambda: NOW_MS / 1000) == 0
    assert again.requests == []
    assert "Jev answers: 0 new, 500 from cache" in capsys.readouterr().out


def test_main_without_a_jev_key_shows_only_the_simple_rules(capsys):
    def no_post(*args, **kwargs):
        raise AssertionError("Jev must not be called")

    argv = ["--bars", "250", "--symbols", "BTC/USDT"]
    assert rc.main(argv, env={}, get=FakeKlines(), post=no_post, now=lambda: NOW_MS / 1000) == 0
    out = capsys.readouterr().out
    assert rc.BUY_HOLD in out and rc.TREND in out and rc.JEV not in out
    assert "JEV_API_KEY is not set" in out


def test_main_rejects_bad_settings_and_explains_data_errors(capsys):
    assert rc.main(["--interval", "7"], env={}) == 2
    assert rc.main(["--bars", "50"], env={}) == 2

    def offline(*args, **kwargs):
        raise requests.ConnectionError("no route to host")

    assert rc.main(["--bars", "250"], env={}, get=offline, now=lambda: NOW_MS / 1000) == 1
    assert "api.bytick.com" in capsys.readouterr().err
