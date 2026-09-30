import json
from dataclasses import replace
from decimal import Decimal

import pytest

from scripts import jev_demo_trader as jt

NOW_S = 1_760_000_000
NOW_MS = NOW_S * 1000
HOUR_MS = 3_600_000


def make_rows(minutes: int, *, count: int = 121, step: float = 0.5):
    """Bybit-style rows, newest first; the first row is the still-forming candle."""
    span = minutes * 60_000
    boundary = (NOW_MS // span) * span
    rows = []
    for k in range(count):
        start = boundary - k * span
        close = 100.0 + (count - k) * step
        rows.append([str(start), str(close - 0.2), str(close + 0.3), str(close - 0.4), str(close), "10", "1000"])
    return rows


def decision(choice="long", confidence=0.8):
    options = {"long": 0.1, "flat": 0.1, "short": 0.1}
    options[choice] = 0.8
    return jt.Decision(choice, confidence, options)


class FakeBroker:
    def __init__(self, position=None, *, price=100.0, equity=10_000.0, fail_stop=False):
        self.position = position or jt.Position()
        self.price = price
        self.equity = equity
        self.fail_stop = fail_stop
        self.orders = []
        self.stops = []
        self.leverage = []

    def get_position(self, symbol):
        return self.position

    def get_equity(self):
        return self.equity

    def last_price(self, symbol):
        return self.price

    def get_klines(self, symbol, minutes, limit):
        return make_rows(minutes)

    def set_leverage(self, symbol, leverage):
        self.leverage.append(leverage)

    def market_order(self, symbol, side, qty, reduce_only=False):
        self.orders.append((side, qty, reduce_only))
        if reduce_only:
            self.position = jt.Position()
        else:
            self.position = jt.Position("long" if side == "buy" else "short", qty, self.price, 0.0)

    def set_stop_loss(self, symbol, position, stop_price):
        if self.fail_stop:
            raise RuntimeError("exchange rejected the stop")
        self.stops.append(stop_price)
        self.position = replace(self.position, stop_loss=stop_price)


def make_trader(broker, answer, tmp_path, **config):
    cfg = jt.Config(**config).validate()
    log = tmp_path / "log.jsonl"
    trader = jt.Trader(broker, cfg, answer, log_path=log, clock=lambda: NOW_S, sleep=lambda _: None)
    return trader, log


def last_log(path):
    return json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])


# ---------------------------------------------------------------- snapshot


def test_parse_klines_drops_forming_candle_and_sorts_oldest_first():
    candles = jt.parse_klines(make_rows(60), interval_min=60, now_ms=NOW_MS)
    assert len(candles) == 120
    assert candles[0].ts_ms < candles[-1].ts_ms
    assert candles[-1].ts_ms + HOUR_MS <= NOW_MS


def test_frame_features_describe_an_uptrend():
    features = jt.frame_features(jt.parse_klines(make_rows(60), interval_min=60, now_ms=NOW_MS))
    assert features["ema20_vs_ema50_pct"] > 0
    assert features["close_vs_ema50_pct"] > 0
    assert features["rsi14"] > 50
    assert features["chg_24_bars_pct"] > 0
    assert features["vs_24_bar_high_pct"] <= 0 <= features["vs_24_bar_low_pct"]


def test_frame_features_need_enough_candles():
    short = jt.parse_klines(make_rows(60, count=30), interval_min=60, now_ms=NOW_MS)
    with pytest.raises(ValueError, match="not_enough_candles"):
        jt.frame_features(short)


# ---------------------------------------------------------------- Jev answers


def answer(choice="long", probabilities=None, confidence=0.8):
    probabilities = probabilities or {"long": 0.8, "flat": 0.15, "short": 0.05}
    return {"answers": {"position": {"choice": choice, "probabilities": probabilities, "confidence": confidence}}}


def test_parse_jev_answer_accepts_a_valid_choice():
    choice, probabilities, confidence = jt.parse_jev_answer(answer(), ("long", "flat", "short"))
    assert (choice, confidence) == ("long", 0.8)
    assert probabilities["flat"] == 0.15


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"answers": {}},
        answer(choice="buy"),
        answer(probabilities={"long": 0.8, "flat": 0.2}),
        answer(probabilities={"long": 0.5, "flat": 0.3, "short": 0.3}),
        answer(choice="flat"),
        answer(confidence=None),
        answer(confidence=1.5),
    ],
)
def test_parse_jev_answer_rejects_unusable_answers(payload):
    with pytest.raises(jt.JevError):
        jt.parse_jev_answer(payload, ("long", "flat", "short"))


def test_parse_jev_answer_reads_a_named_question():
    payload = {"answers": {"regime": {"choice": "trend", "probabilities": {"trend": 0.7, "range": 0.3},
                                      "confidence": 0.7}}}
    assert jt.parse_jev_answer(payload, ("trend", "range"), "regime")[0] == "trend"
    with pytest.raises(jt.JevError, match="regime_answer_missing"):
        jt.parse_jev_answer(answer(), ("trend", "range"), "regime")


def test_parse_jev_answer_rejects_short_when_shorting_is_off():
    with pytest.raises(jt.JevError):
        jt.parse_jev_answer(answer(choice="short", probabilities={"long": 0.1, "flat": 0.1, "short": 0.8}),
                            ("long", "flat"))


def test_ask_jev_posts_a_typed_question_to_systemone():
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return answer(probabilities={"long": 0.7, "flat": 0.3}, confidence=0.7)

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    result = jt.ask_jev(
        {"symbol": "BTC/USDT"},
        options=("long", "flat"),
        api_key="k",
        base_url="https://api.typesafe.ai/v1/",
        model="jev-latest",
        timeout=8,
        post=post,
    )
    url, kwargs = calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert kwargs["headers"]["Authorization"] == "Bearer k"
    assert kwargs["json"]["model"] == "jev-latest"
    assert kwargs["json"]["state"] == {"symbol": "BTC/USDT"}
    criteria = kwargs["json"]["questions"]["position"]["criteria"]
    assert set(criteria) == {"long", "flat"}
    assert result.choice == "long" and result.confidence == 0.7


def test_jev_settings_require_a_key():
    with pytest.raises(SystemExit):
        jt.jev_settings({})
    settings = jt.jev_settings({"JEV_API_KEY": " k "})
    assert settings == {"api_key": "k", "base_url": "https://api.typesafe.ai/v1", "model": "jev-latest", "timeout": 8.0}


# ---------------------------------------------------------------- decision rules


@pytest.mark.parametrize(
    ("dec", "current", "allow_short", "expected"),
    [
        (None, "flat", True, ("hold", "jev_unavailable")),
        (decision("long", 0.5), "flat", True, ("hold", "low_confidence")),
        (decision("long"), "long", True, ("hold", "already_long")),
        (decision("long"), "flat", True, ("long", "jev_long")),
        (decision("flat"), "long", True, ("flat", "jev_flat")),
        (decision("short"), "long", True, ("short", "jev_short")),
        (decision("short"), "long", False, ("flat", "jev_short")),
    ],
)
def test_plan_position(dec, current, allow_short, expected):
    plan = jt.plan_position(dec, current_side=current, allow_short=allow_short, min_confidence=0.6)
    assert (plan.action, plan.reason) == expected


def test_sizing_and_stop_prices():
    assert jt.position_qty(10_000, 100, position_pct=0.1, leverage=2) == pytest.approx(20.0)
    assert jt.position_qty(0, 100, position_pct=0.1, leverage=1) == 0.0
    assert jt.protective_stop_price(100, "long", 0.02) == pytest.approx(98.0)
    assert jt.protective_stop_price(100, "short", 0.02) == pytest.approx(102.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"interval_min": 7},
        {"position_pct": 0.9},
        {"leverage": 10},
        {"stop_loss_pct": 0.0},
        {"min_confidence": 1.5},
        {"max_opens_per_day": 0},
    ],
)
def test_config_rejects_unsafe_values(bad):
    with pytest.raises(ValueError):
        jt.Config(**bad).validate()


# ---------------------------------------------------------------- trader


def test_dry_run_logs_the_decision_and_sends_no_orders(tmp_path):
    broker = FakeBroker()
    trader, log = make_trader(broker, lambda state: decision("long"), tmp_path)
    result = trader.run_cycle()
    assert result["actions"] == ["would_open_long"]
    assert broker.orders == [] and broker.stops == []
    record = last_log(log)
    assert record["jev"]["choice"] == "long"
    assert set(record["state"]["timeframes"]) == {"1h", "4h"}
    assert record["state"]["account"]["equity_usdt"] == 10_000.0


def test_execute_opens_a_long_with_leverage_size_and_stop(tmp_path):
    broker = FakeBroker()
    trader, _ = make_trader(broker, lambda state: decision("long"), tmp_path, execute=True, position_pct=0.1)
    result = trader.run_cycle()
    assert result["actions"] == ["opened_long"]
    assert broker.leverage == [1]
    assert broker.orders == [("buy", pytest.approx(10.0), False)]
    assert broker.stops == [pytest.approx(98.0)]


def test_flip_closes_first_then_opens_the_other_side(tmp_path):
    broker = FakeBroker(jt.Position("long", 5.0, 100.0, 98.0))
    trader, _ = make_trader(broker, lambda state: decision("short"), tmp_path, execute=True, allow_short=True)
    result = trader.run_cycle()
    assert result["actions"] == ["closed_long", "opened_short"]
    assert broker.orders[0] == ("sell", 5.0, True)
    assert broker.orders[1][0] == "sell" and broker.orders[1][2] is False
    assert broker.stops == [pytest.approx(102.0)]


def test_flat_answer_closes_the_position(tmp_path):
    broker = FakeBroker(jt.Position("long", 5.0, 100.0, 98.0))
    trader, _ = make_trader(broker, lambda state: decision("flat"), tmp_path, execute=True)
    assert trader.run_cycle()["actions"] == ["closed_long"]
    assert broker.position.side == "flat"


def test_cooldown_blocks_a_new_open_but_not_a_close(tmp_path):
    broker = FakeBroker()
    trader, _ = make_trader(broker, lambda state: decision("long"), tmp_path, execute=True, cooldown_bars=2)
    trader.last_open_ms = NOW_MS - HOUR_MS
    assert trader.run_cycle()["actions"] == ["open_skipped:cooldown"]
    assert broker.orders == []

    broker = FakeBroker(jt.Position("long", 5.0, 100.0, 98.0))
    trader, _ = make_trader(broker, lambda state: decision("flat"), tmp_path, execute=True, cooldown_bars=2)
    trader.last_open_ms = NOW_MS - HOUR_MS
    assert trader.run_cycle()["actions"] == ["closed_long"]


def test_daily_limit_blocks_new_opens(tmp_path):
    broker = FakeBroker()
    trader, _ = make_trader(broker, lambda state: decision("long"), tmp_path, execute=True, max_opens_per_day=1)
    assert trader.run_cycle()["actions"] == ["opened_long"]
    broker.position = jt.Position()
    trader.last_open_ms = 0
    assert trader.run_cycle()["actions"] == ["open_skipped:daily_limit"]


def test_jev_failure_changes_nothing(tmp_path):
    broker = FakeBroker(jt.Position("long", 5.0, 100.0, 98.0))

    def failing(state):
        raise jt.JevError("probabilities_incomplete")

    trader, log = make_trader(broker, failing, tmp_path, execute=True)
    result = trader.run_cycle()
    assert result["plan"].reason == "jev_unavailable" and result["actions"] == []
    assert broker.orders == []
    assert "probabilities_incomplete" in last_log(log)["jev_error"]


def test_low_confidence_keeps_the_current_position(tmp_path):
    broker = FakeBroker(jt.Position("long", 5.0, 100.0, 98.0))
    trader, _ = make_trader(broker, lambda state: decision("flat", 0.4), tmp_path, execute=True)
    assert trader.run_cycle()["actions"] == []
    assert broker.orders == []


def test_a_position_without_stop_gets_one_before_anything_else(tmp_path):
    broker = FakeBroker(jt.Position("long", 5.0, 100.0, 0.0))
    trader, _ = make_trader(broker, lambda state: decision("long"), tmp_path, execute=True)
    assert trader.run_cycle()["actions"] == ["stop_loss_restored"]
    assert broker.stops == [pytest.approx(98.0)]


def test_position_is_closed_when_its_stop_cannot_be_set(tmp_path):
    broker = FakeBroker(fail_stop=True)
    trader, log = make_trader(broker, lambda state: decision("long"), tmp_path, execute=True)
    with pytest.raises(RuntimeError, match="stop_loss_failed_position_closed"):
        trader.run_cycle()
    assert broker.position.side == "flat"
    assert [order[2] for order in broker.orders] == [False, True]
    assert "stop_loss_failed" in last_log(log)["cycle_error"]


# ---------------------------------------------------------------- Bybit demo broker


class FakeBybitClient:
    def __init__(self, base_url="https://api-demo.bybit.com", positions=None):
        self.base_url = base_url
        self.positions = positions or []
        self.signed = []

    def get_positions(self, *, symbol=""):
        return {"result": {"list": self.positions}}

    def _normalize_price(self, *, symbol, price):
        return Decimal("98.1"), 1

    def _dec_str(self, value, strict_precision=None):
        return "98.1"

    def _signed_request(self, method, path, *, params=None, json_body=None):
        self.signed.append((method, path, json_body))
        return {}


def broker_for(client):
    return jt.BybitDemoBroker(client, to_symbol=lambda symbol: symbol.replace("/", ""))


def test_broker_refuses_a_live_host():
    with pytest.raises(RuntimeError, match="refusing to trade"):
        broker_for(FakeBybitClient(base_url="https://api.bybit.com"))


def test_broker_reads_a_one_way_position():
    row = {"side": "Sell", "size": "0.5", "avgPrice": "100.5", "stopLoss": "", "positionIdx": 0}
    position = broker_for(FakeBybitClient(positions=[row])).get_position("BTC/USDT")
    assert position == jt.Position("short", 0.5, 100.5, 0.0, 0)
    assert broker_for(FakeBybitClient(positions=[{"side": "", "size": "0", "positionIdx": 0}])).get_position(
        "BTC/USDT"
    ) == jt.Position()


def test_broker_rejects_hedge_mode():
    rows = [{"side": "", "size": "0", "positionIdx": 1}, {"side": "", "size": "0", "positionIdx": 2}]
    with pytest.raises(RuntimeError, match="hedge mode"):
        broker_for(FakeBybitClient(positions=rows)).get_position("BTC/USDT")


def test_broker_sets_the_stop_on_the_position():
    client = FakeBybitClient()
    broker_for(client).set_stop_loss("BTC/USDT", jt.Position("long", 1.0, 100.0, 0.0, 0), 98.123)
    method, path, body = client.signed[0]
    assert (method, path) == ("POST", "/v5/position/trading-stop")
    assert body == {
        "category": "linear",
        "symbol": "BTCUSDT",
        "positionIdx": 0,
        "tpslMode": "Full",
        "stopLoss": "98.1",
        "slTriggerBy": "MarkPrice",
    }


# ---------------------------------------------------------------- CLI


def test_main_rejects_unsafe_settings_before_touching_any_service():
    assert jt.main(["--leverage", "10"], env={}) == 2


def test_seconds_until_next_close_waits_for_the_candle_boundary():
    assert jt.seconds_until_next_close(3600 * 5 + 10, 60, delay_s=5) == pytest.approx(3600 - 10 + 5)


# ---------------------------------------------------------------- real BybitClient, fake HTTP


class FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeBybitHttp:
    """Tiny stateful imitation of the Bybit v5 endpoints the trader touches (demo host only)."""

    def __init__(self, position=None):
        self.calls = []
        self.position = position or {"side": "", "size": "0", "avgPrice": "0", "stopLoss": "", "positionIdx": 0}

    def __call__(self, *, method, url, params=None, json=None, data=None, headers=None, timeout=None, verify=None):
        host = "https://api-demo.bybit.com"
        assert url.startswith(host), f"request left the demo host: {url}"
        path = url[len(host):]
        body = globals()["json"].loads(data) if data else {}
        self.calls.append((method, path, body or dict(params or {})))
        return FakeResponse({"retCode": 0, "retMsg": "OK", "result": self.route(path, dict(params or {}), body)})

    def route(self, path, params, body):
        if path == "/v5/market/time":
            return {"timeSecond": str(NOW_S)}
        if path == "/v5/market/kline":
            return {"list": make_rows(int(params["interval"]))}
        if path == "/v5/market/tickers":
            return {"list": [{"symbol": "BTCUSDT", "lastPrice": "100.0"}]}
        if path == "/v5/market/instruments-info":
            return {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001"},
                        "priceFilter": {"tickSize": "0.1"},
                        "leverageFilter": {"maxLeverage": "100"},
                    }
                ]
            }
        if path == "/v5/account/wallet-balance":
            return {"list": [{"accountType": "UNIFIED", "totalEquity": "10000.5"}]}
        if path == "/v5/position/list":
            return {"list": [dict(self.position)]}
        if path == "/v5/order/create":
            if body.get("reduceOnly"):
                self.position = {"side": "", "size": "0", "avgPrice": "0", "stopLoss": "", "positionIdx": 0}
            else:
                side = "Buy" if body["side"] == "Buy" else "Sell"
                self.position = {"side": side, "size": body["qty"], "avgPrice": "100", "stopLoss": "", "positionIdx": 0}
            return {"orderId": "1", "orderLinkId": body.get("orderLinkId", "")}
        if path == "/v5/position/trading-stop":
            self.position["stopLoss"] = body["stopLoss"]
            return {}
        return {}


def real_demo_broker(monkeypatch, http):
    pytest.importorskip("flask")
    from app.services.live_trading.factory import create_client

    monkeypatch.setattr("requests.request", http)
    client = create_client(
        {
            "exchange_id": "bybit",
            "api_key": "key",
            "secret_key": "secret",
            "environment": "demo",
            "market_scope": "swap",
        },
        market_type="swap",
    )
    return jt.BybitDemoBroker(client)


def posts(http, path):
    return [body for method, p, body in http.calls if method == "POST" and p == path]


def test_full_cycle_through_the_real_client_opens_a_long_with_a_stop(monkeypatch, tmp_path):
    http = FakeBybitHttp()
    trader, log = make_trader(real_demo_broker(monkeypatch, http), lambda state: decision("long"), tmp_path, execute=True)
    assert trader.run_cycle()["actions"] == ["opened_long"]

    [order] = posts(http, "/v5/order/create")
    assert order["category"] == "linear" and order["symbol"] == "BTCUSDT"
    assert order["side"] == "Buy" and order["orderType"] == "Market" and order["positionIdx"] == 0
    assert float(order["qty"]) == pytest.approx(10.0) and "reduceOnly" not in order
    assert order["orderLinkId"].startswith("jev")
    assert posts(http, "/v5/position/set-leverage")[0]["buyLeverage"] == "1"
    [stop] = posts(http, "/v5/position/trading-stop")
    assert stop["positionIdx"] == 0 and stop["tpslMode"] == "Full" and stop["slTriggerBy"] == "MarkPrice"
    assert float(stop["stopLoss"]) == pytest.approx(98.0, abs=0.1)
    state = last_log(log)["state"]
    assert state["timeframes"]["4h"]["last_close"] > 0 and state["account"]["equity_usdt"] == 10000.5


def test_full_cycle_through_the_real_client_closes_with_a_reduce_only_order(monkeypatch, tmp_path):
    http = FakeBybitHttp({"side": "Buy", "size": "10", "avgPrice": "100", "stopLoss": "98", "positionIdx": 0})
    trader, _ = make_trader(real_demo_broker(monkeypatch, http), lambda state: decision("flat"), tmp_path, execute=True)
    assert trader.run_cycle()["actions"] == ["closed_long"]
    [order] = posts(http, "/v5/order/create")
    assert order["side"] == "Sell" and order["reduceOnly"] is True and float(order["qty"]) == 10.0


def test_dry_run_through_the_real_client_only_reads(monkeypatch, tmp_path):
    http = FakeBybitHttp()
    trader, _ = make_trader(real_demo_broker(monkeypatch, http), lambda state: decision("long"), tmp_path)
    assert trader.run_cycle()["actions"] == ["would_open_long"]
    assert all(method == "GET" for method, _, _ in http.calls)
