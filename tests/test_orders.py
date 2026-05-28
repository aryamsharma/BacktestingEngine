import pytest

from Engine.Backtester.Orders import MKT, LMT, TRAIL_LMT, ADAPT


class TestMKT:
    def test_init(self):
        o = MKT(100)
        assert o.order_type == "MKT"
        assert o.requested_shares == 100

    def test_init_rejects_float(self):
        with pytest.raises(AssertionError):
            MKT(10.5)

    def test_buy_execute(self, broker_data, current_tick):
        o = MKT(10)
        result = o._execute(broker_data, current_tick)
        assert result["return"] is True
        assert result["shares"] == 10
        assert result["price"] == 97.32
        assert broker_data.available_cash == pytest.approx(10000 - 97.32 * 10 - 2)
        assert broker_data.shares == 10

    def test_buy_insufficient_cash(self, broker_data, current_tick):
        broker_data.available_cash = 5.0
        o = MKT(10)
        result = o._execute(broker_data, current_tick)
        assert result["return"] is False
        assert result["code"] == 1

    def test_sell(self, broker_data, current_tick):
        broker_data.shares = 20
        o = MKT(-10)
        result = o._execute(broker_data, current_tick)
        assert result["return"] is True
        assert result["shares"] == -10
        assert broker_data.shares == 10

    def test_sell_insufficient_shares(self, broker_data, current_tick):
        broker_data.shares = 5
        o = MKT(-10)
        result = o._execute(broker_data, current_tick)
        assert result["return"] is False
        assert result["code"] == 5


class TestLMT:
    def test_init(self):
        o = LMT(50, 100.0)
        assert o.order_type == "LMT"
        assert o.requested_shares == 50
        assert o.requested_price == 100.0

    def test_init_rejects_float_shares(self):
        with pytest.raises(AssertionError):
            LMT(50.5, 100.0)

    def test_buy_executes_below_limit(self, broker_data, current_tick):
        o = LMT(10, 98.0)
        result = o._execute(broker_data, current_tick)
        assert result["return"] is True

    def test_buy_does_not_execute_above_limit(self, broker_data, current_tick):
        o = LMT(10, 97.0)
        result = o._execute(broker_data, current_tick)
        assert result["return"] is False

    def test_sell_executes_above_limit(self, broker_data, current_tick):
        broker_data.shares = 20
        o = LMT(-10, 97.0)
        result = o._execute(broker_data, current_tick)
        assert result["return"] is True

    def test_sell_does_not_execute_below_limit(self, broker_data, current_tick):
        broker_data.shares = 20
        o = LMT(-10, 98.0)
        result = o._execute(broker_data, current_tick)
        assert result["return"] is False

    def test_none_price(self, broker_data, current_tick):
        o = LMT(10, None)
        result = o._execute(broker_data, current_tick)
        assert result["return"] is False
        assert result["code"] == 2


class TestTRAIL_LMT:
    def test_init_price_delta(self):
        o = TRAIL_LMT(-10, price_delta=-0.5)
        assert o.order_type == "TRAIL_STOP"
        assert o.requested_shares == -10
        assert o.trail_delta(100) == 100 - 0.5

    def test_init_percentage_delta(self):
        o = TRAIL_LMT(-10, percentage_delta=-0.02)
        assert o.trail_delta(100) == 100 * -0.02

    def test_first_call_returns_false(self, broker_data, current_tick):
        o = TRAIL_LMT(-10, price_delta=-0.5)
        result = o._execute(broker_data, current_tick)
        assert result["return"] is False
        assert o.old_trail_price is not None

    def test_sell_triggers_when_price_drops(self, broker_data, current_tick):
        broker_data.shares = 10
        o = TRAIL_LMT(-10, price_delta=-0.5)
        o._execute(broker_data, current_tick)

        lower_tick = dict(current_tick, close=96.00)
        result = o._execute(broker_data, lower_tick)
        assert result["return"] is True
        assert result["shares"] == -10

    def test_sell_updates_trail_on_higher_close(self, broker_data, current_tick):
        o = TRAIL_LMT(-10, price_delta=-0.5)
        o._execute(broker_data, current_tick)
        old_trail = o.old_trail_price

        higher_tick = dict(current_tick, close=98.00)
        result = o._execute(broker_data, higher_tick)
        assert result["return"] is False
        assert o.old_trail_price > old_trail

    def test_buy_triggers_when_price_rises(self, broker_data, current_tick):
        o = TRAIL_LMT(10, price_delta=-0.5)
        o._execute(broker_data, current_tick)
        higher_tick = dict(current_tick, close=98.00)
        result = o._execute(broker_data, higher_tick)
        assert result["return"] is True
        assert result["shares"] == 10


class TestADAPT:
    def test_init(self):
        o = ADAPT(10, "sma", go_under=True)
        assert o.order_type == "ADAPT"
        assert o.requested_shares == 10
        assert o.following == "sma"
        assert o.go_under is True

    def test_buy_goes_under(self, broker_data, current_tick):
        tick_with_sma = dict(current_tick, sma=98.0)
        o = ADAPT(10, "sma", go_under=True)
        result = o._execute(broker_data, tick_with_sma)
        assert result["return"] is True

    def test_buy_stays_above(self, broker_data, current_tick):
        tick_with_sma = dict(current_tick, sma=96.0)
        o = ADAPT(10, "sma", go_under=True)
        result = o._execute(broker_data, tick_with_sma)
        assert result["return"] is False

    def test_sell_goes_above(self, broker_data, current_tick):
        broker_data.shares = 20
        tick_with_sma = dict(current_tick, sma=96.0)
        o = ADAPT(-10, "sma", go_under=False)
        result = o._execute(broker_data, tick_with_sma)
        assert result["return"] is True
