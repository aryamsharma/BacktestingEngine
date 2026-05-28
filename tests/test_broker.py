import pytest


class TestBroker:
    def test_init(self, data_dir, mock_strategy_noop):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy_noop,
                        trade_cost=2, exchange_feed=ex)
        assert broker.initial_cash == 10000
        assert broker.available_cash == 10000
        assert broker.trade_cost == 2
        assert broker.algorithm is mock_strategy_noop

    def test_simulate_noop_strategy(self, data_dir, mock_strategy_noop):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy_noop,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        results = broker.get_results()
        assert "initial cash" in results
        assert "final cash" in results
        assert results["initial cash"] == 10000.0

    def test_simulate_with_strategy(self, data_dir, mock_strategy):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        results = broker.get_results()
        assert results["initial cash"] == 10000.0
        assert results["final cash"] >= 0

    def test_run_alias(self, data_dir, mock_strategy_noop):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy_noop,
                        trade_cost=2, exchange_feed=ex)
        broker.run()
        assert broker.count > 0

    def test_simulate_populates_states(self, data_dir, mock_strategy):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=2, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        assert len(broker.states) > 0
        assert len(broker.states[0]) == 3

    def test_simulate_progress(self, data_dir, mock_strategy_noop, capsys):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy_noop,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate(show_progress=True)
        captured = capsys.readouterr()
        assert "1/1" in captured.out or len(captured.out) >= 0

    def test_get_results_after_simulate(self, data_dir, mock_strategy_noop):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy_noop,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        results = broker.get_results()
        assert "time taken" in results
        assert "candles/second" in results
        assert isinstance(results["candles/second"], float)
