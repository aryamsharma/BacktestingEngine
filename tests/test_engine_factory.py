import pytest


class TestEngineFactory:
    def test_create_backtest(self, data_dir, mock_strategy):
        from Engine.Core.EngineFactory import create, Mode

        engine = create(Mode.BACKTEST, mock_strategy, {
            "data_path": data_dir,
            "cash": 10000,
            "trade_cost": 2,
            "limit": 1,
            "lazy_loading": False,
        })
        from Engine.Backtester.Broker import Broker
        assert isinstance(engine, Broker)

    def test_create_backtest_runs(self, data_dir, mock_strategy):
        from Engine.Core.EngineFactory import create, Mode

        engine = create(Mode.BACKTEST, mock_strategy, {
            "data_path": data_dir,
            "cash": 10000,
            "trade_cost": 2,
            "limit": 1,
            "lazy_loading": False,
        })
        engine.run()
        result = engine.get_results()
        assert result["initial cash"] == 10000.0

    def test_create_paper_raises_without_connection(self, mock_strategy):
        from Engine.Core.EngineFactory import create, Mode

        with pytest.raises((ConnectionError, OSError, ImportError)):
            create(Mode.PAPER, mock_strategy, {
                "symbols": ["SPY"],
                "cash": 10000,
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1,
            })

    def test_invalid_mode(self, mock_strategy):
        from Engine.Core.EngineFactory import create, Mode

        with pytest.raises(ValueError):
            create("invalid", mock_strategy, {})

    def test_backtest_with_sorting(self, data_dir, mock_strategy_noop):
        from Engine.Core.EngineFactory import create, Mode

        engine = create(Mode.BACKTEST, mock_strategy_noop, {
            "data_path": data_dir,
            "cash": 50000,
            "trade_cost": 0,
            "limit": 3,
            "sorting": True,
            "lazy_loading": False,
        })
        engine.run()
        assert engine.initial_cash == 50000

    def test_backtest_lazy_loading(self, data_dir, mock_strategy_noop):
        from Engine.Core.EngineFactory import create, Mode

        engine = create(Mode.BACKTEST, mock_strategy_noop, {
            "data_path": data_dir,
            "cash": 10000,
            "limit": 1,
            "lazy_loading": True,
        })
        assert callable(engine.iter)
