import pytest


class TestStatistics:
    def test_init(self, data_dir, mock_strategy):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        from Engine.Backtester.Statistics import Statistics

        ex = Exchange(data_dir, limit=2, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        stats = Statistics(broker)
        assert stats.initial_cash == 10000
        assert stats.total_files == 2

    def test_states_df_structure(self, data_dir, mock_strategy):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        from Engine.Backtester.Statistics import Statistics

        ex = Exchange(data_dir, limit=2, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        stats = Statistics(broker)
        assert "initial" in stats.states_df.columns
        assert "end" in stats.states_df.columns
        assert "net" in stats.states_df.columns

    def test_get_statistics(self, data_dir, mock_strategy):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        from Engine.Backtester.Statistics import Statistics

        ex = Exchange(data_dir, limit=2, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        stats = Statistics(broker)
        result = stats.get_statistics()
        assert "%mean" in result
        assert "sortino" in result
        assert isinstance(result["%mean"], float)

    def test_get_outliers_single_file(self, data_dir, mock_strategy):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        from Engine.Backtester.Statistics import Statistics

        ex = Exchange(data_dir, limit=1, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        stats = Statistics(broker)
        outliers = stats.get_outliers()
        assert outliers == []

    def test_get_outliers_multiple_files(self, data_dir, mock_strategy):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        from Engine.Backtester.Statistics import Statistics

        ex = Exchange(data_dir, limit=4, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        stats = Statistics(broker)
        outliers = stats.get_outliers()
        assert isinstance(outliers, list)

    def test_get_max_drawdowns(self, data_dir, mock_strategy):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        from Engine.Backtester.Statistics import Statistics

        ex = Exchange(data_dir, limit=3, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        stats = Statistics(broker)
        dd = stats.get_max_drawdowns()
        assert "mdd" in dd
        assert "mddl" in dd
        assert isinstance(dd["mdd"], float)

    def test_cash_graph_no_error(self, data_dir, mock_strategy):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        from Engine.Backtester.Statistics import Statistics

        ex = Exchange(data_dir, limit=1, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        stats = Statistics(broker)

        import matplotlib
        matplotlib.use("Agg")

        try:
            stats.cash_graph()
            stats.cash_graph(scale="log")
        except Exception as e:
            pytest.fail(f"cash_graph raised exception: {e}")

    def test_risk_free_rate(self, data_dir, mock_strategy):
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange
        from Engine.Backtester.Statistics import Statistics

        ex = Exchange(data_dir, limit=2, lazy_loading=False)
        broker = Broker(cash=10000, algorithm=mock_strategy,
                        trade_cost=2, exchange_feed=ex)
        broker.simulate()
        stats = Statistics(broker, risk_free_rate=0.05)
        result = stats.get_statistics()
        assert "sortino" in result
