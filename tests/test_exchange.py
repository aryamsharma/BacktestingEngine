import os
import pytest


class TestExchange:
    def test_init_requires_directory(self):
        from Engine.Backtester.Exchange import Exchange
        with pytest.raises(AssertionError):
            Exchange("/nonexistent/path")

    def test_init_loads_csv_files(self, data_dir):
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir)
        assert len(ex.files) > 0
        assert all(f.endswith(".csv") for f in ex.files)

    def test_setup_data_calls_algorithm_setup(self, data_dir, mock_strategy):
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1)
        result = ex.setup_data(algorithm=mock_strategy)
        assert callable(result)

    def test_preprocessed_generator_length(self, data_dir, mock_strategy_noop):
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1, lazy_loading=False)
        gen = ex.setup_data(algorithm=mock_strategy_noop)
        data = gen()
        assert len(data) > 0

    def test_generator_yields_correct_keys(self, data_dir, mock_strategy_noop):
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1, lazy_loading=False)
        gen = ex.setup_data(algorithm=mock_strategy_noop)
        info_list = gen()
        info = info_list[0]
        assert "current_tick" in info
        assert "current_tick_true" in info
        assert "filepath" in info
        assert "step" in info
        assert "new_file" in info
        assert "end_file" in info
        assert "total_steps" in info
        assert "progress" in info

    def test_generator_slippage_offset(self, data_dir, mock_strategy_noop):
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1, slippage=2, lazy_loading=False)
        gen = ex.setup_data(algorithm=mock_strategy_noop)
        info_list = gen()
        for info in info_list:
            assert info["current_tick_true"]["step"] == info["current_tick"]["step"] + 2

    def test_generator_progress_tracking(self, data_dir, mock_strategy_noop):
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=2, lazy_loading=False)
        gen = ex.setup_data(algorithm=mock_strategy_noop)
        info_list = gen()
        seen_progress = set()
        for info in info_list:
            seen_progress.add(info["progress"])
        assert len(seen_progress) == 2

    def test_slippage_minimum_one(self, data_dir, mock_strategy_noop):
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=1, slippage=0, lazy_loading=False)
        assert ex.slippage == 1

    def test_sorting(self, data_dir, mock_strategy_noop):
        from Engine.Backtester.Exchange import Exchange
        ex = Exchange(data_dir, limit=5, sorting=True, lazy_loading=False)
        gen = ex.setup_data(algorithm=mock_strategy_noop)
        info_list = gen()
        prev_date = None
        for info in info_list:
            if info["new_file"]:
                date_str = os.path.basename(info["filepath"]).split("_")[-1].replace(".csv", "")
                if prev_date is not None:
                    assert date_str >= prev_date
                prev_date = date_str
