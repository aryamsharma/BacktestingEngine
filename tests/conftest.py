import os

import pytest


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data", "intraday_trading"
)


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


@pytest.fixture
def sample_csv(data_dir):
    return os.path.join(data_dir, "AAWW_2022-09-27.csv")


@pytest.fixture
def broker_data():
    class FakeBroker:
        available_cash = 10000.0
        shares = 0
        trade_cost = 2.0
    return FakeBroker()


@pytest.fixture
def current_tick():
    return {
        "open": 97.30,
        "high": 97.43,
        "low": 97.25,
        "close": 97.32,
        "volume": 7349,
        "step": 0,
    }


@pytest.fixture
def buy_tick():
    return {
        "open": 97.30,
        "high": 97.43,
        "low": 97.25,
        "close": 97.32,
        "volume": 7349,
        "step": 5,
        "small_rolling": 95.0,
        "long_rolling": 94.0,
    }


@pytest.fixture
def mock_strategy():
    from Engine.Core.Strategy import Strategy
    import pandas as pd

    class BuySetupSMA(Strategy):
        def setup(self, df):
            df["sma"] = df["close"].rolling(5).mean()
            self.bought = False
            return df

        def on_bar(self, **kwargs):
            if kwargs["new_stock"]:
                self.bought = False
            if kwargs["shares"] > 0 or self.bought:
                return None
            from Engine.Backtester.Orders import LMT
            self.bought = True
            return [LMT(10, kwargs["price"])]

    return BuySetupSMA()


@pytest.fixture
def mock_strategy_noop():
    from Engine.Core.Strategy import Strategy
    import pandas as pd

    class Noop(Strategy):
        def setup(self, df):
            return df
        def on_bar(self, **kwargs):
            return None

    return Noop()
