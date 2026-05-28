# BacktestingEngine

A unified backtesting and paper-trading engine for stock market strategies. Switch between modes by changing a single constant.

## Quick Start

```python
from Engine.Core.EngineFactory import create, Mode
from Engine.Core.Strategy import Strategy

class MyStrategy(Strategy):
    def setup(self, df):
        df["sma"] = df["close"].rolling(20).mean()
        return df

    def on_bar(self, **kwargs):
        # kwargs: current_tick, price, balance, shares, ...
        return None  # or a list of Order objects

engine = create(Mode.BACKTEST, MyStrategy(), {
    "data_path": "Data/intraday_trading",
    "cash": 10000,
    "trade_cost": 2,
    "limit": 10,
})
engine.run()
print(engine.get_results())
```

## Modes

| Mode | Class | Description |
|------|-------|-------------|
| `Mode.BACKTEST` | `Broker` | Historical simulation via CSV data |
| `Mode.PAPER` | `PaperTradingEngine` | Live paper trading via IBKR TWS API |

Switch modes without changing strategy code:

```python
engine = create(Mode.PAPER, MyStrategy(), {
    "symbols": ["SPY", "AAPL"],
    "cash": 10000,
    "host": "127.0.0.1",
    "port": 7497,
})
engine.run()
```

## Writing a Strategy

Inherit from `Strategy` (in `Engine/Core/Strategy.py`) and implement:

- **`setup(self, df)`** — add indicators/columns to the DataFrame before the run
- **`on_bar(self, **kwargs)`** — called on every bar; return an `Order` or list of `Order`s, or `None`
- **`on_start_of_day(self)`** — (optional) called at the start of each trading day
- **`on_end_of_day(self)`** — (optional) called at the end of each trading day

### Using the Notebook

Open `Backtester.ipynb` and write strategies in the `Algos` class. Numbered methods (`setup1`, `algo1`, `setup2`, `algo2`, ...) are dispatched automatically:

```python
class Algos(Strategy):
    def setup(self, df):
        return self.__setup_func(df)

    def on_bar(self, **kwargs):
        return self.__on_bar_func(**kwargs)

    def setup1(self, df):
        df["sma"] = ta.sma(df["close"], 20)
        return df

    def algo1(self, **var):
        # Golden Cross logic
        ...
        return [buy_order, sell_order]
```

## Order Types

Defined in `Engine/Backtester/Orders.py`:

| Class | Description |
|-------|-------------|
| `MKT(shares)` | Market order, fills immediately |
| `LMT(shares, price)` | Limit order |
| `TRAIL_LMT(shares, percentage_delta\|price_delta)` | Trailing stop |
| `ADAPT(shares, following, go_under)` | Adaptive to a column value |

## Running Tests

```bash
python -m pytest tests/
```

## Requirements

Python >= 3.7 + packages in `requirements.txt` (pandas, numpy, matplotlib, yfinance, ibapi).

## License

GNU GPLv3
