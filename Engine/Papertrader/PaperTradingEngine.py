import time

from Engine.Core.Strategy import Strategy
from Engine.Papertrader.PaperDataFeed import PaperDataFeed
from Engine.Papertrader.PaperExecutionHandler import PaperExecutionHandler


class PaperTradingEngine:
    def __init__(self, strategy: Strategy, symbols, cash=0, host="127.0.0.1", port=7497, client_id=1):
        self.strategy = strategy
        self.symbols = symbols
        self.initial_cash = cash
        self.available_cash = cash
        self.shares = 0
        self.initial_day_balance = cash

        self.feed = PaperDataFeed(host=host, port=port, client_id=client_id)
        self.executor = PaperExecutionHandler(cash=cash, host=host, port=port, client_id=client_id + 1)

        self.states = []
        self.count = 0

    def run(self):
        for info in self.feed.generator(self.symbols):
            self.count += 1
            current_tick = info["current_tick"]
            price = current_tick["close"]

            orders = self.strategy.on_bar(
                current_tick=current_tick,
                price=price,
                initial_day_balance=self.initial_day_balance,
                balance=self.available_cash,
                bought_market_value=0,
                market_value=self.shares * price,
                shares=self.shares,
                new_stock=info["new_file"],
            )

            if orders:
                self.executor.submit_orders(orders, symbol=info["filepath"])

            self.executor.process_fills()
            self.shares = self.executor.shares
            self.available_cash = self.executor.available_cash

    def get_results(self):
        return {
            "initial cash": round(self.initial_cash, 2),
            "final cash": round(self.available_cash, 2),
            "bars processed": self.count,
        }

    def disconnect(self):
        self.executor.disconnect()
        self.feed.disconnect()
