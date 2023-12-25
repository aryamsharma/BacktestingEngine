import time
from Engine.Orders import MKT

class Message:
    """
    Quick way to print out warnings/information
    """
    def __init__(self):
        """
        warning lookup
        """
        self.lookup_table = {
            "0": "SUCCESS",
            "1": "NO CASH",
            "2": "WRONG ORDER INPUTS",
            "3": "LONG SHARES EOD",
            "4": "SHORT SHARES EOD"
        }

    def warning(self, condition, error_code, extra_info=""):
        if condition:
            print(f"[WARNING] {self.lookup_table[str(error_code)]} {extra_info}")


    def info(self, condition, msg, extra_info=[]):
        if condition:
            print(f"[INFO] {msg}")
            for info in extra_info:
                print(info)


class Broker:
    def __init__(
            self, cash, algorithm, trade_cost, exchange_feed,
            warnings=False, info=False, progress=False):
        """
        Parameter(s)
        :cash (float)
            How much money should you start off with
        :algorithm (algorithm object)
            Algorithm object (Client.py/ipynb)
        :trade_cost (float)
            Cost per trade
        :exchange feed (float)
            Cost per trade
        :warnings (bool)
            Should warnings be printed
        :info (bool)
            Should extra info be printed
        :progress (bool)
            Simple progress counter
        """

        self.warnings = warnings
        self.info = info
        self.checking = not info or not warnings
        self.progress = progress

        self.initial_cash = cash
        self.available_cash = cash
        self.trade_cost = trade_cost

        self.algorithm = algorithm
        self.exchange_feed = exchange_feed

        self.iter = self.exchange_feed.setup_data(algorithm=self.algorithm)


    def _order_checks(self, current_tick):
        """
        Parameter(s)
        :current tick (dict)
            Current bar

        Output(s)
        :shares (int)
            Amount of shares left
        """
        to_delete = []
        for count, order in enumerate(self.orders):
            if order.requested_shares == 0:
                self.alerts.warning(self.warnings, 2)
                to_delete.append(count)
                continue

            values = order._execute(self, current_tick)

            if values["return"]:
                self.alerts.info(self.info, f"{order.order_type} Order filled ({values['shares']} / {values['price']})")
                self.bought_market_value = values["shares"] * values["price"]
                self.trades.append((values["shares"], values["price"], current_tick["step"]))

            if values.get("code", None) is not None:
                self.alerts.warning(
                    self.warnings and not values["return"], values["code"])

            if order.order_type == "MKT":
                values["return"] = True

            if values["return"]:
                to_delete.append(count)

        for i in to_delete[::-1]:
            del self.orders[i]

        return self.shares


    def reset_vars(self):
        """
        Output(s)
        :flag (bool)
            Return value
        :code (int)
            Error code
        """
        tmp = self.initial_day_balance
        self.initial_day_balance = self.available_cash
        self.bought_market_value = 0
        self.orders = []
        flag, code = True, 0

        if self.shares != 0:
            self.alerts.warning(self.warnings, 3, f"{self.shares} Share(s)")
            self.orders = [MKT(-self.shares)]
            self._order_checks(self.last_info["current_tick"])

            if self.shares > 0:
                flag, code = False, 3
            elif self.shares < 0:
                flag, code = False, 4
            else:
                flag, code = True, 0

        self.alerts.info(self.info, f"{self.last_info['filepath']} end")
        self.states.append((tmp, self.available_cash, (self.last_info['filepath'], *self.trades)))
        self.trades = []
        self.initial_day_balance = self.available_cash
        return flag, code


    def simulate(self):
        """
        Run the simulation
        """
        self.last_info = {}
        self.count = 0
        self.shares = 0
        self.bought_market_value = 0
        self.initial_day_balance = self.available_cash
        self.orders = []
        start = time.time()
        self.alerts = Message()
        self.states = []
        self.trades = []

        for info in self.iter():
            if info["new_file"] and self.progress:
                print(f"{info['progress'][0]}/{info['progress'][1]}", end="\r")

            self.count += 1

            if info["end_file"]:
                ret, code = self.reset_vars()
                if not ret:
                    self.alerts.warning(self.warnings, code, "(FAIL COND)")
                    break
                continue

            current_tick = info["current_tick"]
            open = current_tick["close"]

            current_order = self.algorithm.send_algo(
                df=info["feed"],
                current_tick=current_tick,
                price=open,
                initial_day_balance=self.initial_day_balance,
                balance=self.available_cash,
                bought_market_value=self.bought_market_value,
                market_value=self.shares * open,
                shares=self.shares,
                new_stock=info["new_file"])

            if current_order is not None:
                self.orders.extend(current_order)

            self.shares = self._order_checks(info["current_tick_true"])
            self.last_info = info

        self.total_time_taken = time.time() - start


    def get_results(self):
        """
        Output(s)
        :info (dict of float)
            Information
        """
        return {
            "initial cash": round(self.initial_cash, 2),
            "final cash": round(self.available_cash, 2),
            "time taken": round(self.total_time_taken, 2),
            "candles/second": round(self.count / self.total_time_taken, 2)
        }
