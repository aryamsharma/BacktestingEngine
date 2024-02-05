"""
These classes are logic for different types of orders, once you've made
a new order type with the correct logic you are free to use them in
your Client.py/ipynb algorithms class
"""

class MKT:
    """
    Classic market order
    """
    def __init__(self, shares):
        self.order_type = "MKT"
        assert type(shares) is int, "Shares is a non-integer value"
        self.requested_shares = shares

    def _execute(self, broker_data, current_tick):

        buying = self.requested_shares > 0

        # The punishment for ever using a MKT because these are risky even though they are faster
        shares_cost = current_tick["high"] * self.requested_shares
        shares_cost += broker_data.trade_cost

        if buying and shares_cost > broker_data.available_cash:
            return {"return": False, "code": 1}

        broker_data.available_cash -= shares_cost
        broker_data.shares += self.requested_shares

        return {
            "return": True,
            "shares": self.requested_shares,
            "price": (shares_cost - broker_data.trade_cost) / self.requested_shares}

class LMT:
    """
    Classic LMT order
    """
    def __init__(self, shares, price):
        self.order_type = "LMT"
        assert type(shares) is int, "Shares is a non-integer value"
        self.requested_shares = shares
        self.requested_price = price

    def _execute(self, broker_data, current_tick):
        buying = self.requested_shares > 0

        if self.requested_price is None:
            return {"return": False, "code": 2}

        close_price = current_tick["close"]
        shares_cost = close_price * self.requested_shares
        shares_cost += broker_data.trade_cost

        if buying and close_price <= self.requested_price:
            broker_data.available_cash -= shares_cost
            broker_data.shares += self.requested_shares
            return {
                "return": True,
                "shares": self.requested_shares,
                "price": close_price}

        if not buying and close_price >= self.requested_price:
            broker_data.available_cash += shares_cost
            broker_data.shares += self.requested_shares
            return {
                "return": True,
                "shares": self.requested_shares,
                "price": close_price}

        return {"return": False}

class TRAIL_LMT:
    """
    Trailing limit order, you can give a percentage delta or price delta
    """
    def __init__(self, shares, percentage_delta=None, price_delta=None):
        self.order_type = "TRAIL_STOP"
        assert type(shares) is int, "Shares is a non-integer value"
        self.requested_shares = shares

        if percentage_delta is not None:
            self.trail_delta = lambda x: x * percentage_delta
        else:
            self.trail_delta = lambda x: x + price_delta

        self.old_trail_price = None

    def _execute(self, broker_data, current_tick):
        buying = self.requested_shares > 0
        trail_price = self.trail_delta(current_tick["close"])

        if self.old_trail_price is None:
            self.old_trail_price = trail_price
            return {"return": False}

        if not buying:
            if current_tick["close"] < self.old_trail_price:
                mkt_order = MKT(self.requested_shares)
                return mkt_order._execute(broker_data, current_tick)

            if trail_price > self.old_trail_price:
                self.old_trail_price = trail_price

        else:
            if current_tick["close"] > self.old_trail_price:
                mkt_order = MKT(self.requested_shares)
                return mkt_order._execute(broker_data, current_tick)

            if trail_price < self.old_trail_price:
                self.old_trail_price = trail_price

        return {"return": False}

class ADAPT:
    """
    Following any columns provided in the df that you have setup
    """
    def __init__(self, shares, following, go_under=True):
        self.order_type = "ADAPT"
        assert type(shares) is int, "Shares is a non-integer value"
        self.requested_shares = shares
        self.following = following
        # go is under or above
        self.go_under = go_under

    def _execute(self, broker_data, current_tick):
        price = current_tick["close"]
        follow_price = current_tick[self.following]

        if self.go_under and price < follow_price:
            mkt = MKT(self.requested_shares)
            return mkt._execute(broker_data, current_tick)

        if not self.go_under and price > follow_price:
            mkt = MKT(self.requested_shares)
            return mkt._execute(broker_data, current_tick)

        return {"return": False}

class example:
    def __init__(self):
        pass

    def _execute(self, broker_data, current_tick):
        # the return fields are always either True or False
        return {"return": False}
        # if the return is False you may add an extra field called code (look in the lookup table to see error types)
        return {"return": False, "code": int}
        # if the return is True then the fields for shares and price bought must be there and be calculated
        return {"return": True, "shares": int, "price": float}
