import sys
import threading
import time

from ibapi.client import EClient
from ibapi.common import TickerId
from ibapi.wrapper import EWrapper, Contract


class _ExecIBapi(EWrapper, EClient):
    def __init__(self, handler):
        EClient.__init__(self, self)
        self.handler = handler

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        print(f"[IBKR ORDER ERROR] {reqId} {errorCode} {errorString}")
        if errorCode == 502:
            sys.exit()

    def connectAck(self):
        self.handler.connected_event.set()

    def nextValidId(self, orderId):
        self.handler.valid_id = orderId
        self.handler.valid_id_event.set()

    def openOrder(self, orderId, contract, order, orderState):
        self.handler.open_orders[orderId] = {
            "symbol": contract.symbol,
            "shares": order.totalQuantity * (1 if order.action == "BUY" else -1),
            "status": orderState.status,
        }

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        if status == "Filled":
            order_info = self.handler.open_orders.pop(orderId, {})
            symbol = order_info.get("symbol", "?")
            shares = order_info.get("shares", 0)
            self.handler.fills.append({
                "symbol": symbol,
                "shares": shares,
                "price": round(avgFillPrice, 2),
                "time": time.time(),
            })


class PaperExecutionHandler:
    def __init__(self, cash=0, host="127.0.0.1", port=7497, client_id=2):
        self.initial_cash = cash
        self.available_cash = cash
        self.shares = 0

        self.connected_event = threading.Event()
        self.valid_id_event = threading.Event()
        self.valid_id = -1
        self.open_orders = {}
        self.fills = []
        self._pending_orders = []

        self.ib = _ExecIBapi(self)
        self._thread = threading.Thread(target=self.ib.run, daemon=True)
        self._thread.start()

        if not self._connect():
            self.ib.disconnect()
            raise ConnectionError("Failed to connect to TWS for order execution")

    def _connect(self, retry=10):
        for i in range(retry):
            self.connected_event.clear()
            self.ib.connect("127.0.0.1", 7497, 2)
            if self.connected_event.wait(timeout=2):
                self.valid_id_event.wait(timeout=2)
                return True
            print(f"Establishing execution connection {i + 1}/{retry}")
        return False

    def submit_orders(self, orders, symbol="SPY", exchange="SMART"):
        for order in orders:
            if order.requested_shares == 0:
                continue

            action = "BUY" if order.requested_shares > 0 else "SELL"
            abs_shares = abs(order.requested_shares)

            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = exchange
            contract.currency = "USD"

            ib_order = type("IBOrder", (), {})()
            ib_order.action = action
            ib_order.totalQuantity = abs_shares
            ib_order.orderType = "MKT"
            ib_order.tif = "DAY"

            if hasattr(order, "requested_price") and order.requested_price is not None:
                ib_order.orderType = "LMT"
                ib_order.lmtPrice = order.requested_price

            self.ib.placeOrder(self.valid_id, contract, ib_order)
            self.valid_id += 1

    def process_fills(self):
        for fill in self.fills:
            self.shares += fill["shares"]
            self.available_cash -= fill["shares"] * fill["price"] + 2
        self.fills.clear()

    def disconnect(self):
        self.ib.disconnect()
