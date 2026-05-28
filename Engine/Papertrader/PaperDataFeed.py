import sys
import threading
import time

from ibapi.client import EClient
from ibapi.common import TickerId
from ibapi.wrapper import EWrapper, Contract


class _IBapi(EWrapper, EClient):
    def __init__(self, feed):
        EClient.__init__(self, self)
        self.feed = feed

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        print(f"[IBKR ERROR] {reqId} {errorCode} {errorString}")
        if errorCode == 502:
            print("TWS is not connecting")
            sys.exit()
        if errorCode in [200, 162]:
            self.feed._cancel(reqId)

    def connectAck(self):
        self.feed.connected_event.set()

    def nextValidId(self, orderId):
        self.feed.valid_id = orderId
        self.feed.valid_id_event.set()

    def realtimeBar(self, reqId, time_, open_, high, low, close, volume, wap, count):
        self.feed._on_bar(reqId, time_, open_, high, low, close, volume)

    def historicalData(self, reqId, bar):
        self.feed._on_bar(reqId, bar.date, bar.open, bar.high, bar.low, bar.close, int(bar.volume))

    def historicalDataEnd(self, reqId, start, end):
        self.feed._historical_done(reqId)

    def connectionClosed(self):
        print("IBKR connection closed")


class PaperDataFeed:
    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id

        self.connected_event = threading.Event()
        self.valid_id_event = threading.Event()
        self.valid_id = -1

        self._bar_queue = {}
        self._req_barrier = {}

        self.ib = _IBapi(self)
        self._thread = threading.Thread(target=self.ib.run, daemon=True)
        self._thread.start()

        if not self._connect():
            self.ib.disconnect()
            raise ConnectionError("Failed to connect to TWS")

    def _connect(self, retry=10):
        for i in range(retry):
            self.connected_event.clear()
            self.ib.connect(self.host, self.port, self.client_id)
            if self.connected_event.wait(timeout=2):
                self.valid_id_event.wait(timeout=2)
                return True
            print(f"Establishing connection {i + 1}/{retry}")
        return False

    def _cancel(self, req_id):
        if req_id in self._req_barrier:
            self._req_barrier[req_id].set()

    def _on_bar(self, req_id, time_, open_, high, low, close, volume):
        tick = {
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": int(volume),
            "time": time_,
            "req_id": req_id,
        }
        symbol = self._symbols_by_req.get(req_id, f"req_{req_id}")
        self._bar_queue.setdefault(symbol, []).append(tick)

    def _historical_done(self, req_id):
        if req_id in self._req_barrier:
            self._req_barrier[req_id].set()

    def subscribe(self, symbol: str, exchange="SMART", bar_size=5):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = "USD"

        req_id = self.valid_id
        self.valid_id += 1

        self._symbols_by_req[req_id] = symbol
        self._req_barrier[req_id] = threading.Event()

        self.ib.reqRealTimeBars(req_id, contract, bar_size, "TRADES", 0, [])
        return req_id

    def generator(self, symbols, warmup_bars=390):
        self._bar_queue.clear()
        self._req_barrier = {}
        self._symbols_by_req = {}

        for symbol in symbols:
            self.subscribe(symbol)

        step = 0
        try:
            while True:
                all_done = all(
                    len(self._bar_queue.get(s, [])) > 0 for s in symbols
                )
                if not all_done:
                    time.sleep(0.05)
                    continue

                for s in symbols:
                    tick = self._bar_queue[s].pop(0)
                    yield {
                        "current_tick": tick,
                        "filepath": s,
                        "step": step,
                        "new_file": step == 0,
                        "end_file": False,
                        "total_steps": None,
                        "progress": (step + 1, "live"),
                    }
                    step += 1
        except KeyboardInterrupt:
            pass
        finally:
            self.disconnect()

    def disconnect(self):
        for req_id in self._symbols_by_req:
            self.ib.cancelRealTimeBars(req_id)
        self.ib.disconnect()
