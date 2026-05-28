import sys
import time
import threading

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from ibapi.client import EClient
from ibapi.common import TickerId
from ibapi.wrapper import EWrapper, Contract


# YYYYMMDD
DATE = "20240712"

class IBapi(EWrapper, EClient):
    def __init__(self, interface):
        EClient.__init__(self, self)
        self.interface = interface

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        print(reqId, errorCode, errorString, advancedOrderRejectJson)
        if errorCode == 502:
            print("TWS is not connecting")
            sys.exit()

        if errorCode in [200, 162]:
            print(f"Cancelled {reqId}", end="\r")
            return

    def connectAck(self):
        self.interface.connected = True

    def nextValidId(self, orderId):
        self.interface.returned = True
        self.interface.validID = orderId
        print(f"next valid id {orderId}")
    
    def realtimeBar(self, reqId, time, open, high, low, close, volume, wap, count):
        symbol = self.interface.reference_sheet[reqId]

        if not self.interface.market_data.setdefault(symbol, False):
            self.interface.market_data[symbol] = []

        self.interface.market_data[symbol] += [[time, open, high, low, close, int(volume)]]

    def connectionClosed(self):
        print("Connection closed")


class TradingInterface:
    def __init__(self):
        self.connected = False
        self.validID = -1
        self.market_data = {}
        self.reference_sheet = {}

        self.ib = IBapi(self)
        ib_thread = threading.Thread(target=self.run_api, daemon=True)

        if not self._establish_connection():
            self.ib.disconnect()
            exit()

        ib_thread.start()

    def run_api(self):
        self.ib.run()

    def _establish_connection(self, retry=10):
        for i in range(retry):
            self.ib.connect('127.0.0.1', 7497, 1)
            if self.connected:
                time.sleep(2)
                return True
            print(f"Establishing connection {i + 1}/{retry}")
            time.sleep(i + 1)

        return False
    
    def _get_new_validID(self, retry=10):
        self.returned = False
        self.ib.reqIds(-1)

        for i in range(retry):
            if self.returned:
                break
            print(f"{i + 1}/{retry}")
            time.sleep(i * 0.05)
        
        return self.validID

    def disconnect(self):
        self.ib.disconnect()

    def start_market_line(self, symbol="SPY"):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        id = self._get_new_validID()
        self.reference_sheet[id] = symbol

        self.ib.reqRealTimeBars(
            id,
            contract=contract,
            barSize=5,
            whatToShow="TRADES",
            useRTH=0,
            realTimeBarsOptions=[]
            )
    
    def end_realtime_bars(self):
        for key in self.reference_sheet.keys():
            print(f"Ending {self.reference_sheet[key]} data line")
            self.ib.cancelRealTimeBars(key)

interface = TradingInterface()

try:
    interface.start_market_line("SPY")
    time.sleep(5)
    interface.disconnect()

except Exception as e:
    print("--------------------")
    print("Error")
    print(e)
    interface.disconnect()