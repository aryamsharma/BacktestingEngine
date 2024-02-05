import os
import sys
import threading
import time
import datetime

import pandas as pd
from ibapi.client import EClient
from ibapi.common import TickerId
from ibapi.wrapper import EWrapper, Contract

class DownloadTickerData:
    def daily(tickers, filepath, batch_size=25):
        """
        tickers
        filepath
        threshold basically is asking; how many days ago do you want it to check last,
        """
        current_date = datetime.datetime.today().date()

        if not os.path.isfile(filepath + "A.csv"):
            # 21 days is just an arbitrary number
            last_date = (datetime.datetime.now() - datetime.timedelta(21)).date()

        else:
            last_date = (datetime.datetime.now() - datetime.timedelta(21)).date()

            # sample_file = open(filepath + "A.csv", "r+")
            # last_date = sample_file.readlines()[-1].split(",")[0]
            # last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()

        delta_days = f"{(current_date - last_date).days} D"

        columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        dfs = {}

        year = last_date.strftime("%Y")
        month = last_date.strftime("%m")
        day = last_date.strftime("%d")
        end_date = f"{year}{month}{day} 16:00:00 US/Eastern"

        app = TradingInterface()
        time.sleep(1)

        data = app.req_data(
            tickers, end_date, delta_days, "1 day", batch_size=batch_size)

        app.disconnect()

        for ticker_data in data.items():
            df = pd.DataFrame(ticker_data[1], columns=columns)
            df.index = pd.to_datetime(
                df.set_index("Date").index, format="%Y%m%d")

            dfs[f"{ticker_data[0]}"] = df.drop(columns=["Date"])

        return dfs


    def minute(data: dict, batch_size=25):
        columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        dfs = {}

        app = TradingInterface()
        time.sleep(1)

        for pair in list(data.items()):
            end_date = datetime.datetime.strptime(pair[0], "%Y-%m-%d")
            end_date_key = end_date.strftime("%Y-%m-%d")
            end_date_req = f"{end_date.strftime('%Y%m%d')} 16:00:00 US/Eastern"


            pairs = app.req_data(
                pair[1], end_date_req, "1 D", "1 min", batch_size=batch_size)

            for ticker_data in pairs.items():
                df = pd.DataFrame(ticker_data[1], columns=columns)
                df["Date"] = df["Date"].str[:-11]

                df.index = pd.to_datetime(
                    df.set_index("Date").index, format="%Y%m%d %H:%M:%S")

                df = df.drop(columns=["Date"])

                dfs[f"{ticker_data[0]} {end_date_key}"] = df

        app.disconnect()
        return dfs

class Util:
    def chunks(elements, n):
        for i in range(0, len(elements), n):
            yield elements[i:i + n]


    def write_daily_data(dfs, filepath):
        for pair in dfs.items():
            if not os.path.exists(f"{filepath}{pair[0]}.csv"):
                pair[1].to_csv(f"{filepath}{pair[0]}.csv")
                continue

            df = pd.read_csv(f"{filepath}{pair[0]}.csv", index_col=0)
            df.index = pd.to_datetime(df.index)

            new_df = pair[1]
            new_df.set_index("Date")
            new_df.index = pd.to_datetime(new_df.index)
            new_df = new_df.drop(columns=["Date"])

            final = pd.concat([df, new_df])
            final = final[~final.index.duplicated()]
            final = final.sort_index()

            final.to_csv(f"{filepath}{pair[0]}.csv")

    def generate_stock_contract(symbol, exchange="NYSE"):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "NYSE"
        contract.currency = "USD"
        return contract



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
            self.interface.ids_active.remove(reqId)
            return


    def connectAck(self):
        self.interface.connected = True


    def nextValidId(self, orderId):
        self.interface.validID = orderId
        print(f"next valid id {orderId}")


    def historicalData(self, reqId, bar):
        if self.interface.data.setdefault(reqId, None) is None:
            self.interface.data[reqId] = []

        self.interface.data[reqId] += [
            [bar.date, bar.open, bar.high,
                bar.low, bar.close, int(bar.volume)]]


    def historicalDataEnd(self, reqId, start, end):
        print(f"Done {reqId}", end="\r")
        self.interface.ids_active.remove(reqId)
        # print(f"{reqId} Start: {start}, End: {end}")


class TradingInterface:
    def __init__(self):
        self.connected = False
        self.validID = -1

        self.ib = IBapi(self)
        ib_thread = threading.Thread(target=self.run_api, daemon=True)

        if not self.establish_connection():
            self.ib.disconnect()
            exit()

        ib_thread.start()


    def run_api(self):
        self.ib.run()


    def establish_connection(self, retry=10):
        for i in range(retry):
            self.ib.connect('127.0.0.1', 7497, 1)
            if self.connected:
                time.sleep(2)
                return True
            print(f"Establishing connection {i + 1}/{retry}")
            time.sleep(i + 1)

        return False


    def req_data(
            self, tickers, ending_date,
            delta_days, granularity, batch_size=25):

        self.data = {}
        self.ids_active = []
        id_ticker_pairs = {}

        for ticker_chunk in Util.chunks(tickers, batch_size):
            print("Downloading",  " ".join(ticker_chunk))
            for ticker in ticker_chunk:

                self.ib.reqHistoricalData(
                    self.validID, Util.generate_stock_contract(ticker),
                    ending_date, delta_days, granularity,
                    "TRADES", 1, 1, 0, [])

                self.ids_active.append(self.validID)
                id_ticker_pairs[self.validID] = ticker
                self.validID += 1

            while self.ids_active:
                time.sleep(0.05)

        for pair in id_ticker_pairs.items():
            if self.data.setdefault(pair[0], None) is None:
                del self.data[pair[0]]
                continue

            self.data[pair[1]] = self.data.pop(pair[0])
        print()
        return self.data


    def disconnect(self):
        self.ib.disconnect()
