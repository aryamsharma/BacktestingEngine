import os
from numpy import int16

import pandas as pd
import pandas_ta as ta
import Engine.DataGatherer.ticker_scraper as ts

def filter1(df):
    df["avg_vol"] = ta.sma(df["Volume"], 10)
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], 14)

    dates = df.shift(-1).loc[
        (df["avg_vol"] >= 500000) & (df["avg_vol"] <= 50000000) &
        (df["Open"] >= 10) & (df["Open"] <= 100) &
        (df["atr"] >= 2) &
        (df["Low"].shift(1) >= df["Open"])]["Date"]

    return dates

def filter2(df):
    df["avg_vol"] = ta.sma(df["Volume"], 10)
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], 14)

    dates = df.shift(-1).loc[
        (df["avg_vol"] >= 500000) & (df["avg_vol"] <= 50000000) &
        (df["Open"] >= 10) & (df["Open"] <= 100) &
        (df["atr"] >= 2) &
        (df["Close"] >= df["Open"] * 1.05)]["Date"]

    return dates


if __name__ == "__main__":
    update_tickers = False
    update_daily_data = False
    backtesting_selection = True
    filepath = "Data/stock_screener/"

    print(f"Updating tickers      {update_tickers}")
    print(f"Update daily data     {update_daily_data}")
    print(f"backtesting_selection {backtesting_selection}")
    input("\nEnter to continue\n")

    if update_tickers:
        if not os.path.exists(filepath):
            os.makedirs("Data/stock_screener/")

        quotes = ts.update_quotes(
            f"{filepath}NYSE_quotes.txt", update=update_tickers)

    if update_daily_data:
        import Engine.DataGatherer.download_ibkr_data as ibkr

        quotes = ts.read_quotes(f"{filepath}NYSE_quotes.csv")

        dfs = ibkr.DownloadTickerData.daily(
            quotes, filepath=filepath)

        ibkr.Util.write_daily_data(dfs, filepath)

    if backtesting_selection:
        import Engine.DataGatherer.download_ibkr_data as ibkr
        chosen_dates = {}
        column_dtypes = {
            "Open": "float16",
            "High": "float16",
            "Low": "float16",
            "Close": "float16",
            "Volume": "int32"
        }

        # files = files[:5]

        file_count = 0
        chosen_dates_count = 0
        filter_function = filter2

        files = [
            file for file in os.listdir(filepath) if file.endswith(".csv") and file != "NYSE_quotes.csv"]
        total_file_count = len(files)

        for file in files:
            print(f"{file_count}/{total_file_count}", end="\r")

            df_dates = filter_function(pd.read_csv(
                f"{filepath}{file}", dtype=column_dtypes))

            for date in df_dates:
                if date is None:
                    continue
                chosen_dates_count += 1

                if chosen_dates.get(date, None) is None:
                    chosen_dates[date] = [file[:-4]]
                    continue
                chosen_dates[date] += [file[:-4]]

            file_count += 1

        print(f"Total chosen dates: {chosen_dates_count}")
        if chosen_dates_count == 0:
            print("Chosen dates is zero, exiting")
            exit()
        input("Press enter to continue\n")

        downloaded_data = ibkr.DownloadTickerData.minute(chosen_dates)

        folder = "Data/day_after_5_percent_rise_again/"
        if not os.path.exists(folder):
            os.mkdir(folder)

        for data in downloaded_data.items():
            ticker, date = data[0].split()
            data[1].to_csv(f"{folder}{ticker}_{date}.csv")
