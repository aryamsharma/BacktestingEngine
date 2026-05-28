import os
from datetime import datetime, timedelta

from numpy import int16

import pandas as pd
import pandas_ta as ta
import Engine.DataGatherer.ticker_scraper as ts

import yfinance as yf


def filter1(df, ticker):
	df["avg_vol"] = ta.sma(df["Volume"], 10)
	df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], 14)

	dates = df.shift(-1).loc[
		(df["avg_vol"] >= 500000) & (df["avg_vol"] <= 50000000) &
		(df["Open"] >= 10) & (df["Open"] <= 100) &
		(df["atr"] >= 2)]["Date"]

	return dates

def filter2(df, ticker):
	df["avg_vol"] = ta.sma(df["Volume"], 10)
	df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], 14)

	dates = df.loc[
		(df["avg_vol"] >= 500000) & (df["avg_vol"] <= 50000000) &
		(df["Open"] >= 10) & (df["Open"] <= 50) &
		(df["atr"] >= 2)]["Date"]

	return dates

def filter3(df, ticker):
	df["avg_vol"] = ta.sma(df["Volume"], 10)
	df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], 14)

	dates = df.shift(-1).loc[
		(df["avg_vol"] >= 500000) & (df["avg_vol"] <= 50000000) &
		(df["Open"] >= 10) & (df["Open"] <= 100) &
		(df["atr"] >= 2)]["Date"]

	if dates.any():
		try:
			edates = yf.Ticker(ticker[:-4]).get_earnings_dates(limit=8)
		except Exception as e:
			print(f"Error: {e}")
			return []

		if edates is None:
			return []

		edates = list(edates.index)

		adj_edates = []
		ret = []

		if not edates:
			return []

		for edate in edates:
			datetime_edate = datetime.date(edate)

			if edate.hour < 9:
				adj_edates.append(str(datetime_edate))
			else:
				adj_edates.append(str(datetime_edate + timedelta(days=1)))

		for date in dates:
			if date in adj_edates:
				ret.append(date)

		return ret

	return []


if __name__ == "__main__":
	update_tickers = False
	update_daily_data = True
	backtesting_selection = False
	filepath = "Data/stock_screener/"

	print(f"Updating tickers      {update_tickers}")
	print(f"Update daily data     {update_daily_data}")
	print(f"backtesting_selection {backtesting_selection}")
	input("\nEnter to continue\n")

	if update_tickers:
		if not os.path.exists(filepath):
			os.makedirs("Data/stock_screener/")

		quotes = ts.update_quotes(
			f"{filepath}NYSE_quotes.txt", update=True)

	if update_daily_data:
		import Engine.DataGatherer.download_ibkr_data as ibkr

		quotes = ts.read_quotes(f"{filepath}NYSE_quotes.csv")

		dfs = ibkr.DownloadTickerData.daily(
			quotes, filepath=filepath)

		ibkr.Util.write_daily_data(dfs, filepath)

	if backtesting_selection:
		import Engine.DataGatherer.download_ibkr_data as ibkr
		chosen_dates = {}
		filter_function = filter2
		files = [
			ticker_file for ticker_file in os.listdir(filepath) if ticker_file.endswith(".csv") and ticker_file != "NYSE_quotes.csv"]

		current_files_count = 0
		current_total_files_count = len(files)

		for ticker_file in files:
			print(f"{current_files_count}/{current_total_files_count}", end="\r")

			ticker_dates = filter_function(
				pd.read_csv(
					f"{filepath}{ticker_file}",
					dtype={
						"Open": "float16",
						"High": "float16",
						"Low": "float16",
						"Close": "float16",
						"Volume": "int32"
					}
				),
				ticker_file)

			current_files_count += 1

			# Multiple data type was outputs are bad but you gotta deal with it
			if ticker_dates is None or not list(ticker_dates):
				continue

			chosen_dates = ts.parse_dates(
				chosen_dates, ticker_dates, ticker_file)

		ts.print_dates(chosen_dates)
		chosen_dates_count = len(chosen_dates.items())

		print(f"Total chosen dates: {chosen_dates_count}")

		if chosen_dates_count == 0:
			print("Chosen dates is zero, exiting")
			exit()

		ret = input("Download the chosen dates? (y/n): ")
		if ret.lower() != "y":
			exit()

		folder = "Data/trading_days_ML/"
		if not os.path.isdir(folder):
			print(f"Creating folder ({folder})")
			os.mkdir(folder)

		chosen_dates = ts.remove_downloaded_files(
			[ticker_file for ticker_file in os.listdir(folder) if ticker_file.endswith(".csv")],
			chosen_dates)

		if not chosen_dates:
			print("Done")

		else:
			downloaded_data = ibkr.DownloadTickerData.minute(chosen_dates)

			if not os.path.exists(folder):
				os.mkdir(folder)

			for data in downloaded_data.items():
				ticker, date = data[0].split()
				data[1].to_csv(f"{folder}{ticker}_{date}.csv")