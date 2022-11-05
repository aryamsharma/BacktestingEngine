import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os
import math
import json

class NpEncoder(json.JSONEncoder):
    """
    Custom encoder for JSON library since it can't use Numpy types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super(NpEncoder, self).default(obj)

class Intraday_Simulator:
    def __init__(self, filepath):
        """
        Inputs
        :filepath (String) You can provide either the filepath to a single .csv file or a folder filled with .csv files

        Outputs
        :instance of class
        """
        # self.files = sorted([f"{filepath}/{file}" for file in os.listdir(filepath) if file.endswith(".csv")], key=lambda x: x.split("_")[-1]) if os.path.isdir(filepath) else [filepath]
        # self.is_folder = True if self.files else False
        self.filepath = filepath
        if os.path.isdir(filepath):
            self.is_folder = True
            self.files = [f"{filepath}/{file}" for file in os.listdir(filepath) if file.endswith(".csv")]
            self.files = sorted(self.files, key=lambda x: x.split("_")[-1])
        else:
            self.is_folder = False
            self.files = [filepath]

    def _cost_per_trade(self, shares, func):
        """TODO tiered cost per order put in"""
        return func(shares)
    
    def filter_files(self, cycle=False):
        """
        Filter files so that there are only one stock being traded per day

        Inputs
        :cycle (bool) Should cycle to create multiple "timelines"
        I.E.
        Input:    [Stock A day 1, Stock B day 1, Stock A day 2, Stock C day 3]
        if True:  [Stock A day 1, Stock A day 2, Stock C day 3]
        if False: [Stock A day 1, Stock A day 2, Stock C day 3, Stock B day 1, ]

        Outputs
        :N/A
        """
        if not self.is_folder: raise Exception("Function made for more than one file")
        dates = [i.split("_") for i in self.files]
        final_dates = []

        while True:
            if not dates:
                break

            set_dates = set([])
            final = []
            to_delete = []

            for date in dates:
                tmp = len(set_dates)
                set_dates.add(date[-1])
                if len(set_dates) != tmp:
                    final.append("_".join(date))
                    to_delete.append(date)

            final_dates.extend(final[:])

            for part in to_delete:
                dates.remove(part)
            
            if not cycle:
                break

        self.filtered = True
        self.cycled = True
        self.files = final_dates

    def setup(self, account_balance=10000, trade_method="fixed", trade_cost=0, limit=None, algorithm=None):
        """
        Setup some variable to start simulated trading

        Inputs
        :account_balance (float) How much money you start out with in your account
        :trade_method (str)      Type of trading method used (fixed / tiered)
        :trade_cost (float)      Only used if the trading_method is set to fixed
        :limit (integer)         How many files to run from the folder provided
        :Sim (class object)      Object that's from the class Algos
        
        Outputs
        :N/A
        """
        self.initial_account_balance = account_balance
        self.account_balance = account_balance
        self.algos = algorithm

        if trade_method == "fixed":
            self.trade_method = False
            self.trade_cost = trade_cost

        # TODO finish tiered trading cost
        if trade_method == "tiered":
            self.trade_method = True
            self.trade_cost = self._cost_per_trade
        
        self.all_DataFrames = []
        if limit != 0 and limit >= 1 and limit != None:
            for file in self.files[:limit]:
                df = pd.read_csv(file, index_col=0)
                df.index = pd.to_datetime(df.index)
                df = df.between_time("09:30:00", "16:00:00")
                df.columns = list(map(str.lower, list(df.columns)))
                assert len(df.index) > 0, "No rows in DataFrame"
                self.all_DataFrames.append((self.algos.send_setup(df), file))
        else:
            for file in self.files:
                df = pd.read_csv(file, index_col=0)
                df.index = pd.to_datetime(df.index)
                df = df.between_time("09:30:00", "16:00:00")
                df.columns = list(map(str.lower, list(df.columns)))
                assert len(df.index) > 0, "No rows in DataFrame"
                self.all_DataFrames.append((self.algos.send_setup(df), file))

    def run(self, warnings=False, verbose=False, progress=False, capture_trades=False, slippage=0):
        """
        Inputs
        :N/A
        
        Outputs (ret)
        :Final (float)          The account balance in the end
        :Net (float)            Net Equity
        :Total trades (Integer) 1 buy and 1 ell is consider to be one trade
        :Trade Win % (float)    if capture_trades is True will return winrate
        """
        shares = 0
        total_trades = 0

        if self.is_folder:
            self.all_returns = [(self.account_balance, 0.0, "N/A")]
            if progress:
                total = len(self.all_DataFrames)
                count = 1

        if capture_trades:
            self.all_trades = []
            day_trades = []

        
        for df, file in self.all_DataFrames:
            tmp_account_balance = self.account_balance
            if progress and self.is_folder: print(f"{count} / {total} {file}", end="\r")
            
            initial_day_balance = self.account_balance
            last_market_exposure = 0
            market_exposure = 0

            if capture_trades: day_trades.append(file)
    
            # Market Open
            for step in range(1, len(df.index) - 1):
                price = df.iloc[step + 1]["open"]
                market_exposure = shares * price

                # Calling the algorithm
                ret = self.algos.send_algo(
                    df=df[:step],
                    # price=df.iloc[step]["close"], // TODO Decide if I need to babyfeed myself
                    balance=self.account_balance,
                    shares=shares,
                    initial_day_balance=initial_day_balance,
                    last_market_exposure=last_market_exposure,
                    market_exposure=market_exposure,
                    new_stock=step == 1,
                    bought=shares > 0)
                # print(ret, shares, self.account_balance)

                if ret is None or (ret > 1 or ret < -1):
                    if warnings:
                        print(f"WARNING ret value is {ret}, setting to 0")
                    ret = 0

                # TODO add function for tiered cost per trade
                # Buy shares
                if ret > 0 and shares == 0:
                    shares += int(self.account_balance // price * abs(ret))
                    if shares == 0:
                        if warnings:
                            print(f"WARNING Bought with {self.account_balance} in account")
                        break
                    price *= 1 + slippage
                    self.account_balance -= shares * price + self.trade_cost
                    if capture_trades: day_trades.append(("B", price, step))
                    last_market_exposure = shares * price
                    if verbose: print(f"BOU {shares} at {price} {self.account_balance}")

                # Sell shares
                if ret < 0 and shares > 0:
                    sol_shares = int(shares * abs(ret))
                    if sol_shares == 0:
                        if warnings:
                            print(f"WARNING Tried to sell {int(shares * abs(ret))} with {shares} shares")
                        continue
                    total_trades += 1
                    price *= 1 - slippage
                    if capture_trades: day_trades.append(("S", price, step))
                    self.account_balance += shares * price
                    self.account_balance -= self.trade_cost
                    market_exposure = 0
                    if verbose: print(f"SOL {sol_shares} at {price} {self.account_balance}")
                    shares -= sol_shares

            # Market closed
            if shares > 0:
                if warnings:
                    if self.is_folder:
                        print(f"WARNING {file} has shares outstanding after market close")
                    else:
                        print(f"WARNING shares outstanding after market close")
    
                total_trades += 1
                price *= 1 - slippage
                if capture_trades: day_trades.append(("S", price, step))
                self.account_balance += shares * price
                self.account_balance -= self.trade_cost
                if verbose: print(f"SOL {shares} at {price}")
                shares = 0

            if self.is_folder:
                final = round(self.account_balance, 2)
                net = round(self.account_balance - tmp_account_balance, 2)
                if verbose:
                    print(f"Done  {file}")
                    print(f"Start {tmp_account_balance}")
                    print(f"Final {final}")
                    print(f"Net   {net}")
                    print()
                self.all_returns.append((final, net, file))

            if progress and self.is_folder: count += 1
            if capture_trades: 
                self.all_trades.append(day_trades)
                day_trades = []
    
        if progress and self.is_folder: print()

        final = round(self.account_balance, 2)
        net = round(self.account_balance - self.initial_account_balance, 2)

        del self.all_DataFrames
        
        ret = {
            "Final": final,
            "Net": net,
            "Total Trades": total_trades
        }
        
        if not capture_trades:
            return ret
        
        if total_trades == 0:
            ret["Winning trades"] = np.nan
            return ret
        
        # TODO: URGENT; This will fail in a scenario where if one buy signal is coupled with multiple sell signals
        # I.E
        # B 10, S 5, S 5;
        winning = 0
        for trades in self.all_trades:
            buy = trades[1::2]
            sell = trades[2::2]
            # print(len(trades), len(buy), len(sell), buy[-1], sell[-1])
            # print(trades)
            for i in range(len(buy)):
                if sell[i][1] > buy[i][1]:
                    winning += 1

        ret["Winning trades"] = winning / total_trades * 100
        return ret
    
    def plot_account_balance(self):
        """
        Plotting your account balance
        
        Inputs
        :N/A

        Outputs
        :N/A
        """
        if not self.is_folder: raise Exception("Function made for more than one file")

        df = pd.DataFrame(self.all_returns)
        df.columns = ["Account Balance", "Net", "File Name"]
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.axes().axes.get_xaxis().set_visible(False)
        plt.plot(df["Account Balance"], color="royalblue", linewidth=3)
        plt.title("Account balance")
        plt.show()
    
    def get_history(self, col_no=None):
        """
        Getting results after each day
        
        Inputs
        :col_no (list of int) specify which specific column's to return

        Outputs
        :(list of [filepath (string), account balance (float), net (float)])
        """
        if not self.is_folder: raise Exception("Function made for more than one file")
        return self.all_returns if col_no is None else np.asarray(self.all_returns)[:, col_no].tolist() 

    def get_files(self):
        """
        Getting each file that ran
        
        Inputs
        :N/A

        Outputs
        :(list of string) File(s) used
        """
        return self.files
  
    def get_all_trades(self):
        """
        Get every trade that occurred

        Inputs
        :N/A

        Outputs
        :(list of [string, float, integer]) Trades and their info
        """
        return self.all_trades

class StrategyPerformance:
    def __init__(self, save_plots=False, folder_name=None):
        """
        Priming class

        Inputs
        :save_plots (bool)    True if storing results
        :folder_name (string) Name of folder default is a number

        Outputs
        :instance of class
        """
        self.save_plots = save_plots
        if save_plots:
            if folder_name is not None:
                self.folder_name = f"Results/{folder_name}"
            else:
                folder_num = len([i for i in os.listdir("Results") if not i.startswith(".")])
                self.folder_name = f"Results/Result_{folder_num}"
        
            os.mkdir(self.folder_name)

    def _get_stats(self, all_info, drawdowns=False, plot=False, text=""):
        """
        Priming class

        Inputs
        :drawdowns (bool) True if want to calculate drawdowns
        :plot (bool)      True if plot
        :test (string)    Filename of saved plot

        Outputs (ret)
        :mean (float)
        :%mean (float)
        :median (float)
        :%median (float)
        :std dev (float)
        :%std dev (float)
        :variance (float)
        :range (float)
        :min range (float)
        :max range (float)
        :skewness (float)
        :kurtosis (float)
        :sharpe ratio (float)
        :drawdowns (list of int) only returned if drawdown param is True
        """

        if len(all_info) == 0 or all_info is None: raise Exception("Function made for more than one file")

        df = pd.DataFrame(all_info)
        df.columns = ["Account Balance", "Net", "File Name"]
        stats = {}
        all_returns_array = np.array(df["Net"]).astype(np.float)
        all_account_balance_array = np.array(df["Account Balance"]).astype(np.float)
        df["%net"] = df["Net"] / df["Account Balance"]

        # Location
        stats["mean"] = round(np.mean(all_returns_array), 2)
        stats["%mean"] = np.mean(df["%net"])

        stats["median"] = round(np.median(all_returns_array), 2)
        stats["%median"] = np.median(df["%net"])

        # all_stats["mode"] = scipy.stats.mode(all_returns_array) # Useless for now

        # Spread
        stats["std dev"] = np.std(all_returns_array)
        stats["%std dev"] = np.std(df["%net"])
        stats["variance"] = np.var(all_returns_array)

        stats["range"] = round(all_returns_array.ptp(), 2)
        stats["min range"] = round(all_returns_array.min(), 2)
        stats["max range"] = round(all_returns_array.max(), 2)
        
        
        # Shape
        stats["skewness"] = df["Net"].skew()
        stats["kurtosis"] = df["Net"].kurtosis()
        
        # Custom
        # Sharpe ratio = return of strategy - Risk-Free Return (S&P500 5.9%)/ std dev of strategy
        stats["sharpe ratio"] = (stats["%mean"] * 100 / 11.9) / stats["%std dev"]

        # Time = (252 trading days) × (6.5 trading hours per trading day) = 1,638
        # stats["annualized sharpe ratio"] = (stats["mean"] * 12) / stats["std dev"]

        if drawdowns:
            ## Drawdowns
            ### Max drawdown
            end = np.argmax((np.maximum.accumulate(all_account_balance_array) - all_account_balance_array) / np.maximum.accumulate(all_account_balance_array))
            if len(all_account_balance_array[:end]) > 0:
                start = np.argmax(all_account_balance_array[:end]) # start of period
            else:
                start = 0
        
            stats["max drawdown"] = (1 - df.iloc[end]["Account Balance"] / df.iloc[start]["Account Balance"]) * 100

            MDD_horizontal_y = np.repeat(df.iloc[start]["Account Balance"], 2)
            MDD_horizontal_x = [start, end]

            MDD_vertical_x = [end, end]
            MDD_vertical_y = [df.iloc[end]["Account Balance"], df.iloc[start]["Account Balance"]]


            ### Max length drawdown
            df["Cumulative"] = df["Net"].cumsum().round(2)
            df["HighValue"] = df["Cumulative"].cummax()

            df["Drawdown"] = df["Cumulative"] - df["HighValue"]

            tmp = df.loc[df["Drawdown"] >= 0]
            tmp = np.array(tmp.index)

            if len(tmp) == 1:
                vals = [0, 0]
                index = 0
                
            else:
                tmp_diff = np.diff(tmp)
                largest = tmp_diff.max()
                index = list(np.where(tmp_diff == largest))[-1][-1]
                vals = tmp[index], tmp[index + 1]

            stats["max drawdown length"] = [vals]
    
            MDDL_horizontal_x = vals
            MDDL_horizontal_y = np.repeat(df.iloc[tmp[index]]["Account Balance"], 2)
            
        
        if plot:
            plt.rcParams['figure.figsize'] = [20, 10]
            plt.xlabel("Days")
            plt.ylabel("Account Balance")
            if drawdowns:
                # Max Drawndown
                plt.plot(MDD_horizontal_x, MDD_horizontal_y, color="red", linewidth=2)
                plt.plot(MDD_vertical_x, MDD_vertical_y, color="red", linewidth=2)

                # Max Drawndown length
                plt.plot(MDDL_horizontal_x, MDDL_horizontal_y, color="orange", linewidth=2)
            
            plt.plot(df["Account Balance"], color="royalblue", linewidth=3)

            plt.title("Account balance")
            if self.save_plots:
                file_name = f"{self.folder_name}/Equity" if text == "" else f"{self.folder_name}/Equity_{text}"
                plt.savefig(file_name)
            plt.show()

        return stats

    def get_outliers(self, all_info):
        """
        Separate using z-score separate the outliers from the normal data

        Inputs
        :all_info (list) Use Intraday_Simulator.get_history to get

        Outputs
        :(list of lists) outliers
        :(list of lists) normal data
        """
        stats = StrategyPerformance._get_stats(self, all_info, drawdowns=False, plot=False)
        outliers = []
        normal = []

        for info in all_info:
            z_score = (info[1] - stats["mean"]) / stats["std dev"]
            if z_score > 3 or z_score < -3:
                outliers.append(info)
            else:
                normal.append(info)
        
        return outliers, normal

    def find_outliers(self, sim_object, drawdowns=True):
        """
        Gets the sim_object, looks through for any outliers;
        If there are any outliers, remove them and adjust future numbers accordingly and plot

        Inputs
        :sim_object (Intraday_Simulator)

        Outputs
        :N/A
        """
        self.data = sim_object.get_history()
        self.outliers, _ = StrategyPerformance.get_outliers(self, self.data)

        # Basically after taking out the outlier I have to adjust the future account_balances accordingly 
        starting_balance = self.data[0][0]

        data_without_outliers = [section for section in self.data if section not in self.outliers]
        cumulative_returns = np.cumsum([i[1] for i in data_without_outliers])
        self.normal = [(starting_balance, 0, "N/A")]

        for i in range(1, len(data_without_outliers)):
            self.normal.append((starting_balance + cumulative_returns[i], data_without_outliers[i][1], data_without_outliers[i][-1]))

        results = {"Starting Balance": sim_object.initial_account_balance}

        if self.outliers:
            print("Outliers found")

            for sec in self.outliers:
                print(f"{sec} at index {self.data.index(sec)}")

            print("Before removing outliers")
            results["Before"] = StrategyPerformance._get_stats(self, self.data, drawdowns=drawdowns, plot=True, text="before")
            results["Before"]["Ending balance"] = sim_object.account_balance
    
            print("After removing outliers")
            results["After"] = StrategyPerformance._get_stats(self, self.normal, drawdowns=drawdowns, plot=True, text="after")
            results["After"]["Ending balance"] = self.normal[-1][0]
            print(f"Before: {results['Before']}")
            print()
            print(f"After: {results['After']}")

        else:
            print("No outliers")
            results = StrategyPerformance._get_stats(self, self.data, drawdowns=drawdowns, plot=True)
            results["Starting Balance"] = sim_object.initial_account_balance
            results["Ending balance"] = sim_object.account_balance
            print(results)

        if self.save_plots:
            inputs = {
                "Filepath": sim_object.filepath,
                "Filtered": sim_object.filtered,
                "Cyclic": sim_object.cycled,
                "Algorithm number": sim_object.algos.num,
                "Account Balance": sim_object.initial_account_balance,
                "Trade Method": sim_object.trade_method,
                "Trade Cost": sim_object.trade_cost
            }

            with open(f"{self.folder_name}/input.json", "w+") as f:
                json.dump(inputs, f, indent=4)

            with open(f"{self.folder_name}/output.json", "w+") as f:
                json.dump(results, f, indent=4, cls=NpEncoder)

    def remove_outliers(self, sim_object):
        """
        Removes outliers from original list

        Inputs
        :sim_object (Intraday_Simulator)

        Outputs
        :N/A
        """
        sim_object.all_returns = self.normal
    
    def _plot_trades(self, trades, FIGSIZE=[20, 10]):
        """
        Plot certain trading days

        Inputs
        :trades (list) trading activity of a single day
        :FIGZISE (list of int) size of charts, in inches (i.e. [20, 10])

        Outputs
        :N/A
        """
        df = pd.read_csv(trades[0], index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.between_time("09:30:00", "16:00:00")
        df.columns = list(map(str.lower, list(df.columns)))
    
        buy_x = []
        sell_x = []

        for trade in trades[1:]:
            if trade[0] == "B": buy_x.append(df.index[trade[-1]])
            if trade[0] == "S": sell_x.append(df.index[trade[-1]])
        
        plt.rcParams['figure.figsize'] = FIGSIZE
        plt.plot(df["close"], "royalblue", linewidth=2)

        for x in buy_x:
            plt.axvline(x, color="green")
        for x in sell_x:
            plt.axvline(x, color="red")
        
        tmp = df.iloc[::50].index
        plt.xticks(tmp, tmp, rotation=45)

        if self.save_plots:
            plt.savefig(f"{self.folder_name}/{trades[0].split('.')[0].split('/')[-1]}")
        plt.show()

    def check_outliers(self, sim_object, customs=[]):
        """
        Plot trading days for outliers to ensure no funky activity

        Inputs
        :sim_object (Intraday_Simulator)
        :customs (list of int) if number is passed through then it
            will output that trading day's activity (can also pass in "all" to get every trading day)

        Outputs
        :N/A
        """
        trades = sim_object.get_all_trades()
        customs = [i for i in range(len(trades))] if customs == "all" else customs
        FIGSIZE = [20, 10] if not customs else [10, 5]
        plt.xlabel("Time")
        plt.ylabel("Stock price")
        for outlier in self.outliers:
            self._plot_trades(trades[self.data.index(outlier) - 1], FIGSIZE)

        if customs: print("CUSTOM")

        for custom in customs:
            print(custom)
            self._plot_trades(trades[custom], FIGSIZE)
    
    def returns_histogram(self, sim_object, bins_algo="total length", bins_num=None):
        """
        Create histogram for Amount of returns vs Returns

        Inputs
        :bins_algo (string) How to calculate number of bins for histogram; range or total length
        :bins_num (int)     If number is passed through histogram will use this many bins

        Outputs
        :N/A
        """
        plt.xlabel("Returns")
        plt.ylabel("Amount of returns")
        returns = [i[1] for i in sim_object.get_history()]
        if bins_num is None:
            if bins_algo == "total length": bins_algo = int(round(1 + 3.322 * math.log2(len(returns)), 0))
            if bins_algo == "range": bins_algo = int((max(returns) - min(returns)) /  len(returns))
        else:
            bins_algo = bins_num

        print(bins_algo)
        plt.hist(returns, bins=bins_algo)
        if self.save_plots:
            plt.savefig(f"{self.folder_name}/returns_histogram")
        plt.show()
    
    def price_to_returns(self, sim_object):
        """
        Create scatter plot for Open of each stock vs Returns

        Inputs
        :sim_object (Intraday_Simulator)

        Outputs
        :N/A
        """
        quick = []
        # OHLCV
        for i in sim_object.get_history()[1:]:
            tmp = []
            df = pd.read_csv(i[-1]).iloc[0]
            tmp.append(df["Open"])
            tmp.append(df["High"])
            tmp.append(df["Low"])
            tmp.append(df["Close"])
            tmp.append(df["Volume"])
            tmp.append(i[1])
            quick.append(tmp)
    
        opens = [i[0] for i in quick]
        high = [i[1] for i in quick]
        low = [i[2] for i in quick]
        close = [i[3] for i in quick]
        volume = [i[4] for i in quick]

        returns = [i[5] for i in quick]

        plt.xlabel("Returns")
        plt.ylabel("Starting price of stock")

        plt.scatter(returns, opens, color="orange")
        plt.scatter(returns, high, color="green")
        plt.scatter(returns, low, color="red")
        plt.scatter(returns, close, color="blue")
        # plt.scatter(returns, volume)
        plt.plot([0, 0], [0, 100])
        if self.save_plots:
            plt.savefig(f"{self.folder_name}/price_vs_returns")
        plt.show()
    
    def returns_to_trades(self, sim_object, returns_line=None):
        """
        Create scatter plot for Returns vs Trades count

        Inputs
        :sim_object (Intraday_Simulator)
        :returns_line (list of int) If not None, will create a line from min trades to max trades i.e. [min trades, max trades]

        Outputs
        :N/A
        """
        returns = [i[1] for i in sim_object.get_history()[1:]]
        trades = []

        for i in sim_object.get_all_trades():
            trades.append(len(i[1:]) / 2)

        plt.xlabel("Trades")
        plt.ylabel("Returns")
        plt.scatter(trades, returns)
        if returns_line is not None:
            plt.plot(returns_line, [0, 0])

        if self.save_plots:
            plt.savefig(f"{self.folder_name}/returns_vs_trades")
        plt.show()