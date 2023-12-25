from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class Statistics:
    def __init__(self, broker_obj):
        """
        Parameter(s)
        :broker_obj (Broker obj)
        """
        self.states = np.array(broker_obj.states, dtype=object)
        self.states_df = pd.DataFrame(columns=["initial", "end"])

        self.states_df["initial"] = self.states[:, 0]
        self.states_df["end"] = self.states[:, 1]

        self.states_df = self.states_df.apply(pd.to_numeric)

        self.states_df["net"] = self.states_df["end"] - self.states_df["initial"]

        self.sub_file_trades = np.array(self.states[:, 2], dtype=object)
        self.dfs = [(df[0], df[-1]) for df in broker_obj.exchange_feed.dfs]
        self.total_files = len(self.dfs)

        plt.rcParams['figure.figsize'] = [20, 10]

    def cash_graph(self, scale="linear", drawdowns=None):
        """
        Plots a cash graph
        """
        plt.xlabel("Days")
        plt.ylabel("Account Balance")
        plt.title("Equity")

        plt.plot(np.append([5000], self.states_df["end"]), color="royalblue", linewidth=3, alpha=1)

        if drawdowns is not None:
                plt.plot(
                    drawdowns["mdd_h_x"], drawdowns["mdd_h_y"],
                    color="red", linewidth=2)

                plt.plot(
                    drawdowns["mdd_v_x"], drawdowns["mdd_v_y"],
                    color="red", linewidth=2)

                # Max Drawndown length
                plt.plot(
                    drawdowns["mddl_h_x"], drawdowns["mddl_h_y"],
                    color="orange", linewidth=2)

        plt.yscale(scale)
        plt.show()

    def plot_sub_file_trades(self, sub_files=[], include=[]):
        """
        Plot's certain days graph so you can see each trade individually

        Input(s)
        :sub_files (list of int)
            Numbers which correspond to each /day/ in the cash graph
        :include (list of string)
            which column to show on each graph, leave list empty if nothing
        """
        for count, data in enumerate(self.sub_file_trades):
            if count not in sub_files:
                continue

            df = self.dfs[count][0]

            for trade in data[1:]:
                if trade[0] > 0:
                    plt.axvline(trade[2], color="green", alpha=0.5)
                if trade[0] < 0:
                    plt.axvline(trade[2], color="red", alpha=0.5)

            for index, row in df.iterrows():
                plt.title(self.dfs[count][1])
                index = row["step"]
                plt.plot(
                    [index] * 2,
                    [row["low"], row["high"]],
                    color="black", label='_nolegend_')

                if row["open"] > row["close"]:
                    plt.plot(
                        [index] * 2,
                        [row["open"], row["close"]],
                        color="red", linewidth=2, label='_nolegend_')
                else:
                    plt.plot(
                        [index] * 2,
                        [row["close"], row["open"]],
                        color="green", linewidth=2, label='_nolegend_')

            for column in df.columns:
                if column in include:
                    plt.plot(df[column], linewidth=2, label=column, alpha=0.5)
                    include.append(column)

            plt.legend(loc=1)
            plt.show()

    def get_statistics(self):
        """
        Get extra statistics
        """
        stats = {}
        self.states_df["%net"] = self.states_df["net"] / self.states_df["initial"]

        stats["%mean"] = self.states_df["%net"].mean() * 100
        # (expected - risk free) / std of downside
        stats["sortino"] = (stats["%mean"] - (3847 / 63000)) / self.states_df.loc[self.states_df['%net'] <= 0]["%net"].std()

        for key in stats.keys():
            # print(key, stats[key])
            stats[key] = round(stats[key], 4)

        return stats

    def get_outliers(self):
        """
        Find outliers based on z-score

        Output(s)
        :nums (list of ints)
            Trading /days/ which might have too much profit/loss; Used to check that the data isn't causing the problem
        """
        if self.total_files <= 1:
            return []
        df_col = self.states_df["net"]
        self.states_df["z_net"] = (df_col - df_col.mean()) / df_col.std(ddof=0)
        pos = self.states_df.loc[self.states_df["z_net"] > 3]
        neg = self.states_df.loc[self.states_df["z_net"] < -3]
        return sorted(np.append(pos.index, neg.index).tolist())

    def get_max_drawdowns(self):
        drawdowns = {}
        end = np.argmax((np.maximum.accumulate(self.states_df["initial"]) -
                        self.states_df["initial"]) / np.maximum.accumulate(self.states_df["initial"]))

        if len(self.states_df["initial"][:end]) > 0:
            # start of period
            start = np.argmax(self.states_df["initial"][:end])
        else:
            start = 0

        drawdowns["mdd"] = (
            1 - self.states_df.iloc[end]["initial"] / self.states_df.iloc[start]["initial"]) * 100

        drawdowns["mdd_h_y"] = np.repeat(self.states_df.iloc[start]["initial"], 2)
        drawdowns["mdd_h_x"] = [start, end]

        drawdowns["mdd_v_x"] = [end, end]
        drawdowns["mdd_v_y"] = [self.states_df.iloc[end]["initial"],
                            self.states_df.iloc[start]["initial"]]

        # Max length drawdown
        self.states_df["cumulative"] = self.states_df["net"].cumsum().round(2)
        self.states_df["highvalue"] = self.states_df["cumulative"].cummax()

        self.states_df["drawdown"] = self.states_df["cumulative"] - self.states_df["highvalue"]

        tmp = self.states_df.loc[self.states_df["drawdown"] >= 0]
        tmp = np.array(tmp.index)

        if len(tmp) == 1:
            vals = [0, 0]
            index = 0

        else:
            tmp_diff = np.diff(tmp)
            largest = tmp_diff.max()
            index = list(np.where(tmp_diff == largest))[-1][-1]
            vals = tmp[index], tmp[index + 1]

        drawdowns["mddl"] = vals[1] - vals[0]
        drawdowns["mddl_h_x"] = vals
        drawdowns["mddl_h_y"] = tuple(np.repeat(
            self.states_df.iloc[tmp[index]]["initial"], 2))

        return drawdowns
