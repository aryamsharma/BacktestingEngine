import os
import pandas as pd
import numpy as np

class Exchange:
    def __init__(self, filepath, sorting=False, limit=-1, slippage=1, lazy_loading=None, lazy_loading_limit=24):
        """
        Parameter(s)
        :filepath (str)
            Filepath to folder containing files
        :sorting (bool); False
            Whether to sort based on dates (all files must be formatted TICKER_YYYY-MM-DD.csv)
        :limit (int); -1
            How many files in the folder should be used
        :slippage (int); 1
            Delta between requesting an order and the order being active
        :lazy_loading (bool); None
            If lazy loading is turned on the the data will be produced lazily, with the trade off of being /slow/, otherwise it will generate all the data at the start but will be extra fast (~97.3% faster)
        :lazy_loading_limit (bool); 24
            If lazy loading is None, then if there are more files that the limit the program will generate it lazily other it'll be generate at start
        """
        #TODO slippage :_)

        filepath = filepath
        self.lazy_loading = lazy_loading
        self.lazy_loading_limit = lazy_loading_limit
        self.limit = limit

        # 1 bar of slippage at minimum, so that orders aren't instantly active
        self.slippage = 1 if slippage <= 0 else slippage

        assert os.path.isdir(filepath), "Must be a folder"

        self.files = [
            f"{filepath}/{file}" for file in os.listdir(filepath) if file.endswith(".csv")]

        if sorting:
            self.files = sorted(self.files, key=lambda x: x.split("_")[-1])


    def setup_data(self, algorithm):
        """
        Parameter(s)
        :filepath (str)
            Algorithm class (example in Client.py/ipynb)

        Output(s)
        :function (func)
            iterator function
        """
        self.dfs = []
        self.total_files = 0

        # Deciding to slice the list or not
        sliced_files = self.files[:self.limit] if self.limit != 0 and self.limit >= 1 else self.files

        for file in sliced_files:
            self.total_files += 1

            df = pd.read_csv(file, index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.between_time("09:30:00", "16:00:00")
            df.columns = list(map(str.lower, list(df.columns)))

            for column in df.columns:
                df[column] = df[column].round(decimals=2)

            assert len(df.index) > 0, "No rows in DataFrame"

            modified_df = algorithm.send_setup(df)

            # Technique to make searching DataFrames even faster
            df_unique = df
            df["step"] = np.arange(df.shape[0])
            df_unique.index = df_unique["step"]
            df_dict = df_unique.to_dict(orient='index')
            self.dfs.append((modified_df, pd.Series(df_dict), file))

        if self.lazy_loading:
            return self.generator
        # Can't do elif not self.lazy_loading because that will return True for None
        elif self.lazy_loading == False:
            self.info = [data for data in self.generator()]
            return self.preprocessed

        if len(self.dfs) <= self.lazy_loading_limit:
            self.info = [data for data in self.generator()]
            return self.preprocessed
        else:
            return self.generator


    def preprocessed(self):
        """
        Output(s)
        :info (list of dict)
            List of dictionaries (dict schema in generator function)
        """
        return self.info


    def generator(self):
        """
        Output(s)
        :info (dict)
            Dictionary with information
        """
        total_length = len(self.dfs)
        count = 0

        for df, df_dict, file in self.dfs:
            total_steps = df.shape[0] - self.slippage
            count += 1
            for step in range(0, total_steps):
                info = {
                    "feed": df.head(step),
                    "current_tick": df_dict[step],
                    "current_tick_true": df_dict[step + self.slippage],
                    "filepath": file,
                    "step": step,
                    "new_file": step == 0,
                    "end_file": step == total_steps - 1,
                    "total_steps": total_steps,
                    "progress": (count, total_length)
                }
                yield info


if __name__ == "__main__":
    # Ignore this is me just testing
    class Algos:
        def __init__(self): return None
        def setup0(self, df): return df
        def algo0(self, **var):
            if var["new_stock"]:
                return 1
            else:
                return -1
        def send_setup(self, df): return self.setup0(df)
        def send_algo(self, **var): return self.algo0(**var)

    exchange = Exchange("../Data/intra_all_6")
    exchange.modify_data(1, algorithm=Algos())

    count = 0
    print(exchange.dfs[0][0])
    for i in exchange.step():
        print(i["current_tick"])
        break
        # count += 1
        # if count == 2:
        #     break
