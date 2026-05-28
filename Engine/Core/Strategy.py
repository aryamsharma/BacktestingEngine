from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def on_bar(self, **kwargs) -> list | None:
        ...

    def on_start_of_day(self):
        pass

    def on_end_of_day(self):
        pass
