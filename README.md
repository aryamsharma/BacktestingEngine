# BacktestingEngine
## Description
___
This BacktestingEngine allows a trader to program their own ideas with ease. Within the [notebook/file](Client.ipynb), serveral ideas can be programmed and tested on the spot. After the testing, serveral statiscal analysis tools from the [BacktestingEngine](BacktestingEngine.py) can be used to determine the effectiveness.
## Requirements
python >= 3.7

Download all python packages within [requirements.txt](requirements.txt)
## Usage
___
To test out a new algorithm head to the [notebook/file](Client.ipynb) and within the **Algos** class create two new functions;
```python
def setup2(self, df):
    return df
def algo2(self, **var):
    return 0
```
Where _2_ is the algos number (i.e. if _2_ algos have been created, the new functions will end with _3_).
___
Within the setup function you can add any new columns (i.e. moving average) to the [pandas](https://github.com/pandas-dev/pandas) DataFrame to be used later in the algorithm. This is purely for decreasing processing and time during runtime. The function ***must*** return a DataFrame back.

Within the algo function is where the actual algorithm resides. It should return a class from [Engine/Orders.py](Engine/Orders.py). An example on how to make and use the classes should be in [Orders.py](Engine/Orders.py) and the [notebook/file](Client.ipynb) respectively.

An example (Golden Crossover) is provided within the [notebook/file](Client.ipynb) itself.

## Issues
1. Drawdown calculations are wrong, but good enough for a rough estimate.
## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
