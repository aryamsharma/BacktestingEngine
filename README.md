# BacktestingEngine
## Description
___
This BacktestingEngine allows a trader to program their own ideas with ease. Within the [notebook](BacktestingAlgos.ipynb), serveral ideas can be programmed and tested on the spot. After the testing, serveral statiscal analysis tools from the [BacktestingEngine](BacktestingEngine.py) can be used to determine the effectiveness.
## Requirements
python >= 3.7

Download all python packages within [requirements.txt](requirements.txt)
## Usage
___
To test out a new algorithm head to the [notebook](BacktestingAlgos.ipynb) and within the **Algos** class create two new functions;
```python
def setup2(self, df):
    return df
def algo2(self, **var):
    return 0
```
Where _2_ is the algos number (i.e. if _2_ algos have been created, the new functions will end with _3_).

Within the setup function you can add any new columns (i.e. moving average) to the [pandas](https://github.com/pandas-dev/pandas) DataFrame to be used later in the algorithm. This is purely for decreasing processing and time during runtime. The function ***must*** return a DataFrame back.
___
Within the algo function is where the actual algorithm resides. It ***must*** return a value, _n_, between -1 and 1.

* **If n > 0**, n is how much of the account should be used to buy shares. Since no fractional trading in this program exists, returing 0.5 will guarantee only, 50% or less, of the account will be used to buy shares. 

* **If n < 0**, n is how much of the shares will be sold, therefore -1 dictates all shares currently being held will be sold. -0.5 means half the shares will be sold.

* **If n = 0**, nothing will be done.

An example (Golden Crossover) is provided within the [notebook](BacktestingAlgos.ipynb) itself.

## Issues
1. If a buy signal is follwed by more than one sell signal then the code at line 280 in [BacktestingEngine.py](BacktestingEngine.py) will fail; See TODO for further explanation.
## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
