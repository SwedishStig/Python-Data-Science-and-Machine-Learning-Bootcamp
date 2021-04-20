from pandas_datareader import data, wb
import numpy as np
import pandas as pd
import datetime
import seaborn as sea
import matplotlib.pyplot as plt

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)

BAC = data.DataReader("BAC", "yahoo", start, end)
C = data.DataReader("C", "yahoo", start, end)
GS = data.DataReader("GS", "yahoo", start, end)
JPM = data.DataReader("JPM", "yahoo", start, end)
MS = data.DataReader("MS", "yahoo", start, end)
WFC = data.DataReader("WFC", "yahoo", start, end)

tickers = ["BAC", "C", "GS", "JPM", "MS", "WFC"]

bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)

bank_stocks.columns.names = ["bank Ticker", "Stock Info"]

for t in tickers:
    print(bank_stocks[t]["Close"].max())

returns = pd.DataFrame()

for t in tickers:
    returns[t + " Return"] = bank_stocks[t]["Close"].pct_change()

sea.pairplot(returns[1:])
plt.show()

print(returns.idxmin())

print(returns.idxmax())

print(returns.std())

print(returns.loc["2015-01-01":"2015-12-31"].std())

sea.distplot(returns.loc["2015-01-01":"2015-12-31"]["MS Return"], color="green", bins=50)
plt.show()

sea.distplot(returns.loc["2008-01-01":"2008-12-31"]["C Return"], color="red", bins=50)
plt.show()

for t in tickers:
    bank_stocks[t]["Close"].plot(label=t, figsize=(12, 4))
plt.legend()
plt.show()

bank_stocks.xs(key="Close", axis=1, level="Stock Info").plot()
# plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
BAC["Close"].loc["2008-01-01":"2009-01-01"].rolling(window=30).mean().plot(label="30 Day Mov Avg")
BAC["Close"].loc["2008-01-01":"2009-01-01"].plot(label="BAC Close")
plt.legend()
plt.show()

sea.heatmap(bank_stocks.xs(key="Close", axis=1, level="Stock Info").corr(), annot=True)
plt.show()

sea.clustermap(bank_stocks.xs(key="Close", axis=1, level="Stock Info").corr(), annot=True)
plt.show()
