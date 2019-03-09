import numpy as np
import pandas as pd

def etl_calc(r, a):
    r = np.array(r)
    r.sort()
    tail = r[:int(a * len(r))]
    etl = tail.mean()
    return etl

def etr_calc(r, b):
    r = np.array(r)
    r.sort()
    tail = r[int(b * len(r)):]
    etr = tail.mean()
    return etr

def sharpe(r):
    r = np.array(r)
    return r.mean() / r.std()

def high_water(r):
    """
    @param r: a timeseries
    @return: a timeseries with the highwater
    """
    high_ar = np.zeros(len(r))
    high_ar[0] = r[0]
    for i in range(1, len(r)):
        if r[i] > high_ar[i - 1]:
            high_ar[i] = r[i]
        else:
            high_ar[i] = high_ar[i - 1]
    return high_ar

def max_dd(r):
    """
    @param r: a timeseries
    @return: a timeseries with the max drawdown
    """
    high = high_water(r)
    drawdown = [0]
    for i in range(1, len(r)):
        if (r[i] - high[i]) / high[i] < drawdown[-1]:
            drawdown.append((r[i] - high[i]) / high[i])
        else:
            drawdown.append(drawdown[-1])
    return drawdown

def load_data(indices):
    r = pd.read_csv('.\\data\\rtd.csv', index_col=0)
    r.index = pd.to_datetime(r.index)
    r = r[indices]
    return r

def month_ends(index):
    months = index.month
    months_0 = months[1:]
    months_1 = months[:-1]
    month_ends = np.where((months_0 == months_1) == False)[0]
    return month_ends