import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from prettytable import PrettyTable

p1 = pd.read_csv('.\\data\\csi.csv', index_col=0, thousands=',')
p2 = pd.read_csv('.\\data\\gold.csv', index_col=0, thousands=',')

p1.index = pd.to_datetime(p1.index)
p2.index = pd.to_datetime(p2.index)
p1 = p1.sort_index()
p2 = p2.sort_index()

ptd1 = pd.DataFrame({'csi': p1.Price.values}, index = p1.index)
ptd2 = pd.DataFrame({'gold': p2.Price.values}, index = p2.index)

dta = [ptd1, ptd2]
ptd = pd.concat(dta, axis = 1, join = 'outer').iloc[1:, :]
ptd = ptd.fillna(method = 'ffill')

years = ptd.index.year
years_0 = years[1:]
years_1 = years[:-1]
year_ends = np.where((years_0 == years_1) == False)[0]

prof1 = [1]
prof2 = [1]
prof3 = [1]
m1 = 1
m2 = 1
m3 = 1
w1 = 1 / 2
w2 = 1 / 2

for i in range(len(ptd) - 1):
    m1 = ptd.csi[i + 1] / ptd.csi[i] * m1
    m2 = ptd.gold[i + 1] / ptd.gold[i] * m2
    w1 = ptd.csi[i + 1] / ptd.csi[i] * w1
    w2 = ptd.gold[i + 1] / ptd.gold[i] * w2
    m3 = w1 + w2
    if np.any(year_ends == i):
        w1 = m3 / 2
        w2 = m3 / 2
    prof1.append(m1)
    prof2.append(m2)
    prof3.append(m3)

prof = pd.DataFrame({'csi': prof1, 'gold': prof2, '5050': prof3})

fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(ptd.index, prof1, linewidth = 0.75)
ax.plot(ptd.index, prof2, linewidth = 0.75)
ax.plot(ptd.index, prof3, linewidth = 0.75)
ax.legend(['All in CSI 300', 'All in Gold', '50/50 Monthly Rebalancing'])
fig.savefig('.\\fig\\simple_alloc.png')

strats = ['csi', 'gold', '5050']
rprof = np.log(prof) - np.log(prof.shift(1))
rprof = rprof.iloc[1:,]
tVars = PrettyTable(['', 'Cumulative Return', 'Sharpe Ratio', 'Maximum Drawdown', '95%-ETR', '5%-ETL'])
for (i, strat) in enumerate(strats):
    cret = prof.iloc[-1, i] - 1
    sha = sharpe(rprof.iloc[:, i]) * np.sqrt(252)
    mdd = max_dd(prof.iloc[:, i])
    etr = etr_calc(rprof.iloc[:, i], 0.95)
    etl = etr_calc(rprof.iloc[:, i], 0.05)
    tVars.add_row([strat, cret, sha, min(mdd), etr, etl])
print(tVars)
