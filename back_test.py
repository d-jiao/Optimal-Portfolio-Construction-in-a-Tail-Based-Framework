import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from other_approaches import *
from solver import tail_opt
from utils import *

# data
indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx', 'r0']

p = pd.read_csv('.\\data\\ptd.csv', index_col=0)
p.index = pd.to_datetime(p.index)
p = p[indices[:-1]]

r = pd.read_csv('.\\data\\rtd.csv', index_col=0)
r.index = pd.to_datetime(r.index)
r = r[indices]
stuff = pd.DataFrame(dict(zip(indices, [np.nan for i in range(len(indices))])), index = [p.index[0]])
r = pd.concat([stuff, r])

# dates
month_ends = month_ends(p.index)
start = month_ends[5 * 12 - 1]

# params: most of them default
c = np.ones(8) * 0.0001
d = len(indices) - 1
init = False

for i in range(start, len(p)):
    r_ = np.array(r.iloc[i - 60 : i, :])

    if not init:
        # initialize the positions tracker
        w_min, _ = min_vol(r_[:, :-1], c = c)
        w_eq = equal_weighted(d, c = c)
        w_mkw, _ = max_sharpe(r_[:, :-1], c = c)
        w_cvar, _ = mean_cvar(r_, c = c)
        w_tail, _ = tail_opt(r_, c = c)

        prof_min = np.array([w_min.sum()])
        prof_eq = np.array([w_eq.sum()])
        prof_mkw = np.array([w_mkw.sum()])
        prof_cvar = np.array([w_cvar.sum()])
        prof_tail = np.array([w_tail.sum()])

        weight_min = np.array(w_min)
        weight_eq = np.array(w_eq)
        weight_mkw = np.array(w_mkw)
        weight_cvar = np.array(w_cvar)
        weight_tail = np.array(w_tail)

        init = True
        continue

    if np.any(month_ends == i):
        w_min, _ = min_vol(r_[:, :-1], c = c)
        w_eq = equal_weighted(d, c = c, xb = w_eq)
        w_mkw, _ = max_sharpe(r_[:, :-1], c = c, xb = w_mkw)
        w_cvar, _ = mean_cvar(r_, c = c, xb = w_cvar)
        w_tail, _ = tail_opt(r_, c = c, xb = w_tail)

    rr = np.array(p.iloc[i, :]) / np.array(p.iloc[i - 1, :])

    w_min = w_min * rr
    w_eq = np.append(w_eq[:-1] * rr, w_eq[-1] * (1 + r_[-1, -1] / 100))
    w_mkw = w_mkw * rr
    w_cvar = np.append(w_cvar[:-1] * rr, w_cvar[-1] * (1 + r_[-1, -1] / 100))
    w_tail = np.append(w_tail[:-1] * rr, w_tail[-1] * (1 + r_[-1, -1] / 100))

    prof_min = np.append(prof_min, w_min.sum())
    prof_eq = np.append(prof_eq, w_eq.sum())
    prof_mkw = np.append(prof_mkw, w_mkw.sum())
    prof_cvar = np.append(prof_cvar, w_cvar.sum())
    prof_tail = np.append(prof_tail, w_tail.sum())

    weight_min = np.vstack((weight_min, w_min))
    weight_eq = np.vstack((weight_eq, w_eq))
    weight_mkw = np.vstack((weight_mkw, w_mkw))
    weight_cvar = np.vstack((weight_cvar, w_cvar))
    weight_tail = np.vstack((weight_tail, w_tail))

prof = pd.DataFrame({'min': prof_min, 'eq': prof_eq, 'mkw': prof_mkw, 'cvar': prof_cvar, 'tail': prof_tail})
# prof = pd.DataFrame({'eq': prof_eq, 'mkw': prof_mkw, 'cvar': prof_cvar, 'tail': prof_tail})
prof.index = r.index[start: ]
prof.to_csv('.\\data\\prof.csv')

weight_min = pd.DataFrame(weight_min)
weight_min.to_csv('.\\data\\weight_min.csv')
weight_eq = pd.DataFrame(weight_eq)
weight_eq.to_csv('.\\data\\weight_eq.csv')
weight_mkw = pd.DataFrame(weight_mkw)
weight_mkw.to_csv('.\\data\\weight_mkw.csv')
weight_cvar = pd.DataFrame(weight_cvar)
weight_cvar.to_csv('.\\data\\weight_cvar.csv')
weight_tail = pd.DataFrame(weight_tail)
weight_tail.to_csv('.\\data\\weight_tail.csv')

strats = ['Min Vol', 'Equal Weighted', 'Max Sharpe', 'Mean-CVaR', 'Tail Portfolio']
# strats = ['Equal Weighted', 'Max Sharpe', 'Mean-CVaR', 'Tail Portfolio']
fig = plt.figure(figsize = (8, 4))
plt.plot(prof['min'], linewidth = 0.75)
plt.plot(prof['eq'], linewidth = 0.75)
plt.plot(prof['mkw'], linewidth = 0.75)
plt.plot(prof['cvar'], linewidth = 0.75)
plt.plot(prof['tail'], linewidth = 0.75)
plt.legend(strats)
fig.savefig('.\\fig\\prof.png')

rprof = np.log(prof) - np.log(prof.shift(1))
rprof = rprof.iloc[1:,]
tVars = PrettyTable(['', 'Cumulative Return', 'Volatility', 'Sharpe Ratio', 'Maximum Drawdown', '95%-ETR', '5%-ETL'])
for (i, strat) in enumerate(strats):
    cret = prof.iloc[-1, i] - 1
    vol = rprof.iloc[:, i].std() * np.sqrt(252)
    sha = sharpe(rprof.iloc[:, i]) * np.sqrt(252)
    mdd = max_dd(prof.iloc[:, i])
    etr = etr_calc(rprof.iloc[:, i], 0.95)
    etl = etr_calc(rprof.iloc[:, i], 0.05)
    tVars.add_row([strat, cret, vol, sha, min(mdd), etr, etl])
print(tVars)