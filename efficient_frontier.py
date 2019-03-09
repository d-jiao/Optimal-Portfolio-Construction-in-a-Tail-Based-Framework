import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from solver import tail_opt
from other_approaches import mean_cvar
from utils import *

# Data
indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx', 'r0']
r = pd.read_csv('.\\data\\simulated1.csv', index_col=0)
r = r[indices]
r = np.array(r.iloc[-60:, :])

# parameters
m = 3
g = np.arange(0.02, 0.052, 0.002)
a = 0.05
b = 0.95
c = np.ones(8) * 0.0001

etr = []
weights = []
for g_ in g:
    weights_, etr_ = tail_opt(r, a, b, g_, m, c)
    etr.append(etr_)
    weights.append(weights_)
    print('gamma =', str(g_), 'finished.')

weights = np.array(weights)
etr = np.array(etr)
stocks = pd.DataFrame(weights)
stocks.columns = indices
cash = 1 - weights.sum(axis = 1)
data = stocks.join(pd.DataFrame({'cash': cash, 'etr': etr}))
data.index = g

ret = []
weights1 = []
etr1 = []
for g_ in g:
    weights_, ret_ = mean_cvar(r, a, g_, m, c)
    ret.append(ret_)
    weights1.append(weights_)
    print('gamma =', str(g_), 'finished.')

weights1 = np.array(weights1)
ret = np.array(ret)
stocks1 = pd.DataFrame(weights1)
stocks1.columns = indices[:]
stocks1['ret'] = ret
stocks1.index = g

for i in range(len(stocks1)):
    w = np.array(stocks1.iloc[i, :-2])
    rr = np.matmul(r[:, :-1], w)
    etr1.append(etr_calc(rr, b))

stocks1['etr'] = etr1

data.to_csv('.\\data\\tail_opt.csv')
stocks1.to_csv('.\\data\\mean_cvar.csv')

fig = plt.figure(figsize = (4, 3))
plt.plot(data.index, data.etr, linewidth = 0.75)
plt.plot(data.index, stocks1.etr, linewidth = 0.75)
plt.legend(['Tail Portfolio', 'Mean-CVaR'])
plt.xlabel('Expected Tail Loss')
plt.ylabel('Expected Tail Return')
plt.xticks([0.02 , 0.03 , 0.04, 0.05])
plt.tight_layout()
fig.savefig('.\\fig\\ef.png')