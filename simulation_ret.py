import numpy as np
import pandas as pd
from scipy.stats import genpareto
from scipy.stats import t
from utils import *

z = pd.read_csv('.\\data\\simulated1.csv', index_col=0)
r0 = np.array(z.r0)
z = np.array(z.iloc[:, :8])
garch_param = pd.read_csv('.\\data\\garch_param.csv', index_col = 0, header = None)
indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx']

c = np.array(garch_param.iloc[0, :])
w = np.array(garch_param.iloc[1, :])
a = np.array(garch_param.iloc[2, :])
b = np.array(garch_param.iloc[3, :])
g = np.array(garch_param.iloc[4, :])
(n, d) = z.shape

z -= c
r = np.zeros((n, d))
eps = np.zeros((n, d))
r[0, :] = z[0, :] + c
eps[0, :] = z[0, :]
sigma = np.ones(d)

for i in range(1, n):
    for j in range(d):
        sigma[j] = np.sqrt(w[j] + a[j] * eps[i - 1, j] ** 2 + b[j] * sigma[j] ** 2
                           + g[j] * int(eps[i - 1, j] < 0) * eps[i - 1, j] ** 2)
        eps[i, j] = sigma[j] * z[i, j]
        r[i, j] = c[j] + eps[i, j]
        pass

r = pd.DataFrame(r)
r.columns = indices
r['r0'] = r0
r.to_csv('.\\data\\simulated_ret.csv')