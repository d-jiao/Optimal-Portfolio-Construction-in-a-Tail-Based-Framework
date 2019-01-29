import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ip_optimization import tail_opt_ip

# Data
rtd = pd.read_csv('.\\data\\rtd.csv', index_col = 0)
r = np.array(rtd.iloc[-60:, :])
l = -r

# parameters
m = 2.5
alpha = 1 - np.append([0.01], np.arange(0.05, 0.45, 0.05))
beta = 1 - np.append([0.01], np.arange(0.05, 0.45, 0.05))
gamma = np.arange(0.001, 0.1001, 0.001)

# efficient frontier w.r.t. required CVaR
a = alpha[1]
b = beta[1]
etl = []
weights = []
for g in gamma:
    etl_, weights_ = tail_opt_ip(r, a, b, g, m)
    etl.append(etl_)
    weights.append(weights_)

print(etl)
print(weights)

# efficient frontier w.r.t. alpha and beta
g = gamma[0]
etl = []
weights = []
for a in alpha:
    for b in beta:
        etl_, weights_ = tail_opt_ip(r, a, b, g, m)
        etl.append(etl_)
        weights.append(weights_)

lfig = plt.figure()
plt.plot(gamma, etl)
plt.xlabel('Required CVaR')
plt.ylabel('Optimal ETR')