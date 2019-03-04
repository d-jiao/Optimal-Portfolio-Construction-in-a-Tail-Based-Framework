import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

r = pd.read_csv('.\\data\\rtd.csv', index_col=0)
p = pd.read_csv('.\\data\\ptd.csv', index_col=0)
r.index = pd.to_datetime(r.index)
p.index = pd.to_datetime(p.index)

years = r.index.year
years_0 = years[1:]
years_1 = years[:-1]
year_ends = np.where((years_0 == years_1) == False)[0]

p1 = p['csi']
p2 = p['spx']

prof1 = [1]
prof2 = [1]
prof3 = [1]
m1 = 1
m2 = 1
m3 = 1
w1 = p1[0] / (p1[0] + p2[0])
w2 = p2[0] / (p1[0] + p2[0])

for i in range(len(p) - 1):
    m1 = p1[i + 1] / p1[i] * m1
    m2 = p2[i + 1] / p2[i] * m2
    w1 = p1[i + 1] / p1[i] * w1
    w2 = p2[i + 1] / p2[i] * w2
    m3 = w1 + w2
    if np.any(year_ends == i):
        w1 = p1[i] / (p1[i] + p2[i]) * m3
        w2 = p2[i] / (p1[i] + p2[i]) * m3
    prof1.append(m1)
    prof2.append(m2)
    prof3.append(m3)

fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(p.index, prof1, linewidth = 0.75)
ax.plot(p.index, prof2, linewidth = 0.75)
ax.plot(p.index, prof3, linewidth = 0.75)
ax.legend(['All in CSI 300', 'All in FTSE 100', '50/50 Monthly Rebalancing'])
fig.savefig('.\\fig\\simple_alloc.png')
