import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.stattools import acf, pacf
from arch import arch_model

rtd = pd.read_csv('.\\data\\rtd.csv', index_col = 0)
indices_front = ['CSI 300', 'S&P 500', 'Nikkei 225', 'FTSE 100']
indices = ['csi', 'spx', 'nky', 'ukx']

fig, ax = plt.subplots(4, 1, sharex='col', figsize = (8, 6))
for i in range(4):
    ax[i].plot(rtd.iloc[:, i], linewidth = 0.5)
    ax[i].set_ylabel(indices_front[i])
    ax[i].set_xticks(rtd.index[[0, int(len(rtd) / 4), int(len(rtd) / 2), int(len(rtd) * 3 / 4), len(rtd) - 1]])
    ax[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.tight_layout()
fig.savefig('.\\fig\\rtd.png')

fig, ax = plt.subplots(4, 1, sharex='col', figsize = (8, 6))
for i in range(4):
    pacfs = pacf(rtd.iloc[:, i], 20)
    for j in range(21):
        if pacfs[j] >= 0:
            ax[i].vlines(j, 0, pacfs[j], linewidth = 0.75)
        else:
            ax[i].vlines(j, pacfs[j], 0, linewidth = 0.75)
    ax[i].set_ylabel(indices_front[i])
    ax[i].set_xlim([-0.5, 20.5])
    ax[i].hlines([-0.1, 0.1], -0.5, 20.5, colors='b', linestyles='dotted', linewidth = 0.75)
    ax[i].hlines(0, -0.5, 20.5, linewidth = 0.75)
plt.tight_layout()
fig.savefig('.\\fig\\pacf.png')

fig, ax = plt.subplots(4, 1, sharex='col', figsize = (8, 6))
for i in range(4):
    pacfs = acf(rtd.iloc[:, i], 20)
    for j in range(21):
        if pacfs[j] >= 0:
            ax[i].vlines(j, 0, pacfs[j], linewidth = 0.75)
        else:
            ax[i].vlines(j, pacfs[j], 0, linewidth = 0.75)
    ax[i].set_ylabel(indices_front[i])
    ax[i].set_xlim([-0.5, 20.5])
    ax[i].hlines([-0.1, 0.1], -0.5, 20.5, colors='b', linestyles='dotted', linewidth = 0.75)
    ax[i].hlines(0, -0.5, 20.5, linewidth = 0.75)
plt.tight_layout()
fig.savefig('.\\fig\\acf.png')

am = []
res = []
resid = pd.DataFrame()
for i in range(4):
    am.append(arch_model(rtd.iloc[:, i], p=1, o=1, q=1, dist='StudentsT'))
    res.append(am[-1].fit(update_freq = 0))
    resid[indices_front[i]] = res[-1].resid
    print(res[-1].summary())

    fig_r = plt.figure(figsize=(8, 3))
    plt.plot(res[-1].resid, linewidth = 0.5)
    plt.xticks(rtd.index[[0, int(len(rtd) / 4), int(len(rtd) / 2), int(len(rtd) * 3 / 4), len(rtd) - 1]])
    fig_r.savefig('.\\fig\\' + indices[i] + '_res.png')

    fig_v = plt.figure(figsize = (8, 3))
    plt.plot(res[-1].conditional_volatility, linewidth = 0.5)
    plt.xticks(rtd.index[[0, int(len(rtd) / 4), int(len(rtd) / 2), int(len(rtd) * 3 / 4), len(rtd) - 1]])
    fig_v.savefig('.\\fig\\' + indices[i] + '_vol.png')

resid.to_csv('.\\data\\res.csv')