import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.stattools import acf, pacf
from arch import arch_model

rtd = pd.read_csv('.\\data\\rtd.csv', index_col = 0)
indices = ['CSI 300', 'S&P 500', 'Nikkei 225', 'FTSE 100']

fig, ax = plt.subplots(4, 1, sharex='col')
for i in range(4):
    ax[i].plot(rtd.iloc[:, i], linewidth = 0.75)
    ax[i].set_ylabel(indices[i])
    ax[i].set_xticks(rtd.index[[0, int(len(rtd) / 4), int(len(rtd) / 2), int(len(rtd) * 3 / 4), len(rtd) - 1]])
    ax[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.tight_layout()
fig.savefig('.\\fig\\rtd.png')

fig, ax = plt.subplots(4, 1, sharex='col')
for i in range(4):
    pacfs = pacf(rtd.iloc[:, i], 20)
    for j in range(21):
        if pacfs[j] >= 0:
            ax[i].vlines(j, 0, pacfs[j], linewidths=0.75)
        else:
            ax[i].vlines(j, pacfs[j], 0, linewidths=0.75)
    ax[i].set_ylabel(indices[i])
    ax[i].set_xlim([-0.5, 20.5])
    ax[i].hlines([-0.1, 0.1], -0.5, 20.5, colors='b', linestyles='dotted', linewidths=0.75)
    ax[i].hlines(0, -0.5, 20.5, linewidths=0.75)
plt.tight_layout()
fig.savefig('.\\fig\\pacf.png')

fig, ax = plt.subplots(4, 1, sharex='col')
for i in range(4):
    pacfs = acf(rtd.iloc[:, i], 20)
    for j in range(21):
        if pacfs[j] >= 0:
            ax[i].vlines(j, 0, pacfs[j], linewidths=0.75)
        else:
            ax[i].vlines(j, pacfs[j], 0, linewidths=0.75)
    ax[i].set_ylabel(indices[i])
    ax[i].set_xlim([-0.5, 20.5])
    ax[i].hlines([-0.1, 0.1], -0.5, 20.5, colors='b', linestyles='dotted', linewidths=0.75)
    ax[i].hlines(0, -0.5, 20.5, linewidths=0.75)
plt.tight_layout()
fig.savefig('.\\fig\\acf.png')

am = []
res = []
resid = pd.DataFrame()
for i in range(4):
    am.append(arch_model(rtd.iloc[:, i], p=1, o=1, q=1, dist='StudentsT'))
    res.append(am[-1].fit(update_freq = 0))
    resid[indices[i]] = res[-1].resid
    print(res[-1].summary())
    fig = res[-1].plot()
    ax = fig.axes
    ax[0].set_xticks(rtd.index[[0, int(len(rtd) / 4), int(len(rtd) / 2), int(len(rtd) * 3 / 4), len(rtd) - 1]])
    ax[1].set_xticks(rtd.index[[0, int(len(rtd) / 4), int(len(rtd) / 2), int(len(rtd) * 3 / 4), len(rtd) - 1]])
    fig.savefig('.\\fig\\_'.join(indices[i].lower().split() + ['garch']))
resid.to_csv('.\\data\\res.csv')