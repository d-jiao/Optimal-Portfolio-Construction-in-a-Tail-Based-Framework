import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine

res = pd.read_csv('.\\data\\res.csv', index_col = 0)
indices_front = ['CSI 300', 'S&P 500', 'Nikkei 225', 'FTSE 100']
indices = ['csi', 'spx', 'nky', 'ukx']

def pareto(q, loc = 0, scale = 1, shape = 0, lbda = 0):
    if q > loc:
        q = (q - loc) / scale
        if shape == 0:
            g = 1 - np.exp(-q)
        else:
            g = 1 - (1 + shape * q) ** (-1 / shape)
        return 1 - (1 - g) * (1 - lbda)
    else:
        q = (loc - q) / scale
        if shape == 0:
            g = 1 - np.exp(-q)
        else:
            g = 1 - (1 + shape * q) ** (-1 / shape)
        return (1 - g) * lbda

eng = matlab.engine.start_matlab()
output = pd.DataFrame()

upper_quantile = [0.9721086, 0.9750837, 0.9750837, 0.9888434]
lower_quantile = [0.01487542, 0.03756043, 0.01375976, 0.01190033]
upper_threshold = [0.03, -0.15, -0.02, 0.01]
lower_threshold = [-0.04, -0.19, -0.08, -0.043]

for i in range(len(indices)):
    dta = matlab.double(list(res[indices_front[i]]))
    eng.workspace['dta'] = dta

    eng.eval('pdist = paretotails(dta\', ' + str(lower_quantile[i]) + ', ' + str(upper_quantile[i]) + ');', nargout=0)
    eng.eval('tdist = fitdist(dta\', \'tLocationScale\');', nargout=0)
    eng.eval('ndist = fitdist(dta\', \'Normal\');', nargout=0)
    eng.eval('[edist, ex] = ecdf(dta\');', nargout=0)

    x_r = np.linspace(upper_threshold[i], max(dta[0]), 100)
    eng.workspace['x_r'] = matlab.double([x_r[j].item() for j in range(len(x_r))])
    ty_r = eng.eval('cdf(tdist, x_r);')
    ny_r = eng.eval('cdf(ndist, x_r);')
    py_r = eng.eval('cdf(pdist, x_r);')
    ex_r = eng.eval('ex(min(find(ex > ' + str(upper_threshold[i]) + ')) : end);')
    ey_r = eng.eval('edist(min(find(ex > ' + str(upper_threshold[i]) + ')) : end);')

    x_l = np.linspace(min(dta[0]), lower_threshold[i], 100)
    eng.workspace['x_l'] = matlab.double([x_l[j].item() for j in range(len(x_l))])
    ty_l = eng.eval('cdf(tdist, x_l);')
    ny_l = eng.eval('cdf(ndist, x_l);')
    py_l = eng.eval('cdf(pdist, x_l);')
    ex_l = eng.eval('ex(1 : max(find(ex <= ' + str(lower_threshold[i]) + ')));')
    ey_l = eng.eval('edist(1 : max(find(ex <= ' + str(lower_threshold[i]) + ')));')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(x_l, ty_l[0], linewidth = 0.75)
    ax[0].plot(x_l, ny_l[0], linewidth = 0.75)
    ax[0].plot(x_l, py_l[0], linewidth = 0.75)
    ax[0].scatter(ex_l, ey_l, facecolors='none', edgecolors='#9467bd', linewidth = 0.75)
    ax[0].legend(['Student''s t', 'Normal', 'Pareto', 'Empirical'])
    ax[0].set_xlabel('Lower Tail Innovation')
    ax[0].set_ylabel('Cumulative Distribution')

    ax[1].plot(x_r, ty_r[0], linewidth = 0.75)
    ax[1].plot(x_r, ny_r[0], linewidth = 0.75)
    ax[1].plot(x_r, py_r[0], linewidth = 0.75)
    ax[1].scatter(ex_r, ey_r, facecolors='none', edgecolors='#9467bd', linewidth = 0.75)
    ax[1].legend(['Student''s t', 'Normal', 'Pareto', 'Empirical'])
    ax[1].set_xlabel('Upper Tail Innovation')
    ax[1].set_ylabel('Cumulative Distribution')

    plt.tight_layout()
    fig.savefig('.\\fig\\fitted' + indices[i] + '.png')

    upper = eng.eval('pdist.UpperParameters;')
    ushape = upper[0][0]
    uscale = upper[0][1]
    lower = eng.eval('pdist.LowerParameters;')
    lshape = lower[0][0]
    lscale = lower[0][1]

    output[indices[i]] = [ushape, uscale, lshape, lscale]

output.to_csv('.\\data\\pareto.csv')