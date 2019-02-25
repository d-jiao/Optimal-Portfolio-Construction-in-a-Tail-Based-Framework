import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine

res = pd.read_csv('.\\data\\res.csv', index_col = 0)
indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx']

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
params = pd.DataFrame()
copula_data = pd.DataFrame()

upper_quantile = [0.9758274, 0.9877278, 0.9873559, 0.9739680, 0.9854965, 0.9747118, 0.9877278, 0.9817776]
lower_quantile = [0.02082559, 0.02008181, 0.02268501, 0.03161026, 0.01599107, 0.01896616, 0.01041279, 0.02528821]
upper_threshold = [0.03, 0.025, 0.03, 0.02, 0.03, 0.025, 0.03, 0.02]
lower_threshold = [-0.035, -0.025, -0.03, -0.02, -0.035, -0.025, -0.035, -0.02]

for i in range(len(indices)):
    dta = matlab.double(list(res[indices[i]]))
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
    ex_r = eng.eval('ex(find(ex > ' + str(upper_threshold[i]) + '));')
    ey_r = eng.eval('edist(find(ex > ' + str(upper_threshold[i]) + '));')

    x_l = np.linspace(min(dta[0]), lower_threshold[i], 100)
    eng.workspace['x_l'] = matlab.double([x_l[j].item() for j in range(len(x_l))])
    ty_l = eng.eval('cdf(tdist, x_l);')
    ny_l = eng.eval('cdf(ndist, x_l);')
    py_l = eng.eval('cdf(pdist, x_l);')
    ex_l = eng.eval('ex(find(ex < ' + str(lower_threshold[i]) + '));')
    ey_l = eng.eval('edist(find(ex < ' + str(lower_threshold[i]) + '));')

    fig_l = plt.figure(figsize=(4, 4))
    plt.plot(x_l, ty_l[0], linewidth = 0.75)
    plt.plot(x_l, ny_l[0], linewidth = 0.75)
    plt.plot(x_l, py_l[0], linewidth = 0.75)
    plt.scatter(ex_l, ey_l, facecolors='none', edgecolors='#9467bd', linewidth = 0.75)
    plt.legend(['Student\'s t', 'Normal', 'Pareto', 'Empirical'])
    plt.xlabel('Lower Tail Innovation')
    plt.ylabel('Cumulative Distribution')
    plt.tight_layout()
    fig_l.savefig('.\\fig\\lower_fitted_' + indices[i] + '.png')

    fig_r = plt.figure(figsize=(4, 4))
    plt.plot(x_r, ty_r[0], linewidth = 0.75)
    plt.plot(x_r, ny_r[0], linewidth = 0.75)
    plt.plot(x_r, py_r[0], linewidth = 0.75)
    plt.scatter(ex_r, ey_r, facecolors='none', edgecolors='#9467bd', linewidth = 0.75)
    plt.legend(['Student\'s t', 'Normal', 'Pareto', 'Empirical'])
    plt.xlabel('Upper Tail Innovation')
    plt.ylabel('Cumulative Distribution')
    plt.tight_layout()
    fig_r.savefig('.\\fig\\upper_fitted_' + indices[i] + '.png')

    upper = eng.eval('pdist.UpperParameters;')
    ushape = upper[0][0]
    uscale = upper[0][1]
    lower = eng.eval('pdist.LowerParameters;')
    lshape = lower[0][0]
    lscale = lower[0][1]

    params[indices[i]] = [ushape, uscale, lshape, lscale]

    y = []
    py = eng.eval('cdf(pdist, dta);')
    ty = eng.eval('cdf(tdist, dta);')
    for j in range(len(res)):
        if res[indices[i]][j] > upper_threshold[i] or res[indices[i]][j] < lower_threshold[i]:
            y.append(py[0][j])
        else:
            y.append(ty[0][j])
    copula_data[indices[i]] = y

copula_data.index = res.index

params.to_csv('.\\data\\pareto.csv')
copula_data.to_csv('.\\data\\copula_data.csv')