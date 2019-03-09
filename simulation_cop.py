import sys
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from scipy.stats import t
from utils import *

def ppf_evt(x, ul, ur, tparams, gpdparams):
    '''
    :param x: the value of the cdf
    :param ul: left-tail threshold
    :param ur: right-tail threshold
    :param tparams: parameters for the location-scale t distribution
    :param gpdparams: parameters for the generalized Pareto distribution
    :return:
    '''
    if x >= ur:
        return genpareto.ppf(x, c = gpdparams[0], loc = gpdparams[2], scale = gpdparams[1])
    elif x <= ul:
        return genpareto.ppf(x, c = gpdparams[3], loc = gpdparams[5], scale = gpdparams[4])
    else:
        return t.ppf(x, df = tparams[2], loc = tparams[0], scale = tparams[1])

def h(u1, u2, rho, df):
    if u1 == 1:
        u1 -= sys.float_info.epsilon
    if u1 == 0:
        u1 += sys.float_info.epsilon
    if u2 == 1:
        u2 -= sys.float_info.epsilon
    if u2 == 0:
        u2 += sys.float_info.epsilon
    u1_ = t.ppf(u1, df + 1)
    u2_ = t.ppf(u2, df)
    a = (df + u2_ ** 2) * (1 - rho ** 2) / (df + 1)
    b = u1_ * np.sqrt(a) + rho * u2_
    return t.cdf(b, df)

def sim(d, n, ppf, rhos, dfs):
    x = np.zeros((n, d))
    for i in range(n):
        v = np.zeros((d, d))
        w = np.random.rand(d)
        v[0, 0] = w[0]
        x[i, 0] = ppf[0](v[0, 0])
        for j in range(1, d):
            v[j, 0] = w[j]
            for k in range(j - 1, -1, -1):
                v[j, 0] = h(v[j, 0], v[k, k], rhos[k, j], dfs[k, j])
            x[i, j] = ppf[j](v[j, 0])
            if j == d - 1:
                break
            for k in range(j):
                v[j, k + 1] = h(v[j, k], v[k, k], rhos[k, j], dfs[k, j])
    return x

if __name__ == '__main__':
    ul = [0.020825586, 0.020081815, 0.022685013, 0.031610264, 0.007809595, 0.034213462, 0.010412793, 0.025288211]
    ur = [0.9758274, 0.9877278, 0.9873559, 0.9739680, 0.9854965, 0.9747118, 0.9877278, 0.9817776]
    indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx']

    tparams = pd.read_csv('.\\data\\fitted_tparams.csv', index_col = 0)
    gpdparams = pd.read_csv('.\\data\\pareto.csv', index_col = 0)
    # garchparams = pd.read_csv('.\\data\\garch_param.csv', index_col=0, header=None)
    r = pd.read_csv('.\\data\\rtd.csv', index_col=0)
    copparams = pd.read_csv('.\\data\\cop_param.csv', index_col = 0)

    n = 2700
    d = 8
    # c = np.array(garchparams.iloc[0, :])
    r.index = pd.to_datetime(r.index)
    month_ends = month_ends(r.index)
    r0 = r.r0[month_ends[12 * 9 - 1] + 1:].mean()

    ppfs = []
    for j in range(d):
        ppf = lambda x: ppf_evt(x, ul[j], ur[j], tparams[indices[j]], gpdparams[indices[j]])
        ppfs.append(ppf)

    rhos = np.zeros((d, d))
    rhos[np.triu_indices(d, 1)] = copparams['corr']

    dfs = np.zeros((d, d))
    dfs[np.triu_indices(d, 1)] = copparams['dof']

    for k in range(1):
        sample = sim(d, n, ppfs, rhos, dfs) #+ c
        sample = pd.DataFrame(sample)
        sample.columns = indices
        sample['r0'] = np.ones(n) * r0
        sample.to_csv('.\\data\\simulated' + str(k + 1) + '.csv')