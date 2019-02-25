import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx']
upper_threshold = [0.03, 0.025, 0.03, 0.02, 0.03, 0.025, 0.03, 0.02]
lower_threshold = [-0.035, -0.025, -0.03, -0.02, -0.035, -0.025, -0.035, -0.02]

# plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(len(indices)):
    mrl_di = pd.read_csv('.\\data\\' + indices[i] + '_mrl_di.csv', index_col = 0)
    tc = pd.read_csv('.\\data\\' + indices[i] + '_tc.csv', index_col = 0)

    fig, ax = plt.subplots(2, 2, figsize = (8, 4))

    ax[0][0].plot(mrl_di['a.x'], mrl_di['mrl'], color = '#1f77b4', linewidth = 0.75)
    ax[0][0].plot(mrl_di['a.x'], mrl_di['lower'], color = '#ff7f0e', linewidth = 0.75, linestyle = ':')
    ax[0][0].plot(mrl_di['a.x'], mrl_di['upper'], color = '#ff7f0e', linewidth = 0.75, linestyle = ':')
    ax[0][0].axvline(upper_threshold[i], linewidth = 0.75, color = 'b', linestyle = '--')
    ax[0][0].set_ylabel('Mean Excess')

    ax[0][1].plot(mrl_di['b.thresh'], mrl_di['b.DI'], linewidth = 0.75)
    ax[0][1].set_ylabel('Dispersion Index')
    ax[0][1].axvline(upper_threshold[i], linewidth=0.75, color = 'b', linestyle = '--')
    M = len(mrl_di)
    conf = 0.95
    conf_sup = chi2.ppf(1 - (1 - conf) / 2, M - 1)/(M - 1)
    conf_inf = chi2.ppf((1 - conf) / 2, M - 1)/(M - 1)
    left = mrl_di['b.thresh'][sum(mrl_di['b.DI'] < conf_inf)]
    right = max(mrl_di['b.thresh'])
    ax[0][1].fill_between([left, max(mrl_di['b.thresh'])], conf_inf, conf_sup, alpha=0.1)

    nt = len(tc)
    obs_sup = max(mrl_di['a.x'])
    obs_inf = min(mrl_di['a.x'])
    dobs = (obs_sup - obs_inf) / nt
    x = [(obs_inf + i * dobs) for i in range(nt)]
    ax[1][0].scatter(x, tc['mscale'], facecolors='none', edgecolors='#1f77b4')
    ax[1][0].vlines(x, tc['lower'], tc['upper'], linewidth = 0.75, color = '#ff7f0e')
    ax[1][0].axvline(upper_threshold[i], linewidth=0.75, color = 'b', linestyle = '--')
    ax[1][0].set_ylabel('Modified Scale')
    ax[1][0].set_xlabel('Threshold')

    ax[1][1].scatter(x, tc['shape'], facecolors='none', edgecolors='#1f77b4')
    ax[1][1].vlines(x, tc['lower.1'], tc['upper.1'], linewidth = 0.75, color = '#ff7f0e')
    ax[1][1].axvline(upper_threshold[i], linewidth=0.75, color = 'b', linestyle = '--')
    ax[1][1].set_ylabel('Shape')
    ax[1][1].set_xlabel('Threshold')

    plt.tight_layout()

    fig.savefig('.\\fig\\pot_right_' + indices[i] + '.png')

for i in range(len(indices)):
    mrl_di = pd.read_csv('.\\data\\' + indices[i] + '_mrl_di_left.csv', index_col = 0)
    tc = pd.read_csv('.\\data\\' + indices[i] + '_tc_left.csv', index_col = 0)

    fig, ax = plt.subplots(2, 2, figsize = (8, 4))

    ax[0][0].plot(-mrl_di['a.x'][::-1], mrl_di['mrl'][::-1], color = '#1f77b4', linewidth = 0.75)
    ax[0][0].plot(-mrl_di['a.x'][::-1], mrl_di['lower'][::-1], color = '#ff7f0e', linewidth = 0.75, linestyle = ':')
    ax[0][0].plot(-mrl_di['a.x'][::-1], mrl_di['upper'][::-1], color = '#ff7f0e', linewidth = 0.75, linestyle = ':')
    ax[0][0].axvline(lower_threshold[i], linewidth = 0.75, color = 'b', linestyle = '--')
    ax[0][0].set_ylabel('Mean Excess')

    ax[0][1].plot(-mrl_di['b.thresh'][::-1], mrl_di['b.DI'][::-1], linewidth = 0.75)
    ax[0][1].set_ylabel('Dispersion Index')
    ax[0][1].axvline(lower_threshold[i], linewidth=0.75, color = 'b', linestyle = '--')
    M = len(mrl_di)
    conf = 0.95
    conf_sup = chi2.ppf(1 - (1 - conf) / 2, M - 1)/(M - 1)
    conf_inf = chi2.ppf((1 - conf) / 2, M - 1)/(M - 1)
    right = -mrl_di['b.thresh'][sum(mrl_di['b.DI'] < conf_inf)]
    left = -max(mrl_di['b.thresh'])
    ax[0][1].fill_between([left, right], conf_inf, conf_sup, alpha=0.1)

    nt = len(tc)
    obs_sup = max(mrl_di['a.x'])
    obs_inf = min(mrl_di['a.x'])
    dobs = (obs_sup - obs_inf) / nt
    x = [-(obs_sup - i * dobs) for i in range(nt)]
    ax[1][0].scatter(x, tc['mscale'][::-1], facecolors='none', edgecolors='#1f77b4')
    ax[1][0].vlines(x, tc['lower'][::-1], tc['upper'][::-1], linewidth = 0.75, color = '#ff7f0e')
    ax[1][0].axvline(lower_threshold[i], linewidth=0.75, color = 'b', linestyle = '--')
    ax[1][0].set_ylabel('Modified Scale')
    ax[1][0].set_xlabel('Threshold')

    ax[1][1].scatter(x, tc['shape'][::-1], facecolors='none', edgecolors='#1f77b4')
    ax[1][1].vlines(x, tc['lower.1'][::-1], tc['upper.1'][::-1], linewidth = 0.75, color = '#ff7f0e')
    ax[1][1].axvline(lower_threshold[i], linewidth=0.75, color = 'b', linestyle = '--')
    ax[1][1].set_ylabel('Shape')
    ax[1][1].set_xlabel('Threshold')

    plt.tight_layout()

    fig.savefig('.\\fig\\pot_left_' + indices[i] + '.png')