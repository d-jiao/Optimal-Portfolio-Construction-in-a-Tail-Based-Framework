import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

coppdf1 = pd.read_csv('.\\data\\coppdf1.csv', index_col = 0)
coppdf2 = pd.read_csv('.\\data\\coppdf2.csv', index_col = 0)
coppdf3 = pd.read_csv('.\\data\\coppdf3.csv', index_col = 0)

copdat1 = pd.read_csv('.\\data\\copdat1.csv', index_col = 0)
copdat2 = pd.read_csv('.\\data\\copdat2.csv', index_col = 0)
copdat3 = pd.read_csv('.\\data\\copdat3.csv', index_col = 0)

copdat1 = norm.ppf(copdat1) / 10
copdat2 = norm.ppf(copdat2) / 10
copdat3 = norm.ppf(copdat3) / 10

u1 = np.arange(0.02, 1, 0.02)
u2 = np.arange(0.02, 1, 0.02)
u1, u2 = np.meshgrid(u1, u2)

fig = plt.figure(figsize = (8, 3))

ax1 = fig.add_subplot(131, projection = '3d')
ax1.plot_surface(X = u1, Y = u2, Z = coppdf1)
ax1.view_init(elev=None, azim=210)
ax1.set_title(r'$\rho = 0.9, \nu = 9$')

ax2 = fig.add_subplot(132, projection = '3d')
ax2.plot_surface(X = u1, Y = u2, Z = coppdf2)
ax2.view_init(elev=None, azim=210)
ax2.set_title(r'$\rho = 0.9, \nu = 3$')

ax3 = fig.add_subplot(133, projection = '3d')
ax3.plot_surface(X = u1, Y = u2, Z = coppdf3)
ax3.view_init(elev=None, azim=210)
ax3.set_title(r'$\rho = 0.3, \nu = 9$')

fig.savefig('.\\fig\\coppdf.png')

titles = [r'$\rho = 0.9, \nu = 9$',
          r'$\rho = 0.9, \nu = 3$',
          r'$\rho = 0.3, \nu = 9$']
fig, ax = plt.subplots(2, 3, figsize = (9, 6))
cvar = []
for (i, copdat) in enumerate([copdat1, copdat2, copdat3]):
    ax[0][i].plot(copdat[:,0], linewidth = 0.5)
    ax[0][i].plot(copdat[:,1], linewidth = 0.5, alpha = 0.8)
    ax[1][i].scatter(copdat[:,0], copdat[:,1], s = 0.1)
    ax[0][i].set_title(titles[i])
    ax[0][i].xaxis.set_major_formatter(plt.NullFormatter())
    r = copdat.mean(axis = 1)
    r.sort()
    cvar.append(r[:int(0.05 * len(r))].mean())
plt.tight_layout()

fig.savefig('.\\fig\\copsim.png')
