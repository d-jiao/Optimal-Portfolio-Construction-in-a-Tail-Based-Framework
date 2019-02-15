import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

rtd = pd.read_csv('.\\data\\rtd.csv', index_col = 0)
alpha = 0.05
n = len(rtd)

'''
two assets
'''
weights = np.arange(0, 1, 0.0001)
var = []
cvar = []
tr = []
etr = []
critical_weight = []
for weight in weights:
    r = rtd.shsz.values * weight + rtd.spx.values * (1 - weight)
    r.sort()
    var.append(-r[int(n * alpha)])
    cvar.append(-r[: int(n * alpha)].mean())
    tr.append(r[n - int(n * alpha)])
    etr.append(r[n - int(n * alpha): ].mean())
    if len(cvar) > 1:
        if (cvar[-1] < 0.025 and cvar[-2] > 0.025) \
                or (cvar[-1] > 0.025 and cvar[-2] < 0.025):
            critical_weight.append(weight)

lfig = plt.figure(figsize = (4, 5))
plt.plot(weights, var, linewidth = 0.75)
plt.plot(weights, cvar, linewidth = 0.75)
plt.legend(['VaR', 'CVaR'])
plt.xlabel('Weight of CSI 300')
plt.ylabel('Left Tail Statistics')
plt.axvspan(critical_weight[0], critical_weight[1], alpha = 0.2)
plt.hlines(0.025, 0.0, 1.0, linestyle = 'dotted', linewidth = 0.75)
lfig.savefig('.\\fig\\left_tail.png')

rfig = plt.figure(figsize = (4, 5))
plt.plot(weights, tr, linewidth = 0.75)
plt.plot(weights, etr, linewidth = 0.75)
plt.legend(['Tail Return', 'Expected Tail Return'])
plt.xlabel('Weight of CSI 300')
plt.ylabel('Right Tail Statistics')
plt.axvspan(critical_weight[0], critical_weight[1], alpha = 0.2)
level = max(etr[int(critical_weight[0] * 10000)], \
            etr[int(critical_weight[1] * 10000)])
plt.hlines(level, 0.0, 1.0, linestyle = 'dotted', linewidth = 0.75)
rfig.savefig('.\\fig\\right_tail.png')

'''
three assets
'''
xweights = np.arange(0, 1, 0.01)
yweights = np.arange(0, 1, 0.01)
nn = len(xweights)
x, y = np.meshgrid(xweights, yweights)
var3 = []
cvar3 = []
tr3 = []
etr3 = []
for xweight in xweights:
    for yweight in yweights:
        var3_ = []
        cvar3_ = []
        tr3_ = []
        etr3_ = []
        r = rtd.shsz.values * xweight + rtd.spx.values * yweight \
            + rtd.ukx.values * (1 - xweight - yweight)
        r.sort()
        var3.append(-r[int(n * alpha)])
        cvar3.append(-r[: int(n * alpha)].mean())
        tr3.append(r[n - int(n * alpha)])
        etr3.append(r[n - int(n * alpha): ].mean())
var3 = np.array(var3).reshape((nn, nn))
cvar3 = np.array(cvar3).reshape((nn, nn))
tr3 = np.array(tr3).reshape((nn, nn))
etr3 = np.array(etr3).reshape((nn, nn))

lfig3 = plt.figure(figsize = (4, 4))
lax = lfig3.add_subplot(111, projection = '3d')
var3surf = lax.plot_surface(X = x, Y = y, Z = var3)
cvar3surf = lax.plot_surface(X = x, Y = y, Z = cvar3)
lax.set_xlabel('\n Weight of CSI 300', linespacing=1.5)
lax.set_ylabel('\n Weight of S&P 500', linespacing=1.5)
lax.set_zlabel('\n Left Tail Statistics', linespacing=3)
lax.legend(['VaR', 'CVaR'])
lfig3.savefig('.\\fig\\left_tail3.png')

rfig3 = plt.figure(figsize = (4, 4))
rax = rfig3.add_subplot(111, projection = '3d')
rax.plot_surface(X = x, Y = y, Z = tr3)
rax.plot_surface(X = x, Y = y, Z = etr3)
rax.set_xlabel('\n Weight of CSI 300', linespacing=1.5)
rax.set_ylabel('\n Weight of S&P 500', linespacing=1.5)
rax.set_zlabel('\n Right Tail Statistics', linespacing=3)
rax.legend(['Tail Return', 'Expected Tail Return'])
rfig3.savefig('.\\fig\\right_tail3.png')
