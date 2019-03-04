import pandas as pd
import numpy as np
import pandas as pd
from gurobipy import *

def min_vol(cov, m):
    '''
    :param cov: the covariance matrix in np.array
    :param m: leverage constraint
    :return:
    '''
    d = len(cov)
    M = (m + 1) / (m - 1)

    model = Model('min_vol')
    x = model.addVars(d, lb=-M, ub=M)
    x_p = model.addVars(d, lb=0, ub=M)
    x_m = model.addVars(d, lb=0, ub=M)

    obj = QuadExpr(quicksum(x[i] * cov[i, j] * x[j] for i in range(d) for j in range(d)))
    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstr(quicksum(x[i] for i in range(d)), GRB.EQUAL, 1)
    model.addConstrs((x_p[j] >= x[j] for j in range(d)))
    model.addConstrs((x_m[j] >= -x[j] for j in range(d)))
    model.addConstr(m * quicksum(x_m[i] for i in range(d)), GRB.LESS_EQUAL, quicksum(x_p[i] for i in range(d)))

    model.optimize()

    vol = obj.getValue()
    weight = [x[i].x for i in range(d)]
    return vol, weight

def max_sharpe(cov, mu, m):
    '''
    :param cov: covariance matrix
    :param mu: expected return
    :param m: leverage constraint
    :return:
    '''
    d = len(cov)

    model = Model('max_sharpe')
    x = model.addVars(d, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    x_p = model.addVars(d)
    x_m = model.addVars(d)
    alpha = model.addVar()

    obj = QuadExpr(quicksum(x[i] * cov[i, j] * x[j] for i in range(d) for j in range(d)))
    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstr(quicksum(x[i] for i in range(d)), GRB.EQUAL, alpha)
    model.addConstr(quicksum(x[i] * mu[i] for i in range(d)), GRB.EQUAL, 1)
    model.addConstrs((x_p[j] >= x[j] for j in range(d)))
    model.addConstrs((x_m[j] >= -x[j] for j in range(d)))
    model.addConstr(m * quicksum(x_m[i] for i in range(d)), GRB.LESS_EQUAL, quicksum(x_p[i] for i in range(d)))

    model.optimize()

    sharpe = 1 / np.sqrt(obj.getValue())
    weight = [x[i].x / alpha for i in range(d)]
    return sharpe, weight

def mean_cvar(r, a, g, m, c, xb = None):
    '''
    :param r: return series
    :param a: tail threshold
    :param m: leverage constraint
    :param g: level of cvar
    :return:
    '''
    model = Model('mean_cvar')

    # parameters and data
    M = (m + 1) / (m - 1)
    r0 = r[:, -1].mean() / 100
    r = r[:, :-1]
    (n, d) = r.shape
    mu = np.mean(r, axis=0)

    # variables
    x = np.empty(d, dtype=object)
    x_p = np.empty(d, dtype=object)
    x_m = np.empty(d, dtype=object)
    u_p = np.empty(d, dtype=object)
    u_m = np.empty(d, dtype=object)
    y = np.empty(n, dtype=object)
    z = np.empty(n, dtype=object)
    for i in range(n):
        for j in range(d):
            x[j] = model.addVar(lb=-M, ub=M, name='x%d' % j)
            x_p[j] = model.addVar(lb=0, ub=M, name='x_p%d' % j)
            x_m[j] = model.addVar(lb=0, ub=M, name='x_m%d' % j)
            u_p[j] = model.addVar(lb = 0, ub = 2 * M, name = 'u_p%d' % j)
            u_m[j] = model.addVar(lb = 0, ub = 2 * M, name = 'u_m%d' % j)
        y[i] = model.addVar(vtype = GRB.BINARY, name = 'y%d' % i)
        z[i] = model.addVar(lb = 0, name = 'z%d' % i)
    ksai = model.addVar(lb=-1, ub=1, name='ksai')
    x0 = model.addVar(lb=0, ub=M, name='x0')

    # objective function
    obj = LinExpr(mu, x) + x0 * r0
    model.setObjective(obj, GRB.MAXIMIZE)

    # position constraint
    if xb:
        model.addConstr(x0 + LinExpr(np.ones(d), x) + LinExpr(c, x_m) + LinExpr(c, x_m), GRB.EQUAL, 1)
        model.addConstrs((x[j] - xb[j] == u_p[j] - u_m[j] for j in range(d)))
    else:
        model.addConstr(LinExpr(np.ones(d + 1), np.append(x, x0)) + LinExpr(c, x_m) + LinExpr(c, x_p), GRB.EQUAL, 1)

    # leverage constraint
    model.addConstr(m * LinExpr(np.ones(d), x_m), GRB.LESS_EQUAL, LinExpr(np.ones(d), x_p))
    model.addConstrs((x[j] == x_p[j] - x_m[j] for j in range(d)))
    model.addConstrs((x_p[j] >= x[j] for j in range(d)))
    model.addConstrs((x_m[j] >= -x[j] for j in range(d)))

    # left tail constraint
    model.addConstr(ksai + 1 / np.ceil(a * n) * LinExpr(np.ones(n), z), GRB.LESS_EQUAL, g + x0 * r0)
    model.addConstrs((z[i] >= -LinExpr(r[i], x) - ksai for i in range(n)))

    model.optimize()
    model.printAttr('ObjVal')
    return [x[i].x for i in range(d)], obj.getValue()

if __name__ == '__main__':
    a = 0.05
    b = 0.95
    g = 0.5
    m = 3
    c = np.ones(8) * 0.0001

    indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx', 'r0']
    r = pd.read_csv('.\\data\\rtd.csv', index_col=0)
    r = r[indices]
    r = np.array(r.iloc[-1000:, :])

    # sharpe, weight = min_vol(cov, m)
    # vol, weight = max_sharpe(cov, mu, m)
    print(mean_cvar(r, a, g, m, c))
