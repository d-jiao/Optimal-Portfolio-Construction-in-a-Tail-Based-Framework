import pandas as pd
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from gurobipy import *

def min_vol(r, m = 3, c = None, xb = None):
    '''
    :param cov: the covariance matrix in np.array
    :param m: leverage constraint
    :return:
    '''

    model = Model('min_vol')

    # parameters
    cov = np.cov(r, rowvar=0)
    d = len(cov)
    M = (m + 1) / (m - 1)

    # variables
    x = np.empty(d, dtype=object)
    x_p = np.empty(d, dtype=object)
    x_m = np.empty(d, dtype=object)
    for j in range(d):
        x[j] = model.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY, name='x%d' % j)
        x_p[j] = model.addVar(lb = 0, name='x_p%d' % j)
        x_m[j] = model.addVar(lb = 0, name='x_m%d' % j)

    # objective
    obj = QuadExpr(quicksum(x[i] * cov[i, j] * x[j] for i in range(d) for j in range(d)))
    model.setObjective(obj, GRB.MINIMIZE)

    # position constraint
    model.addConstrs((x[j] == x_p[j] - x_m[j] for j in range(d)))
    model.addConstrs((x_p[j] >= x[j] for j in range(d)))
    model.addConstrs((x_m[j] >= -x[j] for j in range(d)))
    if xb is None:
        model.addConstr(LinExpr(np.ones(d), x) + LinExpr(c, x_m) + LinExpr(c, x_p), GRB.EQUAL, 1)
    else:
        u_p = np.empty(d, dtype=object)
        u_m = np.empty(d, dtype=object)
        for j in range(d):
            u_p[j] = model.addVar(lb=0, ub=2 * M, name='u_p%d' % j)
            u_m[j] = model.addVar(lb=0, ub=2 * M, name='u_m%d' % j)
        model.addConstr(LinExpr(np.ones(d), x) + LinExpr(c, u_m) + LinExpr(c, u_p), GRB.EQUAL,
                        xb.sum())
        model.addConstrs((x[j] - xb[j] == u_p[j] - u_m[j] for j in range(d)))

    # leverage constraint
    model.addConstr(m * quicksum(x_m[i] for i in range(d)), GRB.LESS_EQUAL, quicksum(x_p[i] for i in range(d)))

    model.optimize()

    vol = obj.getValue()
    weight = np.array([x[i].x for i in range(d)])
    return weight, vol

def max_sharpe(r, m = 3, c = None, xb = None):
    '''
    No cash is involved
    :param r:
    :param m:
    :param c:
    :param xb:
    :return:
    '''

    model = Model('max_sharpe')

    # parameters
    cov = np.cov(r, rowvar = 0)
    mu = np.mean(r, axis = 0)
    d = len(cov)
    if c is None:
        c = np.zeros(d)

    # variables
    x = np.empty(d, dtype=object)
    x_p = np.empty(d, dtype=object)
    x_m = np.empty(d, dtype=object)
    for j in range(d):
        x[j] = model.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY, name='x%d' % j)
        x_p[j] = model.addVar(lb = 0, name='x_p%d' % j)
        x_m[j] = model.addVar(lb = 0, name='x_m%d' % j)
    alpha = model.addVar()

    # objective
    obj = QuadExpr(quicksum(x[i] * cov[i, j] * x[j] for i in range(d) for j in range(d)))
    model.setObjective(obj, GRB.MINIMIZE)

    # position constraint
    model.addConstr(LinExpr(np.ones(d), x) + LinExpr(c, x_m) + LinExpr(c, x_p), GRB.EQUAL, alpha)
    model.addConstrs((x[j] == x_p[j] - x_m[j] for j in range(d)))
    model.addConstrs((x_p[j] >= x[j] for j in range(d)))
    model.addConstrs((x_m[j] >= -x[j] for j in range(d)))

    # return constraint
    model.addConstr(quicksum(x[i] * mu[i] for i in range(d)), GRB.EQUAL, 1)

    # leverage constraint
    model.addConstr(m * quicksum(x_m[i] for i in range(d)), GRB.LESS_EQUAL, quicksum(x_p[i] for i in range(d)))

    model.optimize()

    # sometimes the model might be infeasible due to negative mu and small m
    try:
        sharpe = 1 / np.sqrt(obj.getValue())
    except:
        if xb is None:
            return np.ones(d) / (d + c.sum())
        else:
            return xb, None

    weight = np.array([x[i].x / alpha.x for i in range(d)])
    # transaction cost
    if xb is None:
        tcost = abs(weight).dot(c)
    else:
        tcost = abs(weight - xb).dot(c)
    weight *= (1 - tcost)

    return weight, sharpe

def mean_cvar(r, a = 0.05, g = 0.02, m = 3, c = None, xb = None):
    '''
    :param r: return series, an np.array
    :param a: tail threshold, a scalar
    :param g: level of cvar, a scalar
    :param m: leverage constraint, a scalar
    :param c: transaction cost, an np.array
    :param xb: starting pos, last one cash, an np.array
    :return:
    '''
    model = Model('mean_cvar')

    # parameters and data
    M = (m + 1) / (m - 1)
    mu = np.mean(r, axis=0)
    mu[-1] /= 100
    r = r[:, :-1]
    (n, d) = r.shape
    print(mu)
    if c is None:
        c = np.zeros(d)

    # variables
    x = np.empty(d, dtype = object)
    x_p = np.empty(d, dtype = object)
    x_m = np.empty(d, dtype = object)
    z = np.empty(n, dtype = object)
    for j in range(d):
        x[j] = model.addVar(lb=-M, ub=M, name='x%d' % j)
        x_p[j] = model.addVar(lb=0, ub=M, name='x_p%d' % j)
        x_m[j] = model.addVar(lb=0, ub=M, name='x_m%d' % j)
    for i in range(n):
        z[i] = model.addVar(lb = 0, name = 'z%d' % i)
    ksai = model.addVar(lb = -1, ub = 1, name = 'ksai')
    x0 = model.addVar(lb = 0, name = 'x0')

    # objective function
    obj = LinExpr(mu, np.append(x, x0))
    model.setObjective(obj, GRB.MAXIMIZE)

    # position constraint
    if xb is None:
        model.addConstr(LinExpr(np.ones(d + 1), np.append(x, x0)) + LinExpr(c, x_m) + LinExpr(c, x_p), GRB.EQUAL, 1)
    else:
        u_p = np.empty(d, dtype=object)
        u_m = np.empty(d, dtype=object)
        for j in range(d):
            u_p[j] = model.addVar(lb=0, ub=2 * M, name='u_p%d' % j)
            u_m[j] = model.addVar(lb=0, ub=2 * M, name='u_m%d' % j)
        model.addConstr(LinExpr(np.ones(d + 1), np.append(x, x0)) + LinExpr(c, u_m) + LinExpr(c, u_p), GRB.EQUAL,
                        xb.sum())
        model.addConstrs((x[j] - xb[j] == u_p[j] - u_m[j] for j in range(d)))

    # leverage constraint
    model.addConstr(m * LinExpr(np.ones(d), x_m), GRB.LESS_EQUAL, LinExpr(np.ones(d), x_p) + x0)
    model.addConstrs((x[j] == x_p[j] - x_m[j] for j in range(d)))
    model.addConstrs((x_p[j] >= x[j] for j in range(d)))
    model.addConstrs((x_m[j] >= -x[j] for j in range(d)))

    # left tail constraint
    model.addConstr(ksai + 1 / np.ceil(a * n) * LinExpr(np.ones(n), z), GRB.LESS_EQUAL, g + x0 * mu[-1], name = 'cvar')
    model.addConstrs((z[i] >= -LinExpr(r[i], x) - ksai for i in range(n)))

    model.optimize()
    # tVars = PrettyTable(['Variable Name', ' Value', 'ReducedCost', 'SensLow', ' SensUp'])
    # for eachVar in model.getVars():
    #     tVars.add_row([eachVar.varName, eachVar.x, eachVar.RC, eachVar.SAObjLow, eachVar.SAObjUp])
    # print(tVars)
    # shadow_price = model.getAttr(GRB.Attr.Pi)
    # print(shadow_price)

    weight = np.array([x[i].x for i in range(d)] + [x0.x])
    return weight, obj.getValue()

def equal_weighted(d, c = None, xb = None):
    '''
    :param c:
    :param xb: starting pos, last one cash, an np.array
    :return:
    '''
    if c is None:
        c = np.zeros(d)
    if xb is None:
        # 1 = (n + 1) * w + w * sum(ci)
        return np.ones(d + 1) / (d + 1 + c.sum())
    else:
        # need x1 <= x2 <= ... <= xk <= w < xk+1 <= ... <= xn
        w0 = xb.sum()
        xb = xb[:-1]
        xb_ = np.array([x for x, _ in sorted(zip(xb, c))])
        c = np.array([x for _, x in sorted(zip(xb, c))])
        for i in range(1, d):
            wd = xb_[:i].dot(c[:i])
            wu = xb_[i:].dot(c[i:])
            cd = c[:i].sum()
            cu = c[i:].sum()
            w = (w0 + wd - wu) / (d + 1 + cd - cu)
            if (w >= xb_[i - 1]) and (w <= xb_[i]):
                return np.ones(d + 1) * w

if __name__ == '__main__':
    a = 0.05
    b = 0.95
    g = 0.02
    m = 3
    c = np.ones(8) * 0.0001

    indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx', 'r0']
    r = pd.read_csv('.\\data\\rtd.csv', index_col=0)
    r = r[indices]
    # r = np.array(r.iloc[2645-30:2645, :])
    r = np.array(r.iloc[-90:, :])

    # weight, sharpe = max_sharpe(r[:,:-1], c = c, xb = xb)
    weight, cvar = mean_cvar(r = r)
    print(weight)
