from gurobipy import *
import pandas as pd
import numpy as np

def tail_opt(r, a = 0.05, b = 0.95, g = 0.02, m = 3, c = None, xb = None):
    '''
    :param r: data of return to day
    :param a: confidence level of left tail
    :param b: confidence level of right tail
    :param g: level of left tail
    :param m: short selling constraint
    :return: optimal right tail, weights for the assets
    '''

    model = Model('tail')
    # model.Params.MIPFocus = 1
    # model.Params.MIPGapAbs = 1e-3
    # model.Params.MIPSepCuts = 2
    # model.Params.MIRCuts = 2
    # model.Params.ModKCuts = 2
    # model.Params.NetworkCuts = 2

    # parameters
    M = (m + 1) / (m - 1)
    r0 = r[:, -1].mean() / 100
    r = r[:, :-1]
    (n, d) = r.shape
    if c is None:
        c = np.zeros(d)

    # variables
    x = np.empty(d, dtype = object)
    x_p = np.empty(d, dtype=object)
    x_m = np.empty(d, dtype=object)
    X = np.empty((n, d), dtype = object)
    y = np.empty(n, dtype = object)
    z = np.empty(n, dtype = object)
    for i in range(n):
        for j in range(d):
            X[i, j] = model.addVar(lb = -M, ub = M, name = "X%d%d" %(i, j))
        y[i] = model.addVar(vtype = GRB.BINARY, name = 'y%d' % i)
        z[i] = model.addVar(lb = 0, name = 'z%d' % i)
    for j in range(d):
        x[j] = model.addVar(lb=-M, ub=M, name='x%d' % j)
        x_p[j] = model.addVar(lb=0, ub=M, name='x_p%d' % j)
        x_m[j] = model.addVar(lb=0, ub=M, name='x_m%d' % j)
    ksai = model.addVar(lb = -1, ub = 1, name = 'ksai')
    x0 = model.addVar(lb = 0, ub = 1, name = 'x0')

    # objective function
    etr = LinExpr(r.reshape(n * d), [X[i, j] for i in range(n) for j in range(d)])
    obj = etr / np.ceil((1 - b) * n) + x0 * r0
    model.setObjective(obj, GRB.MAXIMIZE)

    # objective constraints
    model.addConstr(LinExpr(np.ones(n), y), GRB.LESS_EQUAL, np.ceil((1 - b) * n))
    model.addConstrs((X[i, j] <= M * y[i] for j in range(d) for i in range(n)))
    model.addConstrs((X[i, j] >= -M * y[i] for j in range(d) for i in range(n)))
    model.addConstrs((X[i, j] - x[j] <= 2 * M * (1 - y[i]) for j in range(d) for i in range(n)))
    model.addConstrs((X[i, j] - x[j] >= -2 * M * (1 - y[i]) for j in range(d) for i in range(n)))

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
    model.addConstr(m * LinExpr(np.ones(d), x_m), GRB.LESS_EQUAL, LinExpr(np.ones(d), x_p))
    model.addConstrs((x[j] == x_p[j] - x_m[j] for j in range(d)))
    model.addConstrs((x_p[j] >= x[j] for j in range(d)))
    model.addConstrs((x_m[j] >= -x[j] for j in range(d)))

    # left tail constraint
    model.addConstr(ksai + 1 / np.ceil(a * n) * LinExpr(np.ones(n), z), GRB.LESS_EQUAL, g + x0 * r0)
    model.addConstrs((z[i] >= -LinExpr(r[i], x) - ksai for i in range(n)))

    model.optimize()
    model.printAttr('ObjVal')
    weight = np.array([x[i].x for i in range(d)] + [x0.x])
    value = obj.getValue()
    print(weight)
    return weight, value

if __name__ == '__main__':
    a = 0.05
    b = 0.95
    g = 0.05
    m = 3
    c = np.ones(8) * 0.0001

    indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx', 'r0']
    r = pd.read_csv('.\\data\\rtd.csv', index_col=0)
    r = r[indices]
    r = np.array(r.iloc[-60:, :])

    tail_opt(r, a, b, g, m, c)