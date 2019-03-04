from gurobipy import *
import pandas as pd
import numpy as np

def tail_opt(r, a, b, g, m):
    '''
    :param r: data of return to day
    :param a: confidence level of left tail
    :param b: confidence level of right tail
    :param g: level of left tail
    :param m: short selling constraint
    :return: optimal right tail, weights for the assets
    '''

    model = Model('tail')
    # model.Params.NumericFocus = 1

    # parameters
    M = (m + 1) / (m - 1)
    (n, d) = r.shape

    # # variables
    # x = model.addVars(d, lb = -M, ub = M, name = 'x')
    # x_p = model.addVars(d, lb = 0, ub = M, name = 'x_p')
    # x_m = model.addVars(d, lb = 0, ub = M, name = 'x_m')
    # X = model.addVars(n, d, lb = 0, ub = M, name = 'X')
    # y = model.addVars(n, vtype = GRB.BINARY, name = 'y')
    # z = model.addVars(n, lb = 0, name = 'z')
    #
    # # list reformualtion
    # x_ = [x[i] for i in range(d)]
    # x_p_ = [x_p[i] for i in range(d)]
    # x_m_ = [x_m[i] for i in range(d)]
    # y_ = [y[i] for i in range(n)]
    # z_ = [z[i] for i in range(n)]

    # variables
    x = np.empty(d, dtype=object)
    x_p = np.empty(d, dtype=object)
    x_m = np.empty(d, dtype=object)
    X = np.empty((n, d), dtype=object)
    y = np.empty(n, dtype=object)
    z = np.empty(n, dtype=object)

    for i in range(n):
        for j in range(d):
            x[j] = model.addVar(lb = -M, ub = M, name='x%d' % j)
            x_p[j] = model.addVar(lb = 0, ub = M, name='x_p%d' % j)
            x_m[j] = model.addVar(lb = 0, ub = M, name='x_m%d' % j)
            X[i, j] = model.addVar(lb = 0, ub = M, name="X%d%d" %(i, j))
        y[i] = model.addVar(vtype = GRB.BINARY, name = 'y%d' % i)
        # y[i] = model.addVar(lb = 0, ub = 1, name = 'y%d' % i)
        z[i] = model.addVar(lb = 0, name = 'z%d' % i)
    ksai = model.addVar(lb=-1, ub=1, name='ksai')

    # objective function
    etr = LinExpr(r.reshape(n * d), [X[i, j] for i in range(n) for j in range(d)])
    model.setObjective(etr / np.ceil((1 - b) * n), GRB.MAXIMIZE)

    # objective constraints
    model.addConstr(LinExpr(np.ones(n), y), GRB.EQUAL, np.ceil((1 - b) * n))
    model.addConstrs((X[i, j] <= M * y[i] for j in range(d) for i in range(n)))
    model.addConstrs((X[i, j] >= -M * y[i] for j in range(d) for i in range(n)))
    model.addConstrs((X[i, j] - x[j] <= 2 * M * (1 - y[i]) for j in range(d) for i in range(n)))
    model.addConstrs((X[i, j] - x[j] >= -2 * M * (1 - y[i]) for j in range(d) for i in range(n)))
    # for i in range(n):
    #     model.addConstrs((X[i, j] <= M * y[i] for j in range(d)))
    #     model.addConstrs((X[i, j] >= -M * y[i] for j in range(d)))
    #     model.addConstrs((X[i, j] - x[j] <= 2 * M * (1 - y[i]) for j in range(d)))
    #     model.addConstrs((X[i, j] - x[j] >= -2 * M * (1 - y[i]) for j in range(d)))

    # position constraint
    model.addConstr(LinExpr(np.ones(d), x), GRB.EQUAL, 1)

    # leverage constraint8GB RAM, u5-8250U CPU
    model.addConstr(m * LinExpr(np.ones(d), x_m), GRB.LESS_EQUAL, LinExpr(np.ones(d), x_p))
    model.addConstrs((x_p[j] >= x[j] for j in range(d)))
    model.addConstrs((x_m[j] >= -x[j] for j in range(d)))

    # left tail constraint
    model.addConstr(ksai + 1 / np.ceil(a * n) * LinExpr(np.ones(n), z), GRB.LESS_EQUAL,  g)
    model.addConstrs((z[i] >= -LinExpr(r[i], x) - ksai for i in range(n)))

    model.optimize()
    model.printAttr('ObjVal')
    print([y[i].x for i in range(n)])

if __name__ == '__main__':
    a = 0.05
    b = 0.95
    g = 0.02
    m = 3
    r = pd.read_csv('.\\data\\rtd.csv', index_col=0)
    indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx']
    r = r[indices]
    r = np.array(r.iloc[-90 :, :])
    tail_opt(r, a, b, g, m)
    # 0.0256285