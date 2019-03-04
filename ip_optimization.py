import cvxpy as cvx
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

    # parameters
    M = (m + 1) / (m - 1)
    (n, d) = r.shape

    # variables
    x = cvx.Variable(d)
    x_p = cvx.Variable(d)
    x_m = cvx.Variable(d)
    X = cvx.Variable((n, d))
    y = cvx.Variable(n, boolean = True)
    z = cvx.Variable(n)
    ksai = cvx.Variable(1)

    # objective function
    obj = cvx.Maximize(cvx.sum(cvx.multiply(X, r)) / np.ceil((1 - b) * n))

    # constraints
    cons = []

    # objective constraints
    cons += [sum(y) == np.ceil((1 - b) * n)]
    for i in range(n):
        cons += [X[i][:] <= M * y[i], X[i][:] >= -M * y[i]]
        cons += [X[i][:] - x <= 2 * M * (1 - y[i])]
        cons += [X[i][:] - x >= -2 * M * (1 - y[i])]

    # position constraint
    cons += [cvx.sum(x) == 1]

    # leverage constraint
    cons += [m * cvx.sum(x_m) <= cvx.sum(x_p)]
    cons += [x == x_p - x_m]
    cons += [x_m >= 0, x_m >= -x]
    cons += [x_p >= 0, x_p >= x]

    # left tail constraint
    cons += [ksai + 1 / np.ceil(a * n) * cvx.sum(z) <= g]
    cons += [z >= 0]
    cons += [z >= -r * x - ksai]

    prob = cvx.Problem(obj, cons)
    prob.solve(solver = cvx.GUROBI)
    if (prob.status == 'optimal'):
        return prob.value, x.value
    else:
        return False

if __name__ == '__main__':
    a = 0.05
    b = 0.95
    g = 0.02
    m = 3
    r = pd.read_csv('.\\data\\rtd.csv', index_col=0)
    indices = ['csi', 'spx', 'nky', 'ukx', 'hsi', 'cac', 'dax', 'asx']
    r = r[indices]
    r = np.array(r.iloc[-30 :, :])
    v, w = tail_opt(r, a, b, g, m)
    print(v, w)
