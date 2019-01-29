import cvxpy as cvx
import pandas as pd
import numpy as np

def tail_opt_ip(r, a, b, g, m):
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
    # position constraint
    cons += [x == x_p - x_m]
    cons += [sum(x) == 1]
    # leverage constraint
    cons += [m * sum(x_m) <= sum(x_p)]
    cons += [x_m >= 0, x_p >= 0]
    # objective reformulation and left tail constraint
    cons += [sum(y) == np.ceil((1 - b) * n)]
    cons += [ksai + 1 / (np.ceil(1 - a) * n) * sum(z) <= g]
    for i in range(n):
        cons += [X[i][:] <= M * y[i], X[i][:] >= -M * y[i]]
        cons += [z[i] >= sum(-r[i][:] * x) - ksai]
        cons += [X[i][:] - x <= 2 * M * (1 - y[i])]
        cons += [X[i][:] - x >= -2 * M * (1 - y[i])]

    prob = cvx.Problem(obj, cons)
    prob.solve(solver = cvx.GUROBI)
    if (prob.status == 'optimal'):
        return prob.value, x.value
    else:
        return False

if __name__ == '__main__':
    a = 0.05
    b = 0.05
    g = 0.0001
    m = 2.5
    rtd = pd.read_csv('rtd.csv', index_col=0)
    r = np.array(rtd.iloc[-60:, :])
    v, w = tail_opt_ip(r, a, b, g, m)
    print(v, w)
