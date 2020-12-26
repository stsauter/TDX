import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize


class Tdx:
    def __init__(self, m, h, r, l):
        self.m = m
        self.h = h
        self.r = r
        self.l = l

    def fit(self, x_train, t_train):
        mu = np.linspace(np.quantile(x_train, 0.01), np.quantile(x_train, 0.99), self.m).reshape(1, self.m)
        phi = norm.pdf(x_train.reshape(x_train.shape[0], 1), loc=mu, scale=self.h)

        tmax = 1
        halflife = 0.1
        psi = -np.log(.5) / (halflife * tmax)
        t = np.max(t_train) - t_train
        w = np.exp(-psi * t)

        u_tilde = np.zeros((self.m, self.m - 1)) + (-np.triu(np.ones((self.m, self.m - 1))))
        for col_idx in range(self.m - 1):
            u_tilde[col_idx + 1][col_idx] = col_idx + 1

        vn = np.zeros((1, u_tilde.shape[1]))
        for col_idx in range(u_tilde.shape[1]):
            vn[0][col_idx] = np.sqrt(u_tilde[:, col_idx].dot(u_tilde[:, col_idx]))
        u = np.matmul(u_tilde, np.diagflat(1 / vn))

        bases = np.tile(t_train.reshape(t_train.shape[0], 1), (1, self.r + 1))
        exponents = np.tile(range(self.r + 1), (t_train.shape[0], 1))
        a = np.power(bases, exponents)
        j = self.j_vect(u, a, self.m, self.r + 1)

        d = np.zeros((self.r + 1, self.r))
        d[1:, :] = np.diagflat(np.ones(self.r))

        one_mat = np.ones(phi.shape)

        np.random.seed(32)
        # x = np.random.rand(self.m - 1, self.r + 1)
        x = np.random.rand(self.r + 1, self.m - 1).transpose()
        x = x.flatten('F')

        gfg = 3

    def j_vect(self, u, a, m, n):
        u_rep = np.zeros((m, (m - 1) * n))
        for i in range(m - 1):
            cols_to_append = np.tile(u[:, i].reshape(u.shape[0], 1), (1, n))
            if i == 0:
                u_rep = cols_to_append
            else:
                u_rep = np.hstack((u_rep, cols_to_append))

        j = np.zeros((m, (m - 1) * n, a.shape[0]))
        for i in range(a.shape[0]):
            j[:, :, i] = u_rep * np.tile(a[i, :], (m, m - 1))

        return j

    def fun_vect(self):
