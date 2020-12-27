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
        phi[phi == 0] = 1e-5

        tmax = 1
        halflife = 0.1
        psi = -np.log(.5) / (halflife * tmax)
        t = np.max(t_train) - t_train
        w = np.exp(-psi * t)
        w = w.reshape(w.shape[0], 1)

        u_tilde = np.zeros((self.m, self.m - 1)) + (-np.triu(np.ones((self.m, self.m - 1))))
        for col_idx in range(self.m - 1):
            u_tilde[col_idx + 1][col_idx] = col_idx + 1

        vn = np.zeros((1, u_tilde.shape[1]))
        for col_idx in range(u_tilde.shape[1]):
            vn[0][col_idx] = np.sqrt(u_tilde[:, col_idx].dot(u_tilde[:, col_idx]))
        self.u = np.matmul(u_tilde, np.diagflat(1 / vn))

        bases = np.tile(t_train.reshape(t_train.shape[0], 1), (1, self.r + 1))
        exponents = np.tile(range(self.r + 1), (t_train.shape[0], 1))
        a = np.power(bases, exponents)
        j = self.j_vect(self.u, a, self.m, self.r + 1)

        d = np.zeros((self.r + 1, self.r))
        d[1:, :] = np.diagflat(np.ones(self.r))

        np.random.seed(32)
        # x = np.random.rand(self.m - 1, self.r + 1)
        x = np.random.rand(self.r + 1, self.m - 1).transpose()
        x = x.flatten('F')

        additional_params = phi, self.u, a, j, self.l, d, w
        # self.gradient_vect(x, phi, u, a, j, self.l, d, w)
        # self.gradient_vect(x, additional_params)

        res = minimize(self.fun_vect, x, jac=self.gradient_vect, args=additional_params, method='bfgs', options={'disp': True})
        self.coefs = res.x.reshape(self.m - 1, self.r + 1)

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

    def fun_vect(self, x, phi, u, a, j, lamda, d, w):
        # tdx_coefs = x.reshape(self.r + 1, self.m - 1).transpose()
        tdx_coefs = x.reshape(self.m - 1, self.r + 1)
        e = np.exp(u @ tdx_coefs @ a.transpose())
        dot_products = (phi.transpose() * e).sum(axis=0)
        dot_products = dot_products.reshape(1, dot_products.shape[0])
        f = np.sum((w.transpose() * np.log(dot_products)) - (w.transpose() * np.log(np.sum(e, axis=0))))
        if lamda > 0:
            penalty = lamda * np.trace((d.transpose() @ tdx_coefs.transpose()) @ tdx_coefs @ d)
            f = f - penalty
        return -1 * f

    def gradient_vect(self, x, phi, u, a, j, lamda, d, w):
    # def gradient_vect(self, x, *args):
        tdx_coefs = x.reshape(self.m - 1, self.r + 1)
        yt1 = self.ytilde_vect(tdx_coefs, phi, u, a)
        yt2 = self.ytilde_vect(tdx_coefs, np.ones(phi.shape), u, a)

        g = np.zeros((1, j.shape[1]))
        for i in range(phi.shape[0]):
            yt1_prod = yt1[i, :].reshape(1, yt1.shape[1]) @ j[:, :, i] * w[i]
            yt2_prod = yt2[i, :].reshape(1, yt2.shape[1]) @ j[:, :, i] * w[i]
            g = g + (yt1_prod - yt2_prod)

        if lamda > 0:
            penalty = lamda * tdx_coefs @ d @ d.transpose()
            penalty = penalty.reshape(1, tdx_coefs.shape[0] * tdx_coefs.shape[1])
            g = g - penalty

        return -1 * g.flatten('F')

    def ytilde_vect(self, x, phi, u, a):
        exp = np.exp(u @ x @ a.transpose())
        b = (phi.transpose() * exp).sum(axis=0)
        b = b.reshape(b.shape[0], 1)
        tmp = 1 / b * phi
        tmp[np.isinf(tmp)] = 0
        yt = tmp * exp.transpose()
        return yt
