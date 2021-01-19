import numpy as np

from sklearn.utils import check_random_state
from scipy.stats import norm
from scipy.optimize import minimize


class Tdx:
    def __init__(self, m, h, r, l, random_state=None, verbose=False):
        self.m = m
        self.h = h
        self.r = r
        self.l = l
        self._verbose = verbose
        self._random_state = check_random_state(random_state)
        self._mu = np.array([])
        self._coefs = np.array([])
        self._u = np.array([])

    def fit(self, x_train, t_train):
        self._mu = np.linspace(np.quantile(x_train, 0.01), np.quantile(x_train, 0.99), self.m).reshape(1, self.m)
        phi = norm.pdf(x_train.reshape(x_train.shape[0], 1), loc=self._mu, scale=self.h)
        phi[phi == 0] = 1e-5

        self._u = self._get_u()
        a = self._get_a(t_train)
        j = self.j_vect(self._u, a, self.m, self.r + 1)
        w = self._get_time_weights(t_train, 0.1, 1)

        d = np.zeros((self.r + 1, self.r))
        d[1:, :] = np.diagflat(np.ones(self.r))

        # x = self._random_state.rand(self.m - 1, self.r + 1)
        x = self._random_state.rand(self.r + 1, self.m - 1).T
        x = x.flatten('F')

        additional_params = phi, self._u, a, j, self.l, d, w
        res = minimize(self.fun_vect, x, jac=self._gradient_vect, args=additional_params, method='l-bfgs-b',
                       options={'disp': self._verbose})
        self._coefs = res.x.reshape(self.m - 1, self.r + 1)

    def _get_u(self):
        u_tilde = self._get_u_tilde()
        vn = np.zeros((1, u_tilde.shape[1]))
        for col_idx in range(u_tilde.shape[1]):
            vn[0, col_idx] = np.sqrt(u_tilde[:, col_idx].dot(u_tilde[:, col_idx]))
        u = np.matmul(u_tilde, np.diagflat(1 / vn))
        return u

    def _get_u_tilde(self):
        u_tilde = np.zeros((self.m, self.m - 1)) + (-np.triu(np.ones((self.m, self.m - 1))))
        for col_idx in range(self.m - 1):
            u_tilde[col_idx + 1, col_idx] = col_idx + 1
        return u_tilde

    def _get_a(self, t_train):
        bases = np.tile(t_train.reshape(t_train.shape[0], 1), (1, self.r + 1))
        exponents = np.tile(range(self.r + 1), (t_train.shape[0], 1))
        a = np.power(bases, exponents)
        return a

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
        # tdx_coefs = x.reshape(self.r + 1, self.m - 1).T
        tdx_coefs = x.reshape(self.m - 1, self.r + 1)
        e = np.exp(u @ tdx_coefs @ a.T)
        dot_products = (phi.T * e).sum(axis=0)
        dot_products = dot_products.reshape(1, dot_products.shape[0])
        f = np.sum((w.T * np.log(dot_products)) - (w.T * np.log(np.sum(e, axis=0))))
        if lamda > 0:
            penalty = lamda * np.trace((d.T @ tdx_coefs.T) @ tdx_coefs @ d)
            f = f - penalty
        return -1 * f

    @staticmethod
    def _get_time_weights(t_train, half_life, t_max):
        psi = -np.log(.5) / (half_life * t_max)
        t = np.max(t_train) - t_train
        w = np.exp(-psi * t)
        return w.reshape(w.shape[0], 1)

    def _gradient_vect(self, x, phi, u, a, j, lamda, d, w):
        tdx_coefs = x.reshape(self.m - 1, self.r + 1)
        yt1 = self._y_tilde_vect(tdx_coefs, phi, u, a)
        yt2 = self._y_tilde_vect(tdx_coefs, np.ones(phi.shape), u, a)

        g = np.zeros((1, j.shape[1]))
        for i in range(phi.shape[0]):
            yt1_prod = yt1[i, :].reshape(1, yt1.shape[1]) @ j[:, :, i] * w[i]
            yt2_prod = yt2[i, :].reshape(1, yt2.shape[1]) @ j[:, :, i] * w[i]
            g = g + (yt1_prod - yt2_prod)

        if lamda > 0:
            penalty = lamda * tdx_coefs @ d @ d.T
            penalty = penalty.reshape(1, tdx_coefs.shape[0] * tdx_coefs.shape[1])
            g = g - penalty

        return -1 * g.flatten('F')

    def _y_tilde_vect(self, x, phi, u, a):
        exp = np.exp(u @ x @ a.T)
        b = (phi.T * exp).sum(axis=0)
        b = b.reshape(b.shape[0], 1)
        tmp = 1 / b * phi
        tmp[np.isinf(tmp)] = 0
        yt = tmp * exp.T
        return yt

    def get_gamma(self, t):
        bases = np.tile(t.reshape(t.shape[0], 1), (1, self.r + 1))
        exponents = np.tile(range(self.r + 1), (t.shape[0], 1))
        a = np.power(bases, exponents)
        e_mat = np.exp(self._u @ self._coefs @ a.T)
        i = np.ones((self.m, 1))
        g = (e_mat / (i * e_mat.sum(axis=0))).T
        return g

    def pdf(self, x, t):
        pdf = np.zeros((t.shape[0], x.shape[0]))
        x_reshaped = x.reshape(1, x.shape[0])
        g = self.get_gamma(t)
        for i in range(self.m):
            pdf = pdf + g[:, i].reshape(g.shape[0], 1) @ norm.pdf(x_reshaped, loc=self._mu[0, i], scale=self.h)
        return pdf
