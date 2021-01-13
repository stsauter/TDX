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
        self.mu = np.linspace(np.quantile(x_train, 0.01), np.quantile(x_train, 0.99), self.m).reshape(1, self.m)
        phi = norm.pdf(x_train.reshape(x_train.shape[0], 1), loc=self.mu, scale=self.h)
        phi[phi == 0] = 1e-5

        tmax = 1
        halflife = 0.1
        psi = -np.log(.5) / (halflife * tmax)
        t = np.max(t_train) - t_train
        w = np.exp(-psi * t)
        w = w.reshape(w.shape[0], 1)

        u_tilde = np.zeros((self.m, self.m - 1)) + (-np.triu(np.ones((self.m, self.m - 1))))
        for col_idx in range(self.m - 1):
            u_tilde[col_idx + 1, col_idx] = col_idx + 1

        vn = np.zeros((1, u_tilde.shape[1]))
        for col_idx in range(u_tilde.shape[1]):
            vn[0, col_idx] = np.sqrt(u_tilde[:, col_idx].dot(u_tilde[:, col_idx]))
        self.u = np.matmul(u_tilde, np.diagflat(1 / vn))

        bases = np.tile(t_train.reshape(t_train.shape[0], 1), (1, self.r + 1))
        exponents = np.tile(range(self.r + 1), (t_train.shape[0], 1))
        a = np.power(bases, exponents)
        j = self.j_vect(self.u, a, self.m, self.r + 1)

        d = np.zeros((self.r + 1, self.r))
        d[1:, :] = np.diagflat(np.ones(self.r))

        np.random.seed(32)
        # x = np.random.rand(self.m - 1, self.r + 1)
        x = np.random.rand(self.r + 1, self.m - 1).T
        x = x.flatten('F')

        additional_params = phi, self.u, a, j, self.l, d, w
        # res = minimize(self.fun_vect, x, jac=self.gradient_vect, args=additional_params, method='l-bfgs-b', options={'disp': True})
        # self.coefs = res.x.reshape(self.m - 1, self.r + 1)

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
            penalty = lamda * tdx_coefs @ d @ d.T
            penalty = penalty.reshape(1, tdx_coefs.shape[0] * tdx_coefs.shape[1])
            g = g - penalty

        return -1 * g.flatten('F')

    def ytilde_vect(self, x, phi, u, a):
        exp = np.exp(u @ x @ a.T)
        b = (phi.T * exp).sum(axis=0)
        b = b.reshape(b.shape[0], 1)
        tmp = 1 / b * phi
        tmp[np.isinf(tmp)] = 0
        yt = tmp * exp.T
        return yt

    def get_gamma(self, t):
        self.set_test_data()
        bases = np.tile(t.reshape(t.shape[0], 1), (1, self.r + 1))
        exponents = np.tile(range(self.r + 1), (t.shape[0], 1))
        a = np.power(bases, exponents)
        e_mat = np.exp(self.u @ self.coefs @ a.T)
        i = np.ones((self.m, 1))
        g = (e_mat / (i * e_mat.sum(axis=0))).T
        return g

    def pdf(self, x, t):
        pdf = np.zeros((t.shape[0], x.shape[0]))
        x_reshaped = x.reshape(1, x.shape[0])
        g = self.get_gamma(t)
        for i in range(self.m):
            pdf = pdf + g[:, i].reshape(g.shape[0], 1) @ norm.pdf(x_reshaped, loc=self.mu[0, i], scale=self.h)
        return pdf

    def set_test_data(self):
        self.coefs = np.array([[-0.342003448280851, 0.0412981820972530, 0.0261539510147315, -0.00243336522872218, -0.00496060704550973, -0.0158247294566967],
        [6.73231217398917, 0.383001973860287, 0.262172175811604, 0.196027578702680, 0.178700803788585, 0.139352149656245],
        [-1.12968470771790, -0.124333240891595, -0.103034127843851, -0.0951741165935979, -0.0826102724907291, -0.0538180639988147],
        [-2.75101552899771, -0.194538781801245, -0.150720371147741, -0.0892378299789635, -0.101508744977172, -0.0705711622835548],
        [-1.70734342697976, -0.175475273236360, -0.104038948665231, -0.0997502755291537, -0.0627053770161427, -0.0755430927904746],
        [0.759537751516781, -0.161812038636567, -0.107764109196660, -0.0859201002774936, -0.0579818935728293, -0.0522825921236686],
        [7.60211126667295, -0.941617570643533, -0.586321653277136, -0.332229530687844, -0.212885940923690, -0.147105694359449],
        [1.13833347832646, 0.0106823907199322, -0.0329995934930307, -0.0298614804787476, -0.0317627349019797, -0.0448103051267880],
        [0.761265614243612, 0.0392720436330845, 0.00268679531312246, -0.0340045804065830, -0.0288428979569656, -0.0133397093629450],
        [4.13965212493327, 0.205953235622318, 0.0933081891368810, 0.0332570406580172, -0.0287844896133309, -0.0300942999035468],
        [4.81878881957077, 0.362786258418644, 0.146971124811358, 0.00882798736687461, -0.0638097502721925, -0.111494807369267],
        [-2.21677429318743, -0.0205265189802002, -0.0336955346258527, -0.0316126089099040, -0.0153901172252245, -0.00848074694220209],
        [-4.06404835158050, 0.0147231751115763, 0.000583253012410675, -0.00539264564576391, -0.00603113725308661, -0.0181702964627574]])
