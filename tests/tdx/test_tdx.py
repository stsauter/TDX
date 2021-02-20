import math
import numpy as np
import pandas as pd

from src.tdx.tdx import Tdx


def test_get_time_weights(helpers):
    t_train = np.loadtxt(helpers.get_test_file_path('t_train.csv'), delimiter=',')
    w_exp = np.loadtxt(helpers.get_test_file_path('w.csv'), delimiter=',')
    w_exp = w_exp.reshape(w_exp.shape[0], 1)

    w = Tdx._get_time_weights(t_train, 0.1, 1)
    np.testing.assert_allclose(w_exp, w)


def test_get_u_tilde(helpers):
    u_tilde_exp = np.loadtxt(helpers.get_test_file_path('u_tilde.csv'), delimiter=',')

    model = Tdx(4, 0.6, 5, 2, seed=32)
    u_tilde = model._get_u_tilde()
    np.testing.assert_allclose(u_tilde_exp, u_tilde)


def test_get_u(helpers):
    u_exp = np.loadtxt(helpers.get_test_file_path('u.csv'), delimiter=',')

    model = Tdx(4, 0.6, 5, 2, seed=32)
    u = model._get_u()
    np.testing.assert_allclose(u_exp, u, rtol=1e-3)


def test_get_a(helpers):
    t_train = np.loadtxt(helpers.get_test_file_path('t_train.csv'), delimiter=',')
    a_exp = np.loadtxt(helpers.get_test_file_path('a.csv'), delimiter=',')

    model = Tdx(4, 0.6, 5, 2, seed=32)
    a = model._get_a(t_train)
    np.testing.assert_allclose(a_exp, a)


def test_get_j(helpers):
    u = np.loadtxt(helpers.get_test_file_path('u.csv'), delimiter=',')
    a = np.loadtxt(helpers.get_test_file_path('a.csv'), delimiter=',')
    j_vect_exp = np.loadtxt(helpers.get_test_file_path('j_vect.csv'), delimiter=',')
    j_vect_exp = j_vect_exp.reshape((4, 18, 12), order='F')

    model = Tdx(4, 0.6, 5, 2, seed=32)
    model._u = u
    j_vect = model._get_j(a)
    np.testing.assert_allclose(j_vect_exp, j_vect, rtol=1e-3)


def test_log_likelihood_fun(helpers):
    x = np.loadtxt(helpers.get_test_file_path('initial_coefs.csv'), delimiter=',')
    phi = np.loadtxt(helpers.get_test_file_path('phi.csv'), delimiter=',')
    u = np.loadtxt(helpers.get_test_file_path('u.csv'), delimiter=',')
    a = np.loadtxt(helpers.get_test_file_path('a.csv'), delimiter=',')
    w = np.loadtxt(helpers.get_test_file_path('w.csv'), delimiter=',')
    d = np.zeros((6, 5))
    d[1:, :] = np.diagflat(np.ones(5))

    model = Tdx(4, 0.6, 5, 2, seed=32)
    model._u = u
    fun_val = model._log_likelihood_fun(x, phi, a, 0, d, w)
    assert math.isclose(23.516, fun_val, abs_tol=0.001)


def test_get_y_tilde(helpers):
    x = np.loadtxt(helpers.get_test_file_path('initial_coefs.csv'), delimiter=',')
    x = x.reshape(3, 6)
    phi = np.loadtxt(helpers.get_test_file_path('phi.csv'), delimiter=',')
    u = np.loadtxt(helpers.get_test_file_path('u.csv'), delimiter=',')
    a = np.loadtxt(helpers.get_test_file_path('a.csv'), delimiter=',')
    y_tilde_exp = np.loadtxt(helpers.get_test_file_path('y_tilde.csv'), delimiter=',')

    model = Tdx(4, 0.6, 5, 2, seed=32)
    model._u = u
    y_tilde = model._get_y_tilde(x, phi, a)
    np.testing.assert_allclose(y_tilde_exp, y_tilde, rtol=1e-3)


def test_gradient(helpers):
    x = np.loadtxt(helpers.get_test_file_path('initial_coefs.csv'), delimiter=',')
    phi = np.loadtxt(helpers.get_test_file_path('phi.csv'), delimiter=',')
    u = np.loadtxt(helpers.get_test_file_path('u.csv'), delimiter=',')
    a = np.loadtxt(helpers.get_test_file_path('a.csv'), delimiter=',')
    j = np.loadtxt(helpers.get_test_file_path('j_vect.csv'), delimiter=',')
    j = j.reshape((4, 18, 12), order='F')
    w = np.loadtxt(helpers.get_test_file_path('w.csv'), delimiter=',')
    grad_exp = np.loadtxt(helpers.get_test_file_path('gradient.csv'), delimiter=',')
    d = np.zeros((6, 5))
    d[1:, :] = np.diagflat(np.ones(5))

    model = Tdx(4, 0.6, 5, 2, seed=32)
    model._u = u
    grad = model._get_gradient(x, phi, a, j, d, w)
    np.testing.assert_allclose(grad_exp, grad, rtol=1e-3)


def test_fit(helpers):
    x_train = np.loadtxt(helpers.get_test_file_path('x_train.csv'), delimiter=',')
    t_train = np.loadtxt(helpers.get_test_file_path('t_train.csv'), delimiter=',')
    model = Tdx(4, 0.6, 5, 2, seed=32)
    model.fit(x_train, t_train)

    raw_data = pd.read_csv(helpers.get_test_file_path('coefs.csv'), header=None)
    assert raw_data.shape == model._coefs.shape
    for i, row in raw_data.iterrows():
        for j, column in enumerate(row):
            assert math.isclose(column, model._coefs[i][j], abs_tol=0.06), i


def test_get_gamma(helpers):
    t_train = np.loadtxt(helpers.get_test_file_path('t_train.csv'), delimiter=',')
    gamma_exp = np.loadtxt(helpers.get_test_file_path('gamma.csv'), delimiter=',')
    coefs = np.loadtxt(helpers.get_test_file_path('coefs.csv'), delimiter=',')
    u = np.loadtxt(helpers.get_test_file_path('u.csv'), delimiter=',')

    model = Tdx(4, 0.6, 5, 2, seed=32)
    model._coefs = coefs
    model._u = u
    gamma = model.get_gamma(t_train)
    np.testing.assert_allclose(gamma_exp, gamma, rtol=1e-3)


def test_pdf(helpers):
    pred_dens_exp = np.loadtxt(helpers.get_test_file_path('pred_dens.csv'), delimiter=',')
    coefs = np.loadtxt(helpers.get_test_file_path('coefs.csv'), delimiter=',')
    u = np.loadtxt(helpers.get_test_file_path('u.csv'), delimiter=',')
    mu = np.array([0.85, 2.806, 4.763, 6.72]).reshape(1, 4)
    x = np.linspace(0.85, 6.72, 20)
    t = np.linspace(0.6, 0.95, 5)

    model = Tdx(4, 0.6, 5, 2, seed=32)
    model._coefs = coefs
    model._u = u
    model._mu = mu
    pred_dens = model.pdf(x, t)
    np.testing.assert_allclose(pred_dens_exp, pred_dens, rtol=1e-2)
