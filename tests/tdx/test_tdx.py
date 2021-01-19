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


def test_get_u_tilde_weights(helpers):
    u_tilde_exp = np.loadtxt(helpers.get_test_file_path('u_tilde.csv'), delimiter=',')

    model = Tdx(4, 0.6, 5, 2, random_state=32)
    u_tilde = model._get_u_tilde()
    np.testing.assert_allclose(u_tilde_exp, u_tilde)


def test_get_u(helpers):
    u_exp = np.loadtxt(helpers.get_test_file_path('u.csv'), delimiter=',')

    model = Tdx(4, 0.6, 5, 2, random_state=32)
    u = model._get_u()
    np.testing.assert_allclose(u_exp, u, rtol=1e-3)


def test_get_a(helpers):
    t_train = np.loadtxt(helpers.get_test_file_path('t_train.csv'), delimiter=',')
    a_exp = np.loadtxt(helpers.get_test_file_path('a.csv'), delimiter=',')

    model = Tdx(4, 0.6, 5, 2, random_state=32)
    a = model._get_a(t_train)
    np.testing.assert_allclose(a_exp, a)


def test_j_vect(helpers):
    u = np.loadtxt(helpers.get_test_file_path('u.csv'), delimiter=',')
    a = np.loadtxt(helpers.get_test_file_path('a.csv'), delimiter=',')
    j_vect_exp = np.loadtxt(helpers.get_test_file_path('j_vect.csv'), delimiter=',')
    j_vect_exp = j_vect_exp.reshape((4, 18, 12), order='F')

    model = Tdx(4, 0.6, 5, 2, random_state=32)
    j_vect = model.j_vect(u, a, 4, 6)
    np.testing.assert_allclose(j_vect_exp, j_vect, rtol=1e-3)


def test_fit(helpers):
    x_train = np.array([0.85, 3.36, 3.21, 4.53, 4.23, 3.84, 3.94, 4.12, 6.72, 4.91, 2.59, 3.24])
    t_train = np.loadtxt(helpers.get_test_file_path('t_train.csv'), delimiter=',')
    model = Tdx(4, 0.6, 5, 2, random_state=32)
    model.fit(x_train, t_train)

    raw_data = pd.read_csv(helpers.get_test_file_path('coefs.csv'), header=None)
    assert raw_data.shape == model._coefs.shape
    for i, row in raw_data.iterrows():
        for j, column in enumerate(row):
            assert math.isclose(column, model._coefs[i][j], abs_tol=0.06), i
