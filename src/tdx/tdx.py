import numpy as np

from sklearn.utils import check_random_state
from scipy.stats import norm
from scipy.optimize import minimize


class Tdx:
    """ TDX stream density estimator.

    Parameters
    ----------
    m : int
        Number of basis functions.
    bandwidth: float
        Standard deviation of basis functions.
    r : int
        Order of polynomial.
    lambda_reg : int or float
        Regularization factor.
    seed : None | int | instance of RandomState (default=None)
        Random number generator seed for reproducibility.
    verbose : boolean (default=False)
        Set to True to print convergence messages of optimization algorithm.

    Notes
    -----
    Temporal density extrapolation (TDX) is an approach for predicting the probability density of a univariate feature
    in a data stream. This method is based on the expansion of basis functions, whose weights are modelled as
    functions of compositional data over time by using an isometric log-ratio transformation. Predictions are made
    by extrapolating the density model to time points outside the available training window [1]_.

    References
    ----------
    .. [1] Krempl, G., Lang, D. & Hofer, V. Temporal density extrapolation using a dynamic basis approach.
       In  Data Min Knowl Disc 33, 1323–1356 (2019).

    Examples
    --------
    >>> # Imports
    >>> from src.data.weight_drift_stream import WeightDriftStream
    >>> from src.tdx.tdx import Tdx
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> # Setting up a data stream
    >>> stream = WeightDriftStream(25000, 120, dist_support=[0, 7], seed=1)
    >>>
    >>> # Setup training data
    >>> x_train, x_test, t_train, t_test = train_test_split(stream.x, stream.t, train_size=0.66, shuffle=False)
    >>>
    >>> # Setup TDX model
    >>> model = Tdx(14, 0.6, 5, 2)
    >>>
    >>> # Train TDX model
    >>> model.fit(x_train, t_train)
    >>>
    >>> # Predict density
    >>> x_grid = np.linspace(np.quantile(stream.x, 0.01), np.quantile(stream.x, 0.99), 200)
    >>> predicted_dens = model.pdf(x_grid, t_test)
    """

    def __init__(
            self,
            m: int,
            bandwidth: float,
            r: int,
            lambda_reg: int or float,
            seed: int or np.random.RandomState = None,
            verbose: bool = False
    ):
        self._m = m
        self._bandwidth = bandwidth
        self._r = r
        self._lambda = lambda_reg
        self._verbose = verbose
        self._random_state = check_random_state(seed)
        self._mu = np.array([])
        self._coefs = np.array([])
        self._u = np.array([])

    @property
    def m(self):
        """
        Return the number of basis functions.

        Returns
        -------
        int:
            the number of basis functions
        """
        return self._m

    @property
    def bandwidth(self):
        """
        Return the standard deviation of basis functions.

        Returns
        -------
        float:
            the standard deviation of basis functions
        """
        return self._bandwidth

    @property
    def r(self):
        """
        Return the order of polynomial.

        Returns
        -------
        int:
            the order of polynomial
        """
        return self._r

    @property
    def lambda_reg(self):
        """
        Return the regularization factor.

        Returns
        -------
        int or float:
            the regularization factor
        """
        return self._lambda

    def fit(self, x_train, t_train):
        """ Fit the model.

        Parameters
        ----------
        x_train : numpy.ndarray of shape (n_samples,)
            Feature values of the training samples.
        t_train : numpy.ndarray of shape (n_samples,)
            Time values of the training samples.
        """
        self._mu = np.linspace(np.quantile(x_train, 0.01), np.quantile(x_train, 0.99), self._m).reshape(1, self._m)
        phi = norm.pdf(x_train.reshape(x_train.shape[0], 1), loc=self._mu, scale=self._bandwidth)
        phi[phi == 0] = 1e-5

        self._u = self._get_u()
        a = self._get_a(t_train)
        j = self._get_j(a)
        w = self._get_time_weights(t_train, 0.1, 1)

        c = np.zeros((self._r + 1, self._r))
        c[1:, :] = np.diagflat(np.ones(self._r))

        # x = self._random_state.rand(self._m - 1, self._r + 1)
        # self._random_state.rand((self._r + 1) * (self._m - 1))
        x = self._random_state.rand(self._r + 1, self._m - 1).T
        x = x.flatten('F')

        additional_params = phi, a, j, c, w
        res = minimize(self._log_likelihood_fun, x, jac=self._get_gradient, args=additional_params, method='l-bfgs-b',
                       options={'disp': self._verbose})
        self._coefs = res.x.reshape(self._m - 1, self._r + 1)

    def _get_u(self):
        u_tilde = self._get_u_tilde()
        vn = np.zeros((1, u_tilde.shape[1]))
        for col_idx in range(u_tilde.shape[1]):
            vn[0, col_idx] = np.sqrt(u_tilde[:, col_idx].dot(u_tilde[:, col_idx]))
        u = np.matmul(u_tilde, np.diagflat(1 / vn))
        return u

    def _get_u_tilde(self):
        u_tilde = np.zeros((self._m, self._m - 1)) + (-np.triu(np.ones((self._m, self._m - 1))))
        for col_idx in range(self._m - 1):
            u_tilde[col_idx + 1, col_idx] = col_idx + 1
        return u_tilde

    def _get_a(self, t_train):
        n = self._r + 1
        bases = np.tile(t_train.reshape(t_train.shape[0], 1), (1, n))
        exponents = np.tile(range(n), (t_train.shape[0], 1))
        a = np.power(bases, exponents)
        return a

    def _get_j(self, a):
        n = self._r + 1
        u_rep = np.zeros((self._m, (self._m - 1) * n))
        for i in range(self._m - 1):
            cols_to_append = np.tile(self._u[:, i].reshape(self._u.shape[0], 1), (1, n))
            if i == 0:
                u_rep = cols_to_append
            else:
                u_rep = np.hstack((u_rep, cols_to_append))

        j = np.zeros((self._m, (self._m - 1) * n, a.shape[0]))
        for i in range(a.shape[0]):
            j[:, :, i] = u_rep * np.tile(a[i, :], (self._m, self._m - 1))

        return j

    @staticmethod
    def _get_time_weights(t_train, half_life, t_max):
        psi = np.log(.5) / (half_life * t_max)
        t = np.max(t_train) - t_train
        w = np.exp(psi * t)
        return w.reshape(w.shape[0], 1)

    def _log_likelihood_fun(self, x, phi, a, j, c, w):
        # tdx_coefs = x.reshape(self.r + 1, self.m - 1).T
        tdx_coefs = x.reshape(self._m - 1, self._r + 1)
        e = np.exp(self._u @ tdx_coefs @ a.T)
        dot_products = (phi.T * e).sum(axis=0)
        dot_products = dot_products.reshape(1, dot_products.shape[0])
        f = np.sum((w.T * np.log(dot_products)) - (w.T * np.log(np.sum(e, axis=0))))
        if self._lambda > 0:
            penalty = self._lambda * np.trace(c.T @ tdx_coefs.T @ tdx_coefs @ c)
            f = f - penalty
        return -1 * f

    def _get_gradient(self, x, phi, a, j, c, w):
        tdx_coefs = x.reshape(self._m - 1, self._r + 1)
        yt1 = self._get_y_tilde(tdx_coefs, phi, a)
        yt2 = self._get_y_tilde(tdx_coefs, np.ones(phi.shape), a)

        g = np.zeros((1, j.shape[1]))
        for i in range(phi.shape[0]):
            yt1_prod = yt1[i, :].reshape(1, yt1.shape[1]) @ j[:, :, i] * w[i]
            yt2_prod = yt2[i, :].reshape(1, yt2.shape[1]) @ j[:, :, i] * w[i]
            g = g + (yt1_prod - yt2_prod)

        if self._lambda > 0:
            penalty = self._lambda * tdx_coefs @ c @ c.T
            penalty = penalty.reshape(1, tdx_coefs.shape[0] * tdx_coefs.shape[1])
            g = g - penalty

        return -1 * g.flatten('F')

    def _get_y_tilde(self, coefs, y, a):
        e = np.exp(self._u @ coefs @ a.T)
        beta = (y.T * e).sum(axis=0)
        beta = beta.reshape(beta.shape[0], 1)
        tmp = 1 / beta * y
        tmp[np.isinf(tmp)] = 0
        yt = tmp * e.T
        return yt

    def get_gamma(self, t):
        """ Compute the basis weights at at a given time point t.

        Parameters
        ----------
        t : numpy.ndarray of shape (n_time_values,)
            Time values at which the basis weights should be computed.

        Returns
        -------
        numpy.ndarray of shape (n_time_values, m)
            Basis weights at a given time point t.
        """
        bases = np.tile(t.reshape(t.shape[0], 1), (1, self._r + 1))
        exponents = np.tile(range(self._r + 1), (t.shape[0], 1))
        a = np.power(bases, exponents)
        e = np.exp(self._u @ self._coefs @ a.T)
        i = np.ones((self._m, 1))
        gamma = (e / (i * e.sum(axis=0))).T
        return gamma

    def pdf(self, x, t):
        """ Predict the probability density at point x and at a given time point t.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_samples,)
            Data points at which the probability density function should be evaluated.
        t : numpy.ndarray of shape (n_time_values,)
            Time values at which the probability densities should be predicted.

        Returns
        -------
        numpy.ndarray of shape (n_time_values, n_samples)
            Probability density function evaluated at point x and time point t.
        """
        pdf = np.zeros((t.shape[0], x.shape[0]))
        x_reshaped = x.reshape(1, x.shape[0])
        g = self.get_gamma(t)
        for i in range(self._m):
            pdf = pdf + g[:, i].reshape(g.shape[0], 1) @ norm.pdf(x_reshaped, loc=self._mu[0, i], scale=self._bandwidth)
        return pdf
