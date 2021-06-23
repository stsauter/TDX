import concurrent
import math

import numpy as np
import pandas as pd

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt

from river.base.density_estimator import MiniBatchDensityEstimator


class Tdx(MiniBatchDensityEstimator):
    """TDX stream density estimator.

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
    n_start_points : int (default=1)
        If value is > 1, the solver tries to find n local solutions by starting from randomly choosen points when
        fitting the model.
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
    >>> stream = WeightDriftStream(25000, 120, dist_support=[0, 7])
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
            n_start_points: int = 1,
            cache_size: int = 1000,
            grace_period: int = 1000,
            seed: int or np.random.RandomState = None,
            verbose: bool = False
    ):
        self._m = m
        self._bandwidth = bandwidth
        self._r = r
        self._lambda = lambda_reg
        self._n_start_points = n_start_points
        self._cache_size = cache_size
        self._grace_period = grace_period
        self._verbose = verbose
        self._random_state = check_random_state(seed)
        self._mu = np.array([])
        self._coeffs = np.array([])
        self._u = np.array([])
        self._max_optimization_attempts = 5
        self._n_training_instances = 0
        self._partial_fit_counter = 0
        self._t_min = 0
        self._t_max = 0

        if self._n_start_points < 1:
            raise ValueError("Number of starting points has to be greater than 0")
        self._temp_multi_start_coeffs = np.array([])

        if self._grace_period > self._cache_size:
            raise ValueError("Parameter cache size may not be greater than the cache size")
        self._queue = deque(maxlen=self._cache_size)

    @property
    def m(self):
        """Return the number of basis functions.

        Returns
        -------
        int:
            the number of basis functions
        """
        return self._m

    @property
    def bandwidth(self):
        """Return the standard deviation of basis functions.

        Returns
        -------
        float:
            the standard deviation of basis functions
        """
        return self._bandwidth

    @property
    def r(self):
        """Return the order of polynomial.

        Returns
        -------
        int:
            the order of polynomial
        """
        return self._r

    @property
    def lambda_reg(self):
        """Return the regularization factor.

        Returns
        -------
        int or float:
            the regularization factor
        """
        return self._lambda

    @property
    def n_start_points(self):
        return self._n_start_points

    @property
    def cache_size(self):
        return self._cache_size

    @property
    def grace_period(self):
        return self._grace_period

    @property
    def seed(self):
        return self._random_state

    @property
    def verbose(self):
        return self._verbose

    def learn_one(self, x: dict):
        self._queue.append(x)
        self._n_training_instances = self._n_training_instances + 1
        if self._n_training_instances == self._grace_period:
            self._n_training_instances = 0
            if len(self._queue) == self._cache_size:
                x_train, t_train = self._get_train_arrays_from_cache()
                self.fit_partial(x_train, t_train)
        return self

    def _get_train_arrays_from_cache(self):
        grouped_dict = dict()
        instances = list(self._queue)
        for k, v in [(key, instance[key]) for instance in instances for key in instance]:
            if k not in grouped_dict:
                grouped_dict[k] = [v]
            else:
                grouped_dict[k].append(v)

        t_train = np.array(grouped_dict['timestamp'])
        for key in grouped_dict.keys():
            if key != 'timestamp':
                x_train = np.array(grouped_dict[key])
                break

        return x_train, t_train

    def fit_partial(self, x_train, t_train):
        """Fit the model partially.

        Parameters
        ----------
        x_train : numpy.ndarray of shape (n_samples,)
            Feature values of the training samples.
        t_train : numpy.ndarray of shape (n_samples,)
            Time values of the training samples.
        """
        if self._partial_fit_counter > 0:
            target_min_in_src = self._transform_timestamps(t_train[0], self._t_min, self._t_max)
            target_max_in_src = self._transform_timestamps(t_train[-1], self._t_min, self._t_max)
            src_coefs = self._coeffs
            self._coeffs = self._transform_tdx_coeffs(self._coeffs, target_min_in_src, target_max_in_src)
            if self._n_start_points > 1:
                for i in range(self._temp_multi_start_coeffs.shape[0]):
                    src_coeffs = self._temp_multi_start_coeffs[i, :].reshape(self._m - 1, self._r + 1)
                    target_coeffs = self._transform_tdx_coeffs(src_coeffs, target_min_in_src, target_max_in_src)
                    self._temp_multi_start_coeffs[i, :] = target_coeffs.flatten()
            self.tgt_timestamps = self._transform_timestamps(self.orig_timestamps, np.min(t_train), np.max(t_train))
            # self.plot_coefs(src_coefs, self.src_timestamps, self._coeffs, self.tgt_timestamps)

        self.orig_timestamps = np.linspace(0, 1, 100)
        self.src_timestamps = self._transform_timestamps(self.orig_timestamps, t_train[0], t_train[-1])

        self.fit(x_train, t_train)
        self._partial_fit_counter = self._partial_fit_counter + 1

    def learn_many(self, X: pd.DataFrame):
        t_train = np.array(X['timestamp'])
        for column in X:
            if column != 'timestamp':
                x_train = np.array(X[column])
                break
        self.fit(x_train, t_train)
        return self

    def fit(self, x_train, t_train):
        """Fit the model.

        Parameters
        ----------
        x_train : numpy.ndarray of shape (n_samples,)
            Feature values of the training samples.
        t_train : numpy.ndarray of shape (n_samples,)
            Time values of the training samples.
        """
        self._mu = np.linspace(np.quantile(x_train, 0.01), np.quantile(x_train, 0.99), self._m).reshape(1, self._m)
        phi = norm.pdf(x_train.reshape(x_train.shape[0], 1), loc=self._mu, scale=self._bandwidth)

        # Set zero elements of phi to small non-zero value.
        # This should prevent runtime errors, since if any entry of phi is equal zero the computation of the function
        # value includes multiplying by zero. This leads to log(0), which is -Inf.
        phi[phi == 0] = 1e-5

        self._t_min = t_train[0]
        self._t_max = t_train[-1]
        if self._t_min >= self._t_max:
            raise ValueError('Training window has to consist of different timestamps')
        t_train_scaled = self._transform_timestamps(t_train, self._t_min, self._t_max)

        self._u = self._compute_matrix_u()
        a = self._compute_matrix_a(t_train_scaled)
        j = self._compute_matrix_j(a)
        w = self._compute_time_weights(t_train_scaled, 0.1, 1)

        c = np.zeros((self._r + 1, self._r))
        c[1:, :] = np.diagflat(np.ones(self._r))

        res = self._compute_tdx_coeffs(phi, a, j, c, w)
        self._coeffs = res.x.reshape(self._m - 1, self._r + 1)

    def _compute_matrix_u(self):
        """Compute matrix U used for the ilr-transformation."""
        u_tilde = self._compute_matrix_u_tilde()
        vn = np.zeros((1, u_tilde.shape[1]))
        for col_idx in range(u_tilde.shape[1]):
            vn[0, col_idx] = np.sqrt(u_tilde[:, col_idx].dot(u_tilde[:, col_idx]))
        u = np.matmul(u_tilde, np.diagflat(1 / vn))
        return u

    def _compute_matrix_u_tilde(self):
        """Return matrix U tilde needed for computing matrix U."""
        u_tilde = np.zeros((self._m, self._m - 1)) + (-np.triu(np.ones((self._m, self._m - 1))))
        for col_idx in range(self._m - 1):
            u_tilde[col_idx + 1, col_idx] = col_idx + 1
        return u_tilde

    def _compute_matrix_a(self, t_train):
        """Generate the design matrix of the regression problem."""
        n = self._r + 1
        bases = np.tile(t_train.reshape(t_train.shape[0], 1), (1, n))
        exponents = np.tile(range(n), (t_train.shape[0], 1))
        a = np.power(bases, exponents)
        return a

    def _compute_matrix_j(self, a):
        """Generate matrix J needed to compute the gradient."""
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
    def _compute_time_weights(t_train, half_life, t_max):
        psi = np.log(.5) / (half_life * t_max)
        t = np.max(t_train) - t_train
        w = np.exp(psi * t)
        return w.reshape(w.shape[0], 1)

    def _log_likelihood_fun(self, x, phi, a, j, c, w):
        """Computes the objective function."""
        tdx_coeffs = x.reshape(self._m - 1, self._r + 1)
        e = np.exp(self._u @ tdx_coeffs @ a.T)
        dot_products = (phi.T * e).sum(axis=0)
        dot_products = dot_products.reshape(1, dot_products.shape[0])
        f = np.sum((w.T * np.log(dot_products)) - (w.T * np.log(np.sum(e, axis=0))))
        if self._lambda > 0:
            penalty = self._lambda * np.trace(c.T @ tdx_coeffs.T @ tdx_coeffs @ c)
            f = f - penalty

        return -1 * f

    def _compute_gradient(self, x, phi, a, j, c, w):
        tdx_coeffs = x.reshape(self._m - 1, self._r + 1)
        yt1 = self._compute_y_tilde(tdx_coeffs, phi, a)
        yt2 = self._compute_y_tilde(tdx_coeffs, np.ones(phi.shape), a)

        g = np.einsum('ij,jki->ik', yt1, j) * w - np.einsum('ij,jki->ik', yt2, j) * w
        g = g.sum(axis=0)

        if self._lambda > 0:
            penalty = self._lambda * tdx_coeffs @ c @ c.T
            penalty = penalty.reshape(1, tdx_coeffs.shape[0] * tdx_coeffs.shape[1])
            g = g - penalty

        return -1 * g.flatten('F')

    def _compute_y_tilde(self, coeffs, y, a):
        e = np.exp(self._u @ coeffs @ a.T)
        beta = (y.T * e).sum(axis=0)
        beta = beta.reshape(beta.shape[0], 1)
        tmp = 1 / beta * y

        # Some elements of beta may be 0, therefore the resulting Inf cells from the division need to be handled.
        tmp[np.isinf(tmp)] = 0
        yt = tmp * e.T
        return yt

    def _compute_tdx_coeffs(self, phi, a, j, c, w):
        """Compute TDX coefficients by solving optimization problem."""
        retry_counter = 0
        succeeded = False
        res = None
        additional_params = phi, a, j, c, w

        while not succeeded:
            try:
                if self._n_start_points == 1:
                    res = self._optimize(additional_params)
                else:
                    res = self._multi_start_optimize(additional_params)
            except Exception as exc:
                print('An exception occurred during optimization: {}'.format(exc))
            finally:
                if not res or not res.success:
                    retry_counter += 1
                    self._handle_optimization_error(retry_counter)
                else:
                    succeeded = True
        return res

    def _optimize(self, additional_params):
        """Solve optimization problem."""
        if self._partial_fit_counter > 0:
            start_points = self._coeffs.flatten()
        else:
            start_points = self._generate_random_start_points()

        return self._optimize_log_likelihood(start_points, additional_params)

    def _generate_random_start_points(self):
        """Generate starting points for optimization."""
        x = self._random_state.rand(self._m - 1, self._r + 1)
        x = x.flatten('F')
        return x

    def _optimize_log_likelihood(self, x, additional_params):
        """Use BFGS algorithm to solve optimization problem."""
        res = minimize(self._log_likelihood_fun, x, jac=self._compute_gradient, args=additional_params,
                       method='l-bfgs-b', options={'disp': self._verbose, 'maxls': 80})
        return res

    def _multi_start_optimize(self, additional_params):
        """Tries to find multiple local solutions by starting from various points. Finally only the best of these
        solutions is returned.
        """
        self._init_multi_start_points()
        best_fun = float('inf')
        best_result = None
        updated_multi_start_coeffs = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(self._temp_multi_start_coeffs.shape[0]):
                start_points = self._temp_multi_start_coeffs[i, :]
                futures.append(executor.submit(self._optimize_log_likelihood, x=start_points,
                                               additional_params=additional_params))
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    optimize_result = future.result()
                    if optimize_result.success:
                        updated_multi_start_coeffs.append(optimize_result.x)
                    if optimize_result.success and optimize_result.fun < best_fun:
                        best_fun = optimize_result.fun
                        best_result = optimize_result
                except Exception as exc:
                    print('An exception occurred during multistart optimization: {}'.format(exc))

        if len(updated_multi_start_coeffs) > 0:
            self._temp_multi_start_coeffs = np.array(updated_multi_start_coeffs)
        return best_result

    def _init_multi_start_points(self):
        """Generate start points for multistart optimization """
        if self._partial_fit_counter == 0:
            self._temp_multi_start_coeffs = np.zeros((self._n_start_points, (self._m - 1) * (self._r + 1)))
            for i in range(self._temp_multi_start_coeffs.shape[0]):
                self._temp_multi_start_coeffs[i, :] = self._generate_random_start_points()

    def _handle_optimization_error(self, retry_counter):
        if retry_counter < self._max_optimization_attempts and self._partial_fit_counter < 1:
            print('Optimization failed, retrying with different random points...')
        else:
            raise RuntimeError('Unable to solve optimisation problem!')

    @staticmethod
    def _transform_timestamps(t, min_t, max_t):
        """Min-max scaling timestamps into range [0;0.5]."""
        return (t - min_t) / (2 * (max_t - min_t))

    def _transform_tdx_coeffs(self, tdx_coeffs, min_src, max_src):
        """Transforms the TDX coeffcients from source to target time domain.

        Parameters
        ----------
        tdx_coeffs : numpy.ndarray of shape (m-1, r+1)
            TDX coefficients
        min_src : float
            Min value specified in source coordinates
        max_src : float
            Max value specified in source coordinates

        Returns
        -------
        numpy.ndarray of shape (order+1,)
            TDX coefficents transformed into target domain.
        """
        transformed_coeffs = np.zeros(tdx_coeffs.shape)
        for i in range(tdx_coeffs.shape[0]):
            transformed_coeffs[i, :] = self._transform_poly_coeffs(tdx_coeffs[i, :], min_src, max_src)

        return transformed_coeffs

    def _transform_poly_coeffs(self, coeffs_src, min_src, max_src):
        """Transforms the coeffcients of a polynomial function f(s) within a source time domain S to a
        polynomial function g(t) in a target domain T, where the interval [0;0.5] in the target domain
        corresponds to the interval [min_src; max_src] in the source domain.

        Parameters
        ----------
        coeffs_src : numpy.ndarray of shape (order+1,)
            Array with coefficients b0, b1, b2, ...
        min_src : float
            Min value specified in source coordinates
        max_src : float
            Max value specified in source coordinates

        Returns
        -------
        numpy.ndarray of shape (order+1,)
            Coefficents transformed into target domain.
        """
        poly_coeffs = []
        unit_src = max_src - min_src
        for rIdx in range(len(coeffs_src)):
            df = self._poly_eval(coeffs_src, min_src, rIdx)
            coeff = df * math.pow(2 * unit_src, rIdx) * math.pow(math.factorial(rIdx), -1)
            poly_coeffs.append(coeff)

        return poly_coeffs

    @staticmethod
    def _poly_eval(b, x, derivative: int = 0):
        """Calculates the value of a polynomial function f(x) or its derivatives f'(x), f''(x), ...
        at the argument (or vector of function arguments) x.

        Parameters
        ----------
        b : numpy.ndarray of shape (order+1,)
            Array with coefficients b0, b1, b2, ...
        x : float
            Value, where to calculate f(x)
        derivative : int
            Number specifying whether the value of the function is requested (derivative == 0, default) or
            a higher order derivative (any non-negative integer value for derivative)

        Returns
        -------
        float
            Value of the polynomial function or its derivative.
        """
        v = 0
        for i in range(derivative, len(b)):
            v = v + (math.factorial(i) / math.factorial(i - derivative)) * b[i] * math.pow(x, (i - derivative))
        return v

    def plot_coefs(self, src_coefs, src_timestamps, tgt_coefs, tgt_timestamps):
        src_fval = np.zeros((src_coefs.shape[0], src_timestamps.shape[0]))
        tgt_fval = np.zeros((src_coefs.shape[0], src_timestamps.shape[0]))
        bla = np.zeros((src_coefs.shape[0], src_timestamps.shape[0]))
        for i in range(src_coefs.shape[0]):
            for j in range(src_timestamps.shape[0]):
                src_fval[i, j] = self._poly_eval(src_coefs[0, :], src_timestamps[j], i)
                tgt_fval[i, j] = self._poly_eval(tgt_coefs[0, :], tgt_timestamps[j], i)
                bla[i, j] = self._poly_eval(src_coefs[0, :], tgt_timestamps[j], i)

        plt.plot(self.orig_timestamps, src_fval[1, :], 'b+')
        plt.plot(self.orig_timestamps, tgt_fval[1, :], 'r+')
        plt.plot(self.orig_timestamps, bla[1, :], 'g+')
        plt.show()

    def predict_one(self, x: dict):
        if self._partial_fit_counter == 0:
            return 0
        t_pred = np.array([x['timestamp']])
        for key in x.keys():
            if key != 'timestamp':
                x_pred = np.array([x[key]])
                break
        return self.pdf(x_pred, t_pred)[0, 0]

    def predict_many(self, X: pd.DataFrame):
        t = X[X['timestamp'].notnull()]['timestamp'].to_numpy()
        for column in X:
            if column != 'timestamp':
                data_col_name = column
                break
        x_grid = X[X[data_col_name].notnull()][data_col_name].to_numpy()

        #dens = np.zeros((t.shape[0]))
        #for i, t in enumerate(t):
            #dens[i] = self.pdf(x_grid.iloc[i].to_numpy(), np.array([t]))[0, 0]

        return pd.DataFrame(self.pdf(x_grid, t))

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

    def get_gamma(self, t):
        """Compute the basis weights at at a given time point t.

        Parameters
        ----------
        t : numpy.ndarray of shape (n_time_values,)
            Time values at which the basis weights should be computed.

        Returns
        -------
        numpy.ndarray of shape (n_time_values, m)
            Basis weights at a given time point t.
        """
        t_scaled = self._transform_timestamps(t, self._t_min, self._t_max)
        bases = np.tile(t_scaled.reshape(t_scaled.shape[0], 1), (1, self._r + 1))
        exponents = np.tile(range(self._r + 1), (t_scaled.shape[0], 1))
        a = np.power(bases, exponents)
        e = np.exp(self._u @ self._coeffs @ a.T)
        i = np.ones((self._m, 1))
        gamma = (e / (i * e.sum(axis=0))).T
        return gamma
