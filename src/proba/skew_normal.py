import numpy as np
from scipy.stats import norm, skewnorm


class SkewNormal:
    """ Skew normal distribution.

    Parameters
    ----------
    xi : float
        Xi parameter.
    omega: float
        Omega parameter.
    alpha: float
        Alpha parameter.
    seed : None | int | instance of RandomState (default=None)
        Random number generator seed for reproducibility.
    """

    def __init__(self, xi: float, omega: float, alpha: float, seed: int or np.random.RandomState = None):
        self._xi = xi
        self._omega = omega
        self._alpha = alpha
        self._is_truncated = False
        self._min_value = -float("inf")
        self._max_value = float("inf")
        self._seed = seed

    def truncate(self, min_value, max_value):
        """ Set the supported value range of this probability distribution.

        Parameters
        ----------
        min_value : int or float
            Minimum value which should be supported by this probability distribution.
        max_value : int or float
            Maximum value which should be supported by this probability distribution.
        """
        self._is_truncated = True
        self._min_value = min_value
        self._max_value = max_value

    def generate_random_numbers(self, n):
        """ Generate n skew normal random variables.

        Parameters
        ----------
        n : int
            Number of random variables to generate.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Array filled with skew normal random variables.
        """
        # u1 = self._random_state.normal(0, 1, (1, 2 * n))
        # u2 = self._random_state.normal(0, 1, (1, 2 * n))
        u1 = norm.ppf(self._seed.rand(1, 2 * n))
        u2 = norm.ppf(self._seed.rand(1, 2 * n))
        ids = u2 > self._alpha * u1
        u1[ids] = -u1[ids]
        x = self._xi + self._omega * u1
        if self._is_truncated:
            x = x[np.logical_and(x > self._min_value, x < self._max_value)]
            return x[:n]
        return x[0, :n]

    def cdf(self, x):
        """ Compute the cumulative distribution function at point x.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_samples,)
            Data points at which the cumulative distribution function should be evaluated.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Cumulative distribution function evaluated at point x.
        """
        return skewnorm.cdf(x, self._alpha, loc=self._xi, scale=self._omega)

    def pdf(self, x):
        """ Compute the probability density at point x.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_samples,)
            Data points at which the probability density function should be evaluated.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Probability density function evaluated at point x.
        """
        pdf = skewnorm.pdf(x, self._alpha, loc=self._xi, scale=self._omega)

        if self._is_truncated:
            in_range = (self._min_value < x) & (x <= self._max_value)
            return (in_range * pdf) / (self.cdf(self._max_value) - self.cdf(self._min_value))

        return pdf




