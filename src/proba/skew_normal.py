import numpy as np
from scipy.stats import norm, skewnorm


class SkewNormal:

    def __init__(self, xi, omega, alpha, random_state=None):
        self._xi = xi
        self._omega = omega
        self._alpha = alpha
        self._is_truncated = False
        self._min_value = -float("inf")
        self._max_value = float("inf")
        self._random_state = random_state

    def truncate(self, min_value, max_value):
        self._is_truncated = True
        self._min_value = min_value
        self._max_value = max_value

    def generate_random_numbers(self, n):
        # u1 = self._random_state.normal(0, 1, (1, 2 * n))
        # u2 = self._random_state.normal(0, 1, (1, 2 * n))
        u1 = norm.ppf(self._random_state.rand(1, 2 * n))
        u2 = norm.ppf(self._random_state.rand(1, 2 * n))
        ids = u2 > self._alpha * u1
        u1[ids] = -u1[ids]
        x = self._xi + self._omega * u1
        if self._is_truncated:
            x = x[np.logical_and(x > self._min_value, x < self._max_value)]
            return x[:n]
        return x[0, :n]

    def cdf(self, x):
        return skewnorm.cdf(x, self._alpha, loc=self._xi, scale=self._omega)

    def pdf(self, x):
        pdf = skewnorm.pdf(x, self._alpha, loc=self._xi, scale=self._omega)

        if self._is_truncated:
            in_range = (self._min_value < x) & (x <= self._max_value)
            return (in_range * pdf) / (self.cdf(self._max_value) - self.cdf(self._min_value))

        return pdf




