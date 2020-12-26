import numpy as np
from scipy.stats import norm, skewnorm


class SkewNormal:

    def __init__(self, xi, omega, alpha):
        self.xi = xi
        self.omega = omega
        self.alpha = alpha
        self.is_truncated = False
        self.min_value = -float("inf")
        self.max_value = float("inf")

    def truncate(self, min_value, max_value):
        self.is_truncated = True
        self.min_value = min_value
        self.max_value = max_value

    def generate_random_numbers(self, n):
        # u1 = np.random.normal(0, 1, (1, 2 * n))
        # u2 = np.random.normal(0, 1, (1, 2 * n))
        u1 = norm.ppf(np.random.rand(1, 2 * n))
        u2 = norm.ppf(np.random.rand(1, 2 * n))
        ids = u2 > self.alpha * u1
        u1[ids] = -u1[ids]
        x = self.xi + self.omega * u1
        if self.is_truncated:
            x = x[np.logical_and(x > self.min_value, x < self.max_value)]
            return x[:n]
        return x[0, :n]

    def cdf(self, x):
        return skewnorm.cdf(x, self.alpha, loc=self.xi, scale=self.omega)

    def pdf(self, x):
        pdf = skewnorm.pdf(x, self.alpha, loc=self.xi, scale=self.omega)

        if self.is_truncated:
            in_range = (self.min_value < x) & (x <= self.max_value)
            return (in_range * pdf) / (self.cdf(self.max_value) - self.cdf(self.min_value))

        return pdf




