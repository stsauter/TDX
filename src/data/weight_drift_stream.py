import numpy as np

from typing import List

from src.data.skew_normal_drift_stream import SkewNormalDriftStream


class WeightDriftStream(SkewNormalDriftStream):
    """Weightdrift stream.

    The weightdrift stream is an artificial generated univariate data stream described in [1]_.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_segments: int
        Number of time segments.
    dist_support : array_like of shape (2,)
        List containing the minimum and maximum value which should be supported the skew normal distributions
    seed : None | int | instance of RandomState (default=None)
        Random number generator seed for reproducibility.

    References
    ----------
    .. [1] Krempl, G., Lang, D. & Hofer, V. Temporal density extrapolation using a dynamic basis approach.
       In  Data Min Knowl Disc 33, 1323–1356 (2019).
    """

    def __init__(
            self,
            n_samples: int,
            n_segments: int,
            dist_support: List[int] = None,
            seed: int or np.random.RandomState = None
    ):
        super().__init__(n_samples, n_segments, 3, dist_support, seed)
        self._location = self._repeat_segment_values([1, 4, 5])
        self._scale = self._repeat_segment_values([0.7, 0.6, 1])
        self._shape = self._repeat_segment_values([1, -0.5, 1.5])
        self._mixture_coefs = np.array([
            np.linspace(0.1, 0.45, n_segments),
            np.linspace(0.7, 0.1, n_segments)
        ])
        self._mixture_coefs = np.append(self._mixture_coefs, [1 - np.sum(self._mixture_coefs, axis=0)], axis=0)
