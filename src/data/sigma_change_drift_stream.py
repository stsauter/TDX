import numpy as np

from typing import List
from src.data.skew_normal_drift_stream import SkewNormalDriftStream


class SigmaChangeDriftStream(SkewNormalDriftStream):
    """Sigmachange stream.

    The sigmachange stream is an artificial generated univariate data stream described in [1]_.

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
        self._location = self._repeat_segment_values([1, 4, 7.5])
        self._scale = np.array([
            np.linspace(0.5, 1.5, n_segments),
            np.linspace(1.7, 0.7, n_segments),
            np.linspace(0.5, 1.6, n_segments),
        ])
        self._shape = self._repeat_segment_values([2, 1, -0.8])
        self._mixture_coefs = np.tile(1 / self._n_components, (self._n_components, n_segments))
