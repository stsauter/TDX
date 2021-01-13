import numpy as np

from src.data.skew_normal_drift_stream import SkewNormalDriftStream


class SigmaChangeDriftStream(SkewNormalDriftStream):

    def __init__(self, n_samples, n_segments, dist_support=None, random_state=None):
        super().__init__(n_samples, n_segments, 3, dist_support, random_state)
        self._location = self._repeat_segment_values([1, 4, 7.5])
        self._scale = np.array([
            np.linspace(0.5, 1.5, n_segments),
            np.linspace(1.7, 0.7, n_segments),
            np.linspace(0.5, 1.6, n_segments),
        ])
        self._shape = self._repeat_segment_values([2, 1, -0.8])
        self._mixture_coefs = np.tile(1 / self._n_components, (self._n_components, n_segments))
        self._generate_data()
