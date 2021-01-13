import numpy as np

from src.data.skew_normal_drift_stream import SkewNormalDriftStream


class MeanDriftStream(SkewNormalDriftStream):

    def __init__(self, n_samples, n_segments, dist_support=None, random_state=None):
        super().__init__(n_samples, n_segments, 4, dist_support, random_state)
        self._location = np.array([
            np.tile(1, n_segments),
            np.linspace(4, 2.5, n_segments),
            np.linspace(6, 8, n_segments),
            np.linspace(9, 8.5, n_segments)
        ])
        self._scale = self._repeat_segment_values([1, 0.7, 1, 1])
        self._shape = np.array([
            np.linspace(3, 0, n_segments),
            np.linspace(2, -1, n_segments),
            np.tile(0.8, n_segments),
            np.tile(2, n_segments)
        ])
        self._mixture_coefs = np.tile(1 / self._n_components, (self._n_components, n_segments))
        self._generate_data()
