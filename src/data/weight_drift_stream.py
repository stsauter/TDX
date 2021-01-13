import numpy as np

from src.data.skew_normal_drift_stream import SkewNormalDriftStream


class WeightDriftStream(SkewNormalDriftStream):

    def __init__(self, n_samples, n_segments, dist_support=None, random_state=None):
        super().__init__(n_samples, n_segments, 3, dist_support, random_state)
        self._location = self._repeat_segment_values([1, 4, 5])
        self._scale = self._repeat_segment_values([0.7, 0.6, 1])
        self._shape = self._repeat_segment_values([1, -0.5, 1.5])
        self._mixture_coefs = np.array([
            np.linspace(0.1, 0.45, n_segments),
            np.linspace(0.7, 0.1, n_segments)
        ])
        self._mixture_coefs = np.append(self._mixture_coefs, [1 - np.sum(self._mixture_coefs, axis=0)], axis=0)
        self._generate_data()
