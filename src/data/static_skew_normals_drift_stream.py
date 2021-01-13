import numpy as np

from src.data.skew_normal_drift_stream import SkewNormalDriftStream


class StaticSkewNormalsDriftStream(SkewNormalDriftStream):

    def __init__(self, n_samples, n_segments, dist_support=None, random_state=None):
        super().__init__(n_samples, n_segments, 4, dist_support, random_state)
        self._location = self._repeat_segment_values([1, 4, 6, 9])
        self._scale = np.tile(1, (self._n_components, n_segments))
        self._shape = self._repeat_segment_values([3, 2, 0.8, 2])
        self._mixture_coefs = np.tile(1 / self._n_components, (self._n_components, n_segments))
        self._generate_data()
