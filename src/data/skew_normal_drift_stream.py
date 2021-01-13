import numpy as np

from collections import namedtuple
from sklearn.utils import check_random_state
from src.data.base_drift_stream import BaseDriftStream
from src.proba.skew_normal import SkewNormal


class SkewNormalDriftStream(BaseDriftStream):

    def __init__(self, n_samples, n_segments, n_components, dist_support=None, random_state=None):
        super().__init__(n_samples, n_segments, n_components)
        self._location = np.array([])
        self._scale = np.array([])
        self._shape = np.array([])
        if dist_support is not None and len(dist_support) != 2:
            raise ValueError('Distribution support should be an array containing 2 elements')
        self._support = dist_support
        self._random_state = check_random_state(random_state)

    @property
    def dist_support(self):
        return self._support

    def _generate_data(self):
        params = self._get_distribution_params()
        for i in range(self._n_segments):
            n_seg_samples = np.rint(self._n_samples * (self._mixture_coefs[:, i] * self._seg_data_per[i])).astype(int)
            x_s = np.array([])
            c_s = np.array([])
            t_s = np.array([])
            for j in range(self._n_components):
                pd = SkewNormal(params[j, i].xi, params[j, i].omega, params[j, i].alpha, self._random_state)
                if self._support is not None:
                    pd.truncate(self._support[0], self._support[1])
                sampled_x = pd.generate_random_numbers(n_seg_samples[j])
                self._pds[j, i] = pd
                x_s = np.append(x_s, sampled_x, axis=0)
                c_s = np.append(c_s, np.tile(j + 1, n_seg_samples[j]), axis=0)
                t_s = np.append(t_s, np.tile(i * self._segment_length, n_seg_samples[j]), axis=0)

            self._x = np.append(self._x, x_s, axis=0)
            self._c = np.append(self._c, c_s, axis=0).astype(int)
            self._t = np.append(self._t, t_s, axis=0)

    def _get_distribution_params(self):
        SkewNormalParams = namedtuple('SkewNormalParams', ['xi', 'omega', 'alpha'])
        params = np.empty(shape=(self._n_components, self._n_segments), dtype=object)
        for i in range(self._n_components):
            for j in range(self._n_segments):
                params[i, j] = SkewNormalParams(self._location[i, j], self._scale[i, j], self._shape[i, j])
        return params
