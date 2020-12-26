import numpy as np

from collections import namedtuple
from src.data.base_drift_stream import BaseDriftStream
from src.proba.skew_normal import SkewNormal


class WeightDriftStream(BaseDriftStream):

    def __init__(self, n_samples, n_segments):
        super().__init__(n_samples, n_segments, 3)
        self.seg_data_per = np.tile(1 / n_segments, n_segments)
        self.location = np.array([
            np.tile(1, n_segments),
            np.tile(4, n_segments),
            np.tile(5, n_segments)
        ])
        self.scale = np.array([
            np.tile(0.7, n_segments),
            np.tile(0.6, n_segments),
            np.tile(1, n_segments)
        ])
        self.shape = np.array([
            np.tile(1, n_segments),
            np.tile(-0.5, n_segments),
            np.tile(1.5, n_segments)
        ])
        self.mixture_coefs = np.array([
            np.linspace(0.1, 0.45, n_segments),
            np.linspace(0.7, 0.1, n_segments)
        ])
        self.mixture_coefs = np.append(self.mixture_coefs, [1 - np.sum(self.mixture_coefs, axis=0)], axis=0)

        SkewNormalParams = namedtuple('SkewNormalParams', ['xi', 'omega', 'alpha'])
        params = np.empty(shape=(self.n_components, self.n_segments), dtype=object)
        for i in range(self.n_components):
            for j in range(self.n_segments):
                params[i][j] = SkewNormalParams(self.location[i][j], self.scale[i][j], self.shape[i][j])

        self._x = np.array([])
        self._c = np.array([])
        self._t = np.array([])
        self.pds = np.empty(shape=(self.n_components, self.n_segments), dtype=object)
        for i in range(self.n_segments):
            n_seg_samples = np.rint(self.n_samples * (self.mixture_coefs[:, i] * self.seg_data_per[i])).astype(int)
            x_s = np.array([])
            c_s = np.array([])
            t_s = np.array([])
            for j in range(self.n_components):
                pd = SkewNormal(params[j][i].xi, params[j][i].omega, params[j][i].alpha)
                pd.truncate(0, 7)
                sampled_x = pd.generate_random_numbers(n_seg_samples[j])
                self.pds[j][i] = pd
                x_s = np.append(x_s, sampled_x, axis=0)
                c_s = np.append(c_s, np.tile(j+1, n_seg_samples[j]), axis=0)
                t_s = np.append(t_s, np.tile(i * self.segment_length, n_seg_samples[j]), axis=0)

            self._x = np.append(self._x, x_s, axis=0)
            self._c = np.append(self._c, c_s, axis=0).astype(int)
            self._t = np.append(self._t, t_s, axis=0)

    @property
    def x(self):
        return self._x

    @property
    def t(self):
        return self._t

    @property
    def c(self):
        return self._c

    def pdf(self, x, t):
        pdf = np.zeros((t.shape[0], x.shape[1]))
        segments = t / self.segment_length
        unique_segments = np.unique(segments).astype(int)
        for segment in unique_segments:
            f = np.zeros((1, x.shape[1]))
            for c in range(self.n_components):
                f = f + (self.mixture_coefs[c][segment] * self.pds[c][segment].pdf(x))
            n_seg_samples = segments[segments == segment].shape[0]
            pdf[segments == segment, :] = np.tile(f, (n_seg_samples, 1))

        return pdf
