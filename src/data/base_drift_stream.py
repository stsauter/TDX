import numpy as np

from abc import ABCMeta, abstractmethod


class BaseDriftStream(metaclass=ABCMeta):

    def __init__(self, n_samples, n_segments, n_components):
        self._n_samples = n_samples
        self._n_segments = n_segments
        self._n_components = n_components
        self._segment_length = 1 / n_segments
        self._seg_data_per = np.tile(1 / n_segments, n_segments)
        self._pds = np.empty(shape=(self._n_components, self._n_segments), dtype=object)
        self._mixture_coefs = np.zeros((self._n_components, self._n_segments))
        self._x = np.array([])
        self._t = np.array([])
        self._c = np.array([])

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def n_segments(self):
        return self._n_segments

    @property
    def n_components(self):
        return self._n_components

    @property
    def x(self):
        return self._x

    @property
    def t(self):
        return self._t

    @property
    def c(self):
        return self._c

    @abstractmethod
    def _generate_data(self):
        raise NotImplementedError

    def _repeat_segment_values(self, values):
        if len(values) != self._n_components:
            raise ValueError("Length of array should be the same as the number of components")

        segment_array = np.zeros((self._n_components, self._n_segments))
        for i, val in enumerate(values):
            segment_array[i, :] = np.tile(val, (1, self._n_segments))
        return segment_array

    def pdf(self, x, t):
        pdf = np.zeros((t.shape[0], x.shape[0]))
        segments = (t / self._segment_length).round().astype(int)
        unique_segments = np.unique(segments).round().astype(int)
        for segment in unique_segments:
            f = np.zeros((1, x.shape[0]))
            for c in range(self._n_components):
                f = f + (self._mixture_coefs[c, segment] * self._pds[c, segment].pdf(x))
            n_seg_samples = segments[segments == segment].shape[0]
            pdf[segments == segment, :] = np.tile(f, (n_seg_samples, 1))

        return pdf
