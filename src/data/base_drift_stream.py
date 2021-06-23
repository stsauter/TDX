import numpy as np

from abc import abstractmethod

from river.datasets import base


class BaseDriftStream(base.SyntheticDataset):
    """Base class for streams with concept drift.

    This abstract class defines the minimum requirements of a stream with concept drift

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_segments: int
        Number of time segments.
    n_components: int
        Number of components the stream consists of.
    """

    def __init__(self, n_samples: int, n_segments: int, n_components: int):
        if n_segments > n_samples:
            raise ValueError('The number of segments may not be greater than the number of samples')

        super().__init__(n_features=2, n_outputs=0, n_samples=n_samples, task=base.DENS_EST)

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
    def n_segments(self):
        """Return the number of time segments.

        Returns
        -------
        int:
            the number of time segments
        """
        return self._n_segments

    @property
    def n_components(self):
        """Return the number of streaming components.

         Returns
         -------
         int:
             the number of streaming components.
         """
        return self._n_components

    @property
    def x(self):
        """Return data points of the data stream.

         Returns
         -------
         numpy.ndarray of shape (n_samples,):
             Data points of the data stream.
         """
        return self._x

    @property
    def t(self):
        """Return array containing time segments.

         Returns
         -------
         numpy.ndarray of shape (n_samples,):
             Array containing the time segments of the data stream.
         """
        return self._t

    @property
    def c(self):
        """Return array containing the streaming components.

         Returns
         -------
         numpy.ndarray of shape (n_samples,):
             Array containing the component numbers which have generated the corresponding data points.
         """
        return self._c

    def __iter__(self):
        for i, x_val in enumerate(self._x):
            x = dict()
            x['timestamp'] = self._t[i]
            x['value'] = x_val
            yield x

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
        """Compute the probability density at point x and at a given time point t.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_samples,)
            Data points at which the probability density function should be evaluated.
        t : numpy.ndarray of shape (n_time_values,)
            Time values at which the probability densities should be computed.

        Returns
        -------
        numpy.ndarray of shape (n_time_values, n_samples)
            Probability density function evaluated at point x and time point t.
        """
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
