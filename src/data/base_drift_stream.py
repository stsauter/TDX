from abc import ABCMeta
import numpy as np


class BaseDriftStream(metaclass=ABCMeta):

    def __init__(self, n_samples, n_segments, n_components):
        self.n_samples = n_samples
        self.n_segments = n_segments
        self.n_components = n_components
        self.segment_length = 1 / n_segments
        self.seg_data_per = np.tile(1 / n_segments, (1, n_segments))
