import numpy as np


class GridSampler:
    @staticmethod
    def generate_eval_locations(t_start: float, t_end: float, step_size: float, x, n_spatial_points: int):
        n_time_points = int(np.rint((t_end - t_start) / step_size)) + 1
        t = np.linspace(t_start, t_end, n_time_points)
        x = np.linspace(np.quantile(x, 0.01), np.quantile(x, 0.99), n_spatial_points)
        return x, t
