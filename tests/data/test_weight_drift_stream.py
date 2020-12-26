import os
import math
import numpy as np
import pandas as pd


import pytest
# pytest.set_trace() python -m pytest

from src.data.weight_drift_stream import WeightDriftStream


def test_stream_data(test_path):
    np.random.seed(1)
    ds = WeightDriftStream(25000, 120)

    test_file = os.path.join(test_path, 'weightdrift_stream.csv')
    raw_data = pd.read_csv(test_file)
    for i, row in raw_data.iterrows():
        assert math.isclose(row[0], ds.x[i], abs_tol=0.001)
        assert math.isclose(row[1], ds.t[i], abs_tol=0.001)
        assert row[2] == ds.c[i]


def test_stream_pdf(test_path):
    np.random.seed(1)
    ds = WeightDriftStream(25000, 120)

    train_idx = range(math.ceil(0.66 * ds.x.shape[0]))
    test_idx = range(train_idx[-1] + 1, ds.x.shape[0])
    t_test = ds.t[test_idx]
    x_grid = np.linspace(np.quantile(ds.x, 0.01), np.quantile(ds.x, 0.99), 20).reshape(1, 20)
    ds_dens = ds.pdf(x_grid, t_test)

    test_file = os.path.join(test_path, 'weightdrift_pdf.csv')
    raw_data = pd.read_csv(test_file, header=None)
    for i, row in raw_data.iterrows():
        for j, column in enumerate(row):
            assert math.isclose(column, ds_dens[i][j], abs_tol=0.005)
