import math

import numpy as np

from src.data.weight_drift_stream import WeightDriftStream
from src.tdx.tdx import Tdx

np.random.seed(1)

ds = WeightDriftStream(25000, 120)

train_idx = range(math.ceil(0.66 * ds.x.shape[0]))
x_train = ds.x[train_idx]
t_train = ds.t[train_idx]
test_idx = range(train_idx[-1] + 1, ds.x.shape[0])
x_test = ds.x[test_idx]
t_test = ds.t[test_idx]

model = Tdx(14, 0.6, 5, 2).fit(x_train, t_train)

x_grid = np.linspace(np.quantile(ds.x, 0.01), np.quantile(ds.x, 0.99), 200).reshape(1, 200)
true_dens = ds.pdf(x_grid, t_test)

print("This line will be printed.")
