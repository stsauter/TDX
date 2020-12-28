import math

import numpy as np
import matplotlib.pyplot as plt

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

model = Tdx(14, 0.6, 5, 2)
model.fit(x_train, t_train)
gamma = model.get_gamma(ds.t)

x_grid = np.linspace(np.quantile(ds.x, 0.01), np.quantile(ds.x, 0.99), 200)
true_dens = ds.pdf(x_grid.reshape(1, 200), t_test)
pred_dens = model.pdf(x_grid.reshape(1, 200), t_test)

time_idxs = [0, math.ceil(x_test.shape[0] / 2), x_test.shape[0] - 1]
fig, axs = plt.subplots(len(time_idxs))
for i, time_idx in enumerate(time_idxs):
    axs[i].plot(x_grid, true_dens[time_idx, :], '-b', x_grid, pred_dens[time_idx, :], '--r')
    axs[i].set_title('Density at t=' + str(round(t_test[time_idx], 4)))
    axs[i].set(xlabel='X', ylabel='P(X)')

for ax in axs.flat:
    ax.label_outer()

plt.show()
