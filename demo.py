import math
import matplotlib.pyplot as plt
import numpy as np

from src.data.mean_drift_stream import MeanDriftStream
from src.data.sigma_change_drift_stream import SigmaChangeDriftStream
from src.data.static_skew_normals_drift_stream import StaticSkewNormalsDriftStream
from src.data.weight_drift_stream import WeightDriftStream
from src.tdx.tdx import Tdx


def get_every_n(x, t, n=2):
    for i in range(x.shape[0] // n):
        last_idx = n*(i+1)
        yield x[n*i:n*(i+1)], t[n*i:n*(i+1)]
    if x.shape[0] % n > 0:
        yield x[last_idx:], t[last_idx:]

ds = WeightDriftStream(25000, 120, dist_support=[0, 7], seed=1)
# ds = MeanDriftStream(25000, 120, dist_support=[0, 12], seed=1)
# ds = SigmaChangeDriftStream(25000, 120, dist_support=[0, 10], seed=1)
# ds = StaticSkewNormalsDriftStream(25000, 120, dist_support=[0, 11], seed=1)

train_idx = range(math.ceil(0.66 * ds.x.shape[0]))
x_train = ds.x[train_idx]
t_train = ds.t[train_idx]
test_idx = range(train_idx[-1] + 1, ds.x.shape[0])
x_test = ds.x[test_idx]
t_test = ds.t[test_idx]

model = Tdx(14, 0.6, 1, 2, seed=32, verbose=True, n_start_points=1)

# model.fit_partial(x_train, t_train)
coefs = np.array([
    [10, -2, 0.1],
    [10, -2, 0.1]])
dd = model._transform_tdx_coeffs(coefs, 5, 10)
#model.fit(x_train[0:5000], t_train[0:5000])
orig_timestamps = t_train
for x_part, t_part in get_every_n(x_train, t_train, n=825):
    model.fit_partial(x_part, t_part)

# model.fit(x_train, t_train)
# gamma = model.get_gamma(ds.t)

x_grid = np.linspace(np.quantile(ds.x, 0.01), np.quantile(ds.x, 0.99), 200)
true_dens = ds.pdf(x_grid, t_test)
pred_dens = model.pdf(x_grid, t_test)

time_idxs = [0, math.ceil(x_test.shape[0] / 2), x_test.shape[0] - 1]
fig, axs = plt.subplots(len(time_idxs))
for i, time_idx in enumerate(time_idxs):
    axs[i].plot(x_grid, true_dens[time_idx, :], '-b', label="True density")
    axs[i].plot(x_grid, pred_dens[time_idx, :], '--r', label="Predicted density")
    axs[i].set_title('Density at t=' + str(round(t_test[time_idx], 4)))
    axs[i].set(xlabel='X', ylabel='P(X)')
    axs[i].legend(loc="upper right")

for ax in axs.flat:
    ax.label_outer()

# plt.subplots_adjust(hspace=0.5)
plt.show()
