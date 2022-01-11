import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
x_data = pd.read_csv('x.csv', header=None)
t_data = pd.read_csv('t.csv', header=None)
x_inc = []
t_inc = []
i = 0
for x in ds:
    x_inc.append(x.get('value'))
    t_inc.append(x.get('timestamp'))
    assert math.isclose(x_data[0].iloc[i], x_inc[i], abs_tol=0.001)
    assert math.isclose(t_data[0].iloc[i], t_inc[i], abs_tol=0.001)
    i = i + 1
# ds = MeanDriftStream(25000, 120, dist_support=[0, 12], seed=1)
# ds = SigmaChangeDriftStream(25000, 120, dist_support=[0, 10], seed=1)
# ds = StaticSkewNormalsDriftStream(25000, 120, dist_support=[0, 11], seed=1)

model = Tdx(m=14, bandwidth=0.6, r=5, lambda_reg=2,seed=32, cache_size=2500, grace_period=2500)

x_values = np.zeros(25000)
t_values = np.zeros(25000)
i = 0
for x in ds:
    x_values[i] = x['value']
    t_values[i] = x['timestamp']
    i = i + 1

dd = list(ds)

for x in ds:
    density = model.predict_density_one(x)
    model.learn_one(x)

train_idx = range(math.ceil(0.66 * ds.x.shape[0]))
x_train = ds.x[train_idx]
t_train = ds.t[train_idx]
test_idx = range(train_idx[-1] + 1, ds.x.shape[0])
x_test = ds.x[test_idx]
t_test = ds.t[test_idx]

model = Tdx(14, 0.6, 5, 2, seed=32, verbose=True, n_start_points=1)
print(model)
df = pd.DataFrame({'timestamp': t_train, 'value': x_train})
model.learn_many(df)

# model = Tdx(14, 0.6, 1, 2, cache_size=4125, grace_period=4125, seed=32, verbose=True, n_start_points=1)
counter = 0
for x in ds:
    density = model.predict_density_one(x)
    #model.learn_one(x)
    counter = counter + 1
    if counter == len(x_train):
        break

# model.fit_partial(x_train, t_train)
coefs = np.array([
    [10, -2, 0.1],
    [10, -2, 0.1]])
dd = model._transform_tdx_coeffs(coefs, 5, 10)
#model.fit(x_train[0:5000], t_train[0:5000])
orig_timestamps = t_train
#for x_part, t_part in get_every_n(x_train, t_train, n=4125):
    #model.fit_partial(x_part, t_part)

# model.fit(x_train, t_train)
# gamma = model.get_gamma(ds.t)

x_grid = np.linspace(np.quantile(ds.x, 0.01), np.quantile(ds.x, 0.99), 200)
true_dens = ds.pdf(x_grid, t_test)
pred_dens = model.pdf(x_grid, t_test)

x_many_test = np.tile(x_grid.reshape(1, x_grid.shape[0]), (t_test.shape[0], 1))
df = pd.concat((pd.DataFrame({'timestamp': t_test}), pd.DataFrame(x_grid)), axis=1)
pred_dens2 = model.predict_density_many(df)


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
