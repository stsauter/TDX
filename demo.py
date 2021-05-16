import math
import dash
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from src.data.weight_drift_stream import WeightDriftStream
from src.density_evaluation.density_evaluator import DensityEvaluator
from src.spatio_temporal_generators.grid_sampler import GridSampler

ds = WeightDriftStream(25000, 120, dist_support=[0, 7], seed=1)

train_idx = range(math.ceil(0.66 * ds.x.shape[0]))
x_train = ds.x[train_idx]
t_train = ds.t[train_idx]
test_idx = range(train_idx[-1] + 1, ds.x.shape[0])
x_test = ds.x[test_idx]
t_test = ds.t[test_idx]

filename = 'tdx_trained'
infile = open(filename, 'rb')
model = pickle.load(infile)
infile.close()

"""
x = np.load('x.npy')
t = np.load('t.npy')
"
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

model = AgglomerativeClustering(distance_threshold=1.0, n_clusters=None)
# model = AgglomerativeClustering(distance_threshold=None, n_clusters=5)
model = model.fit(X_scaled)

Z = hierarchy.linkage(X_scaled, 'ward')
plt.figure(figsize=(20, 10))
dn = hierarchy.dendrogram(Z)

clusters = fcluster(Z, 3, criterion='maxclust')
plt.figure(figsize=(10, 8))
# plt.scatter(x[:, 0], x[:, 1], c=model.labels_, cmap='prism')
plt.scatter(x[:, 0], x[:, 1], c=clusters, cmap='prism')
plt.show()

"""

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

generator = GridSampler()
evaluator = DensityEvaluator(app, ds, model, generator)
app.layout = evaluator.plot_differences()

if __name__ == '__main__':
    app.run_server(debug=True)
