import math

import numpy as np
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler
from src.spatio_temporal_generators.grid_sampler import GridSampler


class DensityEvaluator:
    def __init__(self, app: dash.Dash, model1, model2, spatio_temp_generator: GridSampler):
        self.app = dash.Dash(__name__)
        self._model1 = model1
        self._model2 = model2
        self._spatio_temp_generator = spatio_temp_generator
        app.callback(
            Output('density-graph', 'figure'),
            Input('x_eval_locations', 'value'),
            Input('time-range-slider', 'value'),
            Input('step-size-slider', 'value'),
            Input('threshold', 'value'),
            Input('chart-type', 'value'),
            Input('clustering-active', 'value')
        )(self.update_graph)

    def compute_differences(self, t_start: float, t_end: float, step_size: float, n_spatial_points: int, threshold, chart_type):
        x, t = self._spatio_temp_generator.generate_eval_locations(t_start, t_end, step_size, self._model1.x,
                                                                   n_spatial_points)
        model1_dens = self._model1.pdf(x, t)
        model2_dens = self._model2.pdf(x, t)
        dens_differences = model1_dens - model2_dens
        abs_dens_differences = abs(model1_dens - model2_dens)
        abs_dens_differences[abs_dens_differences < threshold] = 0
        if chart_type == 'scatter':
            # self.cluster(x, t, abs_dens_differences)
            return pd.DataFrame({
                "Time": np.repeat(t, n_spatial_points),
                "X values": np.tile(x, t.shape[0]),
                "difference": dens_differences.flatten(),
                "abs difference": abs_dens_differences.flatten()
            })
        else:
            df = pd.DataFrame(abs_dens_differences.T)
            df.columns = t
            return df

    def cluster(self, x, t, diff):
        """
        x = np.zeros(shape=(t.shape[0] * x_vals.shape[0], 3))
        counter = 0
        for i in range(t.shape[0]):
            for j in range(x_vals.shape[0]):
                x[counter] = [t[i], x_vals[j], diff[i, j]]
                counter = counter + 1
        """
        dataset = pd.concat([t, x, diff], axis=1)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(dataset)
        Z = hierarchy.linkage(x_scaled, 'ward')
        n_clusters = 2
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        cluster_labels = np.empty(clusters.shape[0], dtype=object)
        cluster_diffs = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_diffs[i] = np.sum(diff[clusters == i + 1]) / len(diff[clusters == i + 1])

        dev_cluster_id = np.argmax(cluster_diffs) + 1
        cluster_labels[clusters == dev_cluster_id] = 'High deviations'
        cluster_labels[clusters != dev_cluster_id] = 'Low deviations'

        for i, row in enumerate(x_scaled):
            for j, column in enumerate(row):
                if not math.isclose(x_scaled[i, j], x_scaled[i][j], abs_tol=0.01):
                    a = x_scaled[i, j]
                    aa = x_scaled[i, j]

        return pd.DataFrame({
            "Time": dataset["Time"],
            "X values": dataset["X values"],
            "clusters": cluster_labels,
            "abs difference": dataset["abs difference"]
        })

    def plot_differences(self):
        return html.Div(children=[
            html.H1(children='Density Evaluation'),
            dcc.Graph(id='density-graph'),
            html.Div(children=[
                html.H4(children='Time Filter'),
                dcc.RangeSlider(
                    id='time-range-slider',
                    min=0.1,
                    max=0.9,
                    step=0.1,
                    value=[0.1, 0.9],
                    marks={
                        0.1: '0.1',
                        0.2: '0.2',
                        0.3: '0.3',
                        0.4: '0.4',
                        0.5: '0.5',
                        0.6: '0.6',
                        0.7: '0.7',
                        0.8: '0.8',
                        0.9: '0.9'
                    },
                )
            ], style={'width': '40%', 'margin-bottom': '2em', 'display': 'inline-block'}),
            html.Div(children=[
                html.H4(children='Step Size'),
                dcc.Slider(
                    id='step-size-slider',
                    min=0.01,
                    max=0.1,
                    step=0.05,
                    value=0.1,
                    marks={
                        0.01: '0.01',
                        0.05: '0.05',
                        0.1: '0.1'
                    },
                )
            ], style={'width': '20%', 'margin-bottom': '2em'}),
            html.Div(children=[
                html.H4(children='Number of sampling values'),
                dcc.Input(id="x_eval_locations", type="number", value=20, min=10, max=100, step=1),
            ]),
            html.Div(children=[
                html.H4(children='Threshold'),
                dcc.Input(id="threshold", type="number",  value=0.02, min=0, max=2, step=0.01),
            ]),
            html.Div(children=[
                html.H4(children='Chart Type'),
                dcc.Dropdown(
                    id='chart-type',
                    options=[
                        {'label': 'Scatter Plot', 'value': 'scatter'},
                        {'label': 'Heatmap', 'value': 'heatmap'},
                    ],
                    value='scatter'
                ),
            ], style={'width': '20%', 'display': 'inline-block'}),
            html.Div(children=[
                html.H4(children='Clustering'),
                dcc.Checklist(
                    id='clustering-active',
                    options=[
                        {'label': 'Clustering activated', 'value': 'active'},
                    ]
                ),
            ], style={'width': '20%', 'margin-bottom': '2em'})
        ])

    def update_graph(self, n_spatial_points, time_range_value, step_size, threshold, chart_type, clustering_active):
        if n_spatial_points is None or threshold is None:
            raise PreventUpdate
        df = self.compute_differences(time_range_value[0], time_range_value[1], step_size, n_spatial_points, threshold,
                                 chart_type)

        if clustering_active and len(clustering_active) == 1:
            cluster_df = self.cluster(df['X values'], df['Time'], df['abs difference'])
            fig = px.scatter(cluster_df, x="Time", y="X values", color="clusters", color_discrete_map={
                "High deviations": "red",
                "Low deviations": "blue"})
        else:
            if chart_type == 'scatter':
                fig = px.scatter(df, x="Time", y="X values", color="abs difference", size="abs difference")
            else:
                fig = go.Figure(data=go.Heatmap(
                    z=df,
                    x=df.columns,
                    hoverongaps=False))

        fig.update_layout(transition_duration=500)
        return fig
