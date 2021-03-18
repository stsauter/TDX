import numpy as np
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from src.spatio_temporal_generators.grid_sampler import GridSampler


class DensityEvaluator:
    def __init__(self, app: dash.Dash, model1, model2, spatio_temp_generator: GridSampler):
        self.app = dash.Dash(__name__)
        self._model1 = model1
        self._model2 = model2
        self._spatio_temp_generator = spatio_temp_generator
        self._t = np.array([])
        self._df = pd.DataFrame()
        self._cluster = False
        app.callback(
            Output('density-graph', 'figure'),
            Input('x_eval_locations', 'value'),
            Input('time-range-slider', 'value'),
            Input('step-size-slider', 'value'),
            Input('threshold', 'value'),
            Input('chart-type', 'value')
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
            self._df = pd.DataFrame({
                "Time": np.repeat(t, n_spatial_points),
                "X values": np.tile(x, t.shape[0]),
                "difference": dens_differences.flatten(),
                "abs difference": abs_dens_differences.flatten()
            })
        else:
            self._df = pd.DataFrame(abs_dens_differences.T)
            self._t = t

        if self._cluster:
            self.cluster(x, t, abs_dens_differences)

    def cluster(self, x_vals, t, diff):
        x = np.zeros(shape=(t.shape[0] * x_vals.shape[0], 3))
        counter = 0
        for i in range(t.shape[0]):
            for j in range(x_vals.shape[0]):
                x[counter] = [t[i], x_vals[j], diff[i, j]]
                counter = counter + 1

        model = AgglomerativeClustering(distance_threshold=1.0, n_clusters=None)
        model = model.fit(x)

        Z = hierarchy.linkage(model.children_, 'ward')
        plt.figure(figsize=(20, 10))
        dn = hierarchy.dendrogram(Z)
        sd = 3


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
            ], style={'width': '20%', 'margin-bottom': '2em', 'display': 'inline-block'})
        ])

    def update_graph(self, n_spatial_points, time_range_value, step_size, threshold, chart_type):
        if n_spatial_points is None or threshold is None:
            raise PreventUpdate
        self.compute_differences(time_range_value[0], time_range_value[1], step_size, n_spatial_points, threshold, chart_type)
        if chart_type == 'scatter':
            fig = px.scatter(self._df, x="Time", y="X values", color="abs difference", size="abs difference")
        else:
            fig = go.Figure(data=go.Heatmap(
                z=self._df,
                x=self._t,
                hoverongaps=False))
        fig.update_layout(transition_duration=500)
        return fig
