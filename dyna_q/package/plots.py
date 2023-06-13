import plotly.graph_objects as go
import pandas as pd


def plot_heatmap(matrix, **kwargs):
    return go.Heatmap(z=matrix.values[::-1],
                      x=matrix.columns,
                      y=matrix.index[::-1],
                      colorscale='Viridis',
                      **kwargs)


def plot_bar_chart(dataframe: pd.DataFrame, attribute: 'str', color: str):
    return go.Bar(y=dataframe[attribute],
                  marker=dict(color=dataframe[color]))
