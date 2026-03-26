import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from napari_flow_editor.flow_nodes.visualisation import (
    plot_histogram,
    colocalization_scatter,
    table_heatmap_plot,
)


def _is_supported_plot_object(obj):
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], dict):
        obj = obj[0]
    module_name = getattr(type(obj), "__module__", "")
    is_plotly = "plotly" in module_name and hasattr(obj, "to_html")
    return is_plotly or isinstance(obj, Figure)


def test_plot_histogram_returns_plot_object():
    image = np.random.default_rng(0).normal(size=(32, 32))
    fig = plot_histogram(image)
    assert _is_supported_plot_object(fig)


def test_colocalization_scatter_shape_mismatch_returns_plot_object():
    ch1 = np.zeros((16, 16), dtype=np.float32)
    ch2 = np.zeros((8, 8), dtype=np.float32)
    fig = colocalization_scatter(ch1, ch2)
    assert _is_supported_plot_object(fig)


def test_table_heatmap_returns_plot_object():
    table = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    fig = table_heatmap_plot(table, colormap="viridis", normalize=True)
    assert _is_supported_plot_object(fig)
