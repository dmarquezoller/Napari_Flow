import numpy as np

# Clean Absolute Import!
from napari_flow_editor.flow_nodes.filters import gaussian_blur

def test_gaussian_blur_runs():
    image = np.random.random((100, 100))
    result = gaussian_blur(image, sigma=2.0)
    assert result.shape == (100, 100)