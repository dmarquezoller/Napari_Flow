# Napari Flow Editor

**Visual image processing pipelines for [napari](https://napari.org/).**

`napari-flow-editor` allows you to construct complex image analysis workflows using a node-based graph interface. Connect filters, transformations, and segmentation algorithms visually, adjust parameters in real-time, and execute pipelines directly on your active Napari layers.

## ✨ Features

* **Graph Interface:** Drag-and-drop nodes to create processing pipelines.
* **🚀Smart Caching:** The engine remembers previous results. If you change a parameter at the end of a pipeline, it **only re-runs the affected nodes**, making tuning instant.
* **🚦Visual Feedback:** Nodes display traffic-light status dots (Grey/Yellow/Green/Red) so you know exactly what is running or cached. 
* **🖥️Live Console:** A built-in terminal at the bottom displays real-time execution logs and error traces.
* **Non-Blocking Execution:** Heavy calculations run in the background, keeping the interface responsive.
* **Auto-Generated Library:** Nodes are generated automatically from Python functions using decorators.
* **Extensible:** Import your own `.py` scripts to add custom nodes instantly.
* **Save/Load:** Persist your complex workflows to JSON files.

## 🛠 Installation

### Prerequisites
* Python 3.9+
* Napari (`pip install napari[all]`)

### 1. Clone the repository
```bash
git clone [https://github.com/dmarquezoller/Napari_Flow.git](https://github.com/dmarquezoller/Napari_Flow.git)
cd napari-flow-editor
```
### 2. Install
We recommend installing in **editable mode** so changes to your node library are reflected immediately.

```bash
pip install -e
```
### 3. Run
Launch Napari from your terminal
```bash
napari
```
Then navigate to `Plugins > Napari Flow Editor` in the top menu

## 📖 User Guide

### 1. The Interface

* **The Toolbar** (Top):
  * **Add Node:** Opens a categorized menu of available algorithms (Filters, Segmentation, etc.).
  * **Import .py:** Load custom user scripts containing your own nodes.(more info below)
  * **Save/Load:** Persist your graph structure to a JSON file to use later.
  * **Run Pipeline:** Executes the graph.
* **Properties Panel** (Right):
  * Click any node in the graph to view it here.
  * Adjust parameters (like `sigma` or `radius`) and see the connection status of sockets.
  * **Dynamic Inputs:** For "Get Layer" nodes, select the target layer from the dropdown menu, the available layers will be the layers currently loaded in napari. 
* **Graph View** (Center):
  * **Status Dots:** Every node has a colored light in the top-right corner: 
    * **⚪Grey:** Dirty/Parameter Changed (Will re-run next time)
    * **🟡Yellow:** Running...
    * **🟢Green:** Done/Cached (will skip next time if input or previous nodes don't change)
    * **🔴Red:** Error (Check the console)
  * **Controls:** Left-click to select, drag to connect sockets and middle-click to pan 
* **Execution Console:** (Bottom)
  * Displays the live progress of the pipeline and detailed error messages if node fails.

### 2. How to Create a Pipeline
Follow these steps to build a simple Gaussian Blur workflow:

1.  **Add Input:**
    * Go to `Add Node > Input > Get Active Layer`.
    * This node automatically grabs the currently selected image from the Napari layer list.
2.  **Add Processing:**
    * Go to `Add Node > Filters > Gaussian Blur`.
3.  **Connect:**
    * Click and drag a wire from the **Input** node's `image` output.
    * Drop it onto the **Blur** node's `image_in` input.
4.  **Configure:**
    * Click the **Blur** node to select it.
    * In the Properties Panel, increase the `sigma` value (e.g., to `5.0`).
5.  **Run:**
    * Click the big green **RUN PIPELINE** button.
    * Along the execution, while a given node is running, the status will turn yellow 🟡 and as soon as it finishes it will turn green 🟢 (or red 🔴 if fails).
    * A new layer named `Gaussian Blur (image_out)` will appear in Napari with the results.

### 3. Smart Caching and Optimization
The editor is designed for experimentation.
* If you have a long pipeline (A --> B --> C --> D ) and you only change a parameter in **Node C**:
  * When you click **Run Pipeline** again, the engine sees that **A** and **B** have not changed.
  * It **skips** A and B (instantly retrieving their cached results from memory).
  * In only executes **C** and **D**.
  * This allows you to tune downstream parameters instantly without waiting for heavy upstream filters to re-calculate.

## 👨‍💻 Developer Guide: Adding Custom Nodes

The engine uses a **Wrapper Pattern** to turn standard Python functions into GUI nodes automatically. You do not need to write UI code.

### 1. The Structure
Create a Python file (e.g., `my_nodes.py`). You must import the `register_node` decorator from the package.

```Python
from napari_flow_editor.flow_nodes.decorator import register_node
import skimage.filters

@register_node(
    label="My Custom Blur",
    category="Custom",
    outputs=["image_out"],
    params_config={
        "sigma": {"min": 0.1, "max": 20.0, "step": 0.1}
    }
)
def my_blur(image_in, sigma: float = 1.0):
    # 'image_in' has no default -> Input Socket
    # 'sigma' has a default -> Float SpinBox Widget
    return skimage.filters.gaussian(image_in, sigma=sigma)

```

### 2. Decorator Options

| Parameter | Description |
| :--- | :--- |
| `label` | The human-readable name shown on the Node title bar (e.g., "Gaussian Blur"). |
| `category` | The submenu name where the node will appear (e.g., "Filters" or "Segmentation"). |
| `outputs` | A list of strings defining the output sockets (e.g., `["image_out", "mask"]`). |
| `params_config` | A dictionary to configure specific widget limits (like `min`, `max`, `step` or `options`). |

## 🎨 Advanced Decorator API

The `@register_node` decorator supports advanced features for creating sophisticated, interactive nodes without writing boilerplate code.

### Interactive Nodes (`interactive`)

Create nodes that let users draw ROIs, pick points, or paint labels directly in Napari:

```python
@register_node(
    label="Interactive Crop",
    category="ROI",
    outputs=["cropped"],
    interactive={
        "layer_type": "shapes",      # "shapes", "points", or "labels"
        "tool": "rectangle",          # Tool to activate
        "prompt": "Draw a rectangle to define the crop region",
        "arg_name": "roi_geometry",   # Function argument to inject geometry
        "confirm": True,              # Show confirmation dialog
    }
)
def interactive_crop(image_in, roi_geometry=None):
    if roi_geometry is None:
        return image_in
    # Use roi_geometry to crop the image
    return cropped_image
```

When this node executes:
1. A temporary Napari layer is created (Shapes/Points/Labels)
2. The specified tool is activated (draw rectangle, add points, paint, etc.)
3. A dialog prompts the user to draw
4. The drawn geometry is passed to your function as `roi_geometry`
5. The temporary layer is removed after execution

### Output Metadata (`output_meta`)

Declaratively style output layers without manual metadata handling:

```python
@register_node(
    label="Threshold Otsu",
    category="Filters",
    outputs=["mask"],
    output_meta={
        "layer_type": "labels",    # Napari layer type
        "colormap": "cyan",        # Colormap to apply
        "opacity": 0.5,            # Layer opacity
        "name_suffix": "Mask",     # Append to layer name
    }
)
def threshold_otsu(image):
    return skimage.filters.threshold_otsu(image) > image
```

The engine automatically wraps the result with the specified metadata, ensuring consistent visualization.

### Input Validation (`validate_inputs`)

Validate inputs before execution with clear error messages:

```python
@register_node(
    label="Gaussian Blur",
    category="Filters",
    outputs=["image_out"],
    validate_inputs={
        "image": {
            "required": True,                           # Input must be connected
            "dtype": ["float32", "float64", "uint8"],   # Allowed data types
            "ndim": [2, 3],                             # Allowed dimensions
        }
    }
)
def gaussian_blur(image, sigma: float = 1.0):
    return skimage.filters.gaussian(image, sigma=sigma)
```

The engine checks these conditions **before** calling your function and raises descriptive `ValueError` messages if validation fails.

### Documentation (`doc`)

Add help text that appears in the properties panel:

```python
@register_node(
    label="Gaussian Blur",
    category="Filters",
    doc="Applies a Gaussian blur filter. Higher sigma = more blur."
)
def gaussian_blur(image, sigma: float = 1.0):
    return skimage.filters.gaussian(image, sigma=sigma)
```

Falls back to the function's docstring if not provided.

### Visual Icons (`icon`)

Add emoji or text icons to node titles for better visual organization:

```python
@register_node(
    label="Gaussian Blur",
    category="Filters",
    icon="🔬"
)
def gaussian_blur(image, sigma: float = 1.0):
    return skimage.filters.gaussian(image, sigma=sigma)
```

The icon appears in the node's title bar in the graph view.

### Complete Example

Here's a node using all advanced features:

```python
@register_node(
    label="Smart Edge Detector",
    category="Analysis",
    outputs=["edges"],
    interactive={
        "layer_type": "shapes",
        "tool": "rectangle",
        "prompt": "Draw a region to analyze",
        "arg_name": "roi",
    },
    output_meta={
        "colormap": "viridis",
        "opacity": 0.7,
        "name_suffix": "Edges",
    },
    validate_inputs={
        "image": {
            "required": True,
            "dtype": ["float32", "uint8"],
            "ndim": [2],
        }
    },
    doc="Detects edges in a user-defined ROI with automatic styling",
    icon="🌊"
)
def smart_edge_detector(image, roi=None, sigma: float = 1.0):
    # Apply ROI if provided
    if roi is not None:
        # Crop to ROI...
        pass
    return skimage.filters.sobel(image)
```

See `flow_nodes/examples_advanced.py` for more complete examples.

### 3. How it works

1. **Scan:** On startup (or import), the system scans functions marked with `@register_node`.
2. **Hash:** Before running, the engine calculates a **Signature** for every node (Hash of Inputs + Parameters).
3. **Cache:** If `Current_Signature == Last_Signature`, execution is skipped and cached results are used.
4. **Thread:** The execution runs in a background `QThread` to keep the UI responsive, sending signals back to update the graph and Napari layers.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.