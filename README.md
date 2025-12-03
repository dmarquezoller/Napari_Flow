# Napari Flow Editor

**Visual image processing pipelines for [napari](https://napari.org/).**

`napari-flow-editor` allows you to construct complex image analysis workflows using a node-based graph interface. Connect filters, transformations, and segmentation algorithms visually, adjust parameters in real-time, and execute pipelines directly on your active Napari layers.

## ✨ Features

* **Graph Interface:** Drag-and-drop nodes to create processing pipelines.
* **Live Parameter Tuning:** Adjust `sigma`, `radius`, and `thresholds` in the properties panel.
* **Smart Execution:** Topological sorting ensures operations run in the correct order.
* **Auto-Generated Library:** Nodes are generated automatically from Python functions using decorators.
* **Extensible:** Import your own `.py` scripts to add custom nodes instantly.
* **Save/Load:** Persist your complex workflows to JSON files.

---

## 🛠 Installation

### Prerequisites
* Python 3.9+
* Napari (`pip install napari[all]`)

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/napari-flow-editor.git](https://github.com/YOUR_USERNAME/napari-flow-editor.git)
cd napari-flow-editor
```
### 2. Install
We recommend installing in **editable mode** so changes to your node library are reflected immediately.

```bash
pip install -e
```