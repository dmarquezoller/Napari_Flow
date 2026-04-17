# Napari Flow Editor

Visual, node-based bioimage analysis workflows for [napari](https://napari.org/).

Napari Flow Editor lets you build pipelines with explicit execution flow and data flow, run them interactively, and iterate quickly with caching and reusable graph blocks.

## Current Highlights

- Node graph editor integrated directly in napari (`Flow Editor` plugin widget).
- Separate **exec flow** (white logic sockets) and **data flow** (typed circular sockets).
- Control-flow nodes:
  - `Begin`
  - `Begin Batch` (CSV-driven multi-run)
  - `Loop` (`N times` and `Until confirm`)
- Interactive node support with in-app action bar (Run/Cancel).
- Smart cache with partial recomputation of affected downstream nodes.
- Multiscale and OME-Zarr workflows with metadata-aware propagation.
- Batch dynamic parameters (bind node params to CSV columns).
- Macro workflow grouping for complex pipelines.
- Plot outputs and video-generation workflows.
- Extensible node system via decorators + auto-generated node library.

## Repository Layout

The plugin package lives in:

- `napari-flow-editor/`
  - `src/napari_flow_editor/` (plugin code)
  - `docs/` (MkDocs documentation)

## Installation

### Prerequisites

- Python 3.9+
- napari environment

### 1) Clone

```bash
git clone https://github.com/dmarquezoller/Napari_Flow.git
cd Napari_Flow/napari-flow-editor
```

### 2) Install (editable)

```bash
pip install "napari[all]"
pip install -e .
```

### 3) Run

```bash
napari
```

Then open:

- `Plugins > Flow Editor`

## Docker (optional)

The repository includes Docker scaffolding in `napari-flow-editor/` for reproducible environments.

### Build image

```bash
cd Napari_Flow/napari-flow-editor
docker compose build
```

### Run tests in container

```bash
docker compose --profile tests run --rm tests
```

### Run MkDocs in container

```bash
docker compose --profile docs up docs
```

Then open `http://127.0.0.1:8000`.

### Run napari GUI in container (Linux/X11)

```bash
xhost +local:docker
docker compose --profile gui run --rm gui
xhost -local:docker
```

Notes:
- GUI forwarding depends on host display setup (X11/Wayland).
- For CI or headless usage, prefer `tests` and `docs` profiles.

## Quick Start

Build a first pipeline:

1. Add nodes:
   - `Begin` (Add Control)
   - `Get Layer` (Input)
   - `Gaussian Blur` (Filters)
   - `Save Image` (Outputs)
2. Connect **exec flow**:
   - `Begin -> Gaussian Blur -> Save Image`
3. Connect **data flow**:
   - `Get Layer:data_out -> Gaussian Blur:image`
   - `Gaussian Blur:image_out -> Save Image:image`
4. Set parameters (`layer_name`, `sigma`, output folder).
5. Run pipeline and inspect output layer/log.

## Developer Notes

### Add a custom node

```python
from napari_flow_editor.flow_nodes.decorator import register_node
import skimage.filters

@register_node(
    label="My Custom Blur",
    category="Custom",
    outputs=["image_out"],
    params_config={"sigma": {"min": 0.1, "max": 20.0, "step": 0.1}},
)
def my_blur(image, sigma: float = 1.0):
    return skimage.filters.gaussian(image, sigma=sigma)
```

Regenerate the node library:

```bash
python src/napari_flow_editor/generate_library.py
```

### Run tests

```bash
python -m pytest src/napari_flow_editor/tests -q
```

## Documentation

From `napari-flow-editor/`:

```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000`.

## Status

The software is actively developed and currently positioned as a research-grade workflow tool for iterative bioimage analysis and method development.

## License

MIT License. See `LICENSE`.
