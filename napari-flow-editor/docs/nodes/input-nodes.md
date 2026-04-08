# Input Nodes

## Goal

Document all nodes that introduce data into the pipeline.

## Input Node Families

### Get Layer

Purpose:
- Pull data from an already loaded napari viewer layer.

Typical use:
- Fast interactive workflows where layers are prepared manually in viewer.

Key parameters:
- `layer_name`
- `axis_map`

Notes:
- Works well with dynamic parameter binding in batch mode (`layer_name` from CSV).
- Best for viewer-centric workflows.

### Open Image File

Purpose:
- Read image files from disk (`.tif`, `.tiff`, and common image formats).

Typical use:
- File-driven pipelines, especially with `Begin Batch`.

Key parameters:
- `path`

Notes:
- Returns layer-style data bundle; often followed by `Select Layer`.

### Open Ome-Zarr

Purpose:
- Read OME-Zarr datasets, including multiscale/multi-layer content.

Typical use:
- Large microscopy datasets and pyramidal data.

Key parameters:
- `path` (directory)

Notes:
- Usually pair with `Select Layer` before processing.

### Select Layer

Purpose:
- Choose one layer from an incoming layer collection.

Typical use:
- Disambiguate multi-layer file inputs before filters/segmentation.

Key parameters:
- `layer_name`
- `layer_type`

Notes:
- Supports interactive layer choice flow at runtime.

### Load CSV

Purpose:
- Load tabular CSV data as pipeline input.

Typical use:
- Measurement filtering and table-based operations.

Key parameters:
- `file_path`

## Choosing The Right Input Node

1. Data already in napari:
   - use `Get Layer`
2. Single file input path:
   - use `Open Image File`
3. OME-Zarr dataset:
   - use `Open Ome-Zarr` + `Select Layer`
4. Table input:
   - use `Load CSV`

## Common Gotchas

- `Get Layer` fails if layer name does not match exactly.
- Running processing directly on unselected multi-layer input can lead to wrong target layer.
- CSV input is tabular data, not image data; ensure downstream node expects table-like input.
