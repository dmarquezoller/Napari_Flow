# Layers and Inputs

## Goal

Document practical input strategies: viewer layers, files, and zarr.

## Typical Input Paths

### 1) Viewer Layer Path (`Get Layer`)

Use when data is already loaded in napari.

Typical pipeline:
- `Get Layer -> Processing Node`

Notes:
- `layer_name` must match the current viewer layer name.
- output type can adapt dynamically to selected layer type.

### 2) File Path (`Open Image File`)

Use when you want to read single image files from disk (for example `.tif`, `.tiff`, `.png`, `.jpg`).

Typical pipeline:
- `Open Image File -> Select Layer -> Processing`

Notes:
- returns a list of layer tuples (same style as other reader nodes)
- `Select Layer` lets you choose the exact incoming layer

### 3) OME-Zarr Path (`Open Ome-Zarr`)

Use for multi-layer or multiscale zarr datasets.

Typical pipeline:
- `Open Ome-Zarr -> Select Layer -> Processing`

Notes:
- reader returns one or more layer entries
- `Select Layer` is usually required before image processing nodes

## Metadata Handling

Metadata is propagated through most processing steps when possible.

Common preserved fields include:
- layer name
- colormap / contrast / gamma
- transform metadata (`scale`, `translate`, etc.)
- axis-related metadata when available

Practical advice:
1. Prefer `Select Layer` before heavy processing in multi-layer inputs.
2. If output visualization looks unusual, check inherited display metadata.
3. For custom nodes, return `(data, meta)` or layer-style tuples to keep metadata explicit.
