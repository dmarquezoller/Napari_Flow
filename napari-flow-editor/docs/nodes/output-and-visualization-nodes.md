# Output and Visualization Nodes

## Goal

Document save/export/plot/video nodes and output artifacts.

## Save / Sink Outputs

### Save Image

Purpose:
- save image outputs to disk with incremented naming (`_001`, `_002`, ...)

Typical inputs:
- processed image data

Key parameters:
- `folder`
- `base_name`
- `format`

### Save Table (CSV)

Purpose:
- persist measurement tables/dataframes as CSV

Typical inputs:
- table outputs from measurement nodes

## Plot Outputs

Plotting nodes produce figure outputs shown in the plot dashboard.

Current plotting nodes:
- `Plot Histogram`
- `Colocalization Scatter`
- `Table Heatmap`

Display behavior:
- static embedded plotting in dashboard
- optional interactive Plotly browser flow when available

## Video Output

### Make Video

Purpose:
- render viewer animation to `.mp4` or `.gif`

Type:
- interactive sink node (no outgoing data socket)

Key parameters:
- instructions table (`rotate`, `sweep`, ranges, axes)
- `fps`
- `format`
- `folder`
- `filename`

## Related Non-Node Outputs

These are app actions in the File menu (not graph nodes):
- `Save Pipeline` (JSON graph state)
- `Load Pipeline`
- `Export Script`
- `Save to Zarr`

## Practical Advice

1. Keep output nodes near the end of each branch for readability.
2. Use unique base names in save nodes during batch runs.
3. Add at least one visual output (plot or saved image) while validating new pipelines.
