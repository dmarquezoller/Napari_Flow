# OME-Zarr Workflows

## Goal

Provide recommended workflows for large multiscale microscopy data.

## Basic Pipeline

Recommended baseline:

`Begin -> Open Ome-Zarr -> Select Layer -> Gaussian Blur -> Save Image`

Why this pattern:
1. `Open Ome-Zarr` can return multiple candidate layers.
2. `Select Layer` makes downstream input explicit.
3. Processing nodes receive a single selected image stream.

## Multiscale Notes

Key behavior to keep in mind:

- multiscale inputs are treated as pyramids of levels
- processing can run per-level to preserve interactive rendering behavior
- display metadata from source layers is propagated when possible
- axis metadata may be inferred or preserved depending on source info

Practical checks:
1. Confirm layer axes metadata after selection.
2. Confirm output looks consistent at different zoom levels.
3. If blur intensity looks unexpected, validate sigma behavior per level.

## Common Pitfalls

1. Applying processing directly to multi-layer output without `Select Layer`
   - Fix: insert `Select Layer`.

2. Unexpected visualization changes after processing
   - Fix: compare source/output display metadata (contrast, colormap, gamma).

3. Confusion between viewer display and actual pixel values
   - Fix: validate numerically with a small ROI when needed.

4. Very large datasets feel slow in full-run pipelines
   - Fix: test on a short branch first, then expand pipeline.
