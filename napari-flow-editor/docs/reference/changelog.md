# Changelog

## Goal

Track user-visible changes by release.

This project uses a simplified changelog format:

## Unreleased

### Added

- `flow_nodes/features.py`: 9 new nodes — Canny, Corner Harris, Corner Shi-Tomasi, LBP, Shape Index, HOG, Blob LoG, Blob DoG, Peak Local Max.
- `output_type="points"` + `dask_strategy="chunked_delayed"` in `dispatch()`: coordinate-output nodes process multi-TB Zarr spatially without materialising to RAM.
- `output_coord_cols` option in `dask_options` for stripping extra coordinate columns (e.g. sigma from `blob_log`).
- Points layer support in `napari_plugin_v2.py` (`viewer.add_points()`).
- Dispatcher migration — `filters.py` complete (Batches 1–5 done); `exposure.py` and `morphology.py` in progress.
- Standalone `"cuda"` path now reachable in auto backend selection for NumPy/CuPy inputs with no Dask strategy (e.g. global threshold nodes).
- `pyramid_strategy="per_level"` added to all global threshold nodes.
- New test files: `test_exposure_dispatch.py`, `test_morphology_dispatch.py`, `test_ridge_filter_dispatch.py`, `test_threshold_dispatch.py`.
- Expanded documentation structure (Getting Started, Concepts, Examples, API Reference, Troubleshooting, Developer, Reference).

### Changed

- Auto backend priority: **Dask+CUDA → Dask → CUDA → CPU** (standalone CUDA was previously unreachable in auto mode).
- `threshold_mean` no longer uses an inner wrapper; passes `skimage.filters.threshold_mean` directly to `dispatch()`.
- All 6 remaining global threshold nodes (`otsu`, `li`, `minimum`, `triangle`, `yen`, `isodata`) gained `cuda_function`, `cuda_arg_names`, `cuda_kwarg_names`.
- Documentation nav labels aligned to a clearer audience split (`Examples`, `API Reference`, `API Developer`).

### Fixed

- `decorator.py` `_choose_backend_mode`: standalone `"cuda"` backend was skipped in auto mode — only `"dask_cuda"` was reachable. Fixed by adding a CUDA-only case before the CPU fallback.
- Removed placeholder-only sections in developer/reference docs and replaced with actionable guidance.

### Removed

- N/A

## 0.0.1

### Added

- Initial plugin packaging and napari manifest entry point.

### Changed

- N/A

### Fixed

- N/A

### Removed

- N/A

## Entry Template (copy for future versions)

```md
## X.Y.Z - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Removed
- ...
```
