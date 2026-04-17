# Dispatch Benchmarks

This folder contains a reproducible benchmark harness for validating the
dispatcher behavior (CPU / Dask / CUDA / auto fallback) using a Gaussian
workflow representative of the `Gaussian Blur` node.

## What it produces

Running the benchmark writes 4 files in `benchmarks/results/`:

- `dispatch_benchmark_<timestamp>.json`: full structured report.
- `dispatch_benchmark_<timestamp>_raw.csv`: one row per measured run.
- `dispatch_benchmark_<timestamp>_summary.csv`: aggregated per case/backend.
- `dispatch_benchmark_<timestamp>.md`: readable summary table.

## Run

From the repository root (`napari-flow-editor`):

```bash
python benchmarks/run_dispatch_benchmarks.py --preset quick
```

Recommended deeper run:

```bash
python benchmarks/run_dispatch_benchmarks.py --preset standard --repeats 3 --warmup 1
```

Large stress-oriented run:

```bash
python benchmarks/run_dispatch_benchmarks.py --preset large --repeats 2 --warmup 1
```

Custom backends:

```bash
python benchmarks/run_dispatch_benchmarks.py --backends cpu,dask,auto
```

“Prove Dask is active on larger data” run:

```bash
python benchmarks/run_dispatch_benchmarks.py \
  --preset large \
  --backends dask,dask_cuda,auto \
  --repeats 2 \
  --warmup 1
```

Include your own external Zarr as an out-of-core case:

```bash
python benchmarks/run_dispatch_benchmarks.py \
  --preset standard \
  --backends dask,dask_cuda,auto \
  --zarr-path /path/to/dataset.ome.zarr \
  --zarr-key 0 \
  --repeats 2 \
  --warmup 1
```

If `--zarr-key auto` (default), the runner tries common array keys (`0`, `s0`, `data`, root).

## Notes

- `selected_backend` is parsed from dispatcher logs (so you can verify fallback behavior).
- Comparisons are reported as `MAE` and `Max abs` versus a per-case reference backend
  (CPU when available, otherwise Dask/Auto fallback).
- `rss_delta_mb` is best-effort process RSS delta and is platform dependent.
- `dask_lazy_*` cases are chunked lazy arrays to validate large-data Dask behavior.
