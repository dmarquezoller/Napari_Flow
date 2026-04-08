# Performance

## Goal

Document where runtime costs come from and how to optimize.

## Where Time Is Spent

Most slow runs come from one or more of these:

1. Heavy compute node operations (filters, segmentation, large transforms).
2. Materialization of lazy arrays (`compute()`), especially on large data.
3. Interactive waits (user input nodes, confirmation loops).
4. Repeated reruns caused by parameter churn.
5. UI redraw and large scene updates during graph edits.

## Typical Bottlenecks

### 1) Dask vs NumPy Execution Path

- Dask helps when data is large and chunked, but can be slower on tiny arrays due to scheduling overhead.
- NumPy/skimage may be faster for small arrays fully in memory.

What to do:

1. Benchmark on a representative subset and on a realistic full case.
2. Keep backend behavior explicit in node docs and test expectations.

### 2) Multiscale Processing

- Processing all pyramid levels is expected for full multiscale outputs.
- For scale-sensitive operators (for example Gaussian), use world-consistent parameter policy where needed.

What to do:

1. Validate per-level behavior with a controlled synthetic dataset.
2. Confirm that output looks right at multiple zoom levels, not just one.

### 3) Cache Invalidation

- If upstream changes do not invalidate dependent caches, results appear stale.
- If invalidation is too broad, everything reruns unnecessarily.

What to do:

1. Change one upstream parameter and verify only downstream branch recomputes.
2. Keep invalidation scoped to dependency graph.

### 4) Interactive and Loop Nodes

- Interactive nodes pause execution by design.
- Loop-until-confirm can dominate wall-time depending on user pace.

What to do:

1. For benchmarking, replace interactive nodes with deterministic alternatives.
2. Use fixed iteration loops for repeatable timing.

### 5) Rendering and Viewer State

- Layer creation and large viewer updates can be non-trivial in heavy scenes.

What to do:

1. Distinguish compute time from viewer render time by logging both.
2. Temporarily disable optional visualization steps during profiling.

## 5-Minute Debug Checklist

1. Re-run the same pipeline twice without edits.
2. If run 2 is not faster, inspect cache hit behavior.
3. Replace source data with a smaller ROI or sample image.
4. Remove one heavy node at a time to isolate the bottleneck.
5. Compare with and without interactive nodes.
6. Compare with and without multiscale input.
7. Record timings per node and keep the log.

## Practical Optimization Playbook

1. Start with correctness, then optimize.
2. Keep pipelines modular (use macros) so expensive branches can be isolated.
3. Avoid unnecessary viewer output nodes during performance tests.
4. Use batch mode only after single-case performance is stable.
5. Keep output writes (video/save nodes) at the end of the pipeline.

## Minimal Benchmark Template

Use this structure for reproducible performance notes:

1. Dataset: `<name, shape, dtype, multiscale yes/no>`
2. Pipeline: `<nodes in order>`
3. Environment: `<OS, CPU/GPU, RAM, napari version, plugin commit>`
4. Run 1 total: `<seconds>`
5. Run 2 total: `<seconds>`
6. Slowest node(s): `<name + time>`
7. Notes: `<interactive waits, I/O path, known caveats>`

## Placeholder Assets

- `<PERF_TABLE_PLACEHOLDER: add benchmark table screenshot>`
- `<PERF_TIMELINE_PLACEHOLDER: add per-node timing chart>`
