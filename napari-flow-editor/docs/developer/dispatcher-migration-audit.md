# Dispatcher Migration Audit

This document tracks the migration of flow nodes to the dispatcher architecture.
Gaussian Blur in `flow_nodes/filters.py` is the reference implementation.

The goal is not to force every node into Dask or CUDA. The goal is for every
node to make its execution behavior explicit:

- dispatcher-safe with a declared strategy, or
- intentionally full-image-only with a clear reason.

## Reference Contract

Every migrated processing node should declare or document:

- Input and output layer types.
- Output dtype policy.
- Backend selection behavior.
- Dask strategy:
  - `pointwise`: no neighborhood context.
  - `neighborhood`: uses overlap/halo.
  - `chunked_delayed`: coordinate/points outputs.
  - `full-image-only`: no chunked execution unless algorithm-specific logic exists.
- Halo source, if any: parameter name, constant depth, or explicit map_overlap config.
- Boundary mode source, if any.
- Postprocess hook, if any: for example converting a maintained backend's
  threshold image into this app's boolean mask output.
- Axis policy:
  - `T` and `C` independent by default.
  - `Z`, `Y`, `X` spatial.
  - RGB/channel-last handled intentionally.
  - Nodes that need plane-wise processing can declare
    `spatial_core_axes="YX"` so `Z` is treated as independent while the
    default Gaussian-style behavior still processes `ZYX` as a true volume.
- CUDA behavior:
  - implemented,
  - not implemented,
  - or intentionally unsupported.
- Tests needed for `YX`, `TYX`, `ZYX`, and where relevant `CZYX`/`TCZYX`.

## Production Migration Rules

The dispatcher migration is now treated as production-facing work. The goal is
to make node declarations simple, reviewable, and backed by maintained
libraries wherever possible.

- Prefer maintained GPU implementations before writing any node-local CUDA
  code:
  - `cupyx.scipy.*` for SciPy/ndimage-style functions.
  - `cucim.skimage.*` for scikit-image-style functions.
- Avoid per-node CUDA helper functions unless there is no maintained library
  equivalent or the library function cannot preserve the node behavior.
- If a node-local CUDA helper is unavoidable, document why in this audit and
  add focused parity tests.
- Dotted `cuda_function="..."` paths are preferred because they keep node
  declarations close to the Gaussian Blur reference style and let dispatcher
  preflight handle missing optional dependencies.
- GPU-specific tests should be optional and skip cleanly on CPU-only
  environments. CPU/Dask correctness tests must not be optional.
- Nodes with no `dask_strategy` (e.g. global thresholds) reach CUDA via the
  standalone `"cuda"` path in auto mode (fixed 2026-05-21 in `decorator.py`).
  This path activates for NumPy/CuPy inputs when `cuda_ok=True`. Dask inputs
  for global ops still fall to CPU and compute lazily via NumPy dispatch.
- For publication/review readiness, each migrated node should have an explicit
  limitation note when CUDA, Dask, or chunked execution is intentionally not
  supported.

## Status Legend

| Status | Meaning |
|---|---|
| Reference | Gold-standard implementation. Keep as comparison target. |
| Migrated | Uses dispatcher, but may still need broader tests. |
| Partial | Some dispatcher behavior exists, but strategy or metadata needs review. |
| Direct | Calls NumPy/skimage directly today. Needs migration or explicit full-image note. |
| Full-image | Should intentionally compute the full input because chunking is unsafe. |
| Special | Non-image, interactive, I/O, plotting, or model-specific logic. |
| Uncertain | Needs API/algorithm review before migration. |

## Priority Legend

| Priority | Meaning |
|---|---|
| P0 | Must be settled before calling dispatcher migration complete. |
| P1 | Important for large-data correctness or common workflows. |
| P2 | Useful, but can come after core processing nodes. |
| P3 | Low priority, special-case, or documentation-only. |

## Filters

| Node | Current state | Target dispatcher behavior | Key risk / note | Tests needed | Priority |
|---|---|---|---|---|---|
| Gaussian Blur | Reference | `neighborhood`, halo from `sigma`, boundary from `mode`, pyramid-aware, CUDA path | Gold standard; do not regress | Existing + keep `TYX`, `ZYX`, Dask, CUDA, zarr lazy | P0 |
| Hysteresis Threshold | Direct | Full-image or custom connectivity-aware implementation | Hysteresis connectivity can cross chunk borders | CPU reference; Dask input should either materialize intentionally or fail clearly | P1 |
| Butterworth | Partial/full-image | Full-image only | FFT per chunk is wrong | Verify Dask input behavior; ensure no silent per-chunk FFT | P1 |
| Correlate Sparse | Direct | `neighborhood` if kernel is small and explicit; otherwise full-image | Kernel input is not currently modeled well | 2D kernel parity; `TYX` no time bleed | P2 |
| Difference of Gaussians | Migrated | `neighborhood`, halo from `high_sigma`, CUDA currently via node-local helper | Review for maintained `cupyx`/`cucim` equivalent before production sign-off | CPU vs Dask parity; CUDA smoke; `TYX`, `ZYX` | P0 |
| Farid | Migrated | `neighborhood`, fixed depth | Looks reasonable | CPU vs Dask parity; `TYX`, `ZYX` | P1 |
| Farid Horizontal | Migrated | `neighborhood`, fixed depth | Axis semantics should be checked for ND images | `YX`, `TYX`; expected horizontal axis | P1 |
| Farid Vertical | Migrated | `neighborhood`, fixed depth | Axis semantics should be checked for ND images | `YX`, `TYX`; expected vertical axis | P1 |
| Filter Forward | Direct | Uncertain | Need confirm skimage semantics and output type | API review first | P3 |
| Filter Inverse | Direct | Uncertain/full-image | Needs impulse/filter context review | API review first | P3 |
| Frangi | Full-image | Explicit full-image CPU via `dispatch(...)` | Default `gamma=None` depends on global Hessian norm, so generic chunking is unsafe | Dask input materializes intentionally; NumPy parity; invalid sigma validation | Done |
| Gabor | Direct | Special/multi-output, likely full-image or 2D neighborhood | Returns real/imag tuple; 2D-oriented | Decide output contract first | P2 |
| Median Filter | Migrated | `neighborhood`, fixed depth from `radius`, CUDA currently via node-local helper | Review for maintained `cupyx`/`cucim` equivalent before production sign-off | CPU vs Dask parity; `TYX`, `ZYX`; footprint dimensionality | P0 |
| Sobel Filter | Migrated | `neighborhood`, depth 1, CUDA currently via node-local helper | Review for maintained `cupyx`/`cucim` equivalent before production sign-off | CPU vs Dask parity; `TYX`, `ZYX` | P0 |
| Sobel Split | Direct | Special multi-output, likely `neighborhood` depth 1 | Multi-output image tuple needs dispatcher handling or two dispatch calls | Output tuple display; parity | P2 |
| Prewitt Filter | Migrated | `neighborhood`, depth 1, CUDA currently via node-local helper | Review for maintained `cupyx`/`cucim` equivalent before production sign-off | CPU vs Dask parity; `TYX`, `ZYX` | P0 |
| Unsharp Mask | Migrated | `neighborhood`, halo from `radius`, CUDA currently via node-local helper | Review for maintained `cupyx`/`cucim` equivalent before production sign-off | CPU vs Dask parity; CUDA smoke | P0 |
| Hessian | Migrated | `neighborhood`, spatial halo from max sigma, optional `cucim.skimage.filters.hessian` | Uses fixed gamma default, so chunk-local with sufficient halo is safe in tests | Synthetic ridge across chunk boundary; optional cuCIM | Done |
| Laplace | Migrated | `neighborhood`, halo from `ksize`, CUDA fallback | CUDA ignores `ksize`; note in docs/tests | CPU vs Dask; CUDA fallback behavior | P1 |
| Meijering | Full-image | Explicit full-image CPU via `dispatch(...)` | Algorithm normalizes each sigma response by global max; generic chunking changes values | Dask input materializes intentionally; NumPy parity; invalid sigma validation | Done |
| Rank Order | Direct | Full-image/special tuple output | Returns ordered image and original values | Decide output layer/table semantics | P2 |
| Roberts | Migrated | `neighborhood`, depth 1 | Looks reasonable | CPU vs Dask parity; `TYX` | P1 |
| Sato | Migrated | `neighborhood`, spatial halo from max sigma, optional `cucim.skimage.filters.sato` | Needs larger halo than Gaussian truncate alone; tests use ridge crossing chunks | Synthetic tube/ridge across chunk boundary; `TYX`; optional cuCIM | Done |
| Scharr | Migrated | `neighborhood`, depth 1 | Looks reasonable | CPU vs Dask parity; `TYX`, `ZYX` | P1 |
| Threshold Otsu | Migrated | Full-image global statistic plus lazy comparison when possible | Scalar threshold is computed per spatial image; Dask mask remains lazy after scalar compute; CUDA path is declared optional pending real cuCIM validation | NumPy/Dask parity; `TYX` independence; not-per-chunk regression; optional cuCIM/fallback tests | Done |
| Threshold Local | Migrated | `neighborhood`, halo from `block_size`, `postprocess=image > threshold`, `cucim.skimage.filters.threshold_local` CUDA path | `map_overlap` uses `boundary="none"` because skimage/cuCIM handle image borders internally | Added NumPy promotion, exact Dask parity, `TYX`, odd-size validation, optional cuCIM tests | P0 |
| Threshold Niblack | Migrated | `neighborhood`, halo from `window_size`, `postprocess=image > threshold`, `cucim.skimage.filters.threshold_niblack` CUDA path | Uses maintained cuCIM threshold image plus dispatcher postprocess to preserve mask output | Added NumPy promotion, exact Dask parity, odd-size validation, optional cuCIM tests | P0 |
| Threshold Sauvola | Migrated | `neighborhood`, halo from `window_size`, `postprocess=image > threshold`, `cucim.skimage.filters.threshold_sauvola` CUDA path | Uses maintained cuCIM threshold image plus dispatcher postprocess to preserve mask output | Added NumPy promotion, exact Dask parity, odd-size validation, optional cuCIM tests | P0 |
| Threshold Li | Migrated | Full-image global statistic plus lazy comparison when possible | Iterative scalar threshold; computes per spatial image; CUDA path is declared optional pending real cuCIM validation | NumPy/Dask parity; optional cuCIM/fallback tests | Done |
| Threshold Mean | Migrated | Global reduction plus lazy comparison when possible | Scalar threshold remains global; tiny CuPy mean helper exists for forced CUDA | NumPy/Dask parity; `TYX` independence; not-per-chunk regression | Done |
| Threshold Minimum | Migrated | Full-image global statistic plus lazy comparison when possible | Histogram/global; can fail on unsuitable histograms just like skimage; CUDA path is declared optional pending real cuCIM validation | NumPy/Dask parity; optional cuCIM/fallback tests | Done |
| Threshold Triangle | Migrated | Full-image global statistic plus lazy comparison when possible | Histogram/global; CUDA path is declared optional pending real cuCIM validation | NumPy/Dask parity; optional cuCIM/fallback tests | Done |
| Threshold Yen | Migrated | Full-image global statistic plus lazy comparison when possible | Histogram/global; CUDA path is declared optional pending real cuCIM validation | NumPy/Dask parity; optional cuCIM/fallback tests | Done |
| Threshold Isodata | Migrated | Full-image global statistic plus lazy comparison when possible | Histogram/global; CUDA path is declared optional pending real cuCIM validation | NumPy/Dask parity; optional cuCIM/fallback tests | Done |
| Wiener Deconvolution | Direct | Full-image only unless algorithm-specific tiling exists | Deconvolution needs global context/PSF | Explicit full-image behavior | P2 |

## Exposure

| Node | Current state | Target dispatcher behavior | Key risk / note | Tests needed | Priority |
|---|---|---|---|---|---|
| Convert to Grayscale | Custom Dask-aware | Keep custom or wrap as special pointwise RGB transform | Must not collapse non-RGB `C`/`T` axes accidentally | RGB/RGBA, `TYXC`, non-RGB unchanged | P0 |
| Adjust Gamma | Migrated | `pointwise`, image float policy, `cucim.skimage.exposure.adjust_gamma` CUDA path | Uses maintained cuCIM skimage mirror instead of node-local CUDA helper | Added NumPy promotion, Dask laziness/parity, `TYX`, optional cuCIM parity tests | P0 |
| Adjust Log | Migrated | `pointwise`, image float policy, `cucim.skimage.exposure.adjust_log` CUDA path | Uses maintained cuCIM skimage mirror instead of node-local CUDA helper | Added NumPy promotion, Dask laziness/parity, `CZYX`, optional cuCIM parity tests | P0 |
| Adjust Sigmoid | Migrated | `pointwise`, image float policy, `cucim.skimage.exposure.adjust_sigmoid` CUDA path | Maps node `cut_off` to skimage/cuCIM `cutoff` | Added NumPy promotion, Dask laziness/parity, optional cuCIM parity tests | P0 |
| Cumulative Distribution | Direct | Full-image/table-like output | skimage returns distribution data, not image-like output | Decide output type before migration | P2 |
| Equalize Adaptive | Direct | Full-image or algorithm-specific block/overlap | CLAHE tile behavior may not be safe with generic chunks | CPU reference; Dask policy decision | P1 |
| Equalize Histogram | Direct | Full-image global histogram | Global histogram must be over whole image | Dask input materialization/logging | P1 |
| Rescale Intensity | Migrated | `pointwise`, image float policy, `cucim.skimage.exposure.rescale_intensity` CUDA path | Explicit range UI maps to skimage/cuCIM `in_range` and `out_range`; tuple bug fixed | Added NumPy promotion, Dask laziness/parity, optional cuCIM parity tests | P0 |

## Features

| Node | Current state | Target dispatcher behavior | Key risk / note | Tests needed | Priority |
|---|---|---|---|---|---|
| Canny | Partial/full-image | Full-image only unless custom halo+global hysteresis exists | Hysteresis can cross chunk borders | Explicit materialization; 2D-only expectations | P1 |
| Corner Harris | Migrated | `neighborhood`, halo from `sigma`, CUDA currently via node-local helper | CUDA helper is 2D-specific; review for maintained `cucim.skimage.feature` equivalent | `YX`, `TYX`, 3D fallback | P1 |
| Corner Shi-Tomasi | Migrated | `neighborhood`, halo from `sigma`, CUDA currently via node-local helper | CUDA helper is 2D-specific; review for maintained `cucim.skimage.feature` equivalent | `YX`, `TYX`, 3D fallback | P1 |
| Local Binary Pattern | Migrated | `neighborhood`, halo from `R`, CPU/Dask only | 2D-specific; needs clear behavior for `TYX`/`ZYX` | `TYX` independent; reject/handle true 3D | P1 |
| Shape Index | Migrated | `neighborhood`, halo from `sigma`, CUDA currently via node-local helper | CUDA helper is 2D-specific; review for maintained `cucim.skimage.feature` equivalent | `YX`, `TYX`, 3D fallback | P1 |
| HOG | Partial/full-image | Full-image CPU | Cell/block grid spans whole image | Explicit materialization; 2D/RGB behavior | P2 |
| Blob LoG | Migrated points | `chunked_delayed`, points metadata, halo from `max_sigma` | Needs ongoing visual/parameter validation | `YX`, `TYX`, `ZYX`, point size metadata | P0 |
| Blob DoG | Migrated points | `chunked_delayed`, points metadata, halo from `max_sigma` | Recently fixed axis and point-display issues | Existing tests; manual napari visual check | P0 |
| Peak Local Max | Migrated points | `chunked_delayed`, halo from `min_distance` | Needs duplicate suppression behavior review | Chunk boundary duplicate test | P1 |

## Morphology

| Node | Current state | Target dispatcher behavior | Key risk / note | Tests needed | Priority |
|---|---|---|---|---|---|
| Dilation | Migrated | Explicit per-node `dispatch(...)`, `neighborhood`, halo `radius` | Axis-aware disk/ball footprint after T/C slicing; optional `cucim.skimage.morphology.dilation` | CPU vs Dask parity; `TYX`, `ZYX`; optional cuCIM | Done |
| Erosion | Migrated | Explicit per-node `dispatch(...)`, `neighborhood`, halo `radius` | Axis-aware disk/ball footprint after T/C slicing; optional `cucim.skimage.morphology.erosion` | CPU vs Dask parity; optional cuCIM | Done |
| Binary Closing | Migrated | Explicit per-node `dispatch(...)`, `neighborhood`, halo `2 * radius` | Uses `closing` instead of deprecated `binary_closing`; bool output | Axis-aware footprint tests; optional cuCIM | Done |
| Binary Opening | Migrated | Explicit per-node `dispatch(...)`, `neighborhood`, halo `2 * radius` | Uses `opening` instead of deprecated `binary_opening`; bool output | Axis-aware footprint tests; optional cuCIM | Done |
| White Tophat | Migrated | Explicit per-node `dispatch(...)`, `neighborhood`, halo `2 * radius` | Tophat depends on opening; preserve dtype | CPU vs Dask parity; optional cuCIM | Done |
| Black Tophat | Migrated | Explicit per-node `dispatch(...)`, `neighborhood`, halo `2 * radius` | Tophat depends on closing; preserve dtype | CPU vs Dask parity; optional cuCIM | Done |
| Skeletonize | Migrated | `pointwise` + `full_core` rechunk; T/C independent via dispatcher default; no CUDA (no cucim equivalent) | Full spatial ZYX core delivered per T/C frame; `image > 0` passed in args | NumPy parity; Dask laziness; TYX independence | Done |
| Remove Small Objects | Migrated | `pointwise` + `full_core` rechunk; T/C independent via dispatcher default; no CUDA | Full spatial ZYX core per frame; duplicate node definition removed | NumPy parity; Dask laziness; TYX independence | Done |
| Remove Small Holes | Migrated | `pointwise` + `full_core` rechunk; T/C independent via dispatcher default; no CUDA | Full spatial ZYX core per frame | NumPy parity; Dask laziness; TYX independence | Done |
| Convex Hull | Direct | Full-image only | Whole-image hull by definition | Explicit materialization | P2 |
| Label Objects | Migrated | `pointwise` + `full_core` rechunk; T/C independent via dispatcher default; CUDA via `cucim.skimage.measure.label` | Full spatial ZYX core per frame; labels are frame-local (restart per T/C frame) | NumPy parity; Dask laziness; TYX independence; int32 output; optional cuCIM | Done |

## Segmentation

| Node | Current state | Target dispatcher behavior | Key risk / note | Tests needed | Priority |
|---|---|---|---|---|---|
| Clear Border | Direct | Full-image connectivity | Border-connected objects can cross chunks | Explicit materialization | P1 |
| Expand Labels | Direct | Full-image unless custom distance transform/reconcile | Label competition crosses chunk borders | Explicit materialization | P1 |
| Find Boundaries | Direct | `neighborhood`, small halo | Likely chunk-safe with overlap | CPU vs Dask parity | P1 |
| Mark Boundaries | Direct | `neighborhood`/special two-input | Needs image+labels alignment | Two-input Dask parity | P1 |
| SLIC | Direct | Full-image only initially | Superpixel IDs and clustering are global-ish | Explicit full-image behavior | P1 |
| Felzenszwalb | Direct | Full-image only initially | Graph segmentation not chunk-safe naively | Explicit full-image behavior | P1 |
| Quickshift | Direct | Full-image only initially | Mode seeking not chunk-safe naively | Explicit full-image behavior | P2 |
| Watershed | Direct | Full-image connectivity unless custom tiled watershed exists | Basin connectivity and marker reconciliation | Explicit full-image behavior | P1 |
| Random Walker | Direct | Full-image only | Solver/global graph | Explicit full-image behavior | P2 |
| Chan Vese | Direct | Full-image only | Iterative global evolution | Explicit full-image behavior | P2 |

## Math And Transform

| Node | Current state | Target dispatcher behavior | Key risk / note | Tests needed | Priority |
|---|---|---|---|---|---|
| Blend Images | Direct arithmetic | `pointwise` or keep lazy arithmetic with metadata | Shape alignment and dtype policy | NumPy/Dask parity; two-input chunks | P1 |
| Interactive Crop | Direct and materializes via `_ensure_numpy` | Preserve lazy slicing when possible | Current eager conversion is bad for large zarr | Dask crop stays lazy; metadata/axes update | P0 |
| Interactive Crop Legacy | Direct and materializes | Deprecate or align with new crop | Duplicated behavior | Decide removal/deprecation | P2 |
| Project 3D to 2D | Direct NumPy reductions | Dask reductions; axis-aware labels | Current uses NumPy reductions, may compute eagerly | Dask lazy max/mean/sum/std | P0 |
| Rotate | Direct | Full-image or per-plane special | Coordinate transform across chunks is not generic-safe | Explicit full-image/per-slice decision | P2 |
| Rescale | Direct | Full-image or per-plane special | RGB axis heuristic only; chunked interpolation risky | RGB and `TYX` behavior | P2 |
| Downscale Local Mean | Direct | Special chunk-aware reduction | Requires chunk alignment to factor | Dask exactness with chunk boundaries | P1 |
| Swirl | Direct | Full-image/per-plane special | Global coordinate transform | Explicit full-image behavior | P3 |
| Warp Polar | Direct | Full-image/per-plane special | Global coordinate transform | Explicit full-image behavior | P3 |

## Measure, I/O, Visualization, Deep Learning

These nodes should not be blindly migrated to the image dispatcher. They need
explicit behavior, but many are not image-filter nodes.

| Node group | Current state | Target behavior | Key risk / note | Priority |
|---|---|---|---|---|
| Get Layer / Open Image / Open OME-Zarr | Special input nodes | Preserve lazy data, metadata, chunk info | Critical for large-data workflows | P0 |
| Save Image / Save Table | Special output nodes | Explicit compute/write behavior and progress logs | Avoid accidental huge eager compute without logs | P1 |
| Region Properties | Full-image/table | Keep explicit full-image or implement slice-by-slice lazy mode | Measurements depend on labels and tables | P1 |
| Filter Labels by CSV | Full-image/table | Full-image label rewrite unless blockwise relabel designed | Label IDs/frame mapping | P2 |
| Plot Histogram / Scatter / Heatmap | Visualization | Materialize intentionally or sample/downsample with warning | Plotting huge zarr can be dangerous | P2 |
| Video Node | Special visualization/export | Cache management and lazy frame reads | Already known memory pressure risk | P1 |
| Cellpose / StarDist / Ultrack | Model-specific | Separate tiling/model backend strategy, not generic dispatcher | GPU/model memory and dependency variability | P2 |

## First Migration Batches

### Batch 1: Easy pointwise exposure nodes

Nodes:

- `adjust_gamma` - migrated
- `adjust_log` - migrated
- `adjust_sigmoid` - migrated
- `rescale_intensity` - migrated

Why first:

- Low algorithmic risk.
- Good testbed for dtype policy and Dask laziness.
- Exposure module is currently mostly unmigrated.

Acceptance:

- NumPy output matches skimage reference.
- Dask output remains lazy for array-sized output.
- `TYX` and `CZYX` do not mix axes.
- `rescale_intensity` call bug is fixed.
- Optional cuCIM/CuPy parity tests skip cleanly when GPU dependencies are not
  installed.

### Batch 2: Local threshold nodes

Nodes:

- `threshold_local` - migrated
- `threshold_niblack` - migrated
- `threshold_sauvola` - migrated

Why second:

- Important segmentation preprocessing.
- Clear `neighborhood` strategy with halo from window/block size.

Acceptance:

- CPU vs Dask parity away from image borders.
- No time/channel bleed.
- Odd window/block size validation.
- Maintained cuCIM paths are used for CUDA; no node-local CUDA helper.

### Batch 3: Morphology neighborhood nodes

Nodes:

- `dilation` - migrated
- `erosion` - migrated
- `binary_opening` - migrated
- `binary_closing` - migrated
- `white_tophat` - migrated
- `black_tophat` - migrated

Why third:

- Common workflows.
- Axis-aware footprint selection will exercise dispatcher context.

Acceptance:

- `TYX` treats time independently.
- `ZYX` uses a true 3D footprint.
- Dask output matches CPU reference for synthetic masks.
- CUDA uses maintained `cucim.skimage.morphology` functions where available.
- Deprecated scikit-image `binary_opening` / `binary_closing` calls are avoided.
- Node implementations keep the Gaussian-style explicit `dispatch(...)` block
  inside each node.
- `pyramid_strategy="per_level"` was missing from all 6 neighborhood nodes on
  initial migration and was added 2026-05-25.

### Batch 4: Fix partial migrated filters

Nodes:

- `frangi` - full-image CPU
- `hessian` - migrated
- `meijering` - full-image CPU
- `sato` - migrated

Why fourth:

- They already use dispatcher, but current `pointwise` strategy is suspicious.
- Need chunk-boundary tests for multiscale filters.

Acceptance:

- `hessian` and `sato` have synthetic ridge/tube chunk-boundary parity tests.
- `hessian` and `sato` derive spatial-only halo from the explicit sigma list.
- `frangi` and `meijering` intentionally materialize Dask inputs because their
  default algorithms use global response maxima.
- Node descriptions no longer say `pointwise`.
- Each node keeps the Gaussian-style explicit `dispatch(...)` block.
- Maintained CUDA paths are used only where parameter-compatible; unsupported
  full-image/global cases are explicit and tested.

Implementation notes:

- The focused tests showed that `frangi` and `meijering` are not generic
  chunk-local. `frangi` defaults `gamma=None`, which uses a global Hessian norm;
  `meijering` normalizes each sigma response by a global max. They are therefore
  explicit full-image CPU nodes for now.
- `hessian` and `sato` are chunk-safe with sufficient halo. The implemented
  spatial halo is `ceil(max_sigma * 6)`, which passed synthetic ridge/tube
  boundary tests where `4 * sigma` was not enough.
- CUDA paths are declared for `hessian` and `sato` through maintained
  `cucim.skimage.filters` dotted paths and skip/fallback cleanly when unavailable.

### Batch 5: Global threshold nodes

Nodes:

- `threshold_otsu` - migrated
- `threshold_li` - migrated
- `threshold_mean` - migrated
- `threshold_minimum` - migrated
- `threshold_triangle` - migrated
- `threshold_yen` - migrated
- `threshold_isodata` - migrated

Why fifth:

- These are common segmentation preprocessors and were still direct calls.
- The intended behavior is scientifically simple but must be explicit:
  compute one scalar threshold on the full spatial image, then apply
  `image > threshold`.

Acceptance:

- Each node keeps an explicit Gaussian-style `dispatch(...)` block.
- Dask inputs compute the scalar threshold intentionally, then return a lazy
  boolean mask when possible.
- `T` and `C` axes remain independent through the dispatcher layout policy.
- NumPy/Dask results match direct scikit-image references.
- Declared CUDA paths are optional and guarded by dispatcher preflight/fallback;
  real cuCIM parity is validated only when CuPy/cuCIM are installed.

Implementation notes:

- These nodes are not chunk-local algorithms. Generic `map_blocks` would compute
  a different threshold per chunk and is therefore wrong.
- `threshold_mean` uses a Dask reduction for Dask inputs; histogram/iterative
  thresholds materialize the relevant spatial slice to compute the scalar.
- Tests include a not-per-chunk regression so future edits do not accidentally
  add `pointwise`/`neighborhood` Dask strategies to global thresholds.
- `threshold_mean` uses `cuda_func=_cuda_threshold_mean` (a module-level callable
  using `cp.mean`) because `cp.mean` has no cucim equivalent. All other 6
  histogram-based global threshold nodes use
  `cuda_function="cucim.skimage.filters.*"` (lazy dotted string) — no
  module-level imports needed, preflight handles missing cucim gracefully.

### Batch 6: Connectivity morphology and labels

Nodes:

- `remove_small_objects` - migrated
- `remove_small_holes` - migrated
- `label_objects` - migrated
- `skeletonize` - migrated

Plan:

- Prefer plane-wise large-data modes using `spatial_core_axes="YX"` where the
  user explicitly wants independent `YX` processing.
- Preserve `T`/`C` independence where sensible, but do not silently split
  connected spatial dimensions in volume mode.
- Do not implement naive chunked labels or object filtering; cross-chunk
  connected components need reconciliation.

Acceptance:

- All four nodes use `dask_options={"strategy": "pointwise", "numpy_chunks": "full_core", "rechunk": "full_core"}` so each T/C frame's full ZYX spatial core is delivered as one numpy array — no naive partial-chunk connectivity.
- No `process_as`, `spatial_core_axes`, or `layout_policy` — the dispatcher's default T/C independence handles axis splitting.
- `image > 0` is passed in `args` (not inside a closure) — lazy for Dask, immediate for NumPy, on-GPU for CuPy.
- NumPy parity against scikit-image references.
- Duplicate `remove_small_objects` definition cleaned up.
- `label_objects` adds `cuda_function="cucim.skimage.measure.label"` — labels are frame-local and restart per T/C frame (not globally unique across volume).

Implementation notes:

- The prior plan proposed `layout_policy="full_nd"` and `process_as` to support plane/volume modes. This was rejected: Gaussian Blur has neither and the dispatcher default already handles T/C independence correctly. The simpler clean pattern was used instead.
- `numpy_chunks="full_core"` + `rechunk="full_core"` are the genuine requirement — connectivity operations need the full ZYX core in one array. This differs from neighborhood filters which use `"spatial_auto"` and chunk with a halo.
- For 2 TB images, peak memory per Dask task ≈ 2× one T/C frame's ZYX block, which is manageable. Full-volume connectivity (3D labels spanning Z) would require a separate dedicated node.
- `skeletonize`, `remove_small_objects`, and `remove_small_holes` have no cucim equivalent — intentionally CPU/Dask only.

### Batch 7: Special filters and multi-output filters

Nodes:

- `hysteresis_threshold`
- `correlate_sparse`
- `gabor`
- `sobel_split`
- `rank_order`
- `wiener`
- `butterworth` audit/finalization

Plan:

- Classify each node as `neighborhood`, `full-image-only`, or `special`.
- For `sobel_split`, use two explicit dispatcher calls or a documented
  multi-output path; do not hide it in a category wrapper.
- For `gabor` and `rank_order`, decide the output contract before optimizing.

Acceptance:

- No direct skimage calls remain without an explicit documented reason.
- Tuple/multi-output behavior is covered by tests.
- Full-image nodes say why chunking is unsafe.

### Batch 8: Large-data transform/math core

Nodes:

- `interactive_crop`
- `project_3d_to_2d`
- `downscale_local_mean`
- `blend_images`
- `rotate_image`
- `rescale_image`

Plan:

- Prioritize lazy-preserving operations first: crop, projection reductions,
  blending, and aligned downscale.
- Treat coordinate transforms as full-image/per-plane unless a safe lazy
  strategy is proven.

Acceptance:

- Cropping a Dask/Zarr layer stays lazy.
- Projection reductions use Dask reductions where possible.
- Blend operations preserve chunks and shape/dtype rules.
- Resampling nodes have explicit full-image/per-plane policies.

## Test Matrix Template

Each migrated image node should get at least one category-level test. Individual
node tests can be generated from a parameter table where possible.

| Test | Purpose |
|---|---|
| NumPy reference parity | Compare node output to direct skimage/NumPy on a small array. |
| Dask laziness | Confirm output is Dask when input is Dask and strategy supports laziness. |
| Dask parity | Compute Dask output and compare to NumPy reference. |
| TYX independence | Detect time bleed or overlap across timepoints. |
| C independence | Detect channel bleed. |
| ZYX spatial halo | Confirm Z gets halo for spatial 3D operations. |
| RGB/RGBA handling | Confirm color channel is not mistaken for Z/C/time. |
| Full-image policy | Confirm full-image nodes materialize intentionally with clear logs or metadata. |
| Optional GPU parity | Compare maintained GPU backend to skimage when CuPy/cuCIM are installed; skip cleanly otherwise. |

## Immediate Next Step

Batches 1, 2, 3, 4, 5, and 6 are complete. Move to Batch 7:
special filters and multi-output filters.

Why this next:

- Several remaining filter nodes still call skimage directly or return
  multi-output/special data structures.
- Batch 7 should classify each remaining filter as `neighborhood`,
  `full-image-only`, or `special` before optimizing.
