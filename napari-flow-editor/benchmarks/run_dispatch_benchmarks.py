#!/usr/bin/env python3
"""
Dispatch benchmark harness for backend validation (CPU / Dask / CUDA / auto).

This script benchmarks a representative Gaussian workflow using the same
dispatch configuration used by the Gaussian Blur node, then writes:
  - raw CSV (all runs),
  - summary CSV (aggregated),
  - JSON (full structured report),
  - Markdown (human-readable report).

Usage examples:
  python benchmarks/run_dispatch_benchmarks.py --preset quick
  python benchmarks/run_dispatch_benchmarks.py --preset standard --repeats 3
  python benchmarks/run_dispatch_benchmarks.py --backends cpu,dask,auto
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "napari_flow_editor_mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

import dask.array as da
import numpy as np
import skimage.filters

from napari_flow_editor.flow_nodes.decorator import (
    dispatch,
    pop_dispatch_context,
    push_dispatch_context,
)
from napari_flow_editor.flow_nodes.filters import gaussian_blur


_SELECTED_BACKEND_RE = re.compile(r"selected=([a-z_]+)")
_MERMAID_MAX_LAYERS = 24
_MERMAID_MAX_EDGES = 48


@dataclass(frozen=True)
class CaseSpec:
    name: str
    kind: str  # "numpy" | "multiscale" | "dask_lazy" | "zarr_lazy_external"
    shape: tuple[int, ...]
    axes: str | None = None
    levels: int = 1
    description: str = ""
    chunks: str | tuple[int, ...] | None = None
    allowed_backends: tuple[str, ...] | None = None
    zarr_path: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dispatch backend benchmarks.")
    parser.add_argument(
        "--preset",
        choices=("notebook", "quick", "standard", "large", "xlarge"),
        default="quick",
        help="Dataset preset size.",
    )
    parser.add_argument(
        "--backends",
        default="cpu,dask,cuda,dask_cuda,auto",
        help="Comma-separated backend requests to test.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Measured runs per (case, backend).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per (case, backend), excluded from report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for synthetic datasets.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma.",
    )
    parser.add_argument(
        "--mode",
        default="nearest",
        choices=("nearest", "reflect", "wrap", "constant"),
        help="Boundary mode for Gaussian.",
    )
    parser.add_argument(
        "--chunks",
        default="spatial_auto",
        help=(
            "Chunk spec for NumPy→Dask promotion. Examples: spatial_auto, auto, "
            "or 1,256,256"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "benchmarks" / "results"),
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--zarr-path",
        default="",
        help=(
            "Optional external Zarr array or group path to include as an out-of-core case. "
            "For OME-Zarr groups, use --zarr-key (default: auto -> tries common keys)."
        ),
    )
    parser.add_argument(
        "--zarr-key",
        default="auto",
        help=(
            "Array key inside --zarr-path when it is a group (examples: 0, s0, data). "
            "Use empty string to read root as array."
        ),
    )
    parser.add_argument(
        "--case-filter",
        default="",
        help=(
            "Optional regular expression used to keep only matching case names. "
            "Useful for running one large case without allocating every preset case."
        ),
    )
    parser.add_argument(
        "--app-interaction",
        action="store_true",
        help=(
            "Benchmark app-like lazy access instead of full output materialization: "
            "build the layer, request a visible slice, request ROIs, and scrub time."
        ),
    )
    parser.add_argument(
        "--interaction-rois",
        default="512,1024",
        help="Comma-separated spatial ROI sizes to compute in --app-interaction mode.",
    )
    parser.add_argument(
        "--interaction-scrub-count",
        type=int,
        default=4,
        help="Number of timepoints to request when scrubbing in --app-interaction mode.",
    )
    parser.add_argument(
        "--save-graphs",
        action="store_true",
        help=(
            "Render Dask array task-graph diagrams for first successful runs "
            "(one file per case/backend with Dask output)."
        ),
    )
    parser.add_argument(
        "--graph-format",
        choices=("svg", "png"),
        default="svg",
        help="Image format used when --save-graphs is enabled.",
    )
    return parser.parse_args()


def _current_rss_mb() -> float | None:
    # Linux fast path without extra deps.
    statm_path = "/proc/self/statm"
    if os.path.exists(statm_path):
        try:
            with open(statm_path, "r", encoding="utf-8") as fh:
                fields = fh.read().strip().split()
            if len(fields) >= 2:
                rss_pages = int(fields[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return float(rss_pages * page_size) / (1024.0 ** 2)
        except Exception:
            pass

    # Generic UNIX fallback.
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux usually KB, macOS usually bytes.
        if value > 10_000_000:
            return value / (1024.0 ** 2)
        return value / 1024.0
    except Exception:
        return None


def _parse_chunks(raw: str) -> str | tuple[int, ...]:
    value = str(raw).strip()
    normalized = value.lower().replace("-", "_")
    if normalized in {"auto", "spatial_auto", "spatial"}:
        return "spatial_auto" if normalized in {"spatial_auto", "spatial"} else "auto"
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return "auto"
    return tuple(int(p) for p in parts)


def _parse_roi_sizes(raw: str) -> tuple[int, ...]:
    sizes: list[int] = []
    for part in str(raw or "").split(","):
        value = part.strip()
        if not value:
            continue
        size = int(value)
        if size <= 0:
            raise ValueError(f"ROI sizes must be positive; got {size}.")
        sizes.append(size)
    return tuple(sizes or [512, 1024])


def _make_case_specs(preset: str) -> list[CaseSpec]:
    if preset == "notebook":
        return [
            CaseSpec(
                name="notebook_overlap_tyx_4x50x50",
                kind="notebook_overlap",
                shape=(4, 50, 50),
                axes="TYX",
                chunks=(1, 25, 25),
                description=(
                    "Notebook-style tiny TYX case executed through the real "
                    "gaussian_blur node, with Dask chunks (1,25,25)."
                ),
                allowed_backends=("cpu", "dask"),
            ),
        ]

    if preset == "xlarge":
        return [
            CaseSpec(
                name="numpy_2d_8192",
                kind="numpy",
                shape=(8192, 8192),
                axes="YX",
                description="XL 2D image (float32).",
            ),
            CaseSpec(
                name="numpy_3d_tyx_40x2048x2048",
                kind="numpy",
                shape=(40, 2048, 2048),
                axes="TYX",
                description="XL 3D timeline-like volume (T,Y,X).",
            ),
            CaseSpec(
                name="multiscale_tyx_l5_24x3072x3072",
                kind="multiscale",
                shape=(24, 3072, 3072),
                axes="TYX",
                levels=5,
                description="XL 5-level multiscale stack.",
            ),
            CaseSpec(
                name="dask_lazy_tyx_64x4096x4096",
                kind="dask_lazy",
                shape=(64, 4096, 4096),
                axes="TYX",
                chunks=(1, 1024, 1024),
                description="XL lazy Dask array (chunked).",
                allowed_backends=("dask", "dask_cuda", "auto"),
            ),
        ]

    if preset == "large":
        return [
            CaseSpec(
                name="numpy_2d_4096",
                kind="numpy",
                shape=(4096, 4096),
                axes="YX",
                description="Large 2D image (float32).",
            ),
            CaseSpec(
                name="numpy_3d_tyx_24x1536x1536",
                kind="numpy",
                shape=(24, 1536, 1536),
                axes="TYX",
                description="Large 3D timeline-like volume (T,Y,X).",
            ),
            CaseSpec(
                name="multiscale_tyx_l5_16x2048x2048",
                kind="multiscale",
                shape=(16, 2048, 2048),
                axes="TYX",
                levels=5,
                description="5-level large multiscale stack.",
            ),
            CaseSpec(
                name="dask_lazy_tyx_32x2048x2048",
                kind="dask_lazy",
                shape=(32, 2048, 2048),
                axes="TYX",
                chunks=(1, 512, 512),
                description="Large lazy Dask array (chunked).",
                allowed_backends=("dask", "dask_cuda", "auto"),
            ),
        ]

    if preset == "standard":
        return [
            CaseSpec(
                name="numpy_2d_2048",
                kind="numpy",
                shape=(2048, 2048),
                axes="YX",
                description="2D single image (float32).",
            ),
            CaseSpec(
                name="numpy_3d_tyx_16x1024x1024",
                kind="numpy",
                shape=(16, 1024, 1024),
                axes="TYX",
                description="3D timeline-like volume (T,Y,X).",
            ),
            CaseSpec(
                name="multiscale_tyx_l4_8x1024x1024",
                kind="multiscale",
                shape=(8, 1024, 1024),
                axes="TYX",
                levels=4,
                description="4-level multiscale stack.",
            ),
            CaseSpec(
                name="dask_lazy_tyx_16x1024x1024",
                kind="dask_lazy",
                shape=(16, 1024, 1024),
                axes="TYX",
                chunks=(1, 256, 256),
                description="Lazy Dask array (chunked).",
                allowed_backends=("dask", "dask_cuda", "auto"),
            ),
        ]

    # quick
    return [
        CaseSpec(
            name="numpy_2d_1024",
            kind="numpy",
            shape=(1024, 1024),
            axes="YX",
            description="2D single image (float32).",
        ),
        CaseSpec(
            name="numpy_3d_tyx_8x512x512",
            kind="numpy",
            shape=(8, 512, 512),
            axes="TYX",
            description="3D timeline-like volume (T,Y,X).",
        ),
        CaseSpec(
            name="multiscale_tyx_l4_8x512x512",
            kind="multiscale",
            shape=(8, 512, 512),
            axes="TYX",
            levels=4,
            description="4-level multiscale stack.",
        ),
        CaseSpec(
            name="dask_lazy_tyx_8x512x512",
            kind="dask_lazy",
            shape=(8, 512, 512),
            axes="TYX",
            chunks=(1, 256, 256),
            description="Lazy Dask array (chunked).",
            allowed_backends=("dask", "dask_cuda", "auto"),
        ),
    ]


def _make_base_array(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.normal(loc=127.0, scale=35.0, size=shape).astype(np.float32)
    return np.clip(arr, 0.0, 255.0, out=arr)


def _resolve_external_zarr_array(path_raw: str, key_raw: str) -> tuple[da.Array, str]:
    path = Path(path_raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Zarr path not found: {path}")

    key = str(key_raw or "").strip()
    if key.lower() == "auto":
        key = "__auto__"

    candidates = []
    if key == "__auto__":
        candidates.extend(
            [
                str(path / "0"),
                str(path / "s0"),
                str(path / "data"),
                str(path),
            ]
        )
    elif key == "":
        candidates.append(str(path))
    else:
        candidates.append(str(path / key))

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    last_error = None
    for candidate in unique_candidates:
        try:
            arr = da.from_zarr(candidate)
            return arr, candidate
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to read a zarr array from '{path_raw}' with key '{key_raw}'. "
        f"Last error: {last_error}"
    )


def _downsample_last_two(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 2:
        return arr
    slicer = (slice(None),) * (arr.ndim - 2) + (slice(None, None, 2), slice(None, None, 2))
    return arr[slicer]


def _build_case_payload(case: CaseSpec, seed: int) -> Any:
    if case.kind == "notebook_overlap":
        rng = np.random.default_rng(seed)
        return rng.random(case.shape, dtype=np.float32)

    base = _make_base_array(case.shape, seed=seed)
    if case.kind == "numpy":
        return base
    if case.kind == "multiscale":
        levels = [base]
        for _ in range(max(1, int(case.levels)) - 1):
            levels.append(_downsample_last_two(levels[-1]).astype(np.float32, copy=False))
        return levels
    if case.kind == "dask_lazy":
        rs = da.random.RandomState(seed)
        chunks = case.chunks if case.chunks is not None else "auto"
        arr = rs.normal(
            loc=127.0,
            scale=35.0,
            size=case.shape,
            chunks=chunks,
        ).astype(np.float32)
        return da.clip(arr, 0.0, 255.0)
    if case.kind == "zarr_lazy_external":
        if not case.zarr_path:
            raise ValueError("External zarr case is missing zarr_path.")
        return da.from_zarr(case.zarr_path)
    raise ValueError(f"Unknown case kind: {case.kind!r}")


def _estimate_nbytes(payload: Any) -> int:
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, da.Array):
        return int(payload.nbytes)
    if isinstance(payload, (list, tuple)):
        return int(sum(_estimate_nbytes(x) for x in payload))
    return 0


def _estimate_elements(payload: Any) -> int:
    if isinstance(payload, (np.ndarray, da.Array)):
        try:
            return int(np.prod(payload.shape, dtype=np.int64))
        except Exception:
            return 0
    if isinstance(payload, (list, tuple)):
        return int(sum(_estimate_elements(x) for x in payload))
    return 0


def _safe_div(numerator: float, denominator: float) -> float:
    den = float(denominator)
    if den == 0.0:
        return 0.0
    return float(numerator) / den


def _backend_request_honored(requested_backend: str, selected_backend: str) -> bool:
    req = str(requested_backend or "").strip().lower()
    sel = str(selected_backend or "").strip().lower()
    if req == "auto":
        return True
    return req == sel


def _dispatch_reason(dispatch_log: str) -> str:
    text = str(dispatch_log or "").strip()
    if not text:
        return ""
    if ". " in text:
        return text.split(". ", 1)[1].strip()
    return text


def _extract_selected_backend(dispatch_log: str) -> str:
    match = _SELECTED_BACKEND_RE.search(dispatch_log or "")
    if not match:
        return "unknown"
    return str(match.group(1))


def _dispatch_gaussian(
    payload: Any,
    *,
    sigma: float,
    mode: str,
    backend: str,
    chunks: str | tuple[int, ...],
) -> Any:
    return dispatch(
        default=skimage.filters.gaussian,
        args=(payload,),
        kwargs={"sigma": sigma, "mode": mode, "preserve_range": True},
        cuda_function="cupyx.scipy.ndimage.gaussian_filter",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["sigma", "mode"],
        cuda_output_dtype=np.float64,
        backend=backend,
        gpu_min_nbytes=0,
        dask_strategy="neighborhood",
        dask_halo_from_param="sigma",
        dask_boundary_from_param="mode",
        independent_axes_param="sigma",
        dask_output_dtype=np.float64,
        pyramid_strategy="per_level",
        pyramid_param_policy={"sigma": "fixed_world"},
        numpy_to_dask_chunks=chunks,
    )


def _dispatch_notebook_overlap(
    payload: Any,
    *,
    sigma: float,
    mode: str,
    backend: str,
    chunks: str | tuple[int, ...],
) -> Any:
    """
    Replicate the tiny notebook scenario, but through the real Gaussian node.

    We steer backend selection explicitly:
    - CPU run: force the dispatcher CPU path as the correctness reference.
    - Dask run: pass Dask input with notebook chunks through the real node path.
    """

    requested = str(backend or "auto").strip().lower()
    if requested not in {"cpu", "dask", "auto"}:
        raise ValueError(
            "Notebook overlap preset supports only cpu/dask/auto backends in node mode; "
            f"got {backend!r}."
        )

    if requested == "cpu":
        return dispatch(
            default=skimage.filters.gaussian,
            args=(payload,),
            kwargs={"sigma": sigma, "mode": mode, "preserve_range": True},
            backend="cpu",
            dask_options={
                "strategy": "neighborhood",
                "halo_from_param": "sigma",
                "boundary_from_param": "mode",
                "independent_axes_param": "sigma",
                "output_dtype": np.float64,
                "numpy_chunks": chunks,
            },
        )

    node_input = payload
    if requested == "dask":
        node_input = (
            payload
            if isinstance(payload, da.Array)
            else da.from_array(payload, chunks=chunks)
        )

    out = gaussian_blur(node_input, sigma=float(sigma), mode=mode)
    input_kind = "dask" if isinstance(node_input, da.Array) else "numpy"
    print(
        f"[dispatch:notebook_overlap] requested={requested}. "
        f"running real gaussian_blur node path via {input_kind} input"
    )
    return out


def _materialize_output(value: Any) -> Any:
    if isinstance(value, da.Array):
        return value.compute()
    if isinstance(value, list):
        return [_materialize_output(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_materialize_output(v) for v in value)
    if isinstance(value, dict):
        return {k: _materialize_output(v) for k, v in value.items()}
    return value


def _first_dask_array(value: Any) -> da.Array | None:
    if isinstance(value, da.Array):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_dask_array(item)
            if found is not None:
                return found
        return None
    if isinstance(value, dict):
        for item in value.values():
            found = _first_dask_array(item)
            if found is not None:
                return found
        return None
    return None


def _dask_output_info(value: Any) -> dict[str, Any]:
    arr = _first_dask_array(value)
    if arr is None:
        return {
            "dask_output_chunks": "",
            "dask_output_chunksize": "",
            "dask_output_numblocks": "",
            "dask_output_npartitions": None,
            "dask_output_tasks": None,
        }
    tasks = None
    try:
        tasks = int(len(arr.__dask_graph__()))
    except Exception:
        tasks = None
    return {
        "dask_output_chunks": str(getattr(arr, "chunks", "")),
        "dask_output_chunksize": str(getattr(arr, "chunksize", "")),
        "dask_output_numblocks": str(getattr(arr, "numblocks", "")),
        "dask_output_npartitions": int(getattr(arr, "npartitions", 0) or 0),
        "dask_output_tasks": tasks,
    }


def _first_array_like(value: Any) -> Any | None:
    if isinstance(value, (np.ndarray, da.Array)):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_array_like(item)
            if found is not None:
                return found
        return None
    if isinstance(value, dict):
        for item in value.values():
            found = _first_array_like(item)
            if found is not None:
                return found
        return None
    return None


def _last_array_like(value: Any) -> Any | None:
    if isinstance(value, (np.ndarray, da.Array)):
        return value
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            found = _last_array_like(item)
            if found is not None:
                return found
        return None
    if isinstance(value, dict):
        for item in reversed(list(value.values())):
            found = _last_array_like(item)
            if found is not None:
                return found
        return None
    return None


def _axes_for_ndim(axes: str | None, ndim: int) -> str | None:
    if not isinstance(axes, str):
        return None
    axes_norm = axes.strip().upper()
    if len(axes_norm) == ndim:
        return axes_norm
    if len(axes_norm) == ndim - 1 and axes_norm.endswith("YX") and "C" not in axes_norm:
        return f"{axes_norm}C"
    return None


def _display_view_for_array(
    arr: Any,
    axes: str | None,
    *,
    t_index: int = 0,
    roi_size: int | None = None,
) -> Any:
    ndim = int(getattr(arr, "ndim", 0) or 0)
    if ndim <= 0:
        return arr

    axes_norm = _axes_for_ndim(axes, ndim)
    if ndim <= 2:
        view = arr
    else:
        slicer: list[Any] = []
        for axis_index in range(ndim):
            axis_label = axes_norm[axis_index] if axes_norm else None
            axis_len = int(getattr(arr, "shape", ())[axis_index])
            if axis_label in {"Y", "X"}:
                slicer.append(slice(None))
            elif axes_norm is None and axis_index >= ndim - 2:
                slicer.append(slice(None))
            elif axis_label == "T":
                slicer.append(min(max(0, int(t_index)), max(0, axis_len - 1)))
            else:
                slicer.append(0)
        view = arr[tuple(slicer)]

    if roi_size is not None and int(getattr(view, "ndim", 0) or 0) >= 2:
        roi = int(roi_size)
        y_len = int(getattr(view, "shape", ())[-2])
        x_len = int(getattr(view, "shape", ())[-1])
        spatial_slicer = (
            (slice(None),) * (int(getattr(view, "ndim", 0) or 0) - 2)
            + (slice(0, min(roi, y_len)), slice(0, min(roi, x_len)))
        )
        view = view[spatial_slicer]
    return view


def _display_view(value: Any, axes: str | None, *, t_index: int = 0, roi_size: int | None = None) -> Any:
    arr = _first_array_like(value)
    if arr is None:
        return value
    return _display_view_for_array(arr, axes, t_index=t_index, roi_size=roi_size)


def _lowres_display_view(value: Any, axes: str | None) -> Any | None:
    if not isinstance(value, (list, tuple)) or len(value) <= 1:
        return None
    arr = _last_array_like(value)
    if arr is None:
        return None
    return _display_view_for_array(arr, axes, t_index=0, roi_size=None)


def _time_axis_length(value: Any, axes: str | None) -> int:
    arr = _first_array_like(value)
    if arr is None:
        return 0
    axes_norm = _axes_for_ndim(axes, int(getattr(arr, "ndim", 0) or 0))
    if not axes_norm or "T" not in axes_norm:
        return 0
    axis = axes_norm.index("T")
    try:
        return int(arr.shape[axis])
    except Exception:
        return 0


def _time_materialize_view(value: Any) -> float:
    t0 = time.perf_counter()
    _materialize_output(value)
    return float(time.perf_counter() - t0)


def _dot_quote(value: str) -> str:
    return json.dumps(str(value))


def _task_label(key: Any) -> str:
    if isinstance(key, tuple) and key:
        name = str(key[0])
        short_name = name.rsplit("-", 1)[0] if "-" in name else name
        suffix = ",".join(str(part) for part in key[1:])
        label = f"{short_name}\\n{suffix}" if suffix else short_name
    else:
        label = str(key)
    if len(label) > 90:
        return label[:87] + "..."
    return label


def _task_fillcolor(key: Any) -> str:
    text = str(key)
    if "overlap_apply" in text or "gaussian" in text:
        return "#dff3df"
    if "overlap" in text:
        return "#ffe8c2"
    if "trim" in text:
        return "#eadfff"
    if "array" in text:
        return "#dbeafe"
    return "#f8fafc"


def _render_dask_graph_with_dot(
    arr: da.Array,
    *,
    path: Path,
    total_tasks: int | None,
) -> tuple[str, Path | None, str, str]:
    dot_exe = shutil.which("dot")
    if not dot_exe:
        return "system dot executable unavailable", None, "mermaid_fallback", ""

    try:
        from dask.core import get_deps

        dsk: Any = arr.__dask_graph__()
        try:
            dsk = arr.__dask_optimize__(dsk, arr.__dask_keys__())
        except Exception:
            pass
        dsk_dict = dsk.to_dict() if hasattr(dsk, "to_dict") else dict(dsk)
        deps, _ = get_deps(dsk_dict)
        keys = sorted(dsk_dict.keys(), key=repr)
        node_ids = {key: f"n{i}" for i, key in enumerate(keys)}

        lines = [
            "digraph dask {",
            "  rankdir=LR;",
            '  graph [bgcolor="white"];',
            '  node [shape=box, style="rounded,filled", color="#334155", fontname="Helvetica", fontsize=10];',
            '  edge [color="#64748b", arrowsize=0.7];',
        ]
        for key in keys:
            attrs = {
                "label": _task_label(key),
                "fillcolor": _task_fillcolor(key),
            }
            attr_text = ", ".join(
                f"{name}={_dot_quote(value)}" for name, value in attrs.items()
            )
            lines.append(f"  {node_ids[key]} [{attr_text}];")

        for key in keys:
            for dep in sorted(deps.get(key, ()), key=repr):
                if dep in node_ids:
                    lines.append(f"  {node_ids[dep]} -> {node_ids[key]};")
        lines.append("}")

        fmt = path.suffix.lstrip(".") or "svg"
        with tempfile.TemporaryDirectory() as tmp_dir:
            dot_path = Path(tmp_dir) / "graph.dot"
            dot_path.write_text("\n".join(lines), encoding="utf-8")
            subprocess.run(
                [dot_exe, f"-T{fmt}", "-o", str(path), str(dot_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        if not path.exists():
            return "system dot did not create the graph file", None, "mermaid_fallback", ""
        message = "rendered optimized array task graph via system dot"
        if total_tasks is not None:
            message = f"{message}; task graph has {total_tasks} tasks"
        return "", path, "array_task_graph", message
    except Exception as exc:
        compact = " ".join(str(exc).split())
        if len(compact) > 240:
            compact = compact[:237] + "..."
        return (
            f"{type(exc).__name__}: {compact}",
            None,
            "mermaid_fallback",
            "",
        )


def _render_dask_graph(value: Any, *, path: Path) -> tuple[str, Path | None, str, str]:
    arr = _first_dask_array(value)
    if arr is None:
        return (
            "no dask output graph available",
            None,
            "none",
            "no dask output graph available",
        )
    total_tasks = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = path.suffix.lstrip(".") or None
        # graphviz appends ".<format>" when `format` is supplied, so pass the
        # filename stem to avoid creating links like `graph.svg` -> `graph.svg.svg`.
        render_target = path.with_suffix("") if fmt else path
        try:
            total_tasks = int(len(arr.__dask_graph__()))
        except Exception:
            total_tasks = None
        rendered = arr.visualize(
            filename=str(render_target),
            format=fmt,
            optimize_graph=True,
            collapse_outputs=True,
            rankdir="LR",
        )
        graph_kind = "array_task_graph"
        message = "rendered optimized array task graph"
        if total_tasks is not None:
            message = f"{message}; task graph has {total_tasks} tasks"
        rendered_path = (
            Path(rendered) if isinstance(rendered, (str, os.PathLike)) else path
        )
        if not rendered_path.exists() and path.exists():
            rendered_path = path
        return "", rendered_path, graph_kind, message
    except Exception as exc:
        lowered = str(exc).lower()
        if "graphviz" in lowered:
            dot_error, dot_path, dot_kind, dot_message = _render_dask_graph_with_dot(
                arr,
                path=path,
                total_tasks=total_tasks,
            )
            if not dot_error:
                return "", dot_path, dot_kind, dot_message
            compact = " ".join(str(exc).split())
            if len(compact) > 160:
                compact = compact[:157] + "..."
            message = (
                f"Dask visualizer unavailable ({type(exc).__name__}: {compact}); "
                f"system dot fallback failed ({dot_error})"
            )
            return (
                message,
                None,
                dot_kind,
                message,
            )
        compact = " ".join(str(exc).split())
        if len(compact) > 240:
            compact = compact[:237] + "..."
        message = f"{type(exc).__name__}: {compact}"
        return message, None, "mermaid_fallback", message


def _graph_stats(value: Any) -> dict[str, Any]:
    arr = _first_dask_array(value)
    if arr is None:
        return {
            "graph_layers": None,
            "graph_dependency_edges": None,
            "graph_total_tasks": None,
        }
    hlg = getattr(arr, "dask", None)
    layers = getattr(hlg, "layers", None)
    deps = getattr(hlg, "dependencies", None)
    total_tasks = None
    try:
        total_tasks = int(len(arr.__dask_graph__()))
    except Exception:
        total_tasks = None
    if not isinstance(layers, dict) or not isinstance(deps, dict):
        return {
            "graph_layers": None,
            "graph_dependency_edges": None,
            "graph_total_tasks": total_tasks,
        }
    dep_edges = 0
    for dep_set in deps.values():
        try:
            dep_edges += len(dep_set)
        except Exception:
            continue
    return {
        "graph_layers": int(len(layers)),
        "graph_dependency_edges": int(dep_edges),
        "graph_total_tasks": total_tasks,
    }


def _build_mermaid_hlg(value: Any) -> str:
    arr = _first_dask_array(value)
    if arr is None:
        return ""
    hlg = getattr(arr, "dask", None)
    layers = getattr(hlg, "layers", None)
    deps = getattr(hlg, "dependencies", None)
    if not isinstance(layers, dict) or not isinstance(deps, dict):
        return ""

    def _node_id(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", str(name))

    layer_items: list[tuple[str, int]] = []
    for layer_name, layer_graph in layers.items():
        lname = str(layer_name)
        try:
            n_tasks = len(layer_graph)
        except Exception:
            n_tasks = 0
        layer_items.append((lname, int(n_tasks)))
    if not layer_items:
        return ""

    layer_items.sort(key=lambda item: (-item[1], item[0]))
    selected_layers = {
        layer_name for layer_name, _ in layer_items[:_MERMAID_MAX_LAYERS]
    }
    omitted_layers = max(0, len(layer_items) - len(selected_layers))

    lines = ["graph LR"]
    for layer_name, n_tasks in layer_items:
        if layer_name not in selected_layers:
            continue
        nid = _node_id(layer_name)
        label = f"{layer_name} ({n_tasks} tasks)"
        safe_label = label.replace('"', "'")
        lines.append(f'    {nid}["{safe_label}"]')

    kept_edges = 0
    omitted_edges = 0
    for layer_name, depends_on in deps.items():
        if str(layer_name) not in selected_layers:
            omitted_edges += len(depends_on)
            continue
        dst = _node_id(layer_name)
        for src_name in sorted(depends_on):
            if str(src_name) not in selected_layers:
                omitted_edges += 1
                continue
            if kept_edges >= _MERMAID_MAX_EDGES:
                omitted_edges += 1
                continue
            src = _node_id(src_name)
            lines.append(f"    {src} --> {dst}")
            kept_edges += 1

    if omitted_layers:
        lines.append(f"    %% omitted {omitted_layers} layers for readability")
    if omitted_edges:
        lines.append(f"    %% omitted {omitted_edges} dependency edges for readability")

    return "\n".join(lines)


def _output_shape_signature(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"array:{tuple(int(x) for x in value.shape)}:{value.dtype}"
    if isinstance(value, da.Array):
        shape = tuple(int(x) for x in value.shape)
        return f"dask:{shape}:{value.dtype}:chunksize={getattr(value, 'chunksize', '')}"
    if isinstance(value, list):
        inner = ", ".join(_output_shape_signature(v) for v in value)
        return f"list[{inner}]"
    if isinstance(value, tuple):
        inner = ", ".join(_output_shape_signature(v) for v in value)
        return f"tuple({inner})"
    return str(type(value).__name__)


def _compare_outputs(reference: Any, candidate: Any) -> dict[str, Any]:
    if isinstance(reference, np.ndarray) and isinstance(candidate, np.ndarray):
        if reference.shape != candidate.shape:
            return {"compatible": False, "mae": None, "max_abs": None, "details": "shape mismatch"}
        ref64 = reference.astype(np.float64, copy=False)
        can64 = candidate.astype(np.float64, copy=False)
        diff = np.abs(can64 - ref64)
        return {
            "compatible": True,
            "mae": float(np.mean(diff)),
            "max_abs": float(np.max(diff)),
            "details": "",
        }

    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            return {"compatible": False, "mae": None, "max_abs": None, "details": "level count mismatch"}
        per_level = []
        for i, (r, c) in enumerate(zip(reference, candidate)):
            cmp_res = _compare_outputs(r, c)
            cmp_res["level"] = i
            per_level.append(cmp_res)
        valid = [x for x in per_level if x.get("compatible")]
        if not valid:
            return {
                "compatible": False,
                "mae": None,
                "max_abs": None,
                "details": "no compatible levels",
                "per_level": per_level,
            }
        return {
            "compatible": all(x.get("compatible", False) for x in per_level),
            "mae": float(np.mean([x["mae"] for x in valid])),
            "max_abs": float(np.max([x["max_abs"] for x in valid])),
            "details": "",
            "per_level": per_level,
        }

    return {
        "compatible": False,
        "mae": None,
        "max_abs": None,
        "details": f"type mismatch ({type(reference).__name__} vs {type(candidate).__name__})",
    }


def _infer_layout_kind(axes: str | None) -> str | None:
    if not isinstance(axes, str):
        return None
    mapping = {
        "YX": "2d_image",
        "YXC": "2d_image_channels",
        "ZYX": "3d_image",
        "ZYXC": "3d_image_channels",
        "TYX": "3d_timeline",
        "TYXC": "3d_timeline_channels",
        "TZYX": "4d_timeline",
        "TZYXC": "4d_timeline_channels",
    }
    return mapping.get(axes.upper())


def _run_single(
    payload: Any,
    *,
    case_kind: str,
    sigma: float,
    mode: str,
    backend: str,
    chunks: str | tuple[int, ...],
    axes: str | None,
) -> tuple[dict[str, Any], Any, Any]:
    rss_before = _current_rss_mb()
    dispatch_log = ""
    output = None

    t0 = time.perf_counter()
    buf = io.StringIO()
    dispatch_ctx = {}
    if isinstance(axes, str) and axes.strip():
        axes_norm = axes.strip().upper()
        dispatch_ctx["metadata"] = {
            "axes": axes_norm,
            "axis_labels": tuple(a.lower() for a in axes_norm),
            "layout_kind": _infer_layout_kind(axes_norm),
        }

    dispatch_token = None
    if dispatch_ctx:
        dispatch_token = push_dispatch_context(dispatch_ctx)
    try:
        with redirect_stdout(buf):
            if case_kind == "notebook_overlap":
                output = _dispatch_notebook_overlap(
                    payload,
                    sigma=sigma,
                    mode=mode,
                    backend=backend,
                    chunks=chunks,
                )
            else:
                output = _dispatch_gaussian(
                    payload,
                    sigma=sigma,
                    mode=mode,
                    backend=backend,
                    chunks=chunks,
                )
    finally:
        if dispatch_ctx:
            pop_dispatch_context(dispatch_token)
    t1 = time.perf_counter()
    dask_info = _dask_output_info(output)

    materialized = _materialize_output(output)
    t2 = time.perf_counter()
    rss_after = _current_rss_mb()
    dispatch_log = buf.getvalue()
    selected = _extract_selected_backend(dispatch_log)

    record = {
        "requested_backend": backend,
        "selected_backend": selected,
        "dispatch_time_s": float(t1 - t0),
        "compute_time_s": float(t2 - t1),
        "total_time_s": float(t2 - t0),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": (
            None
            if (rss_before is None or rss_after is None)
            else float(rss_after - rss_before)
        ),
        "output_signature": _output_shape_signature(materialized),
        "dispatch_log": dispatch_log.strip(),
        **dask_info,
    }
    return record, materialized, output


def _run_app_interaction_single(
    payload: Any,
    *,
    case_kind: str,
    sigma: float,
    mode: str,
    backend: str,
    chunks: str | tuple[int, ...],
    axes: str | None,
    roi_sizes: tuple[int, ...],
    scrub_count: int,
) -> tuple[dict[str, Any], Any]:
    rss_before = _current_rss_mb()
    dispatch_log = ""
    output = None

    t0 = time.perf_counter()
    buf = io.StringIO()
    dispatch_ctx = {}
    if isinstance(axes, str) and axes.strip():
        axes_norm = axes.strip().upper()
        dispatch_ctx["metadata"] = {
            "axes": axes_norm,
            "axis_labels": tuple(a.lower() for a in axes_norm),
            "layout_kind": _infer_layout_kind(axes_norm),
        }

    dispatch_token = None
    if dispatch_ctx:
        dispatch_token = push_dispatch_context(dispatch_ctx)
    try:
        with redirect_stdout(buf):
            if case_kind == "notebook_overlap":
                output = _dispatch_notebook_overlap(
                    payload,
                    sigma=sigma,
                    mode=mode,
                    backend=backend,
                    chunks=chunks,
                )
            else:
                output = _dispatch_gaussian(
                    payload,
                    sigma=sigma,
                    mode=mode,
                    backend=backend,
                    chunks=chunks,
                )
    finally:
        if dispatch_ctx:
            pop_dispatch_context(dispatch_token)
    t1 = time.perf_counter()

    dask_info = _dask_output_info(output)
    dispatch_log = buf.getvalue()
    selected = _extract_selected_backend(dispatch_log)
    errors: list[str] = []

    def _measure(key: str, view: Any) -> None:
        try:
            record[key] = _time_materialize_view(view)
        except Exception as exc:
            record[key] = None
            errors.append(f"{key}: {type(exc).__name__}: {exc}")

    record: dict[str, Any] = {
        "requested_backend": backend,
        "selected_backend": selected,
        "build_time_s": float(t1 - t0),
        "dispatch_time_s": float(t1 - t0),
        "compute_time_s": 0.0,
        "total_time_s": float(t1 - t0),
        "rss_before_mb": rss_before,
        "output_signature": _output_shape_signature(output),
        "dispatch_log": dispatch_log.strip(),
        **dask_info,
    }

    _measure("first_slice_time_s", _display_view(output, axes, t_index=0))
    for roi_size in roi_sizes:
        _measure(
            f"roi_{int(roi_size)}_time_s",
            _display_view(output, axes, t_index=0, roi_size=int(roi_size)),
        )

    requested_scrub = max(0, int(scrub_count))
    available_t = _time_axis_length(output, axes)
    actual_scrub = min(requested_scrub, available_t)
    scrub_times: list[float] = []
    for t_index in range(actual_scrub):
        key = f"scrub_t{t_index}_time_s"
        _measure(key, _display_view(output, axes, t_index=t_index))
        if record.get(key) is not None:
            scrub_times.append(float(record[key]))
    if scrub_times:
        record["scrub_count"] = len(scrub_times)
        record["scrub_mean_time_s"] = float(statistics.mean(scrub_times))
        record["scrub_total_time_s"] = float(sum(scrub_times))
        record["scrub_times_s"] = json.dumps(scrub_times)
    else:
        record["scrub_count"] = 0
        record["scrub_mean_time_s"] = None
        record["scrub_total_time_s"] = 0.0
        record["scrub_times_s"] = ""

    lowres_view = _lowres_display_view(output, axes)
    if lowres_view is not None:
        _measure("lowres_slice_time_s", lowres_view)
    else:
        record["lowres_slice_time_s"] = None

    rss_after = _current_rss_mb()
    materialization_keys = [
        "first_slice_time_s",
        "scrub_total_time_s",
        "lowres_slice_time_s",
        *(f"roi_{int(size)}_time_s" for size in roi_sizes),
    ]
    materialization_total = 0.0
    for key in materialization_keys:
        value = record.get(key)
        if value is not None:
            materialization_total += float(value)

    record.update(
        {
            "app_materialization_time_s": materialization_total,
            "app_total_time_s": float(record["build_time_s"] + materialization_total),
            "total_time_s": float(record["build_time_s"] + materialization_total),
            "rss_after_mb": rss_after,
            "rss_delta_mb": (
                None
                if (rss_before is None or rss_after is None)
                else float(rss_after - rss_before)
            ),
            "status": "ok" if not errors else "partial_error",
            "error": "; ".join(errors),
        }
    )
    return record, output


def _aggregate_values(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _aggregate_interaction_records(
    records: list[dict[str, Any]],
    *,
    requested_backend: str,
) -> dict[str, Any]:
    if not records:
        return {
            "requested_backend": requested_backend,
            "runs_ok": 0,
            "runs_total": 0,
            "status": "error",
            "error": "no records",
        }

    ok_records = [
        r for r in records if str(r.get("status", "")) in {"ok", "partial_error"}
    ]
    base = dict(records[0])
    base.pop("repeat_index", None)
    selected_counts = Counter(r.get("selected_backend", "") for r in ok_records)
    selected_mode = (
        "n/a"
        if not selected_counts
        else ", ".join(f"{k}:{v}" for k, v in sorted(selected_counts.items()))
    )

    numeric_keys = sorted(
        {
            key
            for row in ok_records
            for key, value in row.items()
            if (
                key.endswith("_time_s")
                or key in {"build_time_s", "app_total_time_s", "total_time_s"}
            )
            and isinstance(value, (int, float))
        }
    )
    for key in numeric_keys:
        vals = [
            float(row[key])
            for row in ok_records
            if isinstance(row.get(key), (int, float))
        ]
        mean, std = _aggregate_values(vals)
        base[key] = mean
        base[f"{key}_std"] = std

    rss_vals = [
        float(r["rss_delta_mb"])
        for r in ok_records
        if isinstance(r.get("rss_delta_mb"), (int, float))
    ]
    rss_mean, rss_std = _aggregate_values(rss_vals) if rss_vals else (0.0, 0.0)
    honored_runs = sum(
        1
        for rec in ok_records
        if _backend_request_honored(
            requested_backend, str(rec.get("selected_backend", ""))
        )
    )
    fallback_reasons = set()
    for rec in ok_records:
        if _backend_request_honored(
            requested_backend, str(rec.get("selected_backend", ""))
        ):
            continue
        reason = _dispatch_reason(rec.get("dispatch_log", ""))
        if reason:
            fallback_reasons.add(reason)
    errors = [str(r.get("error", "")) for r in records if str(r.get("error", ""))]
    dominant_backend = (
        "n/a" if not selected_counts else selected_counts.most_common(1)[0][0]
    )
    base.update(
        {
            "requested_backend": requested_backend,
            "runs_ok": len(ok_records),
            "runs_total": len(records),
            "selected_backend_mode": selected_mode,
            "selected_backend_dominant": dominant_backend,
            "backend_honored_runs": honored_runs,
            "backend_honored_ratio": _safe_div(honored_runs, len(ok_records)),
            "fallback_detected": bool(ok_records) and honored_runs < len(ok_records),
            "fallback_reasons": "; ".join(sorted(fallback_reasons)),
            "rss_delta_mean_mb": rss_mean,
            "rss_delta_std_mb": rss_std,
            "status": "ok" if ok_records else "error",
            "error": "; ".join(errors),
        }
    )
    return base


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_optional_time(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "n/a"


def _fmt_optional_int(value: Any) -> str:
    if value in (None, "", 0, "0"):
        return "n/a"
    try:
        return str(int(float(value)))
    except Exception:
        return "n/a"


def _write_markdown(
    path: Path,
    *,
    metadata: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    interaction_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    output_files: dict[str, str],
) -> None:
    lines = []
    lines.append("# Dispatch Benchmark Report")
    lines.append("")
    lines.append(f"- Timestamp: `{metadata['timestamp']}`")
    lines.append(f"- Preset: `{metadata['preset']}`")
    lines.append(f"- Backends requested: `{', '.join(metadata['backends'])}`")
    lines.append(f"- Repeats: `{metadata['repeats']}`")
    lines.append(f"- Warmup: `{metadata['warmup']}`")
    lines.append(f"- Sigma: `{metadata['sigma']}`")
    lines.append(f"- Mode: `{metadata['mode']}`")
    lines.append(f"- Chunks: `{metadata['chunks']}`")
    lines.append(
        f"- Benchmark mode: `{'app-interaction' if metadata.get('app_interaction') else 'full-compute'}`"
    )
    if metadata.get("app_interaction"):
        lines.append(f"- Interaction ROIs: `{metadata.get('interaction_rois')}`")
        lines.append(
            f"- Interaction scrub count: `{metadata.get('interaction_scrub_count')}`"
        )
    lines.append("")
    if summary_rows:
        lines.append("## Summary")
        lines.append("")
        lines.append(
            "`Build mean` is eager execution time for CPU rows. For Dask rows it is graph construction, NumPy-to-Dask promotion, and scheduling setup before `.compute()`. `Compute mean` is the materialization time; this is where Dask parallel execution shows up."
        )
        lines.append("")
        lines.append(
            "| Case | Requested | Selected (mode) | Honored | Ref | Build mean (s) | Compute mean (s) | Total mean (s) | Speedup vs Ref | Dask chunk shape | Dask partitions | Dask numblocks | Dask tasks | Build share (%) | Throughput (GiB/s) | Throughput (MPix/s) | MAE vs Ref | Compatible |"
        )
        lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|")
        for row in summary_rows:
            honored_pct = (
                "n/a"
                if int(row.get("runs_ok", 0)) < 1
                else f"{100.0 * float(row.get('backend_honored_ratio', 0.0)):.1f}"
            )
            speedup = (
                "n/a"
                if row.get("speedup_vs_reference") is None
                else f"{float(row['speedup_vs_reference']):.3f}"
            )
            dispatch_share = (
                "n/a"
                if int(row.get("runs_ok", 0)) < 1
                else f"{float(row.get('dispatch_share_pct', 0.0)):.1f}"
            )
            thr_gib_s = (
                "n/a"
                if int(row.get("runs_ok", 0)) < 1
                else f"{float(row.get('throughput_gib_s', 0.0)):.3f}"
            )
            thr_mpix_s = (
                "n/a"
                if int(row.get("runs_ok", 0)) < 1
                else f"{float(row.get('throughput_mpix_s', 0.0)):.3f}"
            )
            dask_partitions = row.get("dask_output_npartitions")
            dask_partitions = (
                "n/a"
                if dask_partitions in (None, "", 0, "0")
                else str(int(float(dask_partitions)))
            )
            dask_tasks = row.get("dask_output_tasks")
            dask_tasks = (
                "n/a"
                if dask_tasks in (None, "", 0, "0")
                else str(int(float(dask_tasks)))
            )
            dask_numblocks = str(row.get("dask_output_numblocks", "") or "").strip()
            if not dask_numblocks:
                dask_numblocks = "n/a"
            dask_chunksize = str(row.get("dask_output_chunksize", "") or "").strip()
            if not dask_chunksize:
                dask_chunksize = "n/a"
            lines.append(
                "| {case} | {requested} | {selected_mode} | {honored} | {ref} | {dispatch_mean:.4f} | {compute_mean:.4f} | {total_mean:.4f} | {speedup} | {dask_chunksize} | {dask_partitions} | {dask_numblocks} | {dask_tasks} | {dispatch_share} | {thr_gib_s} | {thr_mpix_s} | {mae} | {compatible} |".format(
                    case=row.get("case", ""),
                    requested=row.get("requested_backend", ""),
                    selected_mode=row.get("selected_backend_mode", ""),
                    honored=honored_pct,
                    ref=row.get("reference_backend", "n/a"),
                    dispatch_mean=float(row.get("dispatch_time_mean_s", 0.0)),
                    compute_mean=float(row.get("compute_time_mean_s", 0.0)),
                    total_mean=float(row.get("total_time_mean_s", 0.0)),
                    speedup=speedup,
                    dask_chunksize=dask_chunksize,
                    dask_partitions=dask_partitions,
                    dask_numblocks=dask_numblocks,
                    dask_tasks=dask_tasks,
                    dispatch_share=dispatch_share,
                    thr_gib_s=thr_gib_s,
                    thr_mpix_s=thr_mpix_s,
                    mae=(
                        "n/a"
                        if row.get("mae_vs_reference") is None
                        else f"{float(row['mae_vs_reference']):.6g}"
                    ),
                    compatible=row.get("compatible_vs_reference", "n/a"),
                )
            )
    if interaction_rows:
        roi_sizes = tuple(metadata.get("interaction_rois") or ())
        lines.append("")
        lines.append("## App Interaction")
        lines.append("")
        lines.append(
            "CPU rows are eager: `Build` includes the full CPU filter, and later slice/ROI timings are memory access. Dask rows keep the filtered result lazy, so slice/ROI/scrub timings materialize only the requested view."
        )
        lines.append("")
        headers = [
            "Case",
            "Requested",
            "Selected (mode)",
            "Honored",
            "Build (s)",
            "First slice (s)",
            *[f"ROI {int(size)} (s)" for size in roi_sizes],
            "Scrub mean (s)",
            "Scrub n",
            "Lowres slice (s)",
            "App total (s)",
            "Dask chunk shape",
            "Dask partitions",
            "Dask numblocks",
            "Dask tasks",
            "Status",
        ]
        aligns = [
            "---",
            "---",
            "---",
            "---:",
            "---:",
            "---:",
            *["---:" for _ in roi_sizes],
            "---:",
            "---:",
            "---:",
            "---:",
            "---",
            "---:",
            "---",
            "---:",
            "---",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(aligns) + "|")
        for row in interaction_rows:
            honored_pct = (
                "n/a"
                if int(row.get("runs_ok", 0)) < 1
                else f"{100.0 * float(row.get('backend_honored_ratio', 0.0)):.1f}"
            )
            dask_chunksize = str(row.get("dask_output_chunksize", "") or "").strip()
            if not dask_chunksize:
                dask_chunksize = "n/a"
            dask_numblocks = str(row.get("dask_output_numblocks", "") or "").strip()
            if not dask_numblocks:
                dask_numblocks = "n/a"
            values = [
                str(row.get("case", "")),
                str(row.get("requested_backend", "")),
                str(row.get("selected_backend_mode", row.get("selected_backend", ""))),
                honored_pct,
                _fmt_optional_time(row.get("build_time_s")),
                _fmt_optional_time(row.get("first_slice_time_s")),
                *[
                    _fmt_optional_time(row.get(f"roi_{int(size)}_time_s"))
                    for size in roi_sizes
                ],
                _fmt_optional_time(row.get("scrub_mean_time_s")),
                _fmt_optional_int(row.get("scrub_count")),
                _fmt_optional_time(row.get("lowres_slice_time_s")),
                _fmt_optional_time(row.get("app_total_time_s")),
                dask_chunksize,
                _fmt_optional_int(row.get("dask_output_npartitions")),
                dask_numblocks,
                _fmt_optional_int(row.get("dask_output_tasks")),
                str(row.get("status", "")),
            ]
            lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Backend Selection Notes")
    lines.append("")
    notes = []
    for row in [*summary_rows, *interaction_rows]:
        if not row.get("fallback_detected"):
            continue
        reasons = str(row.get("fallback_reasons", "")).strip() or "n/a"
        notes.append(
            f"- case=`{row.get('case', '')}` requested=`{row.get('requested_backend', '')}` selected=`{row.get('selected_backend_mode', '')}` reasons=`{reasons}`"
        )
    if notes:
        lines.extend(notes)
    else:
        lines.append("- No backend fallbacks detected in measured runs.")

    if graph_rows:
        lines.append("")
        lines.append("## Graph Visuals")
        lines.append("")
        lines.append(
            "These SVGs render the optimized Dask array task graph, so chunk-level parallel branches stay visible. Use `Dask numblocks` and `Dask partitions` as the authoritative chunking summary."
        )
        lines.append("")
        seen_graphs: dict[str, str] = {}
        graph_counter = 0
        for row in graph_rows:
            case = row.get("case", "")
            requested = row.get("requested_backend", "")
            selected = row.get("selected_backend", "")
            status = row.get("status", "")
            graph_kind = str(row.get("graph_kind", "array_task_graph") or "")
            path_str = row.get("graph_file", "")
            msg = row.get("message", "")
            layers = row.get("graph_layers")
            dep_edges = row.get("graph_dependency_edges")
            total_tasks = row.get("graph_total_tasks")
            mermaid = str(row.get("mermaid_hlg", "") or "").strip()
            fingerprint_src = mermaid or f"{case}|{requested}|{selected}|{path_str}|{status}|{msg}"
            fingerprint = hashlib.sha1(
                fingerprint_src.encode("utf-8", errors="replace")
            ).hexdigest()[:10]
            existing_id = seen_graphs.get(fingerprint)
            if existing_id is None:
                graph_counter += 1
                graph_id = f"g{graph_counter}"
                seen_graphs[fingerprint] = graph_id
            else:
                graph_id = existing_id
            metrics = []
            if layers is not None:
                metrics.append(f"layers={layers}")
            if dep_edges is not None:
                metrics.append(f"dep_edges={dep_edges}")
            if total_tasks is not None:
                metrics.append(f"tasks={total_tasks}")
            metrics_str = ", ".join(metrics) if metrics else "n/a"
            lines.append(
                f"- graph=`{graph_id}` kind=`{graph_kind}` case=`{case}` requested=`{requested}` selected=`{selected}` status=`{status}` metrics=`{metrics_str}` message=`{msg}`"
            )
            if existing_id is not None:
                lines.append(f"  - same as graph `{existing_id}`")
                continue
            if status == "ok" and path_str:
                lines.append(f"![]({path_str})")
            if mermaid:
                if status != "ok" or not path_str:
                    lines.append("```mermaid")
                    lines.append(mermaid)
                    lines.append("```")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for key, value in output_files.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    args = _parse_args()
    chunks = _parse_chunks(args.chunks)
    interaction_roi_sizes = _parse_roi_sizes(args.interaction_rois)
    requested_backends = [x.strip() for x in str(args.backends).split(",") if x.strip()]
    if not requested_backends:
        raise ValueError("No backends were requested.")

    # Keep CPU first for stable reference comparison.
    backends = []
    if "cpu" in requested_backends:
        backends.append("cpu")
    for b in requested_backends:
        if b != "cpu":
            backends.append(b)

    cases = _make_case_specs(args.preset)
    zarr_path_input = str(args.zarr_path or "").strip()
    zarr_key_input = str(args.zarr_key or "").strip()
    if zarr_path_input:
        zarr_arr, zarr_component = _resolve_external_zarr_array(
            zarr_path_input, zarr_key_input
        )
        zarr_shape = tuple(int(x) for x in zarr_arr.shape)
        chunk_sig = getattr(zarr_arr, "chunksize", None)
        external_case = CaseSpec(
            name="external_zarr_lazy",
            kind="zarr_lazy_external",
            shape=zarr_shape,
            levels=1,
            description=f"External lazy zarr array: {zarr_component}",
            chunks=tuple(int(x) for x in chunk_sig) if chunk_sig is not None else None,
            allowed_backends=("dask", "dask_cuda", "auto"),
            zarr_path=zarr_component,
        )
        cases.append(external_case)
    case_filter = str(args.case_filter or "").strip()
    if case_filter:
        try:
            case_re = re.compile(case_filter)
        except re.error as exc:
            raise ValueError(f"Invalid --case-filter regex {case_filter!r}: {exc}") from exc
        cases = [case for case in cases if case_re.search(case.name)]
        if not cases:
            raise ValueError(f"--case-filter {case_filter!r} matched no benchmark cases.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"dispatch_benchmark_{timestamp}"
    json_path = out_dir / f"{stem}.json"
    raw_csv_path = out_dir / f"{stem}_raw.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    interaction_raw_csv_path = out_dir / f"{stem}_interaction_raw.csv"
    interaction_csv_path = out_dir / f"{stem}_interaction.csv"
    graph_csv_path = out_dir / f"{stem}_graphs.csv"
    md_path = out_dir / f"{stem}.md"
    graphs_dir = out_dir / f"{stem}_graphs"

    print(f"[bench] writing results to: {out_dir}")
    print(f"[bench] cases: {len(cases)}, backends: {', '.join(backends)}")

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    interaction_raw_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []

    for i, case in enumerate(cases):
        payload = _build_case_payload(case, seed=args.seed + i)
        payload_nbytes = _estimate_nbytes(payload)
        payload_elements = _estimate_elements(payload)
        print(
            f"[bench] case={case.name} kind={case.kind} approx_nbytes={payload_nbytes} approx_elements={payload_elements}"
        )

        per_backend_first_output: dict[str, Any] = {}
        per_backend_first_graph_output: dict[str, Any] = {}
        allowed_backends = set(case.allowed_backends) if case.allowed_backends else None

        for backend in backends:
            run_chunks = case.chunks if case.kind == "notebook_overlap" else chunks
            if allowed_backends is not None and backend not in allowed_backends:
                skipped_row = {
                    "case": case.name,
                    "requested_backend": backend,
                    "case_approx_nbytes": payload_nbytes,
                    "case_approx_elements": payload_elements,
                    "case_approx_gib": float(payload_nbytes) / (1024.0 ** 3),
                    "case_approx_mpix": float(payload_elements) / 1_000_000.0,
                    "runs_ok": 0,
                    "runs_total": 0,
                    "selected_backend_mode": "skipped",
                    "selected_backend_dominant": "n/a",
                    "backend_honored_runs": 0,
                    "backend_honored_ratio": 0.0,
                    "fallback_detected": False,
                    "fallback_reasons": "",
                    "dispatch_time_mean_s": 0.0,
                    "dispatch_time_std_s": 0.0,
                    "compute_time_mean_s": 0.0,
                    "compute_time_std_s": 0.0,
                    "total_time_mean_s": 0.0,
                    "total_time_std_s": 0.0,
                    "dispatch_share_pct": 0.0,
                    "compute_share_pct": 0.0,
                    "throughput_gib_s": 0.0,
                    "throughput_compute_gib_s": 0.0,
                    "throughput_mpix_s": 0.0,
                    "throughput_compute_mpix_s": 0.0,
                    "dask_output_chunks": "",
                    "dask_output_chunksize": "",
                    "dask_output_numblocks": "",
                    "dask_output_npartitions": None,
                    "dask_output_tasks": None,
                    "rss_delta_mean_mb": 0.0,
                    "rss_delta_std_mb": 0.0,
                    "status": "skipped",
                    "error": "backend skipped for this case kind",
                    "reference_backend": "n/a",
                    "reference_total_time_mean_s": None,
                    "speedup_vs_reference": None,
                    "mae_vs_reference": None,
                    "max_abs_vs_reference": None,
                    "compatible_vs_reference": "n/a",
                    "comparison_details": "skipped",
                }
                if args.app_interaction:
                    interaction_rows.append(skipped_row)
                else:
                    summary_rows.append(skipped_row)
                continue

            if args.app_interaction:
                for _ in range(max(0, int(args.warmup))):
                    try:
                        _run_app_interaction_single(
                            payload,
                            case_kind=case.kind,
                            sigma=float(args.sigma),
                            mode=str(args.mode),
                            backend=backend,
                            chunks=run_chunks,
                            axes=case.axes,
                            roi_sizes=interaction_roi_sizes,
                            scrub_count=int(args.interaction_scrub_count),
                        )
                    except Exception:
                        break

                interaction_run_records: list[dict[str, Any]] = []
                first_graph_output = None
                for rep in range(max(1, int(args.repeats))):
                    try:
                        rec, out_lazy = _run_app_interaction_single(
                            payload,
                            case_kind=case.kind,
                            sigma=float(args.sigma),
                            mode=str(args.mode),
                            backend=backend,
                            chunks=run_chunks,
                            axes=case.axes,
                            roi_sizes=interaction_roi_sizes,
                            scrub_count=int(args.interaction_scrub_count),
                        )
                        rec.update(
                            {
                                "case": case.name,
                                "case_kind": case.kind,
                                "case_shape": str(case.shape),
                                "case_levels": case.levels,
                                "case_description": case.description,
                                "case_approx_nbytes": payload_nbytes,
                                "case_approx_elements": payload_elements,
                                "case_approx_gib": float(payload_nbytes) / (1024.0 ** 3),
                                "case_approx_mpix": float(payload_elements) / 1_000_000.0,
                                "repeat_index": rep,
                            }
                        )
                        interaction_run_records.append(rec)
                        interaction_raw_rows.append(rec)
                        if (
                            first_graph_output is None
                            and _first_dask_array(out_lazy) is not None
                        ):
                            first_graph_output = out_lazy
                    except Exception as exc:
                        err_row = {
                            "case": case.name,
                            "case_kind": case.kind,
                            "case_shape": str(case.shape),
                            "case_levels": case.levels,
                            "case_description": case.description,
                            "case_approx_nbytes": payload_nbytes,
                            "case_approx_elements": payload_elements,
                            "case_approx_gib": float(payload_nbytes) / (1024.0 ** 3),
                            "case_approx_mpix": float(payload_elements) / 1_000_000.0,
                            "repeat_index": rep,
                            "requested_backend": backend,
                            "selected_backend": "error",
                            "selected_backend_mode": "error",
                            "build_time_s": 0.0,
                            "first_slice_time_s": None,
                            "scrub_mean_time_s": None,
                            "scrub_count": 0,
                            "app_total_time_s": 0.0,
                            "dispatch_log": "",
                            "status": "error",
                            "error": repr(exc),
                        }
                        interaction_run_records.append(err_row)
                        interaction_raw_rows.append(err_row)
                        break

                if first_graph_output is not None:
                    per_backend_first_graph_output[backend] = first_graph_output
                agg = _aggregate_interaction_records(
                    interaction_run_records,
                    requested_backend=backend,
                )
                agg.update(
                    {
                        "case": case.name,
                        "case_kind": case.kind,
                        "case_shape": str(case.shape),
                        "case_levels": case.levels,
                        "case_description": case.description,
                        "case_approx_nbytes": payload_nbytes,
                        "case_approx_elements": payload_elements,
                        "case_approx_gib": float(payload_nbytes) / (1024.0 ** 3),
                        "case_approx_mpix": float(payload_elements) / 1_000_000.0,
                    }
                )
                interaction_rows.append(agg)
                continue

            # Warmup (ignored in report)
            for _ in range(max(0, int(args.warmup))):
                try:
                    _run_single(
                        payload,
                        case_kind=case.kind,
                        sigma=float(args.sigma),
                        mode=str(args.mode),
                        backend=backend,
                        chunks=run_chunks,
                        axes=case.axes,
                    )
                except Exception:
                    # Warmup failures are handled in measured runs.
                    break

            run_records: list[dict[str, Any]] = []
            first_output = None
            first_graph_output = None
            error_msg = None

            for rep in range(max(1, int(args.repeats))):
                try:
                    rec, out, out_lazy = _run_single(
                        payload,
                        case_kind=case.kind,
                        sigma=float(args.sigma),
                        mode=str(args.mode),
                        backend=backend,
                        chunks=run_chunks,
                        axes=case.axes,
                    )
                    rec.update(
                        {
                            "case": case.name,
                            "case_kind": case.kind,
                            "case_shape": str(case.shape),
                            "case_levels": case.levels,
                            "case_description": case.description,
                            "case_approx_nbytes": payload_nbytes,
                            "case_approx_elements": payload_elements,
                            "repeat_index": rep,
                            "status": "ok",
                            "error": "",
                        }
                    )
                    run_records.append(rec)
                    raw_rows.append(rec)
                    if first_output is None:
                        first_output = out
                    if (
                        first_graph_output is None
                        and _first_dask_array(out_lazy) is not None
                    ):
                        first_graph_output = out_lazy
                except Exception as exc:
                    error_msg = repr(exc)
                    err_row = {
                        "case": case.name,
                        "case_kind": case.kind,
                        "case_shape": str(case.shape),
                        "case_levels": case.levels,
                        "case_description": case.description,
                        "case_approx_nbytes": payload_nbytes,
                        "case_approx_elements": payload_elements,
                        "repeat_index": rep,
                        "requested_backend": backend,
                        "selected_backend": "error",
                        "dispatch_time_s": 0.0,
                        "compute_time_s": 0.0,
                        "total_time_s": 0.0,
                        "rss_before_mb": None,
                        "rss_after_mb": None,
                        "rss_delta_mb": None,
                        "output_signature": "",
                        "dispatch_log": "",
                        "status": "error",
                        "error": error_msg,
                    }
                    raw_rows.append(err_row)
                    run_records.append(err_row)
                    break

            if first_output is not None:
                per_backend_first_output[backend] = first_output
            if first_graph_output is not None:
                per_backend_first_graph_output[backend] = first_graph_output

            ok_records = [r for r in run_records if r.get("status") == "ok"]
            selected_counts = Counter(r.get("selected_backend", "") for r in ok_records)
            selected_mode = (
                "n/a"
                if not selected_counts
                else ", ".join(f"{k}:{v}" for k, v in sorted(selected_counts.items()))
            )
            dispatch_vals = [float(r["dispatch_time_s"]) for r in ok_records]
            compute_vals = [float(r["compute_time_s"]) for r in ok_records]
            total_vals = [float(r["total_time_s"]) for r in ok_records]
            rss_delta_vals = [
                float(r["rss_delta_mb"])
                for r in ok_records
                if r.get("rss_delta_mb") is not None
            ]
            dispatch_mean, dispatch_std = _aggregate_values(dispatch_vals)
            compute_mean, compute_std = _aggregate_values(compute_vals)
            total_mean, total_std = _aggregate_values(total_vals)
            rss_mean, rss_std = _aggregate_values(rss_delta_vals) if rss_delta_vals else (0.0, 0.0)
            dominant_backend = (
                "n/a" if not selected_counts else selected_counts.most_common(1)[0][0]
            )
            honored_runs = 0
            fallback_reasons = set()
            for rec in ok_records:
                selected_backend = str(rec.get("selected_backend", ""))
                if _backend_request_honored(backend, selected_backend):
                    honored_runs += 1
                else:
                    reason = _dispatch_reason(rec.get("dispatch_log", ""))
                    if reason:
                        fallback_reasons.add(reason)
            honored_ratio = _safe_div(honored_runs, len(ok_records))
            fallback_detected = bool(ok_records) and honored_runs < len(ok_records)
            case_gib = float(payload_nbytes) / (1024.0 ** 3)
            case_mpix = float(payload_elements) / 1_000_000.0
            dispatch_share_pct = 100.0 * _safe_div(dispatch_mean, total_mean)
            compute_share_pct = 100.0 * _safe_div(compute_mean, total_mean)
            throughput_gib_s = _safe_div(case_gib, total_mean)
            throughput_compute_gib_s = _safe_div(case_gib, compute_mean)
            throughput_mpix_s = _safe_div(case_mpix, total_mean)
            throughput_compute_mpix_s = _safe_div(case_mpix, compute_mean)
            dask_info_record = next(
                (
                    r
                    for r in ok_records
                    if r.get("dask_output_npartitions") not in (None, "", 0, "0")
                ),
                {},
            )

            summary_rows.append(
                {
                    "case": case.name,
                    "requested_backend": backend,
                    "case_approx_nbytes": payload_nbytes,
                    "case_approx_elements": payload_elements,
                    "case_approx_gib": case_gib,
                    "case_approx_mpix": case_mpix,
                    "runs_ok": len(ok_records),
                    "runs_total": len(run_records),
                    "selected_backend_mode": selected_mode,
                    "selected_backend_dominant": dominant_backend,
                    "backend_honored_runs": honored_runs,
                    "backend_honored_ratio": honored_ratio,
                    "fallback_detected": fallback_detected,
                    "fallback_reasons": "; ".join(sorted(fallback_reasons)),
                    "dispatch_time_mean_s": dispatch_mean,
                    "dispatch_time_std_s": dispatch_std,
                    "compute_time_mean_s": compute_mean,
                    "compute_time_std_s": compute_std,
                    "total_time_mean_s": total_mean,
                    "total_time_std_s": total_std,
                    "dispatch_share_pct": dispatch_share_pct,
                    "compute_share_pct": compute_share_pct,
                    "throughput_gib_s": throughput_gib_s,
                    "throughput_compute_gib_s": throughput_compute_gib_s,
                    "throughput_mpix_s": throughput_mpix_s,
                    "throughput_compute_mpix_s": throughput_compute_mpix_s,
                    "dask_output_chunks": dask_info_record.get("dask_output_chunks", ""),
                    "dask_output_chunksize": dask_info_record.get("dask_output_chunksize", ""),
                    "dask_output_numblocks": dask_info_record.get("dask_output_numblocks", ""),
                    "dask_output_npartitions": dask_info_record.get("dask_output_npartitions"),
                    "dask_output_tasks": dask_info_record.get("dask_output_tasks"),
                    "rss_delta_mean_mb": rss_mean,
                    "rss_delta_std_mb": rss_std,
                    "status": "ok" if ok_records else "error",
                    "error": error_msg or "",
                    "reference_backend": "n/a",
                    "reference_total_time_mean_s": None,
                    "speedup_vs_reference": None,
                    "mae_vs_reference": None,
                    "max_abs_vs_reference": None,
                    "compatible_vs_reference": "n/a",
                    "comparison_details": "",
                    }
                )

        if args.app_interaction:
            if args.save_graphs:
                for backend_name, value in per_backend_first_graph_output.items():
                    selected_mode = "unknown"
                    for row in interaction_rows:
                        if row.get("case") != case.name:
                            continue
                        if row.get("requested_backend") == backend_name:
                            selected_mode = str(
                                row.get("selected_backend_mode", "unknown")
                            )
                            break
                    selected_backend = (
                        selected_mode.split(",", 1)[0].split(":", 1)[0].strip()
                        if selected_mode
                        else "unknown"
                    )
                    graph_path = (
                        graphs_dir / f"{case.name}__{backend_name}.{args.graph_format}"
                    )
                    (
                        render_error,
                        rendered_graph_path,
                        rendered_graph_kind,
                        render_message,
                    ) = _render_dask_graph(value, path=graph_path)
                    mermaid = _build_mermaid_hlg(value)
                    stats = _graph_stats(value)
                    if render_error and mermaid:
                        status = "mermaid_only"
                    elif render_error:
                        status = "error"
                    else:
                        status = "ok"
                    graph_kind = (
                        rendered_graph_kind
                        if status == "ok"
                        else "mermaid_fallback"
                    )
                    if rendered_graph_path is None:
                        graph_ref = ""
                    else:
                        try:
                            graph_ref = str(rendered_graph_path.relative_to(out_dir))
                        except Exception:
                            graph_ref = str(rendered_graph_path)
                    graph_rows.append(
                        {
                            "case": case.name,
                            "requested_backend": backend_name,
                            "selected_backend": selected_backend,
                            "selected_backend_mode": selected_mode,
                            "graph_kind": graph_kind,
                            "status": status,
                            "message": render_error if render_error else render_message,
                            "graph_file": graph_ref,
                            "mermaid_hlg": mermaid,
                            "graph_layers": stats["graph_layers"],
                            "graph_dependency_edges": stats["graph_dependency_edges"],
                            "graph_total_tasks": stats["graph_total_tasks"],
                        }
                    )
            continue

        # Compare first successful output of each backend vs selected reference.
        reference_backend = None
        for candidate in ("cpu", "dask", "auto", "cuda", "dask_cuda"):
            if candidate in per_backend_first_output:
                reference_backend = candidate
                break

        baseline_output = (
            None
            if reference_backend is None
            else per_backend_first_output[reference_backend]
        )

        if baseline_output is None:
            for row in summary_rows:
                if row["case"] == case.name:
                    row["comparison_details"] = "reference baseline unavailable"
            continue

        case_rows = [r for r in summary_rows if r["case"] == case.name]
        ref_row = next(
            (
                r
                for r in case_rows
                if r.get("requested_backend") == reference_backend
                and r.get("status") == "ok"
            ),
            None,
        )
        ref_total = None
        if ref_row is not None:
            ref_total = float(ref_row.get("total_time_mean_s", 0.0))
            if ref_total <= 0.0:
                ref_total = None

        for row in summary_rows:
            if row["case"] != case.name:
                continue
            row["reference_backend"] = reference_backend
            row["reference_total_time_mean_s"] = ref_total
            total_here = float(row.get("total_time_mean_s", 0.0))
            if (
                ref_total is not None
                and total_here > 0.0
                and str(row.get("status", "")) == "ok"
            ):
                row["speedup_vs_reference"] = ref_total / total_here
            else:
                row["speedup_vs_reference"] = None
            backend = row["requested_backend"]
            candidate = per_backend_first_output.get(backend)
            if candidate is None:
                row["comparison_details"] = "no output to compare"
                continue
            cmp_res = _compare_outputs(baseline_output, candidate)
            row["mae_vs_reference"] = cmp_res.get("mae")
            row["max_abs_vs_reference"] = cmp_res.get("max_abs")
            row["compatible_vs_reference"] = str(cmp_res.get("compatible", False))
            row["comparison_details"] = cmp_res.get("details", "")

        if args.save_graphs:
            for backend_name, value in per_backend_first_graph_output.items():
                selected_mode = "unknown"
                for row in summary_rows:
                    if row.get("case") != case.name:
                        continue
                    if row.get("requested_backend") == backend_name:
                        selected_mode = str(row.get("selected_backend_mode", "unknown"))
                        break
                selected_backend = (
                    selected_mode.split(",", 1)[0].split(":", 1)[0].strip()
                    if selected_mode
                    else "unknown"
                )
                graph_path = graphs_dir / f"{case.name}__{backend_name}.{args.graph_format}"
                (
                    render_error,
                    rendered_graph_path,
                    rendered_graph_kind,
                    render_message,
                ) = _render_dask_graph(
                    value, path=graph_path
                )
                mermaid = _build_mermaid_hlg(value)
                stats = _graph_stats(value)
                if render_error and mermaid:
                    status = "mermaid_only"
                elif render_error:
                    status = "error"
                else:
                    status = "ok"
                graph_kind = (
                    rendered_graph_kind
                    if status == "ok"
                    else "mermaid_fallback"
                )
                if rendered_graph_path is None:
                    graph_ref = ""
                else:
                    try:
                        graph_ref = str(rendered_graph_path.relative_to(out_dir))
                    except Exception:
                        graph_ref = str(rendered_graph_path)
                graph_rows.append(
                    {
                        "case": case.name,
                        "requested_backend": backend_name,
                        "selected_backend": selected_backend,
                        "selected_backend_mode": selected_mode,
                        "graph_kind": graph_kind,
                        "status": status,
                        "message": render_error if render_error else render_message,
                        "graph_file": graph_ref,
                        "mermaid_hlg": mermaid,
                        "graph_layers": stats["graph_layers"],
                        "graph_dependency_edges": stats["graph_dependency_edges"],
                        "graph_total_tasks": stats["graph_total_tasks"],
                    }
                )

    metadata = {
        "timestamp": timestamp,
        "preset": args.preset,
        "backends": backends,
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "sigma": float(args.sigma),
        "mode": str(args.mode),
        "chunks": chunks,
        "app_interaction": bool(args.app_interaction),
        "interaction_rois": interaction_roi_sizes,
        "interaction_scrub_count": int(args.interaction_scrub_count),
        "zarr_path": zarr_path_input or None,
        "zarr_key": zarr_key_input or None,
        "python": sys.version,
    }

    report = {
        "metadata": metadata,
        "cases": [case.__dict__ for case in cases],
        "summary": summary_rows,
        "interaction_summary": interaction_rows,
        "interaction_raw": interaction_raw_rows,
        "graphs": graph_rows,
        "raw": raw_rows,
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _write_csv(raw_csv_path, raw_rows)
    _write_csv(summary_csv_path, summary_rows)
    _write_csv(interaction_raw_csv_path, interaction_raw_rows)
    _write_csv(interaction_csv_path, interaction_rows)
    _write_csv(graph_csv_path, graph_rows)
    output_files = {
        "json": str(json_path),
        "markdown": str(md_path),
    }
    if raw_rows:
        output_files["raw_csv"] = str(raw_csv_path)
    if summary_rows:
        output_files["summary_csv"] = str(summary_csv_path)
    if interaction_raw_rows:
        output_files["interaction_raw_csv"] = str(interaction_raw_csv_path)
    if interaction_rows:
        output_files["interaction_csv"] = str(interaction_csv_path)
    if graph_rows:
        output_files["graphs_csv"] = str(graph_csv_path)
    _write_markdown(
        md_path,
        metadata=metadata,
        summary_rows=summary_rows,
        interaction_rows=interaction_rows,
        graph_rows=graph_rows,
        output_files=output_files,
    )

    print("[bench] done.")
    print(f"[bench] json: {json_path}")
    if raw_rows:
        print(f"[bench] raw csv: {raw_csv_path}")
    if summary_rows:
        print(f"[bench] summary csv: {summary_csv_path}")
    if interaction_raw_rows:
        print(f"[bench] interaction raw csv: {interaction_raw_csv_path}")
    if interaction_rows:
        print(f"[bench] interaction csv: {interaction_csv_path}")
    if graph_rows:
        print(f"[bench] graphs csv: {graph_csv_path}")
    print(f"[bench] markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
