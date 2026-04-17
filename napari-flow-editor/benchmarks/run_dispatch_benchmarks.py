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
import io
import json
import os
import re
import statistics
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

from napari_flow_editor.flow_nodes.decorator import dispatch


_SELECTED_BACKEND_RE = re.compile(r"selected=([a-z_]+)")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    kind: str  # "numpy" | "multiscale" | "dask_lazy" | "zarr_lazy_external"
    shape: tuple[int, ...]
    levels: int = 1
    description: str = ""
    chunks: str | tuple[int, ...] | None = None
    allowed_backends: tuple[str, ...] | None = None
    zarr_path: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dispatch backend benchmarks.")
    parser.add_argument(
        "--preset",
        choices=("quick", "standard", "large"),
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
        default="auto",
        help="Chunk spec for NumPy→Dask promotion. Example: auto or 1,256,256",
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
    if value.lower() == "auto":
        return "auto"
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return "auto"
    return tuple(int(p) for p in parts)


def _make_case_specs(preset: str) -> list[CaseSpec]:
    if preset == "large":
        return [
            CaseSpec(
                name="numpy_2d_4096",
                kind="numpy",
                shape=(4096, 4096),
                description="Large 2D image (float32).",
            ),
            CaseSpec(
                name="numpy_3d_tyx_24x1536x1536",
                kind="numpy",
                shape=(24, 1536, 1536),
                description="Large 3D timeline-like volume (T,Y,X).",
            ),
            CaseSpec(
                name="multiscale_tyx_l5_16x2048x2048",
                kind="multiscale",
                shape=(16, 2048, 2048),
                levels=5,
                description="5-level large multiscale stack.",
            ),
            CaseSpec(
                name="dask_lazy_tyx_32x2048x2048",
                kind="dask_lazy",
                shape=(32, 2048, 2048),
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
                description="2D single image (float32).",
            ),
            CaseSpec(
                name="numpy_3d_tyx_16x1024x1024",
                kind="numpy",
                shape=(16, 1024, 1024),
                description="3D timeline-like volume (T,Y,X).",
            ),
            CaseSpec(
                name="multiscale_tyx_l4_8x1024x1024",
                kind="multiscale",
                shape=(8, 1024, 1024),
                levels=4,
                description="4-level multiscale stack.",
            ),
            CaseSpec(
                name="dask_lazy_tyx_16x1024x1024",
                kind="dask_lazy",
                shape=(16, 1024, 1024),
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
            description="2D single image (float32).",
        ),
        CaseSpec(
            name="numpy_3d_tyx_8x512x512",
            kind="numpy",
            shape=(8, 512, 512),
            description="3D timeline-like volume (T,Y,X).",
        ),
        CaseSpec(
            name="multiscale_tyx_l4_8x512x512",
            kind="multiscale",
            shape=(8, 512, 512),
            levels=4,
            description="4-level multiscale stack.",
        ),
        CaseSpec(
            name="dask_lazy_tyx_8x512x512",
            kind="dask_lazy",
            shape=(8, 512, 512),
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
        dask_output_dtype=np.float64,
        pyramid_strategy="per_level",
        pyramid_param_policy={"sigma": "fixed_world"},
        numpy_to_dask_chunks=chunks,
    )


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


def _output_shape_signature(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"array:{tuple(int(x) for x in value.shape)}:{value.dtype}"
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


def _run_single(
    payload: Any,
    *,
    sigma: float,
    mode: str,
    backend: str,
    chunks: str | tuple[int, ...],
) -> tuple[dict[str, Any], Any]:
    rss_before = _current_rss_mb()
    dispatch_log = ""
    output = None

    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        output = _dispatch_gaussian(
            payload,
            sigma=sigma,
            mode=mode,
            backend=backend,
            chunks=chunks,
        )
    t1 = time.perf_counter()

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
    }
    return record, materialized


def _aggregate_values(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    path: Path,
    *,
    metadata: dict[str, Any],
    summary_rows: list[dict[str, Any]],
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
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Case | Requested | Selected (mode) | Ref | Total mean (s) | Compute mean (s) | MAE vs Ref | Max abs vs Ref | Compatible |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for row in summary_rows:
        lines.append(
            "| {case} | {requested} | {selected_mode} | {ref} | {total_mean:.4f} | {compute_mean:.4f} | {mae} | {max_abs} | {compatible} |".format(
                case=row.get("case", ""),
                requested=row.get("requested_backend", ""),
                selected_mode=row.get("selected_backend_mode", ""),
                ref=row.get("reference_backend", "n/a"),
                total_mean=float(row.get("total_time_mean_s", 0.0)),
                compute_mean=float(row.get("compute_time_mean_s", 0.0)),
                mae=(
                    "n/a"
                    if row.get("mae_vs_reference") is None
                    else f"{float(row['mae_vs_reference']):.6g}"
                ),
                max_abs=(
                    "n/a"
                    if row.get("max_abs_vs_reference") is None
                    else f"{float(row['max_abs_vs_reference']):.6g}"
                ),
                compatible=row.get("compatible_vs_reference", "n/a"),
            )
        )
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"dispatch_benchmark_{timestamp}"
    json_path = out_dir / f"{stem}.json"
    raw_csv_path = out_dir / f"{stem}_raw.csv"
    summary_csv_path = out_dir / f"{stem}_summary.csv"
    md_path = out_dir / f"{stem}.md"

    print(f"[bench] writing results to: {out_dir}")
    print(f"[bench] cases: {len(cases)}, backends: {', '.join(backends)}")

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    case_payloads = {
        case.name: _build_case_payload(case, seed=args.seed + i)
        for i, case in enumerate(cases)
    }

    for case in cases:
        payload = case_payloads[case.name]
        payload_nbytes = _estimate_nbytes(payload)
        print(
            f"[bench] case={case.name} kind={case.kind} approx_nbytes={payload_nbytes}"
        )

        per_backend_first_output: dict[str, Any] = {}
        allowed_backends = set(case.allowed_backends) if case.allowed_backends else None

        for backend in backends:
            if allowed_backends is not None and backend not in allowed_backends:
                summary_rows.append(
                    {
                        "case": case.name,
                        "requested_backend": backend,
                        "runs_ok": 0,
                        "runs_total": 0,
                        "selected_backend_mode": "skipped",
                        "dispatch_time_mean_s": 0.0,
                        "dispatch_time_std_s": 0.0,
                        "compute_time_mean_s": 0.0,
                        "compute_time_std_s": 0.0,
                        "total_time_mean_s": 0.0,
                        "total_time_std_s": 0.0,
                        "rss_delta_mean_mb": 0.0,
                        "rss_delta_std_mb": 0.0,
                        "status": "skipped",
                        "error": "backend skipped for this case kind",
                        "reference_backend": "n/a",
                        "mae_vs_reference": None,
                        "max_abs_vs_reference": None,
                        "compatible_vs_reference": "n/a",
                        "comparison_details": "skipped",
                    }
                )
                continue

            # Warmup (ignored in report)
            for _ in range(max(0, int(args.warmup))):
                try:
                    _run_single(
                        payload,
                        sigma=float(args.sigma),
                        mode=str(args.mode),
                        backend=backend,
                        chunks=chunks,
                    )
                except Exception:
                    # Warmup failures are handled in measured runs.
                    break

            run_records: list[dict[str, Any]] = []
            first_output = None
            error_msg = None

            for rep in range(max(1, int(args.repeats))):
                try:
                    rec, out = _run_single(
                        payload,
                        sigma=float(args.sigma),
                        mode=str(args.mode),
                        backend=backend,
                        chunks=chunks,
                    )
                    rec.update(
                        {
                            "case": case.name,
                            "case_kind": case.kind,
                            "case_shape": str(case.shape),
                            "case_levels": case.levels,
                            "case_description": case.description,
                            "case_approx_nbytes": payload_nbytes,
                            "repeat_index": rep,
                            "status": "ok",
                            "error": "",
                        }
                    )
                    run_records.append(rec)
                    raw_rows.append(rec)
                    if first_output is None:
                        first_output = out
                except Exception as exc:
                    error_msg = repr(exc)
                    err_row = {
                        "case": case.name,
                        "case_kind": case.kind,
                        "case_shape": str(case.shape),
                        "case_levels": case.levels,
                        "case_description": case.description,
                        "case_approx_nbytes": payload_nbytes,
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

            summary_rows.append(
                {
                    "case": case.name,
                    "requested_backend": backend,
                    "runs_ok": len(ok_records),
                    "runs_total": len(run_records),
                    "selected_backend_mode": selected_mode,
                    "dispatch_time_mean_s": dispatch_mean,
                    "dispatch_time_std_s": dispatch_std,
                    "compute_time_mean_s": compute_mean,
                    "compute_time_std_s": compute_std,
                    "total_time_mean_s": total_mean,
                    "total_time_std_s": total_std,
                    "rss_delta_mean_mb": rss_mean,
                    "rss_delta_std_mb": rss_std,
                    "status": "ok" if ok_records else "error",
                    "error": error_msg or "",
                    "reference_backend": "n/a",
                    "mae_vs_reference": None,
                    "max_abs_vs_reference": None,
                    "compatible_vs_reference": "n/a",
                    "comparison_details": "",
                }
            )

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

        for row in summary_rows:
            if row["case"] != case.name:
                continue
            row["reference_backend"] = reference_backend
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

    metadata = {
        "timestamp": timestamp,
        "preset": args.preset,
        "backends": backends,
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "sigma": float(args.sigma),
        "mode": str(args.mode),
        "chunks": chunks,
        "zarr_path": zarr_path_input or None,
        "zarr_key": zarr_key_input or None,
        "python": sys.version,
    }

    report = {
        "metadata": metadata,
        "cases": [case.__dict__ for case in cases],
        "summary": summary_rows,
        "raw": raw_rows,
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _write_csv(raw_csv_path, raw_rows)
    _write_csv(summary_csv_path, summary_rows)
    _write_markdown(
        md_path,
        metadata=metadata,
        summary_rows=summary_rows,
        output_files={
            "json": str(json_path),
            "raw_csv": str(raw_csv_path),
            "summary_csv": str(summary_csv_path),
            "markdown": str(md_path),
        },
    )

    print("[bench] done.")
    print(f"[bench] json: {json_path}")
    print(f"[bench] raw csv: {raw_csv_path}")
    print(f"[bench] summary csv: {summary_csv_path}")
    print(f"[bench] markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
