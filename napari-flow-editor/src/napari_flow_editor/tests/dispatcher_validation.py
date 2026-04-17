#!/usr/bin/env python3
"""
Dispatcher validation script.

Runs three practical checks:
1) Large-array laziness (partial compute only)
2) Multiscale per-level laziness (only requested levels compute)
3) Numerical agreement (dask path vs CPU skimage reference)

Exit code:
- 0: all checks passed
- 1: one or more checks failed
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from typing import Callable

import dask.array as da
import numpy as np
import skimage.filters
from dask import delayed
from dask.callbacks import Callback

from napari_flow_editor.flow_nodes.filters import gaussian_blur


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _parse_shape(value: str) -> tuple[int, ...]:
    try:
        parts = [int(x.strip()) for x in value.split(",") if x.strip()]
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid shape: {value!r}") from exc
    if not parts or any(p <= 0 for p in parts):
        raise argparse.ArgumentTypeError(f"Invalid shape: {value!r}")
    return tuple(parts)


def _run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        detail = fn()
        return CheckResult(name=name, ok=True, detail=detail)
    except Exception as exc:
        tb = traceback.format_exc(limit=1).strip().replace("\n", " | ")
        return CheckResult(name=name, ok=False, detail=f"{exc} [{tb}]")


def check_large_array_laziness(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    roi_size: int,
    max_task_fraction: float,
) -> str:
    if len(shape) != len(chunks):
        raise ValueError(f"Shape and chunks rank mismatch: {shape} vs {chunks}")
    if roi_size <= 0:
        raise ValueError("roi_size must be > 0")
    if not (0.0 < max_task_fraction < 1.0):
        raise ValueError("max_task_fraction must be in (0, 1)")

    x = da.zeros(shape, chunks=chunks, dtype=np.float32)
    y = gaussian_blur(x, sigma=1.0, mode="nearest")

    if not isinstance(y, da.Array):
        raise AssertionError(f"Expected dask output, got {type(y)!r}")

    total_parts = int(y.npartitions)

    counter = {"tasks": 0}

    class CountTasks(Callback):
        def _pretask(self, key, dsk, state):
            counter["tasks"] += 1

    # Keep only a tiny ROI to force partial graph execution.
    roi_slices = [slice(0, min(roi_size, shape[0]))]
    for dim in shape[1:]:
        roi_slices.append(slice(0, min(roi_size, dim)))
    roi_slices = tuple(roi_slices)

    with CountTasks():
        _ = y[roi_slices].mean().compute()

    tasks = int(counter["tasks"])
    frac = (tasks / total_parts) if total_parts > 0 else 1.0

    if tasks <= 0:
        raise AssertionError("No tasks executed for ROI compute.")
    if frac >= max_task_fraction:
        raise AssertionError(
            f"Too many tasks executed ({tasks}/{total_parts}, {frac:.3f}). "
            f"Expected < {max_task_fraction:.3f}."
        )

    logical_gb = x.nbytes / (1024**3)
    return (
        f"logical_size={logical_gb:.2f}GB, npartitions={total_parts}, "
        f"tasks_for_roi={tasks}, task_fraction={frac:.4f}"
    )


def check_multiscale_per_level_laziness() -> str:
    calls: list[int] = []

    @delayed
    def mk_level(level_id: int, level_shape: tuple[int, ...]):
        calls.append(level_id)
        return np.full(level_shape, fill_value=level_id, dtype=np.float32)

    shapes = [(8, 64, 64), (8, 32, 32), (8, 16, 16), (8, 8, 8)]
    pyramid = [
        da.from_delayed(mk_level(i, s), shape=s, dtype=np.float32)
        for i, s in enumerate(shapes)
    ]

    out = gaussian_blur(pyramid, sigma=1.0, mode="nearest")
    if not isinstance(out, list):
        raise AssertionError(f"Expected list output for pyramid, got {type(out)!r}")
    if len(out) != len(shapes):
        raise AssertionError(f"Expected {len(shapes)} levels, got {len(out)}")

    if calls:
        raise AssertionError(f"Laziness violated at graph build: calls={calls}")

    # Request level 3 only.
    _ = out[3][0, :4, :4].mean().compute()
    if calls != [3]:
        raise AssertionError(f"Expected calls [3] after level-3 compute, got {calls}")

    # Then request level 1.
    _ = out[1][0, :4, :4].mean().compute()
    if calls != [3, 1]:
        raise AssertionError(f"Expected calls [3, 1] after level-1 compute, got {calls}")

    return "calls_sequence=[3, 1] (only requested levels computed)"


def check_numerical_agreement(seed: int, rtol: float, atol: float) -> str:
    rng = np.random.default_rng(seed)
    image = rng.random((8, 128, 128), dtype=np.float32)
    dask_image = da.from_array(image, chunks=(2, 32, 32))

    out = gaussian_blur(dask_image, sigma=1.2, mode="reflect")
    if not isinstance(out, da.Array):
        raise AssertionError(f"Expected dask output, got {type(out)!r}")
    got = out.compute()

    expected = skimage.filters.gaussian(
        image, sigma=1.2, mode="reflect", preserve_range=True
    )

    if not np.allclose(got, expected, rtol=rtol, atol=atol):
        diff = np.abs(got - expected)
        raise AssertionError(
            f"Numerical mismatch: max={float(diff.max()):.6g}, "
            f"mean={float(diff.mean()):.6g}"
        )

    diff = np.abs(got - expected)
    return f"max_abs_diff={float(diff.max()):.3e}, mean_abs_diff={float(diff.mean()):.3e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dispatcher laziness and correctness.")
    parser.add_argument(
        "--large-shape",
        type=_parse_shape,
        default=(60, 32768, 32768),
        help="Logical shape for large-array laziness check, e.g. 60,32768,32768",
    )
    parser.add_argument(
        "--large-chunks",
        type=_parse_shape,
        default=(1, 1024, 1024),
        help="Chunk shape for large-array laziness check, e.g. 1,1024,1024",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=2048,
        help="ROI edge size used for partial compute check.",
    )
    parser.add_argument(
        "--max-task-fraction",
        type=float,
        default=0.5,
        help="Fail if tasks_for_roi / npartitions is >= this value.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for numerical check.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance.")
    args = parser.parse_args()

    checks = [
        (
            "large_array_laziness",
            lambda: check_large_array_laziness(
                shape=args.large_shape,
                chunks=args.large_chunks,
                roi_size=args.roi_size,
                max_task_fraction=args.max_task_fraction,
            ),
        ),
        ("multiscale_per_level_laziness", check_multiscale_per_level_laziness),
        (
            "numerical_agreement",
            lambda: check_numerical_agreement(
                seed=args.seed, rtol=args.rtol, atol=args.atol
            ),
        ),
    ]

    results = [_run_check(name, fn) for name, fn in checks]

    print("\n=== Dispatcher Validation ===")
    for res in results:
        status = "PASS" if res.ok else "FAIL"
        print(f"[{status}] {res.name}: {res.detail}")

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\nResult: FAILED ({len(failed)}/{len(results)} checks failed)")
        return 1

    print(f"\nResult: PASSED ({len(results)}/{len(results)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
