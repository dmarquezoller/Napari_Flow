from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import dask.array as da
import zarr


LogCallback = Optional[Callable[[str], None]]


@dataclass
class OmeZarrSaveRequest:
    data: Any
    path: str
    name: str = ""
    layer_type: str = "image"
    metadata: dict[str, Any] = field(default_factory=dict)
    axes: Optional[str] = None
    scale: Optional[tuple[float, ...]] = None
    translate: Optional[tuple[float, ...]] = None
    chunks: str | tuple[int, ...] | None = "auto"
    pyramid_levels: int = 1
    pyramid_downsample: str = "xy"
    overwrite: bool = True


@dataclass
class OmeZarrSaveResult:
    path: str
    name: str
    axes: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    levels: int
    requested_levels: int
    source_levels: int
    layer_type: str
    gpu_chunks_converted: bool = False
    omero_channels: int = 0


def _shape_of(data: Any) -> tuple[int, ...]:
    shape = getattr(data, "shape", None)
    if shape is None:
        raise TypeError(f"Object is not array-like: {type(data)!r}")
    return tuple(int(s) for s in shape)


def _is_multiscale_data(data: Any) -> bool:
    if isinstance(data, (np.ndarray, da.Array, tuple, str, bytes)):
        return False
    try:
        if len(data) <= 0:
            return False
        first = data[0]
    except Exception:
        return False
    return hasattr(first, "shape")


def normalize_axes(axes: Any, ndim: int) -> Optional[str]:
    if axes is None:
        return None
    if isinstance(axes, str):
        text = axes.strip()
    else:
        try:
            parts = []
            for item in axes:
                if isinstance(item, dict):
                    item = item.get("name", "")
                parts.append(str(item).strip())
            text = "".join(parts)
        except Exception:
            return None
    text = text.replace(" ", "").replace(",", "").lower()
    if len(text) != int(ndim):
        return None
    allowed = set("tczyx")
    if not set(text).issubset(allowed):
        return None
    return text


def infer_axes(data: Any, metadata: Optional[dict[str, Any]] = None, axes: Any = None) -> str:
    shape = _shape_of(data)
    ndim = len(shape)
    metadata = metadata or {}

    for candidate in (
        axes,
        metadata.get("axes"),
        metadata.get("axis_labels"),
    ):
        normalized = normalize_axes(candidate, ndim)
        if normalized:
            return normalized

    if ndim == 2:
        return "yx"
    if ndim == 3:
        return "zyx"
    if ndim == 4:
        if shape[-1] <= 4:
            return "zyxc"
        return "tzyx"
    if ndim == 5:
        return "tczyx"
    raise ValueError(f"OME-Zarr save supports 2D-5D arrays, got ndim={ndim}.")


def axes_to_ome_metadata(axes: str) -> list[dict[str, str]]:
    out = []
    for axis in axes:
        if axis == "t":
            out.append({"name": "t", "type": "time"})
        elif axis == "c":
            out.append({"name": "c", "type": "channel"})
        elif axis in {"z", "y", "x"}:
            out.append({"name": axis, "type": "space"})
        else:
            out.append({"name": axis, "type": "space"})
    return out


def canonical_ome_axes(axes: str) -> str:
    axes = str(axes or "").lower()
    ordered = []
    for axis in "tczyx":
        if axis in axes:
            ordered.append(axis)
    if len(ordered) != len(axes):
        raise ValueError(f"Cannot map axes={axes!r} to OME axis order.")
    return "".join(ordered)


def _transpose_to_axes(data: Any, source_axes: str, target_axes: str) -> Any:
    if source_axes == target_axes:
        return data
    perm = tuple(source_axes.index(axis) for axis in target_axes)
    if hasattr(data, "transpose"):
        return data.transpose(perm)
    return np.transpose(data, perm)


def _reorder_vector(
    value: tuple[float, ...],
    source_axes: str,
    target_axes: str,
) -> tuple[float, ...]:
    if source_axes == target_axes:
        return value
    return tuple(float(value[source_axes.index(axis)]) for axis in target_axes)


def _coerce_vector(value: Any, ndim: int, default: float) -> tuple[float, ...]:
    if value is None:
        return (float(default),) * int(ndim)
    try:
        seq = tuple(float(v) for v in value)
    except Exception:
        return (float(default),) * int(ndim)
    if len(seq) != int(ndim):
        return (float(default),) * int(ndim)
    return seq


def auto_chunks(shape: tuple[int, ...], axes: str) -> tuple[int, ...]:
    chunks = []
    for size, axis in zip(shape, axes):
        size = max(1, int(size))
        if axis in {"t", "c"}:
            chunks.append(1)
        elif axis == "z":
            chunks.append(min(size, 32))
        elif axis in {"y", "x"}:
            chunks.append(min(size, 512))
        else:
            chunks.append(min(size, 64))
    return tuple(chunks)


def normalize_chunks(
    chunks: str | tuple[int, ...] | None,
    shape: tuple[int, ...],
    axes: str,
) -> tuple[int, ...]:
    if chunks is None or str(chunks).strip().lower() == "auto":
        return auto_chunks(shape, axes)
    try:
        seq = tuple(int(c) for c in chunks)
    except Exception as exc:
        raise ValueError(f"Invalid chunks={chunks!r}") from exc
    if len(seq) != len(shape):
        raise ValueError(f"chunks ndim mismatch: chunks={seq}, shape={shape}")
    return tuple(max(1, min(int(c), int(s))) for c, s in zip(seq, shape))


def _open_group(path: str, mode: str):
    try:
        return zarr.open_group(path, mode=mode, zarr_format=2)
    except TypeError:
        return zarr.open_group(path, mode=mode)


def _to_zarr_v2(darr: da.Array, path: str, component: str):
    kwargs = {
        "url": path,
        "component": component,
        "overwrite": True,
    }
    try:
        return da.to_zarr(darr, **kwargs, zarr_format=2)
    except TypeError:
        try:
            return da.to_zarr(darr, **kwargs, zarr_version=2)
        except TypeError:
            return da.to_zarr(darr, **kwargs)


def _spatial_downsample_axes(
    shape: tuple[int, ...],
    axes: str,
    mode: str = "xy",
) -> dict[int, int]:
    mode = str(mode or "xy").strip().lower()
    if mode in {"zyx", "xyz", "3d", "volume"}:
        downsample_axes = {"z", "y", "x"}
    else:
        downsample_axes = {"y", "x"}

    factors = {}
    for axis_i, axis in enumerate(axes):
        if axis in downsample_axes and int(shape[axis_i]) >= 2:
            factors[axis_i] = 2
    return factors


def _downsample_level(prev: da.Array, axes: str, mode: str = "xy") -> Optional[da.Array]:
    shape = tuple(int(s) for s in prev.shape)
    factors = _spatial_downsample_axes(shape, axes, mode=mode)
    if not factors:
        return None
    return da.coarsen(np.mean, prev, factors, trim_excess=True)


def _as_numpy_backed_dask(array: Any, chunks: tuple[int, ...]) -> tuple[da.Array, bool]:
    if isinstance(array, da.Array):
        darr = array
    else:
        darr = da.from_array(array, chunks=chunks)

    converted_gpu = False
    try:
        import cupy as cp  # type: ignore

        if isinstance(getattr(darr, "_meta", None), cp.ndarray):
            darr = darr.map_blocks(
                cp.asnumpy,
                dtype=darr.dtype,
                meta=np.empty((0,) * int(darr.ndim), dtype=darr.dtype),
            )
            converted_gpu = True
    except Exception:
        converted_gpu = False

    if tuple(int(c) for c in darr.chunksize) != tuple(int(c) for c in chunks):
        darr = darr.rechunk(chunks)
    return darr, converted_gpu


def _coordinate_transformations(
    level_shapes: list[tuple[int, ...]],
    base_scale: tuple[float, ...],
    translate: tuple[float, ...],
) -> list[list[dict[str, list[float] | str]]]:
    base_shape = level_shapes[0]
    transforms = []
    has_translate = any(abs(float(v)) > 1e-12 for v in translate)
    for shape in level_shapes:
        scale = []
        for base_size, level_size, axis_scale in zip(base_shape, shape, base_scale):
            if int(level_size) <= 0:
                ratio = 1.0
            else:
                ratio = float(base_size) / float(level_size)
            scale.append(float(axis_scale) * ratio)
        level_transforms: list[dict[str, list[float] | str]] = [
            {"type": "scale", "scale": scale}
        ]
        if has_translate:
            level_transforms.append(
                {"type": "translation", "translation": [float(v) for v in translate]}
            )
        transforms.append(level_transforms)
    return transforms


def _rgb_to_hex(rgb: Any) -> Optional[str]:
    try:
        values = [float(v) for v in list(rgb)[:3]]
    except Exception:
        return None
    if len(values) < 3:
        return None
    if max(values) <= 1.0:
        values = [v * 255.0 for v in values]
    clipped = [max(0, min(255, int(round(v)))) for v in values]
    return "".join(f"{v:02X}" for v in clipped)


def _colormap_to_omero_color(colormap: Any) -> str:
    name_map = {
        "gray": "FFFFFF",
        "grey": "FFFFFF",
        "red": "FF0000",
        "green": "00FF00",
        "blue": "0000FF",
        "cyan": "00FFFF",
        "magenta": "FF00FF",
        "yellow": "FFFF00",
        "orange": "FFA500",
        "purple": "800080",
        "magma": "FCFDBF",
        "inferno": "FCFFA4",
        "plasma": "F0F921",
        "viridis": "FDE725",
    }

    raw_name = None
    raw_colors = None
    if isinstance(colormap, dict):
        raw_name = colormap.get("name")
        raw_colors = colormap.get("colors")
    elif isinstance(colormap, str):
        raw_name = colormap
    else:
        raw_name = getattr(colormap, "name", None)
        raw_colors = getattr(colormap, "colors", None)

    if raw_name is not None:
        name = str(raw_name).strip().lower()
        if name in name_map:
            return name_map[name]

    if raw_colors is not None:
        try:
            colors = np.asarray(raw_colors)
            if colors.ndim >= 2 and colors.shape[0] > 0 and colors.shape[1] >= 3:
                color = _rgb_to_hex(colors[-1, :3])
                if color:
                    return color
        except Exception:
            pass

    return "FFFFFF"


def _colormap_for_channel(colormap: Any, channel_index: int) -> Any:
    if isinstance(colormap, (str, dict)) or colormap is None:
        return colormap
    try:
        if isinstance(colormap, (list, tuple)) and colormap:
            return colormap[min(channel_index, len(colormap) - 1)]
    except Exception:
        pass
    return colormap


def _contrast_for_channel(contrast: Any, channel_index: int) -> Optional[tuple[float, float]]:
    if contrast is None:
        return None
    try:
        seq = list(contrast)
    except Exception:
        return None
    if len(seq) < 2:
        return None

    first = seq[0]
    try:
        nested = not np.isscalar(first) and len(first) >= 2
    except Exception:
        nested = False

    if nested:
        chosen = seq[min(channel_index, len(seq) - 1)]
        try:
            lo, hi = float(chosen[0]), float(chosen[1])
        except Exception:
            return None
    else:
        try:
            lo, hi = float(seq[0]), float(seq[1])
        except Exception:
            return None

    if hi <= lo:
        return None
    return lo, hi


def _omero_display_metadata(
    name: str,
    axes: str,
    shape: tuple[int, ...],
    metadata: dict[str, Any],
) -> Optional[dict[str, Any]]:
    contrast = metadata.get("contrast_limits")

    channel_count = int(shape[axes.index("c")]) if "c" in axes else 1
    active = bool(metadata.get("visible", True))
    channels = []
    for channel_index in range(max(1, channel_count)):
        color = _colormap_to_omero_color(
            _colormap_for_channel(metadata.get("colormap"), channel_index)
        )
        window = _contrast_for_channel(contrast, channel_index)
        label = str(name)
        if channel_count > 1:
            label = f"{label} {channel_index}"
        channel = {
            "label": label,
            "active": active,
            "color": color,
        }
        if window is not None:
            lo, hi = window
            channel["window"] = {
                "start": lo,
                "end": hi,
                "min": lo,
                "max": hi,
            }
        channels.append(channel)
    if not channels:
        return None
    return {"channels": channels, "rdefs": {"model": "color"}}


def save_ome_zarr(
    request: OmeZarrSaveRequest,
    log: LogCallback = None,
) -> OmeZarrSaveResult:
    path = Path(request.path)
    if path.suffix != ".zarr":
        path = path.with_suffix(".zarr")
    if path.exists() and not request.overwrite:
        raise FileExistsError(f"Output already exists: {path}")

    metadata = dict(request.metadata or {})
    name = str(request.name or metadata.get("name") or path.stem).strip() or path.stem
    data_levels = list(request.data) if _is_multiscale_data(request.data) else [request.data]
    if not data_levels:
        raise ValueError("No image data to save.")
    source_level_count = len(data_levels)
    requested_levels = max(1, int(request.pyramid_levels or 1))
    if len(data_levels) > requested_levels:
        data_levels = data_levels[:requested_levels]

    source_axes = infer_axes(data_levels[0], metadata=metadata, axes=request.axes)
    axes = canonical_ome_axes(source_axes)
    data_levels = [
        _transpose_to_axes(level, source_axes, axes)
        for level in data_levels
    ]
    level_shapes = [_shape_of(level) for level in data_levels]
    base_shape = level_shapes[0]
    if len(axes) != len(base_shape):
        raise ValueError(f"axes={axes!r} does not match shape={base_shape}")

    scale = _coerce_vector(
        request.scale if request.scale is not None else metadata.get("scale"),
        len(source_axes),
        1.0,
    )
    translate = _coerce_vector(
        request.translate if request.translate is not None else metadata.get("translate"),
        len(source_axes),
        0.0,
    )
    scale = _reorder_vector(scale, source_axes, axes)
    translate = _reorder_vector(translate, source_axes, axes)

    gpu_converted = False
    if len(data_levels) < requested_levels:
        last_shape = _shape_of(data_levels[-1])
        last_chunks = normalize_chunks(request.chunks, last_shape, axes)
        last_darr, converted = _as_numpy_backed_dask(data_levels[-1], last_chunks)
        data_levels[-1] = last_darr
        gpu_converted = gpu_converted or converted
        for _ in range(len(data_levels), requested_levels):
            next_level = _downsample_level(
                data_levels[-1],
                axes,
                mode=request.pyramid_downsample,
            )
            if next_level is None:
                break
            data_levels.append(next_level)
        level_shapes = [_shape_of(level) for level in data_levels]

    prepared = []
    for level_index, level in enumerate(data_levels):
        shape = level_shapes[level_index]
        if len(shape) != len(axes):
            raise ValueError(
                f"Level {level_index} ndim mismatch: shape={shape}, axes={axes!r}"
            )
        chunks = normalize_chunks(request.chunks, shape, axes)
        darr, converted = _as_numpy_backed_dask(level, chunks)
        prepared.append((darr, chunks))
        gpu_converted = gpu_converted or converted

    if log:
        log(
            "Writing OME-Zarr "
            f"path={path} shape={base_shape} axes={axes.upper()} chunks={prepared[0][1]} "
            f"levels={len(prepared)}/{requested_levels} source_levels={source_level_count}"
        )

    _open_group(str(path), mode="w")
    for level_index, (darr, _) in enumerate(prepared):
        _to_zarr_v2(darr, str(path), str(level_index))

    root = _open_group(str(path), mode="a")
    coordinate_transformations = _coordinate_transformations(level_shapes, scale, translate)
    datasets = []
    for level_index, transforms in enumerate(coordinate_transformations):
        datasets.append(
            {
                "path": str(level_index),
                "coordinateTransformations": transforms,
            }
        )

    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": name,
            "axes": axes_to_ome_metadata(axes),
            "datasets": datasets,
        }
    ]
    root.attrs["napari_flow"] = {
        "writer": "napari-flow-editor",
        "layer_type": str(request.layer_type or "image"),
        "source_name": str(metadata.get("source_layer", metadata.get("name", ""))),
        "requested_levels": requested_levels,
        "written_levels": len(prepared),
        "source_levels": source_level_count,
        "pyramid_downsample": str(request.pyramid_downsample or "xy"),
    }
    omero = _omero_display_metadata(name, axes, base_shape, metadata)
    if omero is not None:
        root.attrs["omero"] = omero
    omero_channels = len(omero.get("channels", [])) if isinstance(omero, dict) else 0
    if log:
        log(
            "OME-Zarr display metadata "
            f"omero_channels={omero_channels} "
            f"contrast_limits={metadata.get('contrast_limits', None)!r} "
            f"colormap={getattr(metadata.get('colormap', None), 'name', metadata.get('colormap', None))!r}"
        )

    return OmeZarrSaveResult(
        path=str(path),
        name=name,
        axes=axes,
        shape=base_shape,
        chunks=prepared[0][1],
        levels=len(prepared),
        requested_levels=requested_levels,
        source_levels=source_level_count,
        layer_type=str(request.layer_type or "image"),
        gpu_chunks_converted=gpu_converted,
        omero_channels=omero_channels,
    )
