from .decorator import register_node, dispatch
import skimage.feature
import numpy as np
import functools
import math
import dask.array as da


# ---------------------------------------------------------------------------
# features.py — skimage.feature nodes
#
# Output conventions:
#   image-out nodes  → dispatch() with image strategy (neighborhood / pointwise)
#   points-out nodes → dispatch() with output_type="points" and
#                      strategy="chunked_delayed" so large Dask inputs are
#                      processed in spatial chunks without loading the full image
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CUDA helpers (cuda_func path — callable, receives CuPy array)
# All helpers are 2D-specific; dispatch preflight falls back on 3D chunks.
# ---------------------------------------------------------------------------

def _cuda_corner_harris(image, k=0.05, sigma=1.0):
    import cupyx.scipy.ndimage as cnd
    import cupy as cp
    img = image.astype(cp.float64)
    # Gaussian derivatives matching skimage.feature.structure_tensor
    Ix = cnd.gaussian_filter(img, sigma, order=[0, 1])
    Iy = cnd.gaussian_filter(img, sigma, order=[1, 0])
    Axx = cnd.gaussian_filter(Ix * Ix, sigma)
    Axy = cnd.gaussian_filter(Ix * Iy, sigma)
    Ayy = cnd.gaussian_filter(Iy * Iy, sigma)
    det = Axx * Ayy - Axy * Axy
    trace = Axx + Ayy
    return (det - k * trace * trace).astype(cp.float32)


def _cuda_corner_shi_tomasi(image, sigma=1.0):
    import cupyx.scipy.ndimage as cnd
    import cupy as cp
    img = image.astype(cp.float64)
    Ix = cnd.gaussian_filter(img, sigma, order=[0, 1])
    Iy = cnd.gaussian_filter(img, sigma, order=[1, 0])
    Axx = cnd.gaussian_filter(Ix * Ix, sigma)
    Axy = cnd.gaussian_filter(Ix * Iy, sigma)
    Ayy = cnd.gaussian_filter(Iy * Iy, sigma)
    tmp1 = (Axx + Ayy) / 2
    tmp2 = cp.sqrt(((Axx - Ayy) / 2) ** 2 + Axy ** 2)
    return (tmp1 - tmp2).astype(cp.float32)


def _cuda_shape_index(image, sigma=1.0, mode='reflect'):
    import cupyx.scipy.ndimage as cnd
    import cupy as cp
    img = image.astype(cp.float64)
    # Second-order Gaussian derivatives = Hessian components
    Hxx = cnd.gaussian_filter(img, sigma, order=[2, 0], mode=mode)
    Hxy = cnd.gaussian_filter(img, sigma, order=[1, 1], mode=mode)
    Hyy = cnd.gaussian_filter(img, sigma, order=[0, 2], mode=mode)
    tmp = cp.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy ** 2)
    l1 = (Hxx + Hyy + tmp) / 2   # larger eigenvalue
    l2 = (Hxx + Hyy - tmp) / 2   # smaller eigenvalue
    denom = l2 - l1
    denom = cp.where(cp.abs(denom) < 1e-10, cp.full_like(denom, 1e-10), denom)
    return ((2.0 / np.pi) * cp.arctan((l2 + l1) / denom)).astype(cp.float32)


# ---------------------------------------------------------------------------
# CPU-only default wrappers
# These receive the raw image (possibly Dask) and materialise it first.
# Used as default= in dispatch() for nodes that require the full image.
# ---------------------------------------------------------------------------

def _to_numpy(image):
    if isinstance(image, da.Array):
        return image.compute()
    return np.asarray(image)


def _canny_cpu(image, sigma=1.0, low_threshold=0.1, high_threshold=0.2):
    img = _to_numpy(image).astype(np.float32)
    return skimage.feature.canny(
        img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold
    ).astype(np.float32)


def _hog_cpu(image, orientations=9, pixels_per_cell=8, cells_per_block=3):
    img = _to_numpy(image).astype(np.float32)
    _, hog_img = skimage.feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        visualize=True,
    )
    return hog_img.astype(np.float32)


# LBP chunk worker: receives a numpy chunk from map_overlap.
# Uses float64 internally (skimage preference); dispatch casts output to float32.
def _lbp_chunk(image, P=8, R=1.0, method='uniform'):
    return skimage.feature.local_binary_pattern(
        np.asarray(image, dtype=np.float64), P=P, R=R, method=method
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

# --- CANNY ---
@register_node(
    label="Canny Edge Detector",
    category="Features",
    description="Canny edge detection. Returns a binary edge mask. CPU only — hysteresis requires the full image.",
    outputs=["edges"],
    input_types={"image": "image"},
    output_types={"edges": "image"},
    params_config={
        "sigma": {"min": 0.1, "max": 10.0, "step": 0.1},
        "low_threshold": {"min": 0.0, "max": 1.0, "step": 0.01},
        "high_threshold": {"min": 0.0, "max": 1.0, "step": 0.01},
    }
)
def canny(image, sigma: float = 1.0, low_threshold: float = 0.1, high_threshold: float = 0.2):
    return dispatch(
        default=_canny_cpu,
        args=(image,),
        kwargs={"sigma": sigma, "low_threshold": low_threshold, "high_threshold": high_threshold},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        # No dask_options: hysteresis thresholding cannot be applied per-chunk.
        # _canny_cpu materialises any Dask input before calling skimage.
    )


# --- CORNER HARRIS ---
@register_node(
    label="Corner Harris",
    category="Features",
    description="Harris corner detector response map. Supports Dask+CUDA.",
    outputs=["response"],
    input_types={"image": "image"},
    output_types={"response": "image"},
    params_config={
        "k": {"min": 0.0, "max": 0.5, "step": 0.01},
        "sigma": {"min": 0.1, "max": 10.0, "step": 0.1},
    }
)
def corner_harris(image, k: float = 0.05, sigma: float = 1.0):
    return dispatch(
        default=skimage.feature.corner_harris,
        args=(image,),
        kwargs={"k": k, "sigma": sigma},
        cuda_func=_cuda_corner_harris,
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "sigma",
            "numpy_chunks": "spatial_auto",
        }
    )


# --- CORNER SHI-TOMASI ---
@register_node(
    label="Corner Shi-Tomasi",
    category="Features",
    description="Shi-Tomasi minimum-eigenvalue corner response map. Supports Dask+CUDA.",
    outputs=["response"],
    input_types={"image": "image"},
    output_types={"response": "image"},
    params_config={
        "sigma": {"min": 0.1, "max": 10.0, "step": 0.1},
    }
)
def corner_shi_tomasi(image, sigma: float = 1.0):
    return dispatch(
        default=skimage.feature.corner_shi_tomasi,
        args=(image,),
        kwargs={"sigma": sigma},
        cuda_func=_cuda_corner_shi_tomasi,
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "sigma",
            "numpy_chunks": "spatial_auto",
        }
    )


# --- LOCAL BINARY PATTERN ---
@register_node(
    label="Local Binary Pattern",
    category="Features",
    description="LBP texture descriptor. Supports Dask (2-D images). No CUDA equivalent in CuPy.",
    outputs=["lbp"],
    input_types={"image": "image"},
    output_types={"lbp": "image"},
    params_config={
        "P": {"min": 4, "max": 32, "step": 1},
        "R": {"min": 0.5, "max": 10.0, "step": 0.5},
        "method": {"options": ["uniform", "default", "ror", "var"]},
    }
)
def local_binary_pattern(image, P: int = 8, R: float = 1.0, method: str = "uniform"):
    halo = math.ceil(R) + 1
    # Bake P, R, method into the chunk worker so kwargs={} — prevents dispatch from
    # trying to promote these scalars as if they were image arrays.
    fn = functools.partial(_lbp_chunk, P=P, R=R, method=method)
    return dispatch(
        default=fn,
        args=(image,),
        kwargs={},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"depth": halo},
        }
    )


# --- SHAPE INDEX ---
@register_node(
    label="Shape Index",
    category="Features",
    description="Koenderink shape index — encodes local surface curvature in [-1, 1]. Supports Dask+CUDA.",
    outputs=["shape_index"],
    input_types={"image": "image"},
    output_types={"shape_index": "image"},
    params_config={
        "sigma": {"min": 0.1, "max": 10.0, "step": 0.1},
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]},
    }
)
def shape_index(image, sigma: float = 1.0, mode: str = "reflect"):
    return dispatch(
        default=skimage.feature.shape_index,
        args=(image,),
        kwargs={"sigma": sigma, "mode": mode},
        cuda_func=_cuda_shape_index,
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "sigma",
            "boundary_from_param": "mode",
            "numpy_chunks": "spatial_auto",
        }
    )


# --- HOG ---
@register_node(
    label="HOG Visualization",
    category="Features",
    description="Histogram of Oriented Gradients visualisation. CPU only — cell/block grid spans the full image.",
    outputs=["hog_image"],
    input_types={"image": "image"},
    output_types={"hog_image": "image"},
    params_config={
        "orientations": {"min": 4, "max": 16, "step": 1},
        "pixels_per_cell": {"min": 4, "max": 64, "step": 4},
        "cells_per_block": {"min": 1, "max": 8, "step": 1},
    }
)
def hog(image, orientations: int = 9, pixels_per_cell: int = 8, cells_per_block: int = 3):
    return dispatch(
        default=_hog_cpu,
        args=(image,),
        kwargs={"orientations": orientations, "pixels_per_cell": pixels_per_cell,
                "cells_per_block": cells_per_block},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        # No dask_options: HOG cell/block grid requires the full image.
        # _hog_cpu materialises any Dask input before calling skimage.
    )


# ---------------------------------------------------------------------------
# Points-output nodes — return LayerDataTuple (coords, meta, "points")
#
# dispatch() works here unchanged: _cast_array_output and
# _convert_cuda_output_to_numpy both recurse into tuples and leave non-array
# elements (dicts, strings) untouched, so the tuple passes through correctly.
#
# Dask chunking is not applicable (blob/peak algorithms require the full image).
# The CPU wrappers call _to_numpy to materialise any Dask input before calling
# skimage. peak_local_max adds a cuda_func for GPU acceleration.
# ---------------------------------------------------------------------------

# --- BLOB LOG ---
@register_node(
    label="Blob LoG",
    category="Features",
    description="Laplacian-of-Gaussian blob detector. Returns blob centres as a Points layer. Supports Dask (chunked).",
    outputs=["blobs"],
    input_types={"image": "image"},
    output_types={"blobs": "points"},
    params_config={
        "min_sigma": {"min": 0.5, "max": 20.0, "step": 0.5},
        "max_sigma": {"min": 1.0, "max": 50.0, "step": 1.0},
        "num_sigma": {"min": 2, "max": 20, "step": 1},
        "threshold": {"min": 0.0, "max": 1.0, "step": 0.01},
    }
)
def blob_log(image, min_sigma: float = 1.0, max_sigma: float = 10.0,
             num_sigma: int = 10, threshold: float = 0.1):
    return dispatch(
        default=skimage.feature.blob_log,
        args=(image,),
        kwargs={"min_sigma": min_sigma, "max_sigma": max_sigma,
                "num_sigma": num_sigma, "threshold": threshold},
        output_type="points",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "chunked_delayed",
            "halo_from_param": "max_sigma",
            "halo_factor": 2.0,
            "output_coord_cols": slice(None, -1),  # strip sigma column
            "point_size_col": -1,
            "point_size_mode": "blob_diameter",
            "point_kwargs": {
                "face_color": [1.0, 1.0, 1.0, 0.0],
                "border_color": "white",
                "border_width": 2.0,
                "border_width_is_relative": False,
                "opacity": 0.9,
            },
        }
    )


# --- BLOB DOG ---
@register_node(
    label="Blob DoG",
    category="Features",
    description="Difference-of-Gaussians blob detector. Returns blob centres as a Points layer. Supports Dask (chunked).",
    outputs=["blobs"],
    input_types={"image": "image"},
    output_types={"blobs": "points"},
    params_config={
        "min_sigma": {"min": 0.5, "max": 20.0, "step": 0.5},
        "max_sigma": {"min": 1.0, "max": 200.0, "step": 1.0},
        "threshold": {"min": 0.0, "max": 1.0, "step": 0.01},
    }
)
def blob_dog(image, min_sigma: float = 1.0, max_sigma: float = 10.0, threshold: float = 0.1):
    return dispatch(
        default=skimage.feature.blob_dog,
        args=(image,),
        kwargs={"min_sigma": min_sigma, "max_sigma": max_sigma, "threshold": threshold},
        output_type="points",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "chunked_delayed",
            "halo_from_param": "max_sigma",
            "halo_factor": 2.0,
            "output_coord_cols": slice(None, -1),
            "point_size_col": -1,
            "point_size_mode": "blob_diameter",
            "point_kwargs": {
                "face_color": [1.0, 1.0, 1.0, 0.0],
                "border_color": "white",
                "border_width": 2.0,
                "border_width_is_relative": False,
                "opacity": 0.9,
            },
        }
    )


# --- PEAK LOCAL MAX ---
@register_node(
    label="Peak Local Max",
    category="Features",
    description="Finds local intensity maxima above a threshold. Returns peak coordinates as a Points layer. Supports Dask (chunked).",
    outputs=["peaks"],
    input_types={"image": "image"},
    output_types={"peaks": "points"},
    params_config={
        "min_distance": {"min": 1, "max": 50, "step": 1},
        "threshold_rel": {"min": 0.0, "max": 1.0, "step": 0.01},
    }
)
def peak_local_max(image, min_distance: int = 5, threshold_rel: float = 0.1):
    return dispatch(
        default=skimage.feature.peak_local_max,
        args=(image,),
        kwargs={"min_distance": min_distance, "threshold_rel": threshold_rel},
        output_type="points",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "chunked_delayed",
            "halo_from_param": "min_distance",
            "halo_factor": 1.0,
            "point_kwargs": {
                "size": 5,
                "face_color": "white",
                "border_color": "black",
                "border_width": 1.0,
                "border_width_is_relative": False,
                "opacity": 0.9,
            },
        }
    )
