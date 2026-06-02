import inspect

import skimage.measure
import skimage.morphology

from .decorator import register_node, dispatch

# skimage 0.26 renamed min_size/area_threshold to max_size with an off-by-one:
# old min_size=N keeps objects >= N; new max_size=N keeps objects > N.
_SKIMAGE_MAX_SIZE_API = "max_size" in inspect.signature(
    skimage.morphology.remove_small_objects
).parameters


def _validate_radius(radius):
    radius = int(radius)
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius!r}.")
    return radius


def _footprint_for_image(image, radius):
    radius = _validate_radius(radius)
    ndim = int(getattr(image, "ndim", 2) or 2)
    if ndim >= 3:
        return skimage.morphology.ball(radius)
    return skimage.morphology.disk(radius)


# --- ALREADY IMPLEMENTED --- #
# - dilation                  #
# - closing                   #
# - opening                   #
# - skeletonize               #
# - remove small objects      #
# - remove small holes        #
# - white tophat              #
# - black tophat              #
# - erosion                   #
# - convex hull               #
# --- --- --- --- --- --- --- #

# --- DILATION ---
@register_node(
    label="Dilation",
    category="Morphology",
    description="Morphological dilation. Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "radius": {"min": 1, "max": 50}
    }
)
def dilation(image, radius: int = 1):
    radius = _validate_radius(radius)

    def _cpu_dilation(image, radius):
        footprint = _footprint_for_image(image, radius)
        return skimage.morphology.dilation(image, footprint=footprint)

    def _cuda_dilation(image, radius):
        import cupy as cp
        import cucim.skimage.morphology as cucim_morphology

        footprint = cp.asarray(_footprint_for_image(image, radius))
        return cucim_morphology.dilation(image, footprint=footprint)

    _cpu_dilation.__name__ = "dilation"
    _cuda_dilation.__name__ = "cucim_dilation"

    return dispatch(
        default=_cpu_dilation,
        args=(image,),
        kwargs={"radius": radius},
        cuda_func=_cuda_dilation,
        output_dtype_policy="preserve",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "radius",
            "halo_factor": 1.0,
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"boundary": "none"},
        },
    )

# --- CLOSING ---
@register_node(
    label="Binary Closing (Fill Holes)",
    category="Morphology",
    description="Binary closing with an axis-aware footprint. Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={"radius": {"min": 1, "max": 20}}
)
def binary_closing(image, radius: int = 3):
    radius = _validate_radius(radius)

    def _cpu_closing(image, radius):
        footprint = _footprint_for_image(image, radius)
        return skimage.morphology.closing(image, footprint=footprint)

    def _cuda_closing(image, radius):
        import cupy as cp
        import cucim.skimage.morphology as cucim_morphology

        footprint = cp.asarray(_footprint_for_image(image, radius))
        return cucim_morphology.closing(image, footprint=footprint)

    _cpu_closing.__name__ = "closing"
    _cuda_closing.__name__ = "cucim_closing"

    return dispatch(
        default=_cpu_closing,
        args=(image,),
        kwargs={"radius": radius},
        cuda_func=_cuda_closing,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "radius",
            "halo_factor": 2.0,
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"boundary": "none"},
        },
    )

# --- OPENING ---
@register_node(
    label="Binary Opening (Remove Noise)",
    category="Morphology",
    description="Binary opening with an axis-aware footprint. Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={"radius": {"min": 1, "max": 20}}
)
def binary_opening(image, radius: int = 3):
    radius = _validate_radius(radius)

    def _cpu_opening(image, radius):
        footprint = _footprint_for_image(image, radius)
        return skimage.morphology.opening(image, footprint=footprint)

    def _cuda_opening(image, radius):
        import cupy as cp
        import cucim.skimage.morphology as cucim_morphology

        footprint = cp.asarray(_footprint_for_image(image, radius))
        return cucim_morphology.opening(image, footprint=footprint)

    _cpu_opening.__name__ = "opening"
    _cuda_opening.__name__ = "cucim_opening"

    return dispatch(
        default=_cpu_opening,
        args=(image,),
        kwargs={"radius": radius},
        cuda_func=_cuda_opening,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "radius",
            "halo_factor": 2.0,
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"boundary": "none"},
        },
    )

# --- SKELETONIZE ---
@register_node(
    label="Skeletonize",
    category="Morphology",
    description="Skeletonize binary masks. T and C are processed independently; full spatial core is used.",
    outputs=["skeleton"],
    input_types={"image": "image"},
    output_types={"skeleton": "image"},
)
def skeletonize(image):
    return dispatch(
        default=skimage.morphology.skeletonize,
        args=(image > 0,),
        kwargs={},
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
        dask_options={
            "strategy": "pointwise",
            "numpy_chunks": "full_core",
            "rechunk": "full_core",
        },
    )

# --- EROSION (Shrink) ---
@register_node(
    label="Erosion (Shrink)",
    category="Morphology",
    description="Morphological erosion. Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={"radius": {"min": 1, "max": 50}}
)
def erosion(image, radius: int = 1):
    radius = _validate_radius(radius)

    def _cpu_erosion(image, radius):
        footprint = _footprint_for_image(image, radius)
        return skimage.morphology.erosion(image, footprint=footprint)

    def _cuda_erosion(image, radius):
        import cupy as cp
        import cucim.skimage.morphology as cucim_morphology

        footprint = cp.asarray(_footprint_for_image(image, radius))
        return cucim_morphology.erosion(image, footprint=footprint)

    _cpu_erosion.__name__ = "erosion"
    _cuda_erosion.__name__ = "cucim_erosion"

    return dispatch(
        default=_cpu_erosion,
        args=(image,),
        kwargs={"radius": radius},
        cuda_func=_cuda_erosion,
        output_dtype_policy="preserve",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "radius",
            "halo_factor": 1.0,
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"boundary": "none"},
        },
    )

# --- WHITE TOPHAT ---
@register_node(
    label="White Tophat (Bright Spots)",
    category="Morphology",
    description="White tophat background subtraction. Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["tophat"],
    input_types={"image": "image"},
    output_types={"tophat": "image"},
    params_config={"radius": {"min": 1, "max": 50}}
)
def white_tophat(image, radius: int = 15):
    radius = _validate_radius(radius)

    def _cpu_white_tophat(image, radius):
        footprint = _footprint_for_image(image, radius)
        return skimage.morphology.white_tophat(image, footprint=footprint)

    def _cuda_white_tophat(image, radius):
        import cupy as cp
        import cucim.skimage.morphology as cucim_morphology

        footprint = cp.asarray(_footprint_for_image(image, radius))
        return cucim_morphology.white_tophat(image, footprint=footprint)

    _cpu_white_tophat.__name__ = "white_tophat"
    _cuda_white_tophat.__name__ = "cucim_white_tophat"

    return dispatch(
        default=_cpu_white_tophat,
        args=(image,),
        kwargs={"radius": radius},
        cuda_func=_cuda_white_tophat,
        output_dtype_policy="preserve",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "radius",
            "halo_factor": 2.0,
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"boundary": "none"},
        },
    )

# --- BLACK TOPHAT ---
@register_node(
    label="Black Tophat (Dark Spots)",
    category="Morphology",
    description="Black tophat background subtraction. Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["tophat"],
    input_types={"image": "image"},
    output_types={"tophat": "image"},
    params_config={"radius": {"min": 1, "max": 50}}
)
def black_tophat(image, radius: int = 15):
    radius = _validate_radius(radius)

    def _cpu_black_tophat(image, radius):
        footprint = _footprint_for_image(image, radius)
        return skimage.morphology.black_tophat(image, footprint=footprint)

    def _cuda_black_tophat(image, radius):
        import cupy as cp
        import cucim.skimage.morphology as cucim_morphology

        footprint = cp.asarray(_footprint_for_image(image, radius))
        return cucim_morphology.black_tophat(image, footprint=footprint)

    _cpu_black_tophat.__name__ = "black_tophat"
    _cuda_black_tophat.__name__ = "cucim_black_tophat"

    return dispatch(
        default=_cpu_black_tophat,
        args=(image,),
        kwargs={"radius": radius},
        cuda_func=_cuda_black_tophat,
        output_dtype_policy="preserve",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "radius",
            "halo_factor": 2.0,
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"boundary": "none"},
        },
    )

# --- CONVEX HULL ---
@register_node(
    label="Convex Hull",
    category="Morphology",
    outputs=["hull"]
)
def convex_hull(image):
    return skimage.morphology.convex_hull_image(image > 0)

# --- REMOVE SMALL OBJECTS ---
@register_node(
    label="Remove Small Objects",
    category="Morphology",
    description="Remove connected components smaller than min_size. T and C are independent; full spatial core per connectivity check.",
    outputs=["cleaned_mask"],
    input_types={"image": "image"},
    output_types={"cleaned_mask": "image"},
    params_config={
        "min_size": {"min": 10, "max": 1000, "step": 10},
        "connectivity": {"min": 1, "max": 3},
    }
)
def remove_small_objects(image, min_size: int = 64, connectivity: int = 1):
    min_size = int(min_size)
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size!r}.")
    size_kwarg = {"max_size": min_size - 1} if _SKIMAGE_MAX_SIZE_API else {"min_size": min_size}
    return dispatch(
        default=skimage.morphology.remove_small_objects,
        args=(image > 0,),
        kwargs={"connectivity": int(connectivity), **size_kwarg},
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
        dask_options={
            "strategy": "pointwise",
            "numpy_chunks": "full_core",
            "rechunk": "full_core",
        },
    )

# --- REMOVE SMALL HOLES ---
@register_node(
    label="Remove Small Holes",
    category="Morphology",
    description="Fill holes smaller than area_threshold. T and C are independent; full spatial core per connectivity check.",
    outputs=["filled_mask"],
    input_types={"image": "image"},
    output_types={"filled_mask": "image"},
    params_config={
        "area_threshold": {"min": 10, "max": 1000, "step": 10},
        "connectivity": {"min": 1, "max": 3},
    }
)
def remove_small_holes(image, area_threshold: int = 64, connectivity: int = 1):
    area_threshold = int(area_threshold)
    if area_threshold < 1:
        raise ValueError(f"area_threshold must be >= 1, got {area_threshold!r}.")
    size_kwarg = {"max_size": area_threshold - 1} if _SKIMAGE_MAX_SIZE_API else {"area_threshold": area_threshold}
    return dispatch(
        default=skimage.morphology.remove_small_holes,
        args=(image > 0,),
        kwargs={"connectivity": int(connectivity), **size_kwarg},
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
        dask_options={
            "strategy": "pointwise",
            "numpy_chunks": "full_core",
            "rechunk": "full_core",
        },
    )

# --- LABEL OBJECTS ---
@register_node(
    label="Label Objects",
    category="Morphology",
    description="Label connected components. T and C are independent; full spatial core per label operation.",
    outputs=["labels"],
    input_types={"image": "image"},
    output_types={"labels": "labels"},
    params_config={"connectivity": {"min": 1, "max": 3}},
)
def label_objects(image, connectivity: int = 1):
    return dispatch(
        default=skimage.measure.label,
        args=(image > 0,),
        kwargs={"connectivity": int(connectivity)},
        cuda_function="cucim.skimage.measure.label",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["connectivity"],
        output_dtype_policy="label",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
        dask_options={
            "strategy": "pointwise",
            "numpy_chunks": "full_core",
            "rechunk": "full_core",
        },
    )
