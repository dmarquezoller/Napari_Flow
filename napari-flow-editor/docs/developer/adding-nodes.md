# Adding Nodes

## Goal

Provide a clear checklist for adding new flow nodes safely.

## Steps

1. **Create or choose a module**
   - Add your function in `src/napari_flow_editor/flow_nodes/<module>.py`.
   - Keep related operations grouped by domain (`filters`, `measure`, `video`, etc.).

2. **Register with `@register_node`**
   - Minimum required metadata:
     - `label`
     - `category`
     - `outputs`
   - Typical optional metadata:
     - `params_config`
     - `description`
     - `interactive`
     - `logic`
     - `input_types`, `output_types`
     - `dynamic_output_types`

3. **Design function signature correctly**
   - Parameters **without defaults** become required data input sockets.
   - Parameters **with defaults** become editable UI parameters.
   - Reserve `interaction` kwarg for interactive-node runtime payloads.

4. **Add parameter widget config (if needed)**
   - Use `params_config` to define min/max/step/options/widget behavior.
   - Ensure defaults are valid and reflect expected scientific usage.

5. **Declare socket typing**
   - Use `input_types` / `output_types` maps for strict type-safe wiring.
   - Use `dynamic_output_types` for nodes whose output type depends on runtime selection.

6. **Configure logic sockets (optional)**
   - Default logic config enables both exec input and output, multi-connect allowed.
   - Override with:
     - `{"in": bool, "out": bool, "allow_multi_in": bool, "allow_multi_out": bool}`

7. **Use dispatch() for backend-aware execution**
   - Call `dispatch(...)` directly inside the node body — no decorator wrapper needed.
   - The dispatcher selects the backend (CPU / Dask / CUDA) automatically based on input type and available hardware.

8. **Regenerate node library**
   - Run from repo root:
   - `python src/napari_flow_editor/generate_library.py`
   - Confirm your node appears in `src/napari_flow_editor/node_library.json`.

9. **Test**
   - Add/extend tests under `src/napari_flow_editor/tests/`.
   - Run focused tests first, then full suite.

10. **Update docs**
    - Add or update node entry in:
      - `docs/nodes/*.md`
      - Any relevant guide/concept page if behavior is new.

## Minimal Example

```python
import functools
import skimage.morphology
from napari_flow_editor.flow_nodes.decorator import register_node, dispatch

@register_node(
    label="Erosion",
    category="Morphology",
    outputs=["image_out"],
    params_config={"radius": {"type": "int", "min": 1, "max": 25, "step": 1}},
    input_types={"image": "image"},
    output_types={"image_out": "image"},
)
def erosion(image, radius: int = 1):
    footprint = skimage.morphology.disk(radius)
    fn = functools.partial(skimage.morphology.erosion, footprint=footprint)
    return dispatch(
        default=fn,
        args=(image,),
        kwargs={},
        cuda_function="cucim.skimage.morphology.erosion",
        cuda_arg_names=["image"],
        cuda_kwarg_names=[],
        output_dtype_policy="preserve",
        backend="auto",
        pyramid_strategy="per_level",
        dask_options={"strategy": "neighborhood", "halo_from_param": "radius"},
    )
```

> **Note**: The footprint is baked into the callable via `functools.partial` — never pass numpy arrays directly in `kwargs` to `dispatch()`. The `maybe_promote_numpy_to_dask` step inside the dispatcher would convert them to Dask arrays using image-shaped chunks, which breaks non-image arrays like footprints.

## Common Pitfalls

1. Forgetting to regenerate `node_library.json` after adding metadata.
2. Mismatch between declared socket names and function arguments/outputs.
3. Interactive node missing robust behavior for cancel/empty input cases.
4. Overly permissive `any` types where strict typing would prevent user errors.
5. Passing a numpy array (e.g. `footprint`) in `kwargs` to `dispatch()` — use `functools.partial` to bake it into the callable instead.
6. Hardcoding output dtypes in the node — use `output_dtype_policy` (`"image_float"`, `"bool"`, `"preserve"`) so the dispatcher handles dtype consistently across backends.
