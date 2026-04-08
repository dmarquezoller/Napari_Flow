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

7. **Use compute middleware when appropriate**
   - Wrap heavy/image compute nodes with `@smart_compute(...)`.
   - Use `dispatch(...)` inside node function for backend-aware execution.

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
from napari_flow_editor.flow_nodes.decorator import register_node, smart_compute, dispatch
import skimage.filters

@register_node(
    label="Median Filter",
    category="Filters",
    outputs=["image_out"],
    params_config={"radius": {"type": "int", "min": 1, "max": 25, "step": 1}},
    input_types={"image_in": "image"},
    output_types={"image_out": "image"},
)
@smart_compute()
def median_filter(image_in, radius: int = 3):
    return dispatch(skimage.filters.median, image_in, footprint=None)
```

## Common Pitfalls

1. Forgetting to regenerate `node_library.json` after adding metadata.
2. Mismatch between declared socket names and function arguments/outputs.
3. Interactive node missing robust behavior for cancel/empty input cases.
4. Overly permissive `any` types where strict typing would prevent user errors.
