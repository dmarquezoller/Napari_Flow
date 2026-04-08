# Common Errors

## Goal

Catalog recurrent errors with direct fixes.

## Error 1: Node Runs But No Output Layer Appears

### Symptom

- You run a valid pipeline and no new output layer is shown in napari.

### Likely Cause

- The node returned data, but the output metadata (`layer_type`, `name`, `metadata`) was not propagated correctly.
- The node is only forwarding data to downstream nodes and is not intended to add a viewer layer.

### Fix

1. Check the node definition in `flow_nodes/*.py` and verify the return payload follows the expected output contract.
2. Confirm the output socket is connected to a node that adds/updates a viewer layer.
3. Inspect execution log lines for `ADDING:` messages to verify whether layer creation was attempted.

### Prevention

1. Add a simple output node (for example a plotting or save node) while debugging to verify data flow.
2. Keep output naming explicit in node functions.

## Error 2: Socket Looks Disconnected But Is Still Marked Connected

### Symptom

- You remove an edge visually, but the input still appears as connected and blocks reconnection.

### Likely Cause

- The edge graphics item was removed but the internal socket-edge reference was not fully cleared.

### Fix

1. Select and delete the stale edge again with `Delete/Supr`.
2. Reselect the destination node and check `Connections` in the properties panel.
3. If still stale, save pipeline JSON and reload it to force graph reconstruction.

### Prevention

1. Prefer explicit edge selection + delete over partial drag disconnects.
2. Run edge cleanup in graph mutation handlers whenever implementing UI edge behavior.

## Error 3: `AttributeError: 'MultiScaleData' object has no attribute 'astype'`

### Symptom

- A filter node fails on multiscale data with a traceback ending in `image.astype(...)`.

### Likely Cause

- The node backend received napari `MultiScaleData` directly instead of an individual level array.

### Fix

1. Ensure dispatch unwraps multiscale inputs level-by-level before calling NumPy/skimage functions.
2. Confirm the Dask path handles per-level arrays and rebuilds a multiscale output list.
3. Re-test with both workflows:
   - `Open OME-Zarr -> Select Layer -> Filter`
   - `Get Layer -> Filter` on a multiscale viewer layer

### Prevention

1. Add tests that pass both plain arrays and `MultiScaleData`.
2. Keep multiscale-specific handling in shared dispatcher utilities, not per-node ad hoc code.

## Error 4: Plotly Output Says Qt WebEngine Is Missing

### Symptom

- Plot output panel shows a message that interactive Plotly requires Qt WebEngine.

### Likely Cause

- `QtWebEngineWidgets` is unavailable in the current Qt binding or was initialized too late.

### Fix

1. Install Qt WebEngine for your binding/environment.
2. If running from shell entrypoints where WebEngine cannot initialize early, use browser fallback mode.
3. Keep static image fallback enabled for environments without WebEngine.

### Prevention

1. Treat interactive Plotly as optional capability.
2. Document fallback behavior for users in constrained environments.

## Error 5: File/Folder Picker Is Slow Or Shows Unexpected Scope

### Symptom

- Path dialogs take a long time to populate or appear constrained to a subset of folders.

### Likely Cause

- Dialog options were configured with restrictive filters.
- Desktop/Wayland focus quirks make dialog updates appear delayed.
- Background refresh or signal storms block UI responsiveness.

### Fix

1. Use the same dialog options pattern as stable path nodes (for example `Open OME-Zarr`).
2. Disable overly restrictive file mode/filter options unless required.
3. Profile signal handlers triggered on parameter changes and reduce unnecessary full refreshes.

### Prevention

1. Reuse one common helper for file/folder dialogs.
2. Keep parameter-update handlers lightweight and debounced when possible.

## Error 6: Interactive Node Fails If User Clicks Run Without Input

### Symptom

- Interactive workflow crashes or errors if required user selection was not made first.

### Likely Cause

- Missing guard clauses for empty ROI/selection/state.

### Fix

1. Validate interactive state before executing node logic.
2. Show a clear message in the action bar (`Draw ROI first`, `Select layer first`, etc.).
3. Keep Run disabled until minimum required input is present when feasible.

### Prevention

1. Treat interactive nodes as state machines: `Waiting -> Ready -> Running`.
2. Add tests for empty/invalid user action paths.

## Error 7: Parameter Fields Temporarily Refuse Keyboard Input

### Symptom

- Text/number fields become unresponsive after heavy actions (load, run, table row add).

### Likely Cause

- Focus jumps caused by aggressive refresh calls or modal-like event handling.

### Fix

1. Minimize full-widget refreshes on every change.
2. Restore focus explicitly after expensive updates when needed.
3. Move expensive recomputation out of immediate UI callbacks.

### Prevention

1. Separate model updates from UI repaint/update cycles.
2. Audit and throttle event-driven refresh points.

## Still Stuck?

Use this minimum diagnostic bundle when reporting an issue:

1. Pipeline screenshot.
2. Full execution traceback.
3. Node JSON snippet involved.
4. OS + Python + napari + plugin versions.
5. Whether issue reproduces on a fresh session.
