# FAQ

## Goal

Keep short answers for recurring questions.

## Questions

### Do I always need a `Begin` node?

Use exactly one execution entrypoint:

1. `Begin` for single-run pipelines.
2. `Begin Batch` for CSV-driven multi-run pipelines.

Without an entrypoint, data-only nodes may still exist in the graph, but execution flow is undefined.

### Can data nodes exist outside the exec flow line?

Yes. A data source can feed inputs to exec-connected nodes even if the source itself is not on the main white exec line.

### Why is my result unchanged after editing an upstream node?

Usually cache invalidation did not trigger as expected. Force a rerun after changing the upstream parameter and verify the downstream node cache indicators reset.

### Why are some sockets gray?

Gray typically means generic/untyped (`any`) data type or unresolved dynamic type.

### Why can’t I connect two sockets even though colors look similar?

Socket compatibility checks include:

1. Socket direction (in/out).
2. Socket family (exec vs data).
3. Data type compatibility (when strict typing is enabled).
4. Single-connection constraints for that socket.

### Does Dask compute the whole giant image immediately?

Not necessarily. Dask builds a lazy graph. Computation occurs when results are materialized (for example compute or rendering requests). Perceived responsiveness depends on data size, chunking, and what is being displayed/requested.

### Why does a filtered output look very different in contrast/color?

Often this is display metadata propagation (`contrast_limits`, `colormap`, `gamma`) rather than wrong numeric output. Compare raw arrays directly if unsure.

### Can I use interactive nodes in batch pipelines?

Yes, but it may require manual intervention per batch row depending on node behavior. If you want unattended batch runs, avoid interactive nodes or provide deterministic alternatives.

### Why does Plotly open in browser instead of inside the app?

Some environments cannot initialize Qt WebEngine in plugin context. In those cases browser fallback is used for interactive plots and static plots remain available in-app.

### How do I remove nodes and edges quickly?

1. Select node(s) and press `Delete/Supr`.
2. Select edge and press `Delete/Supr` (recommended over fragile drag-disconnect workflows).

### Why does editing parameters sometimes feel blocked for a moment?

It is usually a temporary focus/update contention after heavy UI refresh or run completion. Clicking another app/window and returning can restore focus; long-term fix is reducing aggressive refresh calls.

## Placeholder Assets

- `<FAQ_SCREENSHOT_PLACEHOLDER: add "Delete edge with Supr" UI example>`
