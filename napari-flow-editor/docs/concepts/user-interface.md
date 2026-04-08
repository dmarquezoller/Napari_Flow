# User Interface

## Goal

Explain how the plugin UI is organized so users can build, debug, and run pipelines efficiently.

## Main Layout

The plugin is split into five working areas:

1. **Toolbar (top)** for graph-level actions (`Add Node`, `Add Control`, `Remove Node`, `File`, macro actions).
2. **Node Properties panel** for editing parameters and inspecting connections of the selected node.
3. **Graph canvas** for visual pipeline authoring (nodes + exec/data links).
4. **Execution log panel** for runtime output, warnings, and tracebacks.
5. **Context action bar** for interactive nodes (Run/Cancel) and in-graph floating controls (for example loop stop and macro navigation controls).

## Visual Language

### Nodes

- Node title indicates operation name.
- The top-right status light indicates runtime state:
  - Gray: dirty / needs recomputation.
  - Yellow: running.
  - Green: completed / cached.
  - Red: failed.

### Sockets

- **Exec sockets**: white triangular sockets.
  - Left side: exec input.
  - Right side: exec output.
  - Control nodes can expose multiple named exec outputs (for example `loop_body`, `completed`).
- **Data sockets**: circular sockets, color-coded by data type when defined.
  - Untyped (`any`) sockets appear neutral/gray.

### Edges

- White edges represent execution order.
- Colored edges represent data transport.
- Edge selection highlights the item and allows delete with keyboard.

## Core Interactions

1. **Select** a node or edge with left-click.
2. **Move** nodes by dragging.
3. **Connect** sockets by drag from output to input.
4. **Multi-select** nodes with `Shift+click` (or drag selection if available in your backend).
5. **Delete** selected nodes or edges with `Delete/Supr`.
6. **Pan/zoom** the canvas with view controls (mouse wheel and drag behavior depends on platform/input device).

## Parameter Editing

- Parameters for the selected node are rendered in the properties panel.
- Widget type is inferred from node metadata (`int`, `float`, `bool`, `enum`, `path`, `table`, etc.).
- Dynamic parameter bindings (batch mode) appear only when relevant to the active pipeline context.

## Interactive UX

- Interactive nodes pause execution and expose controls in the action bar.
- The user completes the requested action (for example draw ROI or choose layer), then clicks `Run` or `Cancel`.
- This keeps interactive decisions visible in the same workspace, rather than hidden in separate dialogs.

## UI Design Principles

1. **Immediate feedback**: status lights and logs must explain what is happening now.
2. **Low ambiguity**: exec and data paths are visually distinct.
3. **Progressive complexity**: simple pipelines stay simple, advanced control flow is still available.
4. **Non-blocking execution**: heavy computation should run off the main UI thread.
