# Macros

## Goal

Explain collapsing and expanding groups of nodes for readability.

## What A Macro Is

A macro is a collapsed visual group of nodes.

- It helps reduce graph clutter in complex pipelines.
- It exposes boundary sockets so the macro can still connect to outside nodes.
- Execution still runs the underlying internal nodes.

In other words: macro is primarily a UI abstraction over a real node subset.

## Creating A Macro

Typical workflow:
1. Select multiple nodes in the graph.
2. Collapse selection into a macro.
3. Macro card appears with grouped identity.

Restrictions:
- Begin/Begin Batch nodes cannot be collapsed into macros.
- Nodes already inside another macro group cannot be re-collapsed directly.

## Editing And Expanding

- Click/select macro to inspect high-level details in properties panel.
- Enter macro view to edit internal graph nodes and connections.
- Return to main graph when done.

## Best Practices

1. Group by functional stage, not by random proximity.
2. Name macros with intent (`Preprocess`, `Segmentation`, `Export`).
3. Keep macro boundaries stable to preserve readability and reuse.
