# Export Script

## Goal

Explain how to export a runnable script from a pipeline.

## What Is Exported

Use `File -> Export Script` to generate a `.py` file from the current graph.

The exporter:
- walks nodes in a dependency order
- maps known node types to Python/skimage calls
- serializes node parameters into script arguments
- writes a runnable script skeleton you can edit

Typical use:
1. build and validate pipeline in UI
2. export script
3. review and adapt file paths / environment-specific pieces

## What Is Not Exported

Exported scripts are best-effort and not a full fidelity serialization.

Not fully represented:
- UI-only interactions and manual viewer operations
- complex runtime interactive behavior
- all custom node semantics (especially if no direct function mapping exists)
- exact display-state details

Expected manual edits:
1. replace placeholder input paths
2. verify function mappings for custom nodes
3. adjust imports/environment dependencies
