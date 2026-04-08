# Glossary

## Goal

Keep terminology consistent across docs.

## Terms

### Exec Flow

The control channel that determines **when** nodes run. Represented by white triangular sockets and white edges.

### Data Socket

A typed input/output socket carrying payloads between nodes. Represented by circular sockets and colored data edges.

### Dynamic Parameter

A node parameter whose value is resolved at runtime from external context (for example a CSV column in batch mode).

### Interactive Node

A node that pauses execution and waits for user action (for example ROI drawing or layer choice) before continuing.

### Macro

A grouped subgraph represented as a single node-like block for readability and reuse of workflow structure.

### Batch Row

One row from the batch CSV input. It defines one pipeline run context (path + optional per-run parameters).

### Node Library

The generated registry (`node_library.json`) describing available nodes, parameters, socket metadata, and execution paths.

### Envelope

The `(data, metadata)` payload convention used to propagate arrays together with display/spatial metadata.

### Multiscale

A pyramid representation of image data across resolutions (commonly used by OME-Zarr and large-image workflows).

### Cache Signature

A hash value derived from node parameters and upstream signatures, used to decide whether a node can reuse cached results.

### Begin / Begin Batch

Entrypoint control nodes used to start pipeline execution in single-run or CSV batch mode.

### Loop Body / Completed

Named exec outputs of the loop node:
- `loop_body`: executed each iteration.
- `completed`: executed after loop termination.
