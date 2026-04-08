# Execution Engine Notes

## Goal

Capture non-obvious execution behavior for maintainers.

## Topics

## Exec Traversal Rules

1. Exactly one entrypoint is required: `begin` or `begin_batch`.
2. Traversal follows exec sockets only (`logic_outputs` -> `logic_inputs`).
3. Data nodes without exec input can still execute lazily when requested by downstream nodes.
4. Cycles in exec traversal are treated as errors except for explicit loop semantics in control nodes.

## Loop Handling

- `loop_control` is handled as a dedicated runtime case, not a generic cycle.
- Two named outputs are used:
  - `loop_body`: branch executed per iteration.
  - `completed`: branch executed after loop termination.
- Supported modes:
  - `N times`
  - `Until confirm`
- For loop iterations, body nodes are reset to gray status to reflect per-iteration recompute.

## Batch Row Handling

1. `begin_batch` reads CSV rows with dialect sniffing.
2. Batch context (`row`, `index`, `total`) is set for each iteration.
3. Dynamic parameter bindings are resolved from row columns to node parameters.
4. Empty CSV cells keep node static defaults.
5. Node caches are reset per batch row to avoid cross-row contamination.

## Interactive Synchronization

- Worker thread emits `interaction_request_signal`.
- Main thread collects user action and calls `provide_interaction_result`.
- Worker waits on an internal `threading.Event`.
- Loop stop requests can interrupt an active wait immediately via sentinel unblocking.

## Cache Signatures

- Signature combines:
  - serialized node parameters
  - upstream parent signatures for each connected input
- Signature hash currently uses MD5 over combined string payload.
- Cache hit: node execution is skipped and previous results reused.
- Cache miss: node executes and updates `last_signature` + `cached_results`.

## Metadata Envelope Handling

- Node results can be plain data or `(data, metadata)` envelope.
- Metadata is propagated downstream and merged when applicable.
- `get_layer` constructs rich metadata from napari layer fields and layer tuple payloads.

## Dispatch Context

- Engine pushes dispatch context before node execution via:
  - `push_dispatch_context(...)`
  - `pop_dispatch_context(...)`
- This enables backend dispatch decisions (layout/multiscale/batch-aware behavior) without leaking engine internals into node code.

## Failure Model

- Node-level failure sets node status to red and emits error/log signal.
- Critical exceptions are caught in `run()` and emitted as full traceback in execution log.
- `finished_signal` is emitted in `finally` to guarantee UI unblocking.
