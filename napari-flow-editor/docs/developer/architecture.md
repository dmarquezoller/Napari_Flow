# Architecture

## Goal

Describe core components and how they interact.

## Main Modules

| Module | Path | Responsibility |
|---|---|---|
| Plugin entry/UI | `src/napari_flow_editor/napari_plugin_v2.py` | Main widget, graph scene/view, node/edge graphics, toolbar/actions, napari integration hooks |
| Execution engine | `src/napari_flow_editor/execution_engine.py` | Runtime traversal of exec graph, node execution, cache signatures, loop/batch behavior, interactive synchronization |
| Node decorators/dispatch | `src/napari_flow_editor/flow_nodes/decorator.py` | `@register_node`, `@smart_compute`, backend dispatch (NumPy/Dask/CUDA path abstraction), socket metadata normalization |
| Node implementations | `src/napari_flow_editor/flow_nodes/*.py` | Domain operations exposed as visual nodes |
| Library generator | `src/napari_flow_editor/generate_library.py` | Reflection-based extraction of node metadata into `node_library.json` |
| Script export | `src/napari_flow_editor/script_generator.py` | Converts visual pipelines to Python script stubs |
| Specialized widgets | `src/napari_flow_editor/widgets/*.py` | Reusable UI controls (tables, plotting widgets, etc.) |
| Tests | `src/napari_flow_editor/tests/*.py` | Unit/integration checks for engine, dispatch, nodes, batch, loop, video, plotting |

## Data and Control Paths

```text
User (napari UI)
   |
   v
FlowEditor (QWidget)
   |-- Graph authoring (FlowScene/Node/Socket/Connection)
   |-- Parameter editing / action bars
   |
   +--> RUN PIPELINE
           |
           v
      ExecutionWorker (QThread worker object)
           |
           |-- loads node definitions from node_library.json
           |-- traverses exec edges (white sockets)
           |-- resolves data dependencies from data edges (typed sockets)
           |-- applies cache and dynamic parameter bindings
           |-- runs node function (decorator middleware + dispatch)
           |
           +--> emits signals back to UI:
                 - node status updates
                 - logs / errors
                 - result layers / plotting outputs
                 - interactive requests
```

## Runtime Sequence (High-Level)

1. User builds graph in `FlowScene`.
2. User clicks run.
3. Engine finds single entrypoint (`begin` or `begin_batch`).
4. Engine walks exec thread, resolving required data inputs per node.
5. Node function executes through decorator middleware.
6. Results and metadata are cached and propagated to downstream nodes.
7. Signals update status lights, logs, and viewer outputs.

## Key Design Decisions

1. **Exec and data are separate graph channels** for clarity and deterministic traversal.
2. **Decorator-driven nodes** keep new operation onboarding low-friction.
3. **Reflection-generated library** avoids manual UI registration drift.
4. **Threaded worker model** keeps UI responsive during heavy computation.
5. **Metadata-aware envelopes** preserve viewer semantics (display/axes context) through pipelines.
