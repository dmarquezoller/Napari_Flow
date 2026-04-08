# Execution Flow

## Goal

Explain what determines run order and why some nodes run while others do not.

## Core Rules

1. Execution starts from exactly one start node:
   - `Begin` or `Begin Batch`
2. Exec sockets define run order.
3. Data sockets define value dependencies.
4. A node not reachable from the exec thread does not execute.
5. Cycles are disallowed except through intended loop behavior.
6. Data-only source nodes may still compute lazily when downstream nodes request their outputs.

## Exec Flow vs Data Flow

- Exec flow (logic sockets) answers: "when does this node run?"
- Data flow (data sockets) answers: "what data does this node consume?"

```text
Exec thread:
Begin --> Gaussian Blur --> Save Image

Data dependencies:
Get Layer -(image)-> Gaussian Blur -(image)-> Save Image
```

## Begin vs Begin Batch

### Begin

Runs the connected exec thread once.

### Begin Batch

Runs the same exec thread once per CSV row.

- Row context is applied at runtime.
- Dynamic parameters use values from the current row.
- Node caches are reset between rows to avoid stale outputs.

## Loop Semantics

- `loop_body`: repeated branch.
- `completed`: continuation after loop ends.

```text
Begin --> Loop
           |-- loop_body --> Interactive Crop --> Save Image --> (back to Loop)
           |-- completed --> Next Node
```

## Batch Semantics

One row equals one full pipeline run.

For each row:
1. Batch row context is loaded.
2. Dynamic parameter values are resolved.
3. The exec path runs.

## Interactive Nodes

- Interactive nodes can pause execution awaiting user input.
- In loops, interaction can repeat by iteration.
- In batch mode, interaction can repeat by row and a warning is shown before run.

## Status Signals During Run

Most nodes expose a status light:
- gray: stale / needs recompute
- yellow: currently executing
- green: completed or cached
- red: execution error

Control-flow nodes intentionally emphasize structure over cache-light state.

## Validation Behavior

- Missing start node: blocking error.
- More than one start node: blocking error.
- Missing required data input: blocking error.
- Missing loop body/completed wiring: warning.

## Debug Checklist

1. Is start node connected to exec thread?
2. Are required data inputs connected?
3. Is the target node on the reachable exec path?
4. Are loop outputs wired to the correct sockets?

## Related Pages

- `Core Concepts -> Loops and Control Flow`
- `Core Concepts -> Batch Processing`
- `Troubleshooting -> Common Errors`
