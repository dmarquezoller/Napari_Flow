# Graph Model

## Goal

Define how nodes, sockets, and edges represent computation.

## Main Elements

### Node

A node is a processing unit with:
- a node type (`get_layer`, `gaussian_blur`, `loop_control`, etc.)
- parameters
- zero or more data inputs/outputs
- zero or more exec sockets

### Data Socket

Data sockets move values between nodes (arrays, tables, layer bundles, etc.).

- Input data sockets are on the left side.
- Output data sockets are on the right side.
- Data sockets are typed (for example `image`, `labels`, `table`, `layers`, `any`).
- Type compatibility is validated when connecting edges.

### Exec Socket

Exec sockets define runtime order.

- Exec sockets are the white logic sockets.
- Most runnable nodes use one exec input and one exec output.
- Special nodes:
  - `Begin`: exec output only
  - `Begin Batch`: exec output only
  - `Loop`: one exec input and two exec outputs (`loop_body`, `completed`)

### Edge

An edge connects sockets:
- data edge: carries data values
- exec edge: carries execution order only

Both edge types are necessary for most non-trivial pipelines.

## Typed Data Model

Typed sockets help prevent invalid graphs early.

Examples:
- `image -> image` is valid.
- `table -> image` is invalid unless node expects `table`.
- `any` can connect to or from any typed socket.

## Control vs Data Separation

The graph intentionally separates:
- **Control**: what runs and when
- **Data**: what values are consumed/produced

This separation is what makes loops, batch execution, and data-only sources predictable.

## Mental Model

Think of each run as:
1. Start from exactly one control entry (`Begin` or `Begin Batch`).
2. Follow exec edges to decide node order.
3. For each node, resolve its data inputs from connected data edges.
4. Execute and cache outputs.

```text
Control path:
Begin --> Gaussian Blur --> Adjust Gamma --> Save Image

Data path:
Get Layer -(image)-> Gaussian Blur -(image)-> Adjust Gamma -(image)-> Save Image
```
