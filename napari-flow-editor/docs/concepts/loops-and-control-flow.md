# Loops and Control Flow

## Goal

Describe control nodes and the exec-thread model.

## Loop Node

The `Loop` node is a control-flow node with:
- one exec input
- two exec outputs:
  - `loop_body`
  - `completed`

Supported modes:
1. `N times`: loop body runs exactly `iterations` times.
2. `Until confirm`: loop body repeats until user stops it.

Expected wiring:
- incoming exec thread enters loop input
- loop body branch contains repeated nodes
- completed branch continues normal pipeline after loop

## Validation Rules

- Missing `loop_body` connection: warning.
- Missing `completed` connection: warning.
- General exec cycles are disallowed except loop behavior represented by the loop node itself.

## Stop Behavior (`Until confirm`)

When a stop is requested:
- loop stop flag is set
- if waiting on interaction, worker is unblocked
- current loop exits and execution continues via `completed` branch (if connected)

## Examples

### Example A: Finite Loop

`Begin -> Loop(mode=N times, iterations=3)`

- `Loop.loop_body -> Interactive Crop -> Save Image`
- `Loop.completed -> Next Step`

### Example B: Until Confirm

`Begin -> Loop(mode=Until confirm)`

- `Loop.loop_body -> Interactive Crop -> Save Image`
- user presses stop when enough samples are captured
- pipeline continues through `Loop.completed`
