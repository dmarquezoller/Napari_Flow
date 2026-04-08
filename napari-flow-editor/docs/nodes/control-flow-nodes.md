# Control Flow Nodes

## Goal

Document Begin, Begin Batch, Loop, and control semantics.

## Begin

Role:
- start node for a single-run execution thread

Required connections:
- must connect exec output to first runnable node

Parameters:
- none

Failure modes:
- if not connected, pipeline validation blocks run

## Begin Batch

Role:
- start node for CSV-driven multi-run execution

Required connections:
- same exec-thread role as Begin

Parameters:
- `csv_path` (required)

Failure modes:
- missing CSV path
- unreadable CSV
- missing headers

## Loop

Role:
- define repeated execution over a branch

Required connections:
- one exec input
- `loop_body` output to repeated branch
- `completed` output to continuation branch

Parameters:
- `mode` (`N times`, `Until confirm`)
- `iterations` (for `N times`)

Failure modes:
- missing body/completed branch wiring (warnings)
- unintended graph behavior if loop body is not isolated logically

## Design Notes

1. Keep control flow visually clear before optimizing data branches.
2. Always check that your intended branch is connected to the expected loop output.
3. Use only one start node (`Begin` or `Begin Batch`) per pipeline.
