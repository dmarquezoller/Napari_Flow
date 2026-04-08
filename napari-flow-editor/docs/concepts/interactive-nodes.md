# Interactive Nodes

## Goal

Explain interaction lifecycle for nodes that require user input at runtime.

## Runtime Flow

Interactive nodes pause worker execution and request user input on the main UI thread.

Typical flow:
1. Worker reaches interactive node.
2. UI shows interaction card/controls.
3. User provides input and clicks `Run`, or clicks `Cancel`.
4. Worker resumes with provided interaction payload.

## Inside Loops

If an interactive node is inside a loop body:
- interaction is requested on each iteration
- `Until confirm` loops can be stopped with the loop stop control
- loop body nodes are reset between iterations to reflect a fresh cycle

## In Batch Mode

Interactive nodes are allowed in batch mode.

Behavior:
- a warning is shown before execution starts
- user may need to interact once per CSV row
- useful for controlled semi-automatic workflows

## Cancellation Behavior

- `Cancel` on an interactive step aborts that execution path.
- In loop stop scenarios, the engine can interrupt pending interaction and exit loop immediately.

## Recommended Usage

1. Use interactive nodes for decisions that cannot be pre-encoded as parameters.
2. For fully unattended runs, avoid interactive nodes in batch pipelines.
3. Keep prompts specific so operators know exactly what to do.
