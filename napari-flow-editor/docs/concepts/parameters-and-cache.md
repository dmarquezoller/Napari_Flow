# Parameters and Cache

## Goal

Explain when nodes recompute and when cached results are reused.

## Parameter Changes

When a parameter changes on a node:
1. That node is marked stale.
2. Downstream nodes are marked stale recursively.
3. The next run recomputes stale nodes.

This prevents mixing old outputs with new parameter settings.

## Upstream Dependency Changes

When an upstream output changes:
- all dependent nodes become stale
- stale status propagates through data and exec successors

Status colors:
- gray: stale
- yellow: running
- green: up-to-date
- red: error during execution

## Practical Tips

1. After major rewiring, run once and confirm all expected nodes turned green.
2. If a result looks unchanged after edits, verify the edited node actually became gray before run.
3. In loop and batch workflows, expect forced recompute behavior for loop body / per-row runs.
4. Keep node names meaningful so logs are easier to trace.

## Signature-Based Reuse

Cache reuse is based on a node signature combining:
- node parameter values
- upstream signatures

If signature is unchanged, execution may be skipped and cached outputs reused.
