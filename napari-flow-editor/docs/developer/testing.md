# Testing

## Goal

Define testing strategy for confidence without slowing iteration.

## Test Layers

### Unit Tests

- Focus: deterministic behavior in small isolated pieces.
- Typical scope:
  - dispatch/layout helpers
  - metadata envelope behavior
  - cache signature logic
  - node-specific logic in isolation

### Integration Tests

- Focus: multi-node execution behavior and orchestration correctness.
- Typical scope:
  - exec traversal from begin nodes
  - loop execution semantics
  - batch CSV behavior
  - plotting/video node integration paths

### Manual UI Tests

- Focus: interaction quality and end-user behavior.
- Typical scope:
  - node add/remove/connect/disconnect
  - edge deletion with keyboard
  - interactive node run/cancel flow
  - macro enter/exit behavior
  - plotting and export UX
  - long-path input responsiveness

## Recommended Commands

Run from `napari-flow-editor/`:

```bash
python -m pytest src/napari_flow_editor/tests -q
```

Targeted runs:

```bash
python -m pytest src/napari_flow_editor/tests/test_execution_engine.py -q
python -m pytest src/napari_flow_editor/tests/test_loop_execution.py -q
python -m pytest src/napari_flow_editor/tests/test_batch_execution.py -q
python -m pytest src/napari_flow_editor/tests/test_video_node.py -q
```

Keyword filtering:

```bash
python -m pytest src/napari_flow_editor/tests -k "dispatch or cache or loop" -q
```

Regenerate node library before tests if node metadata changed:

```bash
python src/napari_flow_editor/generate_library.py
```

## Suggested PR Validation Sequence

1. Run targeted tests for changed area.
2. Run full test suite.
3. Run one manual smoke test in napari:
   - simple pipeline (`Begin -> Get Layer -> Gaussian Blur`)
   - one control-flow pipeline (loop or batch)
   - one output workflow (save/plot/video)
4. Confirm docs entry changed if behavior is user-visible.
