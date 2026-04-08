# Video Rendering

## Goal

Explain how video generation works from viewer state and instructions.

## Supported Motion Types

Video rendering is driven by instruction rows in the `Make Video` node.

Current operations:
- `rotate`
- `sweep`

Each row defines:
- operation type
- view mode (`keep`, `2d`, `3d`)
- axis details (for sweep/rotation)
- numeric range (`start`, `end`, `step`)

## Viewer State Dependencies

Rendering uses current viewer state as baseline.

Key points:
- output reflects what is visible in the viewer
- instruction rows modify view/dims over time
- viewer state is restored after rendering completes

## Output Configuration

Main output parameters:
- `fps`
- `format` (`.mp4` or `.gif`)
- `folder`
- `filename`

Execution model:
- `Make Video` is interactive (requires clicking Run in interaction controls)
- render occurs on main thread UI side
- on success, output path and frame count are reported

## Recommended Workflow

1. Prepare viewer (layers, colors, visibility, camera framing).
2. Configure instruction rows with small test ranges first.
3. Render short test clip.
4. Adjust fps/step/range for final export.

## Common Issues

- Empty/incorrect output: check viewer visibility and instruction axis settings.
- Unclear motion: reduce step size and verify axis selection.
- Large files: lower fps or shorten segment ranges.
