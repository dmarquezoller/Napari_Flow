# Quickstart

## Audience

For users who want a first successful pipeline in under five minutes.

## What You Will Build

A minimal pipeline:
`Begin -> Get Layer -> Gaussian Blur`

## Prerequisites

- napari is installed and running
- Flow Editor is installed (see `Getting Started -> Installation`)
- one image layer is loaded (for example, `astronaut`)

## Step 1: Open Flow Editor

1. Open napari.
2. Open Flow Editor from `Plugins -> Add Dock Widget -> Flow Editor`.
3. Confirm the Flow Editor panel is visible.

## Step 2: Add Nodes

1. Add `Begin`.
2. Add `Get Layer`.
3. Add `Gaussian Blur`.

`<screenshot: graph with three nodes, not connected>`

## Step 3: Connect Exec Flow

1. Connect `Begin` exec output to `Gaussian Blur` exec input.

`<screenshot: white exec edge connected>`

## Step 4: Connect Data Flow

1. Connect `Get Layer:data_out` to `Gaussian Blur:image`.

`<screenshot: data edge connected>`

## Step 5: Set Parameters

- `Get Layer.layer_name = astronaut` (or your current layer name)
- `Gaussian Blur.sigma = 1.0`
- `Gaussian Blur.mode = nearest`

## Step 6: Run

1. Click `Run Pipeline`.
2. Confirm a new layer appears in the viewer.

## Expected Result

- A processed output layer appears (usually `Gaussian Blur Output`).
- The executed nodes show successful status.

## If Something Fails

- `Layer not found`: verify `Get Layer.layer_name` matches exactly.
- `Missing required input`: check data edge into Gaussian Blur.
- `No execution`: check Begin is connected to the exec thread.

## Next Step

- `Core Concepts -> Execution Flow`
- `Core Concepts -> Batch Processing`
