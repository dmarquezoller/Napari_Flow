# First Pipeline

## Goal

Build and run a complete beginner pipeline with one processing step and one saved output.

## Example Pipeline

`Begin -> Get Layer -> Gaussian Blur -> Save Image`

`<screenshot: full four-node pipeline with exec and data edges>`

## Build Steps

### 1. Load a sample image

In napari, load any image layer (for example `astronaut`).

### 2. Create nodes

Add these nodes:
1. `Begin` (Control Flow)
2. `Get Layer` (Input)
3. `Gaussian Blur` (Processing)
4. `Save Image` (Output)

### 3. Connect exec flow

Connect white exec sockets:

1. `Begin.exec_out -> Gaussian Blur.logic_in`
2. `Gaussian Blur.exec_out -> Save Image.logic_in`

### 4. Connect data flow

Connect data sockets:

1. `Get Layer.data_out -> Gaussian Blur.image`
2. `Gaussian Blur.image_out -> Save Image.image`  
   (if your socket name differs, connect Gaussian output to Save Image input)

### 5. Configure parameters

`Get Layer`
- `layer_name = <your loaded layer>`

`Gaussian Blur`
- `sigma = 1.0`
- `mode = nearest`

`Save Image`
- `folder = <output folder path>`
- `base_name = first_pipeline`
- `format = .tif` (or preferred format)

### 6. Run pipeline

Click `Run Pipeline`.

### 7. Confirm results

- A blurred output layer appears in napari.
- A saved file appears in `<output folder path>`.

## Common Mistakes

1. `Layer not found` from `Get Layer`
   - Fix: set exact layer name from napari layer list.

2. No output layer appears
   - Fix: confirm exec flow starts from `Begin` and reaches processing node.

3. Save step runs but file missing
   - Fix: verify `folder` exists and process has write permission.

4. Validation blocks run due missing inputs
   - Fix: ensure all required data sockets are connected.

## Optional Next Variation

Replace `Begin` with `Begin Batch`, bind `Get Layer.layer_name` or file path dynamically, and run the same graph over multiple rows.
