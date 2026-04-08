# Processing Nodes

## Goal

Document transformation, filtering, segmentation, and analysis nodes.

## Families

### Filters

Use for smoothing, edge detection, thresholding, and frequency-domain operations.

Representative nodes:
- `Gaussian Blur`
- `Median Filter`
- `Sobel Edge Det.`
- threshold family (`Otsu`, `Li`, `Yen`, etc.)

Typical outputs:
- processed images
- masks (`mask_out`)

### Exposure

Use for intensity remapping and contrast enhancement.

Representative nodes:
- `Adjust Gamma`
- `Rescale Intensity`
- `Equalize Histogram`
- `Equalize Adaptive`

Typical outputs:
- adjusted image data for visualization or downstream segmentation

### Morphology

Use for binary cleanup and shape refinement.

Representative nodes:
- `Binary Opening`
- `Binary Closing`
- `Remove Small Objects`
- `Remove Small Holes`
- `Skeletonize`

Typical outputs:
- cleaned masks
- morphological feature images

### Segmentation

Use for region partitioning and label generation.

Representative nodes:
- `Watershed`
- `SLIC`
- `Random Walker`
- `Felzenszwalb`
- `Clear Border`

Typical outputs:
- label images
- boundary overlays

### Transform

Use for geometric conversion and view-space transformations.

Representative nodes:
- `Rotate`
- `Rescale`
- `Downscale`
- `Warp Polar`

### Math / Utility Processing

Use for workflow-specific image operations and interaction-based transforms.

Representative nodes:
- `Blend Images`
- `Project 3D to 2D`
- `Interactive Crop`

### Measure

Use to extract or filter structured analysis results.

Representative nodes:
- `Region Properties`
- `Filter Labels by CSV`

Typical outputs:
- tables
- filtered labels

### Deep Learning

Use for model-based segmentation and tracking.

Representative nodes:
- `Cellpose Segmentation`
- `StarDist Segmentation`
- `Ultrack Tracking`
- model train/refine nodes

## Practical Selection Strategy

1. Start with deterministic classical nodes first (filters/exposure/morphology).
2. Add segmentation nodes once input quality is stable.
3. Add measurement or DL steps last when preprocessing is validated.

## Common Gotchas

- Some nodes output labels while downstream node expects image (or vice versa).
- Threshold outputs are often binary/mask outputs, not intensity images.
- Deep learning nodes can require additional environment dependencies and valid model setup.
