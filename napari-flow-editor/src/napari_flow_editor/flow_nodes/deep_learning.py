import numpy as np
from .decorator import register_node

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

@register_node(
    label="Cellpose Segmentation",
    category="Deep Learning",
    outputs=["mask_layer"],
    params_config={
        "model_type": {
            "options": ["nuclei", "cyto2", "cyto3", "tissuenet", "livecell"], 
            "value": "nuclei"
        },
        "diameter": {"min": 0.0, "max": 300.0, "step": 1.0, "value": 30.0},
        "flow_threshold": {"min": 0.0, "max": 1.0, "step": 0.1, "value": 0.4},
        "process_3d": {"type": "bool", "value": False, "label": "Process as 3D Vol"},
        "use_gpu": {"type": "bool", "value": True, "label": "Use GPU"}
    }
)
def run_cellpose(image, 
                 model_type: str = 'nuclei', 
                 diameter: float = 30.0, 
                 flow_threshold: float = 0.4, 
                 process_3d: bool = False,
                 use_gpu: bool = True):
    
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not installed.")

    print(f"--- Cellpose Execution ---")
    
    # Initialize Model
    model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
    
    # --- FIX START ---
    
    # 1. Logic for True 3D Volume (Process 3D = True)
    if process_3d and image.ndim == 3:
        print("Running full 3D inference (Volumetric)...")
        
        # We must explicitly tell Cellpose that Axis 0 is Z (Depth)
        # Napari images are typically (Z, Y, X)
        results = model.eval(
            image, 
            diameter=diameter, 
            flow_threshold=flow_threshold, 
            do_3D=True,
            z_axis=0  # <--- CRITICAL FIX: Tells Cellpose "0 is Depth, not Color"
        )
        return results[0] if isinstance(results, tuple) else results

    # 2. Logic for Batch 2D (Process 3D = False)
    # If 3D image but process_3d=False, treat as list of 2D slices
    elif image.ndim == 3:
        print(f"Processing stack of {image.shape[0]} slices (Batch 2D)...")
        final_masks = []
        
        for i, slice_img in enumerate(image):
            print(f"  > Processing slice {i+1}/{image.shape[0]}...")
            res = model.eval(slice_img, diameter=diameter, flow_threshold=flow_threshold, do_3D=False)
            mask = res[0] if isinstance(res, tuple) else res
            final_masks.append(mask)
            
        return np.array(final_masks).astype(np.uint32)

    # 3. Logic for Single 2D Image
    else:
        print("Processing single 2D image...")
        res = model.eval(image, diameter=diameter, flow_threshold=flow_threshold, do_3D=False)
        return res[0] if isinstance(res, tuple) else res