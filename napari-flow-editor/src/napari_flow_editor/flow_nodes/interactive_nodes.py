"""
Interactive nodes that require user input from Napari's GUI.

These nodes demonstrate the use of the generic InteractiveGUIWorker
to pause pipeline execution and collect user input.
"""

from .decorator import register_node
from napari_flow_editor.interactive import interactive_worker
import numpy as np
import napari
import threading


@register_node(
    label="Interactive Seed Points",
    category="Interactive",
    outputs=["seed_coordinates"],
    params_config={
        "point_size": {"type": "float", "min": 1, "max": 50, "step": 1, "default": 10, "label": "Point Size"}
    },
    interactive=True
)
def interactive_seed_points(image_input, point_size=10):
    """
    Interactive node that allows the user to click seed points on an image.
    
    The user clicks points in a Napari Points layer, and the node returns
    the coordinates of all clicked points. This is useful for seeding
    segmentation algorithms, marking features, or selecting regions of interest.
    
    Args:
        image_input: Input image (can be a tuple with metadata)
        point_size: Size of the point markers
        
    Returns:
        numpy array of shape (N, ndim) containing the coordinates of N seed points
    """
    # --- 1. UNPACK INPUT ---
    image = image_input
    meta = {}
    if isinstance(image_input, tuple):
        if len(image_input) >= 2:
            image = image_input[0]
            if isinstance(image_input[1], dict):
                meta = image_input[1]
    
    # Validate image
    if image is None:
        raise ValueError("No input image provided")
    
    # --- 2. INTERACTION (Click Points) ---
    LAYER_NAME = "---- CLICK SEED POINTS (Press Enter when done) ----"
    print(f">> Please click seed points in the '{LAYER_NAME}' layer.")
    print(">> Press Enter in the layer list when finished.")
    
    def setup_points_layer():
        """Create a Points layer for the user to click seed points."""
        viewer = napari.current_viewer()
        if not viewer:
            raise ValueError("No Napari viewer available")
        
        # Remove existing layer if present
        if LAYER_NAME in viewer.layers:
            viewer.layers.remove(LAYER_NAME)
        
        # Create a Points layer
        points_layer = viewer.add_points(
            name=LAYER_NAME,
            ndim=image.ndim,
            size=point_size,
            edge_color='lime',
            face_color='transparent',
            edge_width=2
        )
        
        # Set to add mode so user can click to add points
        points_layer.mode = 'add'
        viewer.layers.selection.active = points_layer
        
        # Wait for user to press Enter (or add at least one point and wait)
        done_event = threading.Event()
        
        # Simple approach: wait for a keypress event on the layer
        # In practice, we could also add a button or use a different signal
        # For now, we'll just use a data change and let user know to press Enter
        def on_key_press(event):
            if event.key == 'Enter':
                done_event.set()
                viewer.layers.events.disconnect(on_key_press)
        
        # Connect to viewer's key press events
        viewer.bind_key('Enter', on_key_press, overwrite=True)
        
        return done_event
    
    def cleanup_points_layer():
        """Retrieve the point coordinates and clean up the layer."""
        viewer = napari.current_viewer()
        if not viewer or LAYER_NAME not in viewer.layers:
            return np.array([])
        
        points_layer = viewer.layers[LAYER_NAME]
        # Copy the coordinates
        coordinates = np.array(points_layer.data) if len(points_layer.data) > 0 else np.array([])
        
        # Unbind the Enter key
        try:
            viewer.bind_key('Enter', None)
        except (KeyError, AttributeError) as e:
            # KeyError: Key binding doesn't exist
            # AttributeError: viewer or bind_key method not available
            print(f"Warning: Could not unbind Enter key: {e}")
        
        # Remove the layer
        viewer.layers.remove(LAYER_NAME)
        
        return coordinates
    
    # Use the generic interactive worker
    seed_coords = interactive_worker.wait_for_user_event(setup_points_layer, cleanup_points_layer)
    
    if seed_coords is None or len(seed_coords) == 0:
        print("⚠️ No seed points were selected.")
        return np.array([])
    
    # --- 3. RETURN RESULTS ---
    print(f"✓ Collected {len(seed_coords)} seed points")
    print(f"  Coordinates shape: {seed_coords.shape}")
    
    # Return coordinates with metadata
    result_meta = meta.copy()
    result_meta["name"] = "Seed Points"
    result_meta["point_count"] = len(seed_coords)
    
    return (seed_coords, result_meta)
