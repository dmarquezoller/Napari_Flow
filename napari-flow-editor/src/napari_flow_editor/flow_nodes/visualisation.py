import numpy as np
import matplotlib
# Force Agg backend so it doesn't try to open a GUI window in a thread (which crashes)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
from PIL import Image
from .decorator import register_node

@register_node(
    label="Plot Histogram",
    category="Plotting",
    outputs=["plot"], # <--- SINK NODE: No output sockets
)
def plot_histogram(image_in):
    """Generates a Matplotlib Figure object from the image."""
    
    # 1. Prepare Data
    if image_in is None:
        return None
    data = image_in.ravel()

    # 2. Create Figure (Pure Python Object, no UI yet)
    # We use a specific size that looks good in a popup
    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    
    # 3. Plot
    ax.hist(data, bins=100, color='#2196F3', alpha=0.7, log=True)
    ax.set_title("Pixel Intensity Distribution")
    ax.set_xlabel("Intensity Value")
    ax.set_ylabel("Count (Log Scale)")
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 4. Return the Figure Object itself
    return fig

@register_node(
    label="Colocalization Scatter",
    category="Plotting",
    outputs=["plot"] # Sink Node
)
def colocalization_scatter(image_ch1, image_ch2):
    """
    Plots intensity of Channel 1 vs Channel 2 to show correlation.
    Requires two images of the same size.
    """
    if image_ch1 is None or image_ch2 is None:
        return None
        
    # 1. Validation: Ensure images match size
    if image_ch1.shape != image_ch2.shape:
        # In a real app, you might return a "Error Figure" with text explaining the error
        fig = plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "Error: Image shapes do not match!", 
                 ha='center', va='center', color='red')
        return fig

    # 2. Flatten data (Pixel 1 vs Pixel 1, Pixel 2 vs Pixel 2...)
    # We subsample if the image is huge to keep plotting fast
    flat1 = image_ch1.ravel()
    flat2 = image_ch2.ravel()
    
    if len(flat1) > 10000:
        indices = np.random.choice(len(flat1), 10000, replace=False)
        flat1 = flat1[indices]
        flat2 = flat2[indices]

    # 3. Plot
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Scatter plot with transparency (alpha) to see density
    ax.scatter(flat1, flat2, alpha=0.1, s=2, c='#2196F3')
    
    # Add a diagonal "Perfect Correlation" line for reference
    max_val = max(flat1.max(), flat2.max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Correlation')
    
    ax.set_title(f"Colocalization (Pearson r: {np.corrcoef(flat1, flat2)[0,1]:.2f})")
    ax.set_xlabel("Intensity Ch 1")
    ax.set_ylabel("Intensity Ch 2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    return fig


# ORTHOGONAL PROJECTION

@register_node(
    label="Orthogonal Projections",
    category="Plotting",
    outputs=["plot"]
)
def ortho_projection_plot(image_in):
    """
    Creates a Maximum Intensity Projection (MIP) along 3 axes.
    Ideal for visualizing 3D volumes (Z-Stacks).
    """
    if image_in is None:
        return None

    # 1. Validation: We need 3 dimensions
    if image_in.ndim < 3:
        fig = plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, f"Error: Input is {image_in.ndim}D.\nNeed 3D (Z, Y, X) input.", 
                 ha='center', va='center', color='red')
        return fig

    # If > 3D (e.g. Time, Z, Y, X), take the first timepoint
    data = image_in
    while data.ndim > 3:
        data = data[0]

    # 2. Calculate Projections (Max Intensity)
    # Axis 0 = Z, Axis 1 = Y, Axis 2 = X
    proj_xy = np.max(data, axis=0) # Top-down
    proj_xz = np.max(data, axis=1) # Side view
    proj_yz = np.max(data, axis=2) # Front view

    # 3. Setup Plot Layout (GridSpec is great for non-square layouts)
    fig = plt.figure(figsize=(8, 8), dpi=100)
    grid = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1], wspace=0.05, hspace=0.05)

    # Main View (XY)
    ax_xy = fig.add_subplot(grid[0, 0])
    ax_xy.imshow(proj_xy, cmap='gray', aspect='auto')
    ax_xy.set_ylabel("Y Axis")
    ax_xy.set_xticks([]) # Hide ticks for cleaner look

    # Side View (YZ) - Rotated to match Y axis of main view
    ax_yz = fig.add_subplot(grid[0, 1])
    ax_yz.imshow(proj_yz.T, cmap='gray', aspect='auto') # Transpose to align Z vertically
    ax_yz.set_xlabel("Z Axis")
    ax_yz.set_yticks([])

    # Bottom View (XZ)
    ax_xz = fig.add_subplot(grid[1, 0])
    ax_xz.imshow(proj_xz, cmap='gray', aspect='auto')
    ax_xz.set_xlabel("X Axis")
    ax_xz.set_ylabel("Z Axis")
    
    # Empty bottom-right corner
    ax_empty = fig.add_subplot(grid[1, 1])
    ax_empty.axis('off')
    ax_empty.text(0.5, 0.5, "MIP\nView", ha='center', va='center', fontweight='bold')

    plt.suptitle(f"Orthogonal MIP (Shape: {data.shape})", y=0.95)
    
    return fig


# INTENSITY HEATMAP


@register_node(
    label="Intensity Heatmap",
    category="Plotting",
    outputs=["plot"],
    params_config={
        "colormap": {"type": "enum", "options": ["inferno", "viridis", "magma", "plasma", "jet"]}
    }
)
def intensity_heatmap(image_in, colormap="inferno"):
    """
    Plots a heatmap of pixel intensities.
    If 3D, extracts the middle slice.
    """
    if image_in is None:
        return None

    data = image_in
    
    # 1. Handle Dimensions: We need 2D for a heatmap
    slice_info = "2D Image"
    if data.ndim > 2:
        # Take the middle Z-slice
        mid_z = data.shape[0] // 2
        data = data[mid_z]
        slice_info = f"Middle Slice (Z={mid_z})"
        
        # Handle 4D/5D if necessary (take first timepoint/channel)
        while data.ndim > 2:
            data = data[0]

    # 2. Subsample if too large (rendering heatmaps of 4k images is slow)
    if data.size > 1024 * 1024:
        scale = int(np.sqrt(data.size / (1024*1024))) + 1
        data = data[::scale, ::scale]
        slice_info += f" (Subsampled {scale}x)"

    # 3. Plot
    fig = plt.figure(figsize=(7, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Imshow with the chosen colormap
    im = ax.imshow(data, cmap=colormap, interpolation='nearest')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pixel Intensity")
    
    ax.set_title(f"Intensity Heatmap\n{slice_info}")
    ax.axis('off') # Turn off axes for cleaner visual
    plt.tight_layout()

    return fig