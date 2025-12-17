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