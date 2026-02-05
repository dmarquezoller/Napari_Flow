import numpy as np
import matplotlib
# Force Agg backend so it doesn't try to open a GUI window in a thread (which crashes)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
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



# 

@register_node(
    label="Table Heatmap",
    category="Plotting",
    outputs=["plot"],  # Outputs a Figure
    params_config={
        "colormap": {
            "type": "enum", 
            "choices": ["viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu", "seismic"], 
            "label": "Colormap"
        },
        "normalize": {
            "type": "bool",
            "value": True,
            "label": "Normalize Columns (0-1)"
        }
    }
)
def table_heatmap_plot(table_in, colormap="viridis", normalize=True):
    """
    Generates a heatmap figure from a Table/DataFrame.
    Automatically selects only numeric columns.
    """
    if table_in is None:
        return None
        
    # 1. Ensure Input is a DataFrame
    # (Plugins might pass a dictionary or a list of dicts)
    if not isinstance(table_in, pd.DataFrame):
        try:
            df = pd.DataFrame(table_in)
        except Exception:
            # Return an error figure if data is invalid
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "Invalid Table Data", ha='center', va='center', color='red')
            return fig
    else:
        df = table_in.copy()

    # 2. Filter Numeric Data Only
    # We drop columns like 'Label' or 'FileName' for the heatmap
    df_num = df.select_dtypes(include=[np.number])
    
    if df_num.empty:
        fig = plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "Table has no numeric columns", ha='center', va='center')
        return fig

    # 3. Normalize Data (Optional)
    # Crucial if comparing "Area" (1000s) with "Circularity" (0.0-1.0)
    data_values = df_num.values
    if normalize:
        # Min-Max Normalization per column
        min_vals = np.nanmin(data_values, axis=0)
        max_vals = np.nanmax(data_values, axis=0)
        range_vals = max_vals - min_vals
        
        # Avoid division by zero for constant columns
        range_vals[range_vals == 0] = 1 
        
        data_values = (data_values - min_vals) / range_vals

    # 4. Create Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw Heatmap
    im = ax.imshow(data_values, aspect='auto', cmap=colormap, interpolation='nearest')
    
    # 5. Styling
    ax.set_title(f"Table Heatmap ({len(df)} rows)")
    
    # X-Axis: Column Names
    ax.set_xticks(np.arange(len(df_num.columns)))
    ax.set_xticklabels(df_num.columns, rotation=45, ha="right", fontsize=9)
    
    # Y-Axis: Row Indices (Hide if too many rows)
    if len(df) < 50:
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df.index, fontsize=8)
    else:
        ax.set_ylabel("Row Index")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    if normalize:
        cbar.set_label("Normalized Value (0-1)")
    
    plt.tight_layout()
    
    return fig