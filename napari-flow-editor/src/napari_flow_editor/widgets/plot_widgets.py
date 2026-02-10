import numpy as np
from qtpy.QtWidgets import QDialog, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QLabel
from qtpy.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.backends.backend_agg import FigureCanvasAgg


class PlotResultDialog(QDialog):
    def __init__(self, fig, title="Plot Result", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 500)
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)


def figure_to_rgb_array(fig):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import numpy as np
        
        # 1. Setup the "Memory Only" Canvas (No Popups)
        canvas = FigureCanvasAgg(fig)
        fig.canvas = canvas
        
        # 2. Render the plot
        canvas.draw()
        
        # 3. Modern Matplotlib Way (Works on v3.8+)
        # buffer_rgba() returns a memory view we can turn directly into a numpy array
        rgba_image = np.asarray(canvas.buffer_rgba())
        
        # 4. Convert RGBA (4 channels) to RGB (3 channels) for simplicity
        # Napari accepts RGBA, but RGB is safer for the logic we wrote earlier.
        rgb_image = rgba_image[:, :, :3]
        
        return rgb_image

    except Exception as e:
        print(f"!!! PLOT CONVERSION FAILED !!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


class PlotDashboard(QWidget):
    """A dedicated widget to display Matplotlib plots inside the UI. Auto-hides when empty."""
    def __init__(self):
        super().__init__()
        # Start hidden!
        self.setVisible(False)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # --- Header (Title + Close Button) ---
        header = QWidget()
        header.setStyleSheet("background-color: #333; border-bottom: 1px solid #555;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 2, 5, 2)
        
        self.title_label = QLabel("📊 Output Plot")
        self.title_label.setStyleSheet("font-weight: bold; color: #ddd;")
        
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setFlat(True)
        self.close_btn.setStyleSheet("color: #aaa; font-weight: bold;")
        self.close_btn.clicked.connect(self.hide_dashboard) # Connect to hide function
        
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.close_btn)
        self.layout.addWidget(header)

        # --- Plot Canvas Area ---
        self.canvas_container = QWidget()
        self.container_layout = QVBoxLayout(self.canvas_container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.canvas_container)
        
        self.current_canvas = None
        self.setMinimumHeight(250) 

    def display(self, fig):
        self.clear_canvas()
        
        # Create new canvas
        self.current_canvas = FigureCanvas(fig) 
        self.current_canvas.setStyleSheet("background-color: #222;") # Dark background
        self.container_layout.addWidget(self.current_canvas)
        self.current_canvas.draw()
        
        # Show widget and title
        self.title_label.setText(f"📊 Output Plot")
        self.setVisible(True) # <--- MAGIC: Expand the layout

    def hide_dashboard(self):
        self.clear_canvas()
        self.setVisible(False) # <--- MAGIC: Collapse the layout
        
    def clear_canvas(self):
        if self.current_canvas:
            self.container_layout.removeWidget(self.current_canvas)
            self.current_canvas.close()
            self.current_canvas = None
