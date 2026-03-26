import numpy as np
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QScrollArea,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
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


def is_plotly_figure(obj):
    """Best-effort runtime check for Plotly figures without hard dependency."""
    if obj is None:
        return False
    module_name = getattr(type(obj), "__module__", "")
    if "plotly" in module_name and hasattr(obj, "to_html"):
        return True
    return False


def _resolve_qwebengine_view():
    """
    Import WebEngine lazily so simply opening the plugin does not initialize
    QtWebEngine unless Plotly output is actually requested.
    """
    try:
        from qtpy.QtWebEngineWidgets import QWebEngineView as _QWebEngineView
        return _QWebEngineView, None
    except Exception as exc:  # pragma: no cover - optional dependency
        return None, str(exc)


class PlotDashboard(QWidget):
    """Display plot outputs (Matplotlib or Plotly) in the bottom panel."""

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
        
        self.current_widget = None
        self.setMinimumHeight(250) 

    def _show(self):
        self.title_label.setText("Output Plot")
        self.setVisible(True)

    def _show_message(self, message):
        self.clear_canvas()
        msg = QLabel(message)
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("color: #d0d0d0; padding: 16px; background-color: #222;")
        self.current_widget = msg
        self.container_layout.addWidget(msg)
        self._show()

    def display(self, figure_obj, plotly_figure=None):
        # Preferred path for plotting nodes: static Matplotlib in-app, with
        # optional user opt-in to interactive browser Plotly.
        if plotly_figure is not None:
            self._show_plot_choice_panel(figure_obj, plotly_figure)
            return
        if is_plotly_figure(figure_obj):
            self.display_plotly(figure_obj)
            return
        self.display_matplotlib(figure_obj)

    def display_matplotlib(self, fig):
        self.clear_canvas()

        # Create new canvas
        self.current_widget = FigureCanvas(fig)
        self.current_widget.setStyleSheet("background-color: #222;")
        self.container_layout.addWidget(self.current_widget)
        self.current_widget.draw()
        self._show()

    def display_plotly(self, fig):
        qwebengine_cls, import_error = _resolve_qwebengine_view()
        if qwebengine_cls is None:
            self._show_plotly_fallback_options(fig, import_error)
            return

        self.clear_canvas()
        web = qwebengine_cls()
        html_doc = fig.to_html(
            full_html=False,
            include_plotlyjs=True,
            config={"responsive": True, "displaylogo": False},
        )
        web.setHtml(html_doc)
        self.current_widget = web
        self.container_layout.addWidget(web)
        self._show()

    def _show_plot_choice_panel(self, fig_matplotlib, fig_plotly):
        self.clear_canvas()

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Plot Output")
        title.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel("Choose display mode for this plot:")
        subtitle.setStyleSheet("color: #c8c8c8;")
        layout.addWidget(subtitle)

        buttons = QHBoxLayout()
        btn_static = QPushButton("Static (Embedded)")
        btn_interactive = QPushButton("Interactive (Browser)")
        btn_static.clicked.connect(lambda: self.display_matplotlib(fig_matplotlib))
        btn_interactive.clicked.connect(lambda: self._on_plotly_browser_selected(fig_plotly))
        buttons.addWidget(btn_static)
        buttons.addWidget(btn_interactive)
        layout.addLayout(buttons)

        hint = QLabel(
            "Static uses the in-app Matplotlib view. "
            "Interactive opens Plotly in your default browser."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa3aa; font-size: 11px;")
        layout.addWidget(hint)

        self.current_widget = panel
        self.container_layout.addWidget(panel)
        self._show()

    def _show_plotly_fallback_options(self, fig, import_error):
        self.clear_canvas()

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Interactive Plotly embed is unavailable in this session.")
        title.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        title.setWordWrap(True)
        layout.addWidget(title)

        subtitle = QLabel(
            "Choose how to display this plot:\n"
            "1) Interactive in browser, or 2) Static preview inside the app."
        )
        subtitle.setStyleSheet("color: #c8c8c8;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        buttons = QHBoxLayout()
        btn_browser = QPushButton("Interactive (Browser)")
        btn_static = QPushButton("Static Preview (Embedded)")
        btn_browser.clicked.connect(
            lambda: self._on_plotly_browser_selected(fig)
        )
        btn_static.clicked.connect(
            lambda: self._on_plotly_static_selected(fig)
        )
        buttons.addWidget(btn_browser)
        buttons.addWidget(btn_static)
        layout.addLayout(buttons)

        details = QLabel(f"Embed reason: {import_error}")
        details.setWordWrap(True)
        details.setStyleSheet("color: #9aa3aa; font-size: 11px;")
        layout.addWidget(details)

        self.current_widget = panel
        self.container_layout.addWidget(panel)
        self._show()

    def _on_plotly_browser_selected(self, fig):
        browser_ok, detail = self._open_plotly_in_browser(fig)
        if browser_ok:
            self._show_message(
                "Opened interactive Plotly in your default browser."
            )
            return
        self._show_message(
            "Could not open browser for interactive Plotly.\n"
            f"Reason: {detail}"
        )

    def _on_plotly_static_selected(self, fig):
        preview_ok, detail = self._show_plotly_static_preview(fig)
        if preview_ok:
            return
        self._show_message(
            "Could not render static preview inside the app.\n"
            "Tip: install `kaleido` for Plotly image export.\n"
            f"Reason: {detail}"
        )

    def _show_plotly_static_preview(self, fig):
        try:
            import plotly.io as pio
            png_bytes = pio.to_image(fig, format="png", scale=2)
        except Exception as exc:
            return False, str(exc)

        image = QImage.fromData(png_bytes, "PNG")
        if image.isNull():
            return False, "Failed to decode Plotly PNG image."

        self.clear_canvas()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #222; border: none;")

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(8, 8, 8, 8)

        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setPixmap(QPixmap.fromImage(image))
        wrapper_layout.addWidget(label)
        scroll.setWidget(wrapper)

        self.current_widget = scroll
        self.container_layout.addWidget(scroll)
        self._show()
        return True, None

    def _open_plotly_in_browser(self, fig):
        try:
            html_doc = fig.to_html(
                full_html=True,
                include_plotlyjs=True,
                config={"responsive": True, "displaylogo": False},
            )

            # Prefer a persistent, user-visible path. Some browser sandbox
            # configurations (e.g. Firefox Snap/Flatpak) cannot access /tmp
            # files created by other processes/namespaces.
            base_candidates = [
                Path.home() / "napari_flow_plots",
                Path.home() / "Documents" / "napari_flow_plots",
                Path(tempfile.gettempdir()) / "napari_flow_plots",
            ]
            output_dir = None
            for candidate in base_candidates:
                try:
                    candidate.mkdir(parents=True, exist_ok=True)
                    output_dir = candidate
                    break
                except Exception:
                    continue

            if output_dir is None:
                return False, "Could not create a writable folder for Plotly output."

            filename = (
                f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.html"
            )
            path = output_dir / filename
            path.write_text(html_doc, encoding="utf-8")

            if not path.exists() or path.stat().st_size == 0:
                return False, f"Failed to write plot HTML to {path}"

            opened = webbrowser.open_new_tab(path.as_uri())
            if not opened:
                return False, "Web browser did not accept open request."
            return True, str(path)
        except Exception as exc:
            return False, str(exc)

    def hide_dashboard(self):
        self.clear_canvas()
        self.setVisible(False) # <--- MAGIC: Collapse the layout
        
    def clear_canvas(self):
        if self.current_widget:
            self.container_layout.removeWidget(self.current_widget)
            self.current_widget.close()
            self.current_widget.deleteLater()
            self.current_widget = None
