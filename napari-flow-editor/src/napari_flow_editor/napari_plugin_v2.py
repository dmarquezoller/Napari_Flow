from qtpy.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsRectItem, QMainWindow,
    QGraphicsDropShadowEffect, QToolBar, QInputDialog, QFileDialog, QVBoxLayout,
    QHBoxLayout, QPushButton, QMenu, QWidget, QGroupBox, QFormLayout, QLabel,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QScrollArea, QBoxLayout,
    QFrame, QMessageBox, QTextEdit, QSplitter, QDialog, QTableWidget, QHeaderView, QAbstractItemView,
    QListWidget, QListWidgetItem
)

from qtpy.QtGui import (
    QBrush, QPen, QColor, QPainterPath, QPainterPathStroker, QLinearGradient, QPainter, QAction, QCursor, QGradient, QImage, QPixmap
)
from qtpy.QtCore import Qt, QPointF, QRectF, QThread, Signal
import os, sys, json, datetime, napari, uuid, importlib.util, inspect, zarr

import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from napari_flow_editor import generate_library
from .execution_engine import ExecutionWorker
from .script_generator import ScriptGenerator
from .widgets.dynamic_table import DynamicTableWidget
from .widgets.plot_widgets import PlotResultDialog, figure_to_rgb_array, PlotDashboard


NODE_LIBRARY = {}

def _normalize_logic_config(logic_cfg):
    cfg = {
        "in": True,
        "out": True,
        "allow_multi_in": True,
        "allow_multi_out": True,
    }
    if isinstance(logic_cfg, dict):
        cfg.update(logic_cfg)
    elif logic_cfg is False:
        cfg["in"] = False
        cfg["out"] = False
    return cfg

# --- SOCKET -----------------------------------------------------
class Socket(QGraphicsEllipseItem):
    def __init__(self, node, socket_type, name, index, total_sockets, max_connections=None):
        super().__init__(-6, -6, 12, 12)
        self.socket_type = socket_type
        self.node = node
        self.name = name # e.g. "image_in"
        self.connected_edges = []
        self.max_connections = max_connections

        # --- Determine whether this is a logic (control-flow) socket ---
        self.is_logic = socket_type in ("logic_in", "logic_out")

        h = node.rect().height()
        w = node.rect().width()

        if self.is_logic:
            is_control_flow = getattr(node, "category", "") == "Control Flow"
            # Control nodes: middle sockets, inverted direction.
            if is_control_flow:
                side = "right" if socket_type == "logic_in" else "left"
                logic_position = "middle"
            # Other nodes: bottom sockets, forward direction.
            else:
                side = "left" if socket_type == "logic_in" else "right"
                logic_position = "bottom"
            x = 0 if side == "left" else w
            y = (h / (total_sockets + 1)) * (index + 1)

            if logic_position == "middle":
                self.local_offset = QPointF(x, y)
            else:
                # Legacy bottom-corner style
                bx = 14 if side == "left" else (w - 14)
                self.local_offset = QPointF(bx, h - 2)
            # Distinct control colors: input=blue, output=lavender.
            color = QColor("#5FA8D3") if socket_type == "logic_in" else QColor("#B08EC6")
            self.setToolTip(f"Logic: {socket_type}")
        else:
            # Data sockets — original layout (left/right sides, evenly spaced)
            y = (h / (total_sockets + 1)) * (index + 1)
            x = 0 if socket_type == "input" else w
            self.local_offset = QPointF(x, y)
            color = QColor("#ffb347") if socket_type == "output" else QColor("#77dd77")
            self.setToolTip(f"Data: {name}")

        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        self.setZValue(3)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.update_position()

    def update_position(self):
        self.setPos(self.node.pos() + self.local_offset)
    
    # ... (keep center_pos and can_accept_connection same as before) ...
    def center_pos(self):
        return self.sceneBoundingRect().center()

    def can_accept_connection(self):
        if self.max_connections is None:
            return True
        return len(self.connected_edges) < self.max_connections


# --- CONNECTION -------------------------------------------------
class Connection(QGraphicsPathItem):
    def __init__(self, start_socket, scene):
        super().__init__()
        self.start_socket = start_socket
        self.end_socket = None
        self.scene_ref = scene
        self.dragging = False
        self.hovered = False
        self.is_logic = getattr(start_socket, "is_logic", False)
        self.setZValue(1)

        # Logic edges: dashed blue-purple line; Data edges: solid grey
        if self.is_logic:
            pen = QPen(QColor("#8888cc"), 2, Qt.PenStyle.DashLine)
        else:
            pen = QPen(QColor("#444"), 2)
        self.setPen(pen)

        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.update_path(self.start_socket.center_pos(), self.start_socket.center_pos())

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(20)
        return stroker.createStroke(self.path())

    def hoverEnterEvent(self, event):
        if self.is_logic:
            self.setPen(QPen(QColor("#C6C6FF"), 3, Qt.PenStyle.DashLine))
        else:
            self.setPen(QPen(QColor("#0078d7"), 3))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if self.is_logic:
            self.setPen(QPen(QColor("#8888cc"), 2, Qt.PenStyle.DashLine))
        else:
            self.setPen(QPen(QColor("#444"), 2))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if self.end_socket and (self.end_socket.center_pos() - event.scenePos()).manhattanLength() < 20:
            event.accept()
            self.detach_end()
            self.dragging = True
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.dragging:
            event.accept()
            self.update_path(self.start_socket.center_pos(), event.scenePos())

    def mouseReleaseEvent(self, event):
        if self.dragging:
            event.accept()
            target = self.scene_ref.find_nearby_socket(
                event.scenePos(), prefer_logic=self.is_logic
            )
            if self.is_logic:
                # Logic edges: logic_out → logic_in, skip cycle check
                valid = (
                    target
                    and target.socket_type == "logic_in"
                    and target.can_accept_connection()
                    and not self.scene_ref.are_already_connected(self.start_socket, target)
                )
            else:
                # Data edges: output → input, enforce DAG
                valid = (
                    target
                    and target.socket_type == "input"
                    and target.can_accept_connection()
                    and not self.scene_ref.are_already_connected(self.start_socket, target)
                    and not self.scene_ref.creates_cycle(self.start_socket.node, target.node)
                )

            if valid:
                self.finalize(target)
            else:
                self.scene_ref.removeItem(self)
            self.dragging = False
        else:
            event.ignore()

    def update_path(self, start, end):
        path = QPainterPath()
        path.moveTo(start)
        dx = (end.x() - start.x()) * 0.5
        c1 = QPointF(start.x() + dx, start.y())
        c2 = QPointF(end.x() - dx, end.y())
        path.cubicTo(c1, c2, end)
        self.setPath(path)

    def finalize(self, end_socket):
        self.end_socket = end_socket
        self.update_path(self.start_socket.center_pos(), self.end_socket.center_pos())
        self.start_socket.connected_edges.append(self)
        self.end_socket.connected_edges.append(self)

    def detach_end(self):
        if self.end_socket and self in self.end_socket.connected_edges:
            self.end_socket.connected_edges.remove(self)
        self.end_socket = None

    def update_positions(self):
        if self.start_socket and self.end_socket:
            self.update_path(
                self.start_socket.center_pos(), self.end_socket.center_pos()
            )


# --- NODE -------------------------------------------------------
class Node(QGraphicsRectItem):
    def __init__(self, x, y, node_type="generic", title=None, uuid_str=None, scene=None):
        # 1. Setup Data
        self.node_type = node_type
        self.category = "Uncategorized"
        self.uid = uuid_str if uuid_str else str(uuid.uuid4())
        self.parameters = {}
        self.logic_config = _normalize_logic_config(None)

        # --- State variables ---
        self.status = "gray"
        self.last_signature = None
        self.cached_results = {}
        
        # Load Definition
        inputs_data = ["in"] # Default if not found
        outputs_data = ["out"]
        
        if node_type in NODE_LIBRARY:
            definition = NODE_LIBRARY[node_type]
            self.title = title if title else definition["label"]
            self.category = definition.get("category", "Uncategorized")
            self.logic_config = _normalize_logic_config(definition.get("logic"))
            inputs_data = definition.get("inputs", ["in"])
            outputs_data = definition.get("outputs", ["out"])
            
            # Load Params
            for key, conf in definition["parameters"].items():
                self.parameters[key] = conf["default"]
        else:
            self.title = title if title else node_type

        # 2. Calculate Height based on params AND sockets
        param_count = len(self.parameters)
        socket_count = max(len(inputs_data), len(outputs_data))
        h = 45 + (param_count * 20) + (socket_count * 10) # Add space for multiple sockets
        h = max(h, 80)
        
        super().__init__(0, 0, 160, h)
        self.setPos(x, y)
        self.scene = scene
        
        # 3. Visual Setup
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(2)

        # 4. Generate Data Sockets dynamically
        self.inputs = []
        for i, name in enumerate(inputs_data):
            self.inputs.append(Socket(self, "input", name, i, len(inputs_data), max_connections=1))

        self.outputs = []
        for i, name in enumerate(outputs_data):
            self.outputs.append(Socket(self, "output", name, i, len(outputs_data), max_connections=None))

        # 5. Generate Logic sockets from node definition.
        logic_in_enabled = bool(self.logic_config.get("in", True))
        logic_out_enabled = bool(self.logic_config.get("out", True))
        logic_in_max = None if self.logic_config.get("allow_multi_in", True) else 1
        logic_out_max = None if self.logic_config.get("allow_multi_out", True) else 1

        self.logic_inputs = []
        if logic_in_enabled:
            self.logic_inputs.append(
                Socket(self, "logic_in", "logic_in", 0, 1, max_connections=logic_in_max)
            )

        self.logic_outputs = []
        if logic_out_enabled:
            self.logic_outputs.append(
                Socket(self, "logic_out", "logic_out", 0, 1, max_connections=logic_out_max)
            )

        if scene:
            all_sockets = self.inputs + self.outputs + self.logic_inputs + self.logic_outputs
            for s in all_sockets:
                scene.addItem(s)

    # Helper to iterate ALL sockets (data + logic)
    def all_sockets(self):
        return self.inputs + self.outputs + self.logic_inputs + self.logic_outputs

    def itemChange(self, change, value):
        if change in (QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged, QGraphicsItem.GraphicsItemChange.ItemPositionChange):
            for s in self.all_sockets():
                s.update_position()
                for e in s.connected_edges:
                    e.update_positions()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget):
        rect = self.rect()
        is_control_flow = self.category == "Control Flow"

        # A. Shadow
        painter.fillRect(rect.adjusted(4, 4, 4, 4), QColor(0, 0, 0, 60))
        
        # B. Main Body Gradient
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        if is_control_flow:
            gradient.setColorAt(0, QColor("#33404f"))
            gradient.setColorAt(1, QColor("#242f3b"))
        else:
            gradient.setColorAt(0, QColor("#3F4242"))
            gradient.setColorAt(1, QColor("#2F3232"))
        painter.setBrush(QBrush(gradient))
        
        # C. Selection Border
        if self.isSelected():
            border_color = QColor("#ff9900")
        else:
            border_color = QColor("#7f9fbe") if is_control_flow else QColor("#727272")
        painter.setPen(QPen(border_color, 2))
        painter.drawRoundedRect(rect, 12, 12)

        # D. Title Header
        title_rect = QRectF(rect.x(), rect.y(), rect.width(), 25)
        title_grad = QLinearGradient(title_rect.topLeft(), title_rect.bottomRight())
        if is_control_flow:
            title_grad.setColorAt(0, QColor("#5a748f"))
            title_grad.setColorAt(1, QColor("#415469"))
        else:
            title_grad.setColorAt(0, QColor("#666"))
            title_grad.setColorAt(1, QColor("#444"))
        painter.setBrush(QBrush(title_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(title_rect, 12, 12)
        
        fix_rect = QRectF(rect.x(), rect.y() + 15, rect.width(), 10)
        painter.drawRect(fix_rect)

        # E. Title Text
        painter.setPen(Qt.GlobalColor.white)
        text_rect = QRectF(rect.x(), rect.y(), rect.width()-15, 25)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.title)

        # F. Status Light (hidden on Control Flow nodes)
        if not is_control_flow:
            status_colors = {
                "gray": QColor("#777777"),   # Dirty/Stale (grey)
                "yellow": QColor("#FFD700"), # Running (yellow)
                "green": QColor("#32CD32"),  # Cached/Done (green)
                "red": QColor("#FF4500")     # Error (red)
            }
            light_color = status_colors.get(self.status, QColor("#777777"))
            
            painter.setBrush(QBrush(light_color))
            painter.setPen(Qt.PenStyle.NoPen)
            # Draw small circle in top-right
            dot_rect = QRectF(rect.x() + rect.width() - 18, rect.y() + 7, 10, 10)
            painter.drawEllipse(dot_rect)
        
        # G. Parameters Text
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        y_offset = 35 
        
        for key, value in self.parameters.items():
            painter.setPen(QColor("#b0b0b0"))
            painter.drawText(QRectF(10, y_offset, rect.width()-20, 20), Qt.AlignmentFlag.AlignLeft, f"{key}:")
            painter.setPen(QColor("#ffffff"))
            if isinstance(value, float):
                val_str = str(round(value, 2))
            else:
                val_str = str(value)

            if len(val_str) > 14: val_str = val_str[:14] + "..."
            painter.drawText(QRectF(10, y_offset, rect.width()-20, 20), Qt.AlignmentFlag.AlignRight, val_str)
            y_offset += 20


# --- SCENE ------------------------------------------------------
class FlowScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.current_connection = None

    def find_nearby_socket(self, pos, radius=15, prefer_logic=None):
        """
        Find the nearest socket within *radius* of *pos*.

        If ``prefer_logic`` is True, logic sockets are preferred over data
        sockets (and vice versa when False).  When None the first match wins.
        """
        area = QRectF(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
        best = None
        best_dist = float("inf")
        for item in self.items(area):
            if not isinstance(item, Socket):
                continue
            d = (item.center_pos() - pos).manhattanLength()
            # Apply preference bias: subtract a large value for the preferred kind
            if prefer_logic is not None and item.is_logic == prefer_logic:
                d -= 1000
            if d < best_dist:
                best_dist = d
                best = item
        return best

    def are_already_connected(self, a, b):
        for e in a.connected_edges:
            if (e.start_socket == a and e.end_socket == b) or (e.start_socket == b and e.end_socket == a):
                return True
        return False

    def creates_cycle(self, start_node, end_node):
        visited = set()

        def dfs(node):
            if node == start_node:
                return True
            for s in node.outputs:
                for e in s.connected_edges:
                    if e.end_socket:
                        next_node = e.end_socket.node
                        if next_node not in visited:
                            visited.add(next_node)
                            if dfs(next_node):
                                return True
            return False

        return dfs(end_node)

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        if (
            isinstance(item, Socket)
            and item.socket_type in ("output", "logic_out")
            and item.can_accept_connection()
        ):
            event.accept()
            self.current_connection = Connection(item, self)
            self.addItem(self.current_connection)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.current_connection:
            self.current_connection.update_path(
                self.current_connection.start_socket.center_pos(), event.scenePos()
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.current_connection:
            pos = event.scenePos()
            start_sock = self.current_connection.start_socket
            is_logic = getattr(start_sock, "is_logic", False)
            target = self.find_nearby_socket(pos, prefer_logic=is_logic)

            if is_logic:
                # Logic edges: logic_out → logic_in, no cycle check
                valid = (
                    target
                    and target.socket_type == "logic_in"
                    and target.can_accept_connection()
                    and not self.are_already_connected(start_sock, target)
                )
            else:
                # Data edges: output → input, enforce DAG
                valid = (
                    target
                    and target.socket_type == "input"
                    and target.can_accept_connection()
                    and not self.are_already_connected(start_sock, target)
                    and not self.creates_cycle(start_sock.node, target.node)
                )

            if valid:
                self.current_connection.finalize(target)
            else:
                self.removeItem(self.current_connection)
            self.current_connection = None
        super().mouseReleaseEvent(event)

    def fit_scene(self):
        """Zoom to fit all items."""
        rect = self.scene().itemsBoundingRect()
        if rect.isNull():
            return
        # Add padding so nodes aren't stuck to edge
        rect.adjust(-50, -50, 50, 50)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)


# --- VIEW -------------------------------------------------------
from qtpy.QtGui import QPainter

class FlowView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.scale_factor = 1.0

    def wheelEvent(self, event):
        zoom_in = 1.1
        zoom_out = 0.9
        factor = zoom_in if event.angleDelta().y() > 0 else zoom_out
        self.scale(factor, factor)

    def reset_view(self):
        items = self.scene().items()
        if not items:
            return
        rect = self.scene().itemsBoundingRect()
        padding = 100
        rect.adjust(-padding, -padding, padding, padding)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(rect.center())

    def fit_scene(self):
        """Zoom to fit all items."""
        rect = self.scene().itemsBoundingRect()
        if rect.isNull():
            return
        # Add padding so nodes aren't stuck to edge
        rect.adjust(-50, -50, 50, 50)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

# --- MAIN WINDOW ------------------------------------------------
class FlowEditor(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        # --- AUTO GENERATE JSON LIBRARY ---
        generate_library.generate()

        global NODE_LIBRARY
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "node_library.json")
        try:
            with open(json_path, "r") as f:
                NODE_LIBRARY.clear()
                NODE_LIBRARY.update(json.load(f))
        except Exception as e:
            print(f"CRITICAL ERROR loading node library: {e}")

        self.viewer = viewer
        self._active_interaction_dialog = None
        
        # 1. Main Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 2. Top Toolbar (Add, Remove, Save, Load, etc.)
        toolbar = QHBoxLayout()
        
        self.btn_add = QPushButton("Add Node")
        self.btn_add.clicked.connect(self.open_add_menu)
        toolbar.addWidget(self.btn_add)

        self.btn_add_control = QPushButton("Add Control")
        self.btn_add_control.clicked.connect(self.open_control_menu)
        self.btn_add_control.setStyleSheet("""
            QPushButton {
                background-color: #4f6780;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #5d7a99; }
            QPushButton:pressed { background-color: #405469; }
        """)
        toolbar.addWidget(self.btn_add_control)

        self.btn_remove = QPushButton("Remove Node")
        self.btn_remove.clicked.connect(self.open_remove_menu)
        toolbar.addWidget(self.btn_remove)

        self.btn_save = QPushButton("Save Pipeline")
        self.btn_save.clicked.connect(self.save_pipeline)
        toolbar.addWidget(self.btn_save)

        self.btn_save_zarr = QPushButton("Save to Zarr")
        self.btn_save_zarr.clicked.connect(self.save_to_zarr)
        toolbar.addWidget(self.btn_save_zarr)

        self.btn_load = QPushButton("Load Pipeline")
        self.btn_load.clicked.connect(self.load_pipeline)
        toolbar.addWidget(self.btn_load)

        self.btn_import = QPushButton("Import Nodes (.py)")
        self.btn_import.clicked.connect(self.import_custom_module)
        toolbar.addWidget(self.btn_import)  

        self.btn_export = QPushButton("Export Script")
        self.btn_export.clicked.connect(self.export_to_python)
        toolbar.addWidget(self.btn_export)      

        self.layout.addLayout(toolbar)

        # 3. Node Properties Panel (Middle)
        self.props_group = QGroupBox("Node Properties")
        group_layout = QVBoxLayout()
        self.props_group.setLayout(group_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        self.props_layout = QFormLayout()
        self.props_layout.setContentsMargins(5, 5, 5, 5)
        scroll_content.setLayout(self.props_layout)
        
        scroll.setWidget(scroll_content)
        group_layout.addWidget(scroll)

        # --- 4. Main Splitter ---
        self.splitter = QSplitter(Qt.Vertical)
        
        # Part A: Properties (Top)
        self.splitter.addWidget(self.props_group)

        # Part B: Bottom Container (Buttons + Graph + Plot + Log)
        self.bottom_container = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(5)

        # --- B1. Action Buttons (Stacked Vertically BELOW Properties, ABOVE Graph) ---
        button_layout = QVBoxLayout() 
        button_layout.setSpacing(5)

        # Fit Button
        self.btn_fit = QPushButton("Fit View")
        button_layout.addWidget(self.btn_fit)

        # Loop stop control (used by "Until confirm" mode)
        self.btn_stop_loop = QPushButton("Stop Loop")
        self.btn_stop_loop.setStyleSheet("""
            QPushButton {
                background-color: #B23A3A;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c94a4a; }
            QPushButton:pressed { background-color: #8f2d2d; }
        """)
        self.btn_stop_loop.setVisible(False)
        self.btn_stop_loop.setEnabled(False)
        button_layout.addWidget(self.btn_stop_loop)

        # Run Button
        self.btn_run = QPushButton("RUN PIPELINE")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32; color: white; font-weight: bold;
                font-size: 14px; border-radius: 4px; border: 1px solid #1B5E20;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:pressed { background-color: #1B5E20; }
        """)
        button_layout.addWidget(self.btn_run)
        
        # Add buttons to the top of the bottom container
        bottom_layout.addLayout(button_layout)

        # --- B2. Inner Splitter (Graph / Plot / Log) ---
        self.inner_splitter = QSplitter(Qt.Vertical)
        
        # 1. Graph View
        self.scene = FlowScene()
        self.view = FlowView(self.scene)
        self.inner_splitter.addWidget(self.view)
        
        # 2. Plot Dashboard (Hidden by default)
        self.plot_dashboard = PlotDashboard()
        self.inner_splitter.addWidget(self.plot_dashboard)
        
        # 3. Console (Restored name 'self.console')
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFixedHeight(100)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Monospace;")
        self.console.setPlaceholderText("Execution log...")
        self.inner_splitter.addWidget(self.console)
        
        # Set Ratios: Graph (70%), Plot (0%), Log (30%)
        self.inner_splitter.setStretchFactor(0, 7) 
        self.inner_splitter.setStretchFactor(1, 0)
        self.inner_splitter.setStretchFactor(2, 3)

        bottom_layout.addWidget(self.inner_splitter)
        
        # Add container to main splitter
        self.splitter.addWidget(self.bottom_container)

        # Set Main Splitter Ratios: Props (30%), Bottom (70%)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 7)

        self.layout.addWidget(self.splitter)
        
        # --- CONNECTIONS ---
        self.btn_fit.clicked.connect(self.view.fit_scene)
        self.btn_stop_loop.clicked.connect(self.on_stop_loop_clicked)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.scene.selectionChanged.connect(self.on_selection)

    # HELPER TO GET UNIQUE NODE TITLE
    def get_unique_title(self, base_title):
        """Generates a unique title like 'Gaussian Blur 1'."""
        existing_titles = [
            item.title for item in self.scene.items() 
            if isinstance(item, Node)
        ]
        
        if base_title not in existing_titles:
            return base_title
            
        counter = 1
        while True:
            new_title = f"{base_title} {counter}"
            if new_title not in existing_titles:
                return new_title
            counter += 1
    
    # --- Node Selection Handler ---
    def on_selection(self):
        """Rebuilds the property panel based on selection."""
        # 1. Clear current widgets
        while self.props_layout.count():
            child = self.props_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        # 2. Get selected node
        sel = self.scene.selectedItems()
        if len(sel) != 1 or not isinstance(sel[0], Node):
            self.props_layout.addRow(QLabel("Select a single node to edit parameters."))
            return

        node = sel[0]
        
        # --- HEADER ---
        # Display the node Title and ID (useful for debugging connections)
        id_label = QLabel(f"<span style='color:#888; font-size:10px;'>ID: {node.uid[:8]}...</span>")
        self.props_layout.addRow(QLabel(f"<b>{node.title}</b>"), id_label)
        
        # Spacer
        self.props_layout.addRow(QLabel("")) 

        # --- SECTION 1: PARAMETERS ---
        self.props_layout.addRow(QLabel("<u>Parameters</u>"))

        if node.node_type not in NODE_LIBRARY:
            self.props_layout.addRow(QLabel("No parameters defined."))
        else:
            params_def = NODE_LIBRARY[node.node_type]["parameters"]
            
            for param_name, conf in params_def.items():
                default_val = conf.get("default", "")
                current_val = node.parameters.get(param_name, default_val)
                # --- A. SPECIAL CASE: Dynamic Layer Selector ---
                # If this is the Input Node, populate the dropdown with REAL Napari layers
                if node.node_type == "get_layer" and param_name == "layer_name":
                    widget = QComboBox()
                    
                    # Get list of layer names from Napari viewer
                    layers = [layer.name for layer in self.viewer.layers]
                    
                    if not layers:
                        widget.addItem("No Layers Found")
                        widget.setEnabled(False)
                    else:
                        widget.addItems(layers)
                        
                        # Try to select the previously saved value
                        index = widget.findText(str(current_val))
                        if index >= 0:
                            widget.setCurrentIndex(index)
                        else:
                            # If saved layer was deleted, default to the first one
                            widget.setCurrentIndex(0)
                            # Update the node param immediately so it's valid
                            self.update_param(node, param_name, layers[0])

                    # Connect signal
                    widget.currentTextChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                    
                    self.props_layout.addRow("Select Layer:", widget)
                    continue  # Skip the standard logic below for this specific parameter
                
                # --- B. STANDARD WIDGETS ---
                
                # FLOAT
                if conf["type"] == "float":
                    widget = QDoubleSpinBox()
                    widget.setRange(conf.get("min", -9999.0), conf.get("max", 9999.0))
                    widget.setSingleStep(conf.get("step", 0.1))
                    widget.setValue(float(current_val))
                    widget.valueChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                
                # INT
                elif conf["type"] == "int":
                    widget = QSpinBox()
                    widget.setRange(conf.get("min", -9999), conf.get("max", 9999))
                    widget.setValue(int(current_val))
                    widget.valueChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                
                # BOOL
                elif conf["type"] == "bool":
                    widget = QCheckBox()
                    widget.setChecked(bool(current_val))
                    widget.toggled.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                
                # ENUM (Standard static dropdowns from library)
                elif conf["type"] == "enum":
                    widget = QComboBox()
                    widget.addItems(conf.get("options", []))
                    widget.setCurrentText(str(current_val))
                    widget.currentTextChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                
                # PATH
                elif conf["type"] == "path":
                    widget_container = QWidget()
                    layout = QHBoxLayout(widget_container)
                    layout.setContentsMargins(0, 0, 0, 0)
                    
                    line_edit = QLineEdit(str(current_val))
                    browse_btn = QPushButton("...")
                    browse_btn.setFixedWidth(30)
                    
                    layout.addWidget(line_edit)
                    layout.addWidget(browse_btn)
                    
                    # --- FIX 2: Pass 'conf' (c=conf) to capture the loop variable properly ---
                    def open_file_dialog(le=line_edit, n=node, k=param_name, c=conf):
                        mode = c.get("mode", "file") 
                        if mode == "directory":
                            path = QFileDialog.getExistingDirectory(self, "Select Directory")
                        else:
                            path, _ = QFileDialog.getOpenFileName(self, "Select File")
                            
                        if path:
                            le.setText(path)
                            self.update_param(n, k, path)

                    browse_btn.clicked.connect(lambda _: open_file_dialog())
                    line_edit.textChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                    
                    widget = widget_container
                
                # TABLE
                elif conf["type"] == "table":
                    limit = conf.get('max_rows', None) 
                    
                    is_row_unique = conf.get('row_unique', False)
                    allow_list = conf.get('allow_duplicates', [])
                    # Pass 'limit' to the class here:
                    widget = DynamicTableWidget(
                        conf.get('columns', []), 
                        max_rows=limit, 
                        row_unique=is_row_unique,
                        allow_duplicates=allow_list
                    )

                    # Load Data
                    val_to_load = current_val if current_val is not None else conf.get('value', [])
                    widget.set_value(val_to_load)

                    # Connect Signal
                    widget.valueChanged.connect(lambda data, n=node, k=param_name: self.update_param(n, k, data))
                    
                    self.props_layout.addRow(param_name.capitalize(), widget)
                    continue


                
                # STRING / OTHER
                else:
                    widget = QLineEdit(str(current_val))
                    widget.textChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))

                self.props_layout.addRow(param_name.capitalize(), widget)

        # Spacer
        self.props_layout.addRow(QLabel("")) 

        # --- SECTION 2: SOCKETS INFO ---
        self.props_layout.addRow(QLabel("<u>Connections</u>"))

        # A. Inputs
        if node.inputs:
            self.props_layout.addRow(QLabel("<b>Inputs:</b>"))
            for s in node.inputs:
                # Check connection status
                if s.connected_edges:
                    status_text = "Connected"
                    style = "color: #77dd77;" # Green
                else:
                    status_text = "Empty"
                    style = "color: #888;"   # Gray
                
                status_lbl = QLabel(status_text)
                status_lbl.setStyleSheet(style)
                self.props_layout.addRow(f"  \u25B8 {s.name}", status_lbl)
        
        # B. Outputs
        if node.outputs:
            self.props_layout.addRow(QLabel("<b>Outputs:</b>"))
            for s in node.outputs:
                count = len(s.connected_edges)
                status_text = f"{count} link(s)"
                style = "color: #ffb347;" if count > 0 else "color: #888;" # Orange if active
                
                status_lbl = QLabel(status_text)
                status_lbl.setStyleSheet(style)
                self.props_layout.addRow(f"  \u25B8 {s.name}", status_lbl)

        # C. Logic (control-flow) connections — always shown
        logic_in_count = sum(len(s.connected_edges) for s in node.logic_inputs)
        logic_out_count = sum(len(s.connected_edges) for s in node.logic_outputs)
        self.props_layout.addRow(QLabel("<b>Logic Flow:</b>"))

        li_text = f"{logic_in_count} link(s)" if logic_in_count else "Empty"
        li_style = "color: #8888cc;" if logic_in_count else "color: #888;"
        li_lbl = QLabel(li_text)
        li_lbl.setStyleSheet(li_style)
        self.props_layout.addRow("  ⟵ logic_in", li_lbl)

        lo_text = f"{logic_out_count} link(s)" if logic_out_count else "Empty"
        lo_style = "color: #8888cc;" if logic_out_count else "color: #888;"
        lo_lbl = QLabel(lo_text)
        lo_lbl.setStyleSheet(lo_style)
        self.props_layout.addRow("  ⟶ logic_out", lo_lbl)
        
        self.props_layout.addRow(QLabel("")) 

        # --- SECTION 3: PREVIEW LAYER ---
        self.props_layout.addRow(QLabel("<u>Result Preview</u>"))
        if hasattr(node, "cached_results") and node.cached_results:
            # Just grab the first value found in the dict
            first_key = next(iter(node.cached_results))
            data = node.cached_results[first_key]
            
            if isinstance(data, np.ndarray):
                # Convert
                pixmap = self.numpy_to_qpixmap(data)
                
                if pixmap:
                    # Create Label container
                    preview_lbl = QLabel()
                    preview_lbl.setPixmap(pixmap)
                    preview_lbl.setAlignment(Qt.AlignCenter)
                    preview_lbl.setStyleSheet("border: 1px solid #444; margin-top: 5px;")
                    
                    # Add to layout
                    self.props_layout.addRow(preview_lbl)
                    
                    # Add Info text (Shape/Type)
                    info_text = f"Shape: {data.shape}\nType: {data.dtype}"
                    self.props_layout.addRow(QLabel(f"<span style='color:#888; font-size:10px;'>{info_text}</span>"))
                else:
                    self.props_layout.addRow(QLabel("<em>Preview not available (Format?)</em>"))
            else:
                self.props_layout.addRow(QLabel("<em>Output is not an image.</em>"))
                
        else:
            # No data yet
            if node.status == "gray":
                self.props_layout.addRow(QLabel("<em>Node has changed. Run to update.</em>"))
            else:
                self.props_layout.addRow(QLabel("<em>No cached result.</em>"))


    def update_param(self, node, param_name, value):
        """Updates a parameter and invalidates the node."""
        node.parameters[param_name] = value
        # When a parameter changes, this node and all downstream nodes become "stale"
        self.set_node_status_recursive(node, "gray")
        self.scene.update() 

    def set_node_status_recursive(self, node, status):
        """Sets status of a node and recursively updates all downstream nodes."""
        # Optimization: If already that status, stop (prevents infinite loops in cyclic graphs)
        if node.status == status:
            return
            
        node.status = status
        
        # If turning gray, we must wipe the 'last_signature' so it knows to re-run next time
        if status == "gray":
            node.last_signature = None 
            
        # Propagate to children (nodes connected to my outputs)
        for output_socket in node.outputs:
            for edge in output_socket.connected_edges:
                if edge.end_socket:
                    child_node = edge.end_socket.node
                    self.set_node_status_recursive(child_node, status)

    # --- Save Pipeline Method ---
    def save_pipeline(self):
        # 1. Build the Pipeline Dictionary (Same as before)
        pipeline = {
            "schema_version": "0.1",
            "nodes": []
        }

        for item in self.scene.items():
            if isinstance(item, Node):
                # Map data connections
                input_connections = {}
                for socket in item.inputs:
                    if socket.connected_edges:
                        edge = socket.connected_edges[0] 
                        if edge.start_socket:
                            src_node = edge.start_socket.node
                            connection_str = f"{src_node.uid}.{edge.start_socket.name}"
                            input_connections[socket.name] = connection_str

                # Map logic connections (logic_in ← logic_out of other nodes)
                logic_connections = []
                for socket in item.logic_inputs:
                    for edge in socket.connected_edges:
                        if edge.start_socket:
                            src_node = edge.start_socket.node
                            logic_connections.append(
                                f"{src_node.uid}.{edge.start_socket.name}"
                            )

                node_data = {
                    "id": item.uid,
                    "type": item.node_type,
                    "label": item.title,
                    "position": {"x": item.pos().x(), "y": item.pos().y()},
                    "parameters": item.parameters,
                    "input_connections": input_connections,
                    "logic_connections": logic_connections,
                }
                pipeline["nodes"].append(node_data)

        # 2. Open File Browser to Save
        # Arguments: Parent, Title, Default Name, File Filter
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Pipeline", 
            "pipeline.json", 
            "JSON Files (*.json)"
        )

        # 3. Write to disk if user selected a file
        if filename:
            # Ensure .json extension is present
            if not filename.endswith(".json"):
                filename += ".json"
                
            try:
                with open(filename, "w") as f:
                    json.dump(pipeline, f, indent=2)
                # Optional: Feedback
                # print(f"Pipeline saved to {filename}")
            except Exception as e:
                print(f"Error saving pipeline: {e}")

    # --- Load Pipeline ---
    def load_pipeline(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Pipeline", "", "JSON Files (*.json)")
        if not filename: return

        self.scene.clear()
        with open(filename, "r") as f:
            pipeline = json.load(f)

        node_map = {} # Map UUID -> Node Object

        # Pass 1: Create all nodes
        for n_data in pipeline.get("nodes", []):
            pos = n_data.get("position", {"x": 0, "y": 0})
            
            # Create node with specific UID and Title from save file
            new_node = self.add_node(
                node_type=n_data.get("type", "generic"),
                pos=QPointF(pos["x"], pos["y"]),
                loaded_params=n_data.get("parameters", {}),
                loaded_uid=n_data.get("id"),
                loaded_title=n_data.get("label")
            )
            node_map[new_node.uid] = new_node

        # Pass 2: Reconnect Edges
        for n_data in pipeline.get("nodes", []):
            target_node = node_map.get(n_data["id"])
            if not target_node: continue
            
            connections = n_data.get("input_connections", {})
            
            # connection format: "target_input_name": "source_uid.source_output_name"
            for tgt_sock_name, src_string in connections.items():
                if "." not in src_string: continue
                
                src_uid, src_sock_name = src_string.split(".", 1)
                src_node = node_map.get(src_uid)
                
                if src_node:
                    # Find the actual socket objects
                    src_socket = next((s for s in src_node.outputs if s.name == src_sock_name), None)
                    tgt_socket = next((s for s in target_node.inputs if s.name == tgt_sock_name), None)
                    
                    if src_socket and tgt_socket:
                        conn = Connection(src_socket, self.scene)
                        conn.finalize(tgt_socket)
                        self.scene.addItem(conn)

            # Reconnect Logic edges
            for src_string in n_data.get("logic_connections", []):
                if "." not in src_string:
                    continue
                src_uid, src_sock_name = src_string.split(".", 1)
                src_node = node_map.get(src_uid)
                if src_node:
                    src_socket = next(
                        (s for s in src_node.logic_outputs if s.name == src_sock_name), None
                    )
                    tgt_socket = target_node.logic_inputs[0] if target_node.logic_inputs else None
                    if src_socket and tgt_socket:
                        conn = Connection(src_socket, self.scene)
                        conn.finalize(tgt_socket)
                        self.scene.addItem(conn)


    # --- Add Node Method ---
    # 1. The UI Logic (Dropdown)
    def open_add_menu(self):
        """Shows a categorized dropdown menu to add data-processing nodes."""
        self._open_add_menu(include_control=False)

    def open_control_menu(self):
        """Shows a dropdown menu with only control-flow nodes."""
        self._open_add_menu(include_control=True, control_only=True)

    def _open_add_menu(self, include_control=True, control_only=False):
        menu = QMenu(self)
        
        # Dictionary to hold reference to created submenus
        # Structure: {"Filters": QMenu_Object, "Segmentation": QMenu_Object}
        submenus = {}

        # Sort items so they appear alphabetically in the menu
        sorted_items = sorted(NODE_LIBRARY.items(), key=lambda x: x[1]["label"])

        for key, data in sorted_items:
            label = data["label"]
            category = data.get("category", "Uncategorized") # Default if missing
            is_control = category == "Control Flow"

            if control_only and not is_control:
                continue
            if not control_only and not include_control and is_control:
                continue

            # 1. Create the Submenu if it doesn't exist yet
            if category not in submenus:
                submenus[category] = menu.addMenu(category)
            
            # 2. Add the Action to the specific Submenu
            action = submenus[category].addAction(label)
            
            # 3. Connect the action using a lambda to capture the specific 'key'
            # We use checked=False (default for triggered) to ignore the boolean arg
            action.triggered.connect(lambda checked=False, k=key: self.add_node(k))

        if not submenus:
            action = menu.addAction("No nodes available")
            action.setEnabled(False)

        # Show the menu at the mouse cursor position
        menu.exec_(QCursor.pos())

    # 2. The Creation Logic (Scene Manipulation)
    def add_node(self, node_type, pos=None, loaded_params=None, loaded_uid=None, loaded_title=None):
        if not pos:
            view_rect = self.view.viewport().rect()
            pos = self.view.mapToScene(view_rect.center())
            
        # Determine title
        base_label = "Node"
        if node_type in NODE_LIBRARY:
            base_label = NODE_LIBRARY[node_type]["label"]
            
        final_title = loaded_title if loaded_title else self.get_unique_title(base_label)

        # Create Node (Pass UID if loading, otherwise None to generate new)
        node = Node(pos.x(), pos.y(), node_type=node_type, title=final_title, uuid_str=loaded_uid, scene=self.scene)
        
        if loaded_params:
            node.parameters.update(loaded_params)
            
        self.scene.addItem(node)
        return node
    
    # --- Remove Node Method ---
    def open_remove_menu(self):
        """Shows a QMenu list of existing nodes to remove."""
        # 1. Find all nodes currently in the scene
        nodes = [item for item in self.scene.items() if isinstance(item, Node)]
        
        if not nodes:
            # Optional: Show a disabled menu item saying "Empty"
            return

        menu = QMenu(self)
        
        # 2. Sort them alphabetically by title so the list is readable
        nodes.sort(key=lambda n: n.title)

        for node in nodes:
            # Create a label. We add the ID just in case you have two nodes named "Gaussian Blur"
            # so you can distinguish them (or you can just use node.title if you prefer)
            label = f"{node.title} ({id(node)})"
            
            action = menu.addAction(label)
            
            # 3. Connect the action. 
            # We pass the specific 'node' object to the lambda.
            action.triggered.connect(lambda checked=False, n=node: self.delete_node(n))

        # 4. Show menu at cursor position
        menu.exec_(QCursor.pos())

    def delete_node(self, node):
        """Helper function to safely remove a specific node, its edges, AND its sockets."""
        
        # Iterate over ALL sockets (data + logic)
        for socket in node.all_sockets():
            # 1. Remove all edges connected to this socket
            for edge in list(socket.connected_edges):
                # Also clean the other end's edge list
                other = edge.end_socket if edge.start_socket is socket else edge.start_socket
                if other and edge in other.connected_edges:
                    other.connected_edges.remove(edge)
                self.scene.removeItem(edge)
            
            # 2. Remove the socket itself from the scene
            self.scene.removeItem(socket)
        
        # 3. Finally, remove the node body
        self.scene.removeItem(node)
    
    # --- Run Pipeline Method ---
    def run_pipeline(self):
        # 1. Disable UI
        self.set_ui_enabled(False)
        self.console.clear()
        
        # 2. Setup Thread
        self.thread = QThread()
        self.worker = ExecutionWorker(self.scene, self.viewer)
        self.worker.moveToThread(self.thread)
        
        # 3. Connect Signals
        self.thread.started.connect(self.worker.run)
        self.worker.node_status_signal.connect(self.update_node_status)
        
        # LOGGING
        self.worker.log_signal.connect(self.append_log)
        
        # RESULTS (Critical Fix: GUI updates happen here in Main Thread)
        self.worker.result_signal.connect(self.handle_execution_result)

        # INTERACTIVE NODES – dialog on the main thread
        self.worker.interaction_request_signal.connect(
            self.handle_interaction_request
        )
        self.worker.loop_control_state_signal.connect(
            self.handle_loop_control_state
        )
        
        # CLEANUP
        self.worker.finished_signal.connect(self.thread.quit)
        self.worker.finished_signal.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # RE-ENABLE UI
        self.worker.finished_signal.connect(lambda: self.set_ui_enabled(True))
        
        # 4. Start
        self.thread.start()

    def on_stop_loop_clicked(self):
        """Signal the worker to stop the active 'Until confirm' loop."""
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.request_loop_stop()
            self.btn_stop_loop.setEnabled(False)
            self.btn_stop_loop.setText("Stopping...")
        # If an interactive dialog is currently open, close it immediately.
        if self._active_interaction_dialog is not None:
            try:
                self._active_interaction_dialog.reject()
            except Exception:
                pass

    def handle_loop_control_state(self, active, loop_node_uid):
        """Show/hide the loop stop button while an 'Until confirm' loop is active."""
        if active:
            self.btn_stop_loop.setVisible(True)
            self.btn_stop_loop.setEnabled(True)
            self.btn_stop_loop.setText("Stop Loop")
        else:
            self.btn_stop_loop.setVisible(False)
            self.btn_stop_loop.setEnabled(False)
            self.btn_stop_loop.setText("Stop Loop")

    def update_node_status(self, node_uid, status):
        """
        Received from Worker Thread. 
        Finds the node by UID and updates its color.
        """
        for item in self.scene.items():
            # We check isinstance to be safe
            if isinstance(item, Node) and item.uid == node_uid:
                item.status = status
                item.update() # Force repaint of the dot
                break

    # -----------------------------------------------------------------
    # INTERACTIVE NODE DIALOG
    # -----------------------------------------------------------------
    def handle_interaction_request(self, node_uid, config):
        """
        Called on the **main thread** when an interactive node needs user input.

        ``config`` example::

            {
                "layer_type": "shapes",
                "mode": "add_rectangle",
                "edge_color": "#00ff00",
                "face_color": [0, 0, 0, 0],
                "edge_width": 3,
                "prompt": "Draw a rectangle on the image, then click Run.",
            }

        Flow:
          1. Create a temporary napari layer with the requested settings.
          2. Show a small dialog with the prompt text and a **Run** / **Cancel** button.
          3. When Run is clicked, collect the drawn data, remove the temp layer,
             and call ``worker.provide_interaction_result(data)``.
        """
        layer_type = config.get("layer_type", "shapes")
        layer_name = f"__interactive_{node_uid[:8]}__"
        prompt = config.get("prompt", "Draw on the layer, then click Run.")

        # --- 1. Create temporary layer ---
        try:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

            if layer_type == "shapes":
                layer_kwargs = {"name": layer_name}
                for key in ("edge_color", "face_color", "edge_width"):
                    if key in config:
                        layer_kwargs[key] = config[key]
                roi_layer = self.viewer.add_shapes(**layer_kwargs)
                mode = config.get("mode", "add_rectangle")
                roi_layer.mode = mode
                self.viewer.layers.selection.active = roi_layer
            else:
                # Future extensibility: points, labels, etc.
                roi_layer = self.viewer.add_shapes(name=layer_name)
                roi_layer.mode = "add_rectangle"
                self.viewer.layers.selection.active = roi_layer
        except Exception as exc:
            print(f"❌ Failed to create interactive layer: {exc}")
            self.worker.provide_interaction_result(None)
            return

        # --- 2. Build dialog ---
        dialog = QDialog(self)
        self._active_interaction_dialog = dialog
        dialog.setWindowTitle("Interactive Node")
        dialog.setMinimumWidth(320)
        dlg_layout = QVBoxLayout(dialog)

        # Prompt label
        lbl = QLabel(prompt)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size: 13px; padding: 8px;")
        dlg_layout.addWidget(lbl)

        # Button row
        btn_layout = QHBoxLayout()
        btn_run = QPushButton("▶  Run")
        btn_run.setStyleSheet(
            "QPushButton { background-color: #2E7D32; color: white; "
            "font-weight: bold; font-size: 13px; padding: 6px 20px; "
            "border-radius: 4px; }"
            "QPushButton:hover { background-color: #388E3C; }"
        )
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(
            "QPushButton { background-color: #555; color: white; "
            "font-size: 13px; padding: 6px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #777; }"
        )
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_run)
        dlg_layout.addLayout(btn_layout)

        # --- 3. Connect buttons ---
        # Guard against double-fire (Cancel button also emits rejected)
        _resolved = {"done": False}

        def _cleanup_layer():
            try:
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)
            except Exception:
                pass

        def _on_run():
            if _resolved["done"]:
                return
            _resolved["done"] = True
            try:
                data = [np.array(s) for s in roi_layer.data] if len(roi_layer.data) > 0 else None
            except Exception:
                data = None
            _cleanup_layer()
            self._active_interaction_dialog = None
            dialog.accept()
            self.worker.provide_interaction_result(data)

        def _on_cancel():
            if _resolved["done"]:
                return
            _resolved["done"] = True
            _cleanup_layer()
            self._active_interaction_dialog = None
            dialog.reject()
            self.worker.provide_interaction_result(None)

        def _on_rejected():
            # Fired by the X button OR by _on_cancel's dialog.reject().
            # The guard ensures we only call provide_interaction_result once.
            if _resolved["done"]:
                return
            _resolved["done"] = True
            _cleanup_layer()
            self._active_interaction_dialog = None
            self.worker.provide_interaction_result(None)

        btn_run.clicked.connect(_on_run)
        btn_cancel.clicked.connect(_on_cancel)
        dialog.rejected.connect(_on_rejected)

        # Show non-modal so the user can still interact with the napari canvas
        dialog.setModal(False)
        dialog.show()

    def handle_execution_result(self, node_title, output_name, data):

        # DEBUG

        if node_title == "Gaussian Blur":
            print(f"\n🟦 UI GOT GAUSS | output_name={output_name!r} type(data)={type(data)}")
            if isinstance(data, tuple):
                print("   tuple len:", len(data), "types:", [type(x) for x in data])
                if len(data) == 2 and isinstance(data[1], dict):
                    print("   meta name:", data[1].get("name"), "multiscale:", data[1].get("multiscale"))

        ###

        # 1. Update Cache

        node_obj = next((item for item in self.scene.items() 
                        if isinstance(item, Node) and item.title == node_title), None)
        if node_obj:
            node_obj.cached_results[output_name] = data
            node_obj.status = "green"

        # --- A. UNPACK ---
        display_data = data
        display_meta = {}
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
            display_data = data[0]
            display_meta = data[1]


        # --- B. PLOT DASHBOARD HANDLER ---
        # If it is a plot, SHOW the dashboard
        is_plot = False
        type_str = str(type(display_data))
        if "matplotlib" in type_str and "Figure" in type_str:
            is_plot = True
        elif hasattr(display_data, "canvas"): 
            is_plot = True

        if is_plot:
            self.plot_dashboard.display(display_data) # <--- Pop up the widget!
            return

        # --- C. NON-VISUAL FILTER ---
        import pandas as pd
        if isinstance(display_data, (pd.DataFrame, dict, str)):
            return

        # --- D. DISPLAY HELPER ---
        def add_layer_to_viewer(layer_data, raw_meta):
            # 1. Prepare Name
            layer_name = raw_meta.get("name", f"{node_title} Output")
            #DEBUG
            if node_title == "Gaussian Blur":
                print("🟩 UI ADD GAUSS layer_name =", layer_name, "layer_type=", raw_meta.get("layer_type"))
            ####

            # 2. Filter Metadata
            valid_napari_args = {
                "name", "opacity", "blending", "visible", "multiscale",
                "colormap", "contrast_limits", "gamma", "rgb",
                "scale", "translate", "rotate", "shear", "affine",
            }
            napari_kwargs = {"name": layer_name}
            custom_metadata = {}

            for k, v in raw_meta.items():
                if k in valid_napari_args:
                    napari_kwargs[k] = v
                else:
                    custom_metadata[k] = v

            napari_kwargs["metadata"] = custom_metadata

            # 3. Create/Update Layer
            try:
                is_multiscale = isinstance(layer_data, list) and len(layer_data) > 0 and hasattr(layer_data[0], "shape")

                # Allow multiscale pyramids: list of arrays
                if not hasattr(layer_data, "shape"):
                    if not is_multiscale:
                        print("⚠️ Not displayable layer_data:", type(layer_data))
                        return
                    napari_kwargs["multiscale"] = True

                # If already exists, try to update in-place
                if layer_name in self.viewer.layers:
                    layer = self.viewer.layers[layer_name]
                    need_recreate = False

                    if is_multiscale:
                        # For multiscale: compare level0 shapes
                        existing_data = layer.data
                        existing_is_multi = isinstance(existing_data, list) and len(existing_data) > 0
                        if existing_is_multi:
                            if existing_data[0].shape != layer_data[0].shape:
                                need_recreate = True
                            else:
                                layer.data = layer_data
                                layer.metadata.update(custom_metadata)
                                return
                        else:
                            # Existing is single array, new is multiscale → recreate
                            need_recreate = True
                    else:
                        # Single array path
                        if hasattr(layer_data, "shape") and hasattr(layer.data, "shape") and layer.data.shape != layer_data.shape:
                            need_recreate = True
                        else:
                            layer.data = layer_data
                            layer.metadata.update(custom_metadata)
                            return

                    if need_recreate:
                        self.viewer.layers.remove(layer_name)

                # Create new
                if is_multiscale:
                    bad = [type(x) for x in layer_data if not (hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "ndim"))]
                    if bad:
                        raise TypeError(f"Multiscale list contains non-array-like items: {bad}")
                    napari_kwargs["multiscale"] = True

                layer_type = raw_meta.get("layer_type", None)

                print("🟢 ADDING:", layer_name, "layer_type=", layer_type,
                      "data_type=", type(layer_data), "multiscale=", napari_kwargs.get("multiscale"))

                if layer_type == "labels":
                    self.viewer.add_labels(layer_data, **napari_kwargs)
                else:
                    self.viewer.add_image(layer_data, **napari_kwargs)

            except Exception as e:
                import traceback
                print(f"❌ Error displaying layer '{layer_name}': {repr(e)}")
                traceback.print_exc()

        # --- E. HANDLE NAPARI READER OUTPUTS (LayerDataTuple) ---
        def is_layer_data_tuple(x):
            return (
                isinstance(x, tuple)
                and len(x) == 3
                and isinstance(x[1], dict)
                and isinstance(x[2], str)
            )

        def merge_meta(base, extra):
            m = dict(base) if isinstance(base, dict) else {}
            if isinstance(extra, dict):
                m.update(extra)
            return m

        # Guard: if a node still returns (layers, plugin_impl), keep only layers
        if isinstance(display_data, tuple) and len(display_data) == 2 and isinstance(display_data[0], list):
            display_data = display_data[0]

        # Case 1: list of LayerDataTuples -> add all layers and stop
        if isinstance(display_data, list) and len(display_data) and isinstance(display_data[0], tuple) and len(display_data[0]) == 3:
            for data, meta, layer_type in display_data:
                meta = dict(meta)
                meta["layer_type"] = layer_type
                add_layer_to_viewer(data, meta)
            return
        # Case 2: single LayerDataTuple -> add it and stop
        if is_layer_data_tuple(display_data):
            ld, lm, lt = display_data
            lm2 = merge_meta(display_meta, lm)
            lm2.setdefault("layer_type", lt)
            add_layer_to_viewer(ld, lm2)
            return

        # Default: treat as a normal (array, meta) output
        add_layer_to_viewer(display_data, display_meta)





    def append_log(self, text):
        self.console.append(text)
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_ui_enabled(self, enabled: bool):
        """Locks/Unlocks buttons during execution."""
        self.btn_run.setEnabled(enabled)
        self.btn_add.setEnabled(enabled)
        self.btn_add_control.setEnabled(enabled)
        self.btn_remove.setEnabled(enabled)
        self.btn_load.setEnabled(enabled)
        self.btn_save.setEnabled(enabled)
        self.btn_import.setEnabled(enabled)
        if enabled:
            self.btn_stop_loop.setVisible(False)
            self.btn_stop_loop.setEnabled(False)
            self.btn_stop_loop.setText("Stop Loop")
        
        if enabled:
            self.btn_run.setText("RUN PIPELINE")
            self.btn_run.setStyleSheet("""
                QPushButton {
                    background-color: #2E7D32;
                    color: white;
                    font-weight: bold;
                    font-size: 14px;
                    border-radius: 4px;
                    border: 1px solid #1B5E20;
                }
                QPushButton:hover {
                    background-color: #388E3C;
                }
                QPushButton:pressed {
                    background-color: #1B5E20;
                }
            """)

        else:
            self.btn_run.setText("Running...")
            self.btn_run.setStyleSheet("background-color: #555; color: #aaa;") # Grayed out

    # --- Import Custom Module Method ---
    def import_custom_module(self):
        """Loads a user-selected .py file and registers its nodes."""
        filename, _ = QFileDialog.getOpenFileName(self, "Import Custom Node", "", "Python Files (*.py)")
        if not filename:
            return

        try:
            # 1. Load the module dynamically from the file path
            module_name = os.path.basename(filename).replace(".py", "")
            spec = importlib.util.spec_from_file_location(module_name, filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module # Register in sys.modules so pickles work
            spec.loader.exec_module(module)

            # 2. Scan for nodes (Reuse generator logic)
            count = 0
            for name, func in inspect.getmembers(module, inspect.isfunction):
                if getattr(func, "_is_flow_node", False):
                    meta = func._node_meta
                    sig = inspect.signature(func)
                    
                    # Extract inputs/params (Simplified logic)
                    inputs = []
                    parameters = {}
                    
                    for param_name, param in sig.parameters.items():
                        # Skip engine-injected interaction kwarg
                        if param_name == "interaction":
                            continue
                        if param.default == inspect.Parameter.empty:
                            inputs.append(param_name)
                        else:
                            # We need a simple type mapper here or import get_type_name
                            p_type = "string"
                            if param.annotation == float: p_type = "float"
                            elif param.annotation == int: p_type = "int"
                            elif param.annotation == bool: p_type = "bool"
                            elif param.annotation == str: p_type = "enum"
                            
                            extra_config = meta["params_config"].get(param_name, {})
                            param_def = {"type": p_type, "default": param.default}
                            param_def.update(extra_config)
                            parameters[param_name] = param_def

                    # 3. Update the Global Library
                    # CRITICAL: We pass the 'executable' function object directly!
                    node_key = name
                    entry = {
                        "label": meta["label"],
                        "category": "Custom", # Force category or use meta['category']
                        "inputs": inputs,
                        "outputs": meta["outputs"],
                        "parameters": parameters,
                        "execution_path": "custom_loaded", 
                        "executable": func # <--- DIRECT REFERENCE
                    }
                    # Persist interactive config if present
                    interactive_cfg = meta.get("interactive")
                    if interactive_cfg:
                        entry["interactive"] = interactive_cfg
                    logic_cfg = meta.get("logic")
                    if logic_cfg:
                        entry["logic"] = logic_cfg
                    NODE_LIBRARY[node_key] = entry
                    count += 1
            
            if count > 0:
                QMessageBox.information(self, "Success", f"Imported {count} nodes from {module_name}")
            else:
                QMessageBox.warning(self, "Warning", "No nodes found with @register_node decorator.")

        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    # --- Export to Python Script Method ---
    def export_to_python(self):
        """Generates and saves the pipeline as a .py file."""
        # 1. Generate Code
        generator = ScriptGenerator(self.scene)
        code = generator.generate()
        
        # 2. Open Save Dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Python Script", 
            "pipeline.py", 
            "Python Files (*.py)"
        )
        
        if filename:
            if not filename.endswith(".py"):
                filename += ".py"
            
            try:
                with open(filename, "w") as f:
                    f.write(code)
                QMessageBox.information(self, "Success", f"Script exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def numpy_to_qpixmap(self, array):
        """Converts NumPy array to QPixmap with automatic coloring for Labels/Masks."""
        if array is None:
                    return None
                
        # --- FIX: Handle Multiscale Pyramid (List) ---
        if isinstance(array, list):
            if len(array) == 0: return None
            # Take the highest resolution (Level 0) for the preview
            array = array[0]

        # --- FIX: Handle Dask Arrays (Compute small slice) ---
        if hasattr(array, "compute"):
            # It's Dask. We cannot check .shape immediately if it's lazy, 
            # but usually dask has .shape. We need a real numpy array for QImage.
            try:
                # Take middle slice if 3D
                if array.ndim == 3:
                     mid = array.shape[0] // 2
                     array = array[mid]
                # Compute just this 2D slice
                array = array.compute() 
            except:
                return None

        # Standard Safety Check
        if not hasattr(array, "ndim") or not hasattr(array, "shape"):
            return None
        if array.ndim == 3 and array.shape[-1] in [3, 4]:
            display_data = array
            height, width, channels = display_data.shape
            
            # Create QImage directly from RGB data
            # Ensure it is uint8
            if display_data.dtype != np.uint8:
                display_data = (display_data * 255).astype(np.uint8)
                
            fmt = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
            q_img = QImage(display_data.data, width, height, channels * width, fmt)
            
            return QPixmap.fromImage(q_img).scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # 1. Handle 3D Data (Take middle slice)
        if array.ndim == 3:
            mid_z = array.shape[0] // 2
            display_data = array[mid_z, :, :]
        elif array.ndim == 2:
            display_data = array
        else:
            return None 
            
        height, width = display_data.shape
        
        # --- A. BOOLEAN MASK (True/False) -> Cyan/Black ---
        if display_data.dtype == bool:
            # Create RGB container (Height, Width, 3)
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Set True pixels to Cyan (0, 255, 255)
            # You can change this color to whatever you like!
            rgb_image[display_data] = [0, 255, 255] 
            
            q_img = QImage(rgb_image.data, width, height, 3 * width, QImage.Format_RGB888)
            
        # --- B. LABELS (Integers) -> Random Colors ---
        elif np.issubdtype(display_data.dtype, np.integer):
            # Normalize to 0-255 just to see structure, OR apply a color map.
            # A simple trick for labels is to multiply by a prime number to scramble colors
            # This makes label 1 and label 2 look very different.
            
            # Create RGB container
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Simple pseudo-random coloring based on label ID
            # We use prime numbers to shuffle the bits for R, G, B
            mask = display_data > 0
            ids = display_data[mask]
            
            rgb_image[mask, 0] = (ids * 157) % 255  # Red channel
            rgb_image[mask, 1] = (ids * 31) % 255   # Green channel
            rgb_image[mask, 2] = (ids * 73) % 255   # Blue channel
            
            q_img = QImage(rgb_image.data, width, height, 3 * width, QImage.Format_RGB888)

        # --- C. STANDARD IMAGE (Floats) -> Grayscale ---
        else:
            # Normalize to 0-255
            data = display_data.astype(float)
            d_min, d_max = np.min(data), np.max(data)
            
            if d_max > d_min:
                norm = (data - d_min) / (d_max - d_min) * 255
            else:
                norm = np.zeros_like(data)
                
            norm = norm.astype(np.uint8)
            
            # Use Grayscale format
            q_img = QImage(norm.data, width, height, width, QImage.Format_Grayscale8)
        
        # Convert to Pixmap and Scale
        pixmap = QPixmap.fromImage(q_img)
        return pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    


    def save_to_zarr(self):
        layer = self.viewer.layers.selection.active
        if not layer:
            QMessageBox.warning(self, "No Selection", "Please select a layer to save.")
            return

        # 1. Select Parent Folder
        path = QFileDialog.getExistingDirectory(self, "Select Zarr Group (Parent Folder)")
        if not path:
            return
            
        # --- SAFETY CHECK: Are we inside an Image Group? ---
        # If the user selected a .zarr folder that already has '0', '1' inside it...
        if os.path.exists(os.path.join(path, "0")) and os.path.exists(os.path.join(path, ".zattrs")):
            reply = QMessageBox.question(
                self, "Structure Warning",
                "The selected Zarr file appears to be a Single Image (it contains raw data at the root).\n\n"
                "You cannot save a new layer INSIDE it without restructuring it first.\n\n"
                "Do you want to create a NEW Zarr file next to it instead?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Move selection up one level (outside the .zarr)
                path = os.path.dirname(path)
            else:
                # User insists on saving inside. This will break standard readers unless they know what they are doing.
                # We proceed, but know that 'drag-and-drop' might fail.
                pass
        # ---------------------------------------------------

        # 2. Ask for Name
        default_name = f"{layer.name}_processed.zarr"
        dataset_name, ok = QInputDialog.getText(self, "Save to Zarr", 
                                            "Dataset Name:", 
                                            text=default_name)
        if not ok or not dataset_name:
            return

        full_path = os.path.join(path, dataset_name)
        
        # ... (Rest of the saving logic remains the same) ...
        # ... (Copy the 'data_to_save' logic from the previous fix) ...
        
        raw_data = layer.data
        is_multiscale = getattr(layer, 'multiscale', False)
        
        if is_multiscale:
            data_to_save = list(raw_data)
            print(f"📦 Detected Napari MultiScaleData. Converted to list.")
        else:
            data_to_save = raw_data

        try:
            root_group = zarr.open_group(full_path, mode='w')

            def save_single_array(array, component_name):
                # (Use the robust helper from before)
                if isinstance(array, da.Array):
                    da.to_zarr(array, url=full_path, component=component_name, overwrite=True)
                elif isinstance(array, np.ndarray):
                    root_group.create_dataset(component_name, data=array, overwrite=True)
                elif isinstance(array, (list, tuple)):
                    save_single_array(array[0], component_name)
                else:
                    root_group.create_dataset(component_name, data=np.asarray(array), overwrite=True)

            if isinstance(data_to_save, (list, tuple)):
                for i, level_array in enumerate(data_to_save):
                    save_single_array(level_array, str(i))
                root_group.attrs["multiscales"] = [{
                    "version": "0.4",
                    "datasets": [{"path": str(i)} for i in range(len(data_to_save))]
                }]
            else:
                save_single_array(data_to_save, "0")

            QMessageBox.information(self, "Success", f"Layer saved to:\n{full_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Could not save Zarr:\n{e}")
