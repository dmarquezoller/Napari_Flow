from qtpy.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsRectItem, QMainWindow,
    QGraphicsDropShadowEffect, QToolBar, QInputDialog, QFileDialog, QVBoxLayout,
    QHBoxLayout, QPushButton, QMenu, QWidget, QGroupBox, QFormLayout, QLabel,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QScrollArea, QBoxLayout,
    QFrame, QMessageBox, QTextEdit, QSplitter, QDialog, QTableWidget, QHeaderView, QAbstractItemView
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
from matplotlib.figure import Figure
from napari_flow_editor import generate_library
from .execution_engine import ExecutionWorker
from .script_generator import ScriptGenerator


NODE_LIBRARY = {}

# --- CUSTOM WIDGET: TABLE WITH ADD/REMOVE BUTTONS ---
class DynamicTableWidget(QWidget):
    valueChanged = Signal(list)

    def __init__(self, column_config, max_rows=None, row_unique=False, allow_duplicates=None):
        """
        Args:
            allow_duplicates (list): Values that are allowed to be repeated 
                                     (e.g. ["-"] for dimension mapping).
        """
        super().__init__()
        self.column_config = column_config 
        self.max_rows = max_rows
        self.row_unique = row_unique
        
        # Convert to set for faster lookup
        self.allow_duplicates = set(allow_duplicates) if allow_duplicates else set()
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableWidget()
        self.table.setColumnCount(len(column_config) + 1)
        
        headers = [c['label'] for c in column_config] + [""]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(len(column_config), QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setMinimumHeight(150)
        
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton(" + Add Rule ")
        btn_layout.addWidget(self.btn_add)
        
        self.layout.addWidget(self.table)
        self.layout.addLayout(btn_layout)
        self.setLayout(self.layout)

        self.btn_add.clicked.connect(self.add_row)
        
        self._update_ui_state()

    def _update_ui_state(self):
        if self.max_rows is not None:
            current_count = self.table.rowCount()
            if current_count >= self.max_rows:
                self.btn_add.setDisabled(True)
                self.btn_add.setText(f"Limit Reached ({self.max_rows} Max)")
            else:
                self.btn_add.setDisabled(False)
                self.btn_add.setText(" + Add Rule ")

    def add_row(self):
        if self.max_rows is not None and self.table.rowCount() >= self.max_rows:
            return

        # We block signals momentarily to prevent partial updates while building
        self.blockSignals(True)
        row_idx = self.table.rowCount()
        self.table.insertRow(row_idx)
        
        for col_idx, col_def in enumerate(self.column_config):
            self._set_cell_widget(row_idx, col_idx, col_def, col_def.get('value'))
            
        self._add_delete_btn(row_idx)
        
        # New Step: Update constraints immediately so the new row respects existing choices
        self._update_dropdown_constraints()
        
        self.blockSignals(False)
        self.emit_change()
        self._update_ui_state()

    def _add_delete_btn(self, row_idx):
        btn_del = QPushButton("✖")
        btn_del.setFixedSize(24, 24)
        btn_del.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        btn_del.clicked.connect(self.remove_row_via_button)
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0,0,0,0)
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(btn_del)
        self.table.setCellWidget(row_idx, len(self.column_config), container)

    def remove_row_via_button(self):
        btn_clicked = self.sender()
        if not btn_clicked: return
        del_col = len(self.column_config)
        for r in range(self.table.rowCount()):
            container = self.table.cellWidget(r, del_col)
            if container and container.findChild(QPushButton) == btn_clicked:
                self.table.removeRow(r)
                self._update_dropdown_constraints() # <--- Update others when a row is deleted
                self.emit_change()
                self._update_ui_state()
                return

    def _set_cell_widget(self, r, c, col_def, value=None):
        val_type = col_def.get('type', 'text')
        
        if val_type == 'enum':
            w = QComboBox()
            # Store the FULL list of options in the widget property 
            # so we can always restore them even after filtering.
            full_options = col_def.get('options', [])
            w.setProperty("all_options", full_options)
            w.addItems(full_options)
            
            if value: w.setCurrentText(str(value))
            
            # Connect change to constraint updater
            w.currentTextChanged.connect(lambda _: self._on_value_changed())
            
        elif val_type == 'float':
            w = QDoubleSpinBox()
            w.setRange(-1e9, 1e9)
            if value is not None: w.setValue(float(value))
            w.valueChanged.connect(lambda _: self.emit_change())
            
        else:
            w = QLineEdit()
            if value: w.setText(str(value))
            w.textChanged.connect(lambda _: self.emit_change())
            
        self.table.setCellWidget(r, c, w)

    def _on_value_changed(self):
        """Helper to run constraints then emit signal"""
        self._update_dropdown_constraints()
        self.emit_change()

    def _update_dropdown_constraints(self):
        rows = self.table.rowCount()
        cols = self.table.columnCount() - 1 

        # 1. Snapshot Current State
        grid_state = {}
        for r in range(rows):
            for c in range(cols):
                w = self.table.cellWidget(r, c)
                if isinstance(w, QComboBox):
                    grid_state[(r, c)] = w.currentText()

        # 2. Update Every ComboBox
        for r in range(rows):
            for c in range(cols):
                w = self.table.cellWidget(r, c)
                if not isinstance(w, QComboBox): continue

                col_def = self.column_config[c]
                forbidden = set()

                # A. Vertical Uniqueness (Column Logic)
                if col_def.get('unique', False):
                    for other_r in range(rows):
                        if other_r == r: continue
                        val = grid_state.get((other_r, c))
                        # ✅ NEW: If val is in allow_duplicates, do NOT forbid it
                        if val and val not in self.allow_duplicates: 
                            forbidden.add(val)

                # B. Horizontal Uniqueness (Row Logic)
                if self.row_unique:
                    for other_c in range(cols):
                        if other_c == c: continue
                        val = grid_state.get((r, other_c))
                        # ✅ NEW: If val is in allow_duplicates, do NOT forbid it
                        if val and val not in self.allow_duplicates: 
                            forbidden.add(val)

                # 3. Refill Items
                current_val = w.currentText()
                all_options = w.property("all_options") or []
                
                # Valid items: Not forbidden OR Is the current value
                valid_items = [opt for opt in all_options if opt not in forbidden or opt == current_val]
                
                current_items = [w.itemText(i) for i in range(w.count())]
                if valid_items != current_items:
                    w.blockSignals(True)
                    w.clear()
                    w.addItems(valid_items)
                    w.setCurrentText(current_val)
                    w.blockSignals(False)

    def emit_change(self):
        self.valueChanged.emit(self.get_value())

    def get_value(self):
        data = []
        for r in range(self.table.rowCount()):
            row_data = {}
            for c, col_def in enumerate(self.column_config):
                w = self.table.cellWidget(r, c)
                if isinstance(w, QComboBox): val = w.currentText()
                elif isinstance(w, (QSpinBox, QDoubleSpinBox)): val = w.value()
                elif isinstance(w, QLineEdit): val = w.text()
                else: val = None
                row_data[col_def['name']] = val
            data.append(row_data)
        return data

    def set_value(self, data):
        self.blockSignals(True)
        self.table.setRowCount(0)
        if isinstance(data, list):
            limit = self.max_rows if self.max_rows is not None else float('inf')
            for i, row_dict in enumerate(data):
                if i >= limit: break 
                row_idx = self.table.rowCount()
                self.table.insertRow(row_idx)
                for c, col_def in enumerate(self.column_config):
                    if col_def['name'] in row_dict:
                        self._set_cell_widget(row_idx, c, col_def, row_dict[col_def['name']])
                self._add_delete_btn(row_idx)
        
        # Apply constraints after loading
        self._update_dropdown_constraints()
        self.blockSignals(False)
        self._update_ui_state()

# --- SOCKET -----------------------------------------------------
class Socket(QGraphicsEllipseItem):
    def __init__(self, node, socket_type, name, index, total_sockets):
        super().__init__(-6, -6, 12, 12)
        self.socket_type = socket_type
        self.node = node
        self.name = name # e.g. "image_in"
        self.connected_edges = []
        
        # Calculate Y position to distribute sockets evenly
        h = node.rect().height()
        y = (h / (total_sockets + 1)) * (index + 1)
        
        x = 0 if socket_type == "input" else node.rect().width()
        
        self.local_offset = QPointF(x, y)
        
        color = QColor("#ffb347") if socket_type == "output" else QColor("#77dd77")
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
        return not (self.socket_type == "input" and len(self.connected_edges) >= 1)


# --- CONNECTION -------------------------------------------------
class Connection(QGraphicsPathItem):
    def __init__(self, start_socket, scene):
        super().__init__()
        self.start_socket = start_socket
        self.end_socket = None
        self.scene_ref = scene
        self.dragging = False
        self.hovered = False
        self.setZValue(1)
        self.setPen(QPen(QColor("#444"), 2))
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.update_path(self.start_socket.center_pos(), self.start_socket.center_pos())

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(20)
        return stroker.createStroke(self.path())

    def hoverEnterEvent(self, event):
        self.setPen(QPen(QColor("#0078d7"), 3))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
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
            target = self.scene_ref.find_nearby_socket(event.scenePos())
            if (
                target
                and target.socket_type == "input"
                and target.can_accept_connection()
                and not self.scene_ref.are_already_connected(self.start_socket, target)
                and not self.scene_ref.creates_cycle(self.start_socket.node, target.node)
            ):
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
        self.uid = uuid_str if uuid_str else str(uuid.uuid4())
        self.parameters = {}

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

        # 4. Generate Sockets dynamically
        self.inputs = []
        for i, name in enumerate(inputs_data):
            self.inputs.append(Socket(self, "input", name, i, len(inputs_data)))

        self.outputs = []
        for i, name in enumerate(outputs_data):
            self.outputs.append(Socket(self, "output", name, i, len(outputs_data)))

        if scene:
            for s in self.inputs + self.outputs:
                scene.addItem(s)

    # ... (keep itemChange and paint methods same as before) ...
    def itemChange(self, change, value):
        if change in (QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged, QGraphicsItem.GraphicsItemChange.ItemPositionChange):
            for s in self.inputs + self.outputs:
                s.update_position()
                for e in s.connected_edges:
                    e.update_positions()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget):
        rect = self.rect()

        # A. Shadow
        painter.fillRect(rect.adjusted(4, 4, 4, 4), QColor(0, 0, 0, 60))
        
        # B. Main Body Gradient
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor("#3F4242"))
        gradient.setColorAt(1, QColor("#2F3232")) 
        painter.setBrush(QBrush(gradient))
        
        # C. Selection Border
        border_color = QColor("#ff9900") if self.isSelected() else QColor("#727272")
        painter.setPen(QPen(border_color, 2))
        painter.drawRoundedRect(rect, 12, 12)

        # D. Title Header
        title_rect = QRectF(rect.x(), rect.y(), rect.width(), 25)
        title_grad = QLinearGradient(title_rect.topLeft(), title_rect.bottomRight())
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

        # F. Status Light
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

    def find_nearby_socket(self, pos, radius=10):
        area = QRectF(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
        for item in self.items(area):
            if isinstance(item, Socket):
                return item
        return None

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
        if isinstance(item, Socket) and item.socket_type == "output":
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
            target = self.find_nearby_socket(pos)
            valid = (
                target
                and target.socket_type == "input"
                and target.can_accept_connection()
                and not self.are_already_connected(self.current_connection.start_socket, target)
                and not self.creates_cycle(self.current_connection.start_socket.node, target.node)
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

        # AUTO GENERATE JSON LIBRARY
        generate_library.generate()

        global NODE_LIBRARY
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "node_library.json")
        try:
            with open(json_path, "r") as f:
                NODE_LIBRARY.clear() # Clear old data
                NODE_LIBRARY.update(json.load(f)) # Update with new data
        except Exception as e:
            print(f"CRITICAL ERROR loading node library: {e}")


        self.viewer = viewer
        
        # 1. Main Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 2. Toolbar (Top Row)
        toolbar = QHBoxLayout()
        
        self.btn_add = QPushButton("Add Node")
        self.btn_add.clicked.connect(self.open_add_menu)
        toolbar.addWidget(self.btn_add)

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

        # Inner Layout for the GroupBox
        group_layout = QVBoxLayout()
        self.props_group.setLayout(group_layout)
        
        # The Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True) # Important!
        scroll.setFrameShape(QFrame.NoFrame) # remove ugly double border
        
        # The Container Widget that holds the Form
        scroll_content = QWidget()
        self.props_layout = QFormLayout() # This is the layout we add rows to
        self.props_layout.setContentsMargins(5, 5, 5, 5)
        scroll_content.setLayout(self.props_layout)
        
        # Put container into scroll area
        scroll.setWidget(scroll_content)
        group_layout.addWidget(scroll)


        self. bottom_container = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        action_layout = QHBoxLayout()
        
        # 4. Fit Scene Button (Below Properties)
        self.btn_fit = QPushButton("Fit to Scene")
        bottom_layout.addWidget(self.btn_fit)

        # RUN button
        self.btn_run = QPushButton("RUN PIPELINE")
        self.btn_run.setFixedHeight(35)
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32; /* Material Design Green */
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 4px;
                border: 1px solid #1B5E20;
            }
            QPushButton:hover {
                background-color: #388E3C; /* Slightly lighter on hover */
            }
            QPushButton:pressed {
                background-color: #1B5E20; /* Dark on press */
            }
        """)

        self.btn_run.clicked.connect(self.run_pipeline)
        action_layout.addWidget(self.btn_run)

        bottom_layout.addLayout(action_layout)

        # 5. Graphics View (Bottom)
        self.scene = FlowScene()
        self.view = FlowView(self.scene)
        bottom_layout.addWidget(self.view)

        # 6. Console (Bottom)
        self.console_label = QLabel("Execution Log:")
        bottom_layout.addWidget(self.console_label)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFixedHeight(100) # Small height like a terminal
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Monospace;")
        bottom_layout.addWidget(self.console)

        # Vertical Spacer
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.props_group)
        self.splitter.addWidget(self.bottom_container)

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 7)

        self.layout.addWidget(self.splitter)

        self.btn_fit.clicked.connect(self.view.fit_scene)
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
                # Map connections
                input_connections = {}
                for socket in item.inputs:
                    if socket.connected_edges:
                        edge = socket.connected_edges[0] 
                        if edge.start_socket:
                            src_node = edge.start_socket.node
                            connection_str = f"{src_node.uid}.{edge.start_socket.name}"
                            input_connections[socket.name] = connection_str

                node_data = {
                    "id": item.uid,
                    "type": item.node_type,
                    "label": item.title,
                    "position": {"x": item.pos().x(), "y": item.pos().y()},
                    "parameters": item.parameters,
                    "input_connections": input_connections
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


    # --- Add Node Method ---
    # 1. The UI Logic (Dropdown)
    def open_add_menu(self):
        """Shows a categorized dropdown menu to add nodes."""
        menu = QMenu(self)
        
        # Dictionary to hold reference to created submenus
        # Structure: {"Filters": QMenu_Object, "Segmentation": QMenu_Object}
        submenus = {}

        # Sort items so they appear alphabetically in the menu
        sorted_items = sorted(NODE_LIBRARY.items(), key=lambda x: x[1]["label"])

        for key, data in sorted_items:
            label = data["label"]
            category = data.get("category", "Uncategorized") # Default if missing

            # 1. Create the Submenu if it doesn't exist yet
            if category not in submenus:
                submenus[category] = menu.addMenu(category)
            
            # 2. Add the Action to the specific Submenu
            action = submenus[category].addAction(label)
            
            # 3. Connect the action using a lambda to capture the specific 'key'
            # We use checked=False (default for triggered) to ignore the boolean arg
            action.triggered.connect(lambda checked=False, k=key: self.add_node(k))

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
        
        # Iterate over all sockets (both inputs and outputs)
        for socket in node.inputs + node.outputs:
            # 1. Remove all edges connected to this socket
            for edge in list(socket.connected_edges):
                self.scene.removeItem(edge)
            
            # 2. Remove the socket itself from the scene (This was missing)
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
        
        # CLEANUP
        self.worker.finished_signal.connect(self.thread.quit)
        self.worker.finished_signal.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # RE-ENABLE UI
        self.worker.finished_signal.connect(lambda: self.set_ui_enabled(True))
        
        # 4. Start
        self.thread.start()

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

    def handle_execution_result(self, node_title, output_name, data):
        # 1. Update Cache (Always save the raw data for downstream nodes!)
        node_obj = next((item for item in self.scene.items() 
                        if isinstance(item, Node) and item.title == node_title), None)
        if node_obj:
            node_obj.cached_results[output_name] = data
            node_obj.status = "green"

        # --- A. UNPACK ENVELOPE ---
        # Separates the Engine's wrapper: (Data, Metadata) -> Data, Metadata
        display_data = data
        display_meta = {}

        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
            display_data = data[0]
            display_meta = data[1]

        # --- B. THE STRIPPER (Fixes "Filter" crashes) ---
        # If the data itself is a Tuple/List (e.g. (Mask, Stats)), extracting the image.
        if isinstance(display_data, (tuple, list)):
            if len(display_data) > 0:
                first_item = display_data[0]
                # If the first item is an image array, use it and discard the rest.
                if hasattr(first_item, "shape") and hasattr(first_item, "dtype"):
                    display_data = first_item
        
        # --- C. THE BOUNCER (Fixes "CSV" crashes) ---
        # If the final extracted data is a Table or Dictionary, do not plot.
        import pandas as pd
        if isinstance(display_data, (pd.DataFrame, dict, str)):
            print(f"ℹ️ Output '{output_name}' is non-visual. Skipping display.")
            return

        # --- D. DISPLAY HELPER ---
        def add_layer_to_viewer(layer_data, raw_meta):
            # 1. Prepare Name
            layer_name = raw_meta.get("name", f"{node_title} Output")
            
            # 2. Filter Metadata (Napari args vs. Custom User Data)
            valid_napari_args = {"name", "opacity", "blending", "visible", "multiscale", "colormap", "contrast_limits", "gamma", "rgb"}
            napari_kwargs = {"name": layer_name}
            custom_metadata = {}

            for k, v in raw_meta.items():
                if k in valid_napari_args: napari_kwargs[k] = v
                else: custom_metadata[k] = v
            
            napari_kwargs["metadata"] = custom_metadata

            # 3. Create/Update Layer
            try:
                if layer_name in self.viewer.layers:
                    layer = self.viewer.layers[layer_name]
                    layer.data = layer_data
                    layer.metadata.update(custom_metadata)
                    if "rgb" in napari_kwargs:
                        if hasattr(layer, "rgb"):
                            layer.rgb = napari_kwargs["rgb"]
                else:
                    # Heuristic: Is it Labels or Image?
                    import numpy as np
                    is_labels = False  
                    if hasattr(layer_data, "dtype"):
                        # Check if it's integer type
                        if layer_data.dtype == bool or np.issubdtype(layer_data.dtype, np.integer):
                            # ONLY make it a Label if it is NOT RGB
                            # (Images can be integers too, e.g. uint8, uint16)
                            if not napari_kwargs.get("rgb", False):
                                is_labels = True

                    if is_labels:
                        self.viewer.add_labels(layer_data, **napari_kwargs)
                    else:
                        self.viewer.add_image(layer_data, **napari_kwargs)
            except Exception as e:
                # If something weird slips through (like a single number), we catch it here
                print(f"❌ Error displaying layer '{layer_name}': {e}")

        # Execute Display
        add_layer_to_viewer(display_data, display_meta)


    def append_log(self, text):
        self.console.append(text)
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_ui_enabled(self, enabled: bool):
        """Locks/Unlocks buttons during execution."""
        self.btn_run.setEnabled(enabled)
        self.btn_add.setEnabled(enabled)
        self.btn_remove.setEnabled(enabled)
        self.btn_load.setEnabled(enabled)
        self.btn_save.setEnabled(enabled)
        self.btn_import.setEnabled(enabled)
        
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
                    NODE_LIBRARY[node_key] = {
                        "label": meta["label"],
                        "category": "Custom", # Force category or use meta['category']
                        "inputs": inputs,
                        "outputs": meta["outputs"],
                        "parameters": parameters,
                        "execution_path": "custom_loaded", 
                        "executable": func # <--- DIRECT REFERENCE
                    }
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

    def show_plot_popup(self, title, figure):
        """Opens a QDialog to display the matplotlib figure."""
        
        # Create a Dialog Window
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Result: {title}")
        dialog.resize(600, 450)
        
        # Create Layout
        layout = QVBoxLayout(dialog)
        
        # The 'Canvas' is the Qt Widget that draws the Figure
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        
        # Add 'Close' button? (Optional, X works fine)
        # show() makes it non-blocking (you can keep working while it's open)
        dialog.show() 

        if not hasattr(self, 'plot_windows'):
            self.plot_windows = []
        
        self.plot_windows.append(dialog)
        dialog.finished.connect(lambda: self.plot_windows.remove(dialog) if dialog in self.plot_windows else None)

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