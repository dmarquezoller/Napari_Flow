from qtpy.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsRectItem, QMainWindow,
    QGraphicsDropShadowEffect, QToolBar, QInputDialog, QFileDialog, QVBoxLayout,
    QHBoxLayout, QPushButton, QMenu, QWidget, QGroupBox, QFormLayout, QLabel,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QScrollArea, QBoxLayout,
    QFrame, QMessageBox, QTextEdit, QSplitter, QDialog, QTableWidget, QHeaderView, QAbstractItemView,
    QToolTip,
    QListWidget, QListWidgetItem
)

from qtpy.QtGui import (
    QBrush, QPen, QColor, QPainterPath, QPainterPathStroker, QLinearGradient, QPainter, QAction, QCursor, QGradient, QImage, QPixmap
)
from qtpy.QtCore import Qt, QPointF, QRectF, QThread, Signal, QTimer
import os, sys, json, datetime, napari, uuid, importlib.util, inspect, zarr, copy, re, time
import html
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
from .widgets.plot_widgets import (
    PlotResultDialog,
    figure_to_rgb_array,
    PlotDashboard,
    is_plotly_figure,
)


NODE_LIBRARY = {}


class InfoHoverButton(QPushButton):
    """Small info button that always shows a transient hover tooltip."""

    def __init__(self, tooltip_html, parent=None):
        super().__init__("i", parent)
        self._tooltip_html = tooltip_html

    def _show_tooltip(self):
        if not self._tooltip_html:
            return
        pos = self.mapToGlobal(self.rect().bottomLeft())
        QToolTip.showText(pos, self._tooltip_html, self)

    def enterEvent(self, event):
        self._show_tooltip()
        super().enterEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._show_tooltip()
            event.accept()
            return
        super().mousePressEvent(event)

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


DATA_TYPE_COLORS = {
    "image": "#4DA3FF",
    "labels": "#7BC96F",
    "table": "#F2C14E",
    "shapes": "#FF7A59",
    "points": "#2EC4B6",
    "vectors": "#E76F51",
    "scalar": "#9B8AFB",
    "layers": "#B6A3FF",
    "any": "#9AA0A6",
}

# Socket layout lanes (single source of truth for spacing).
SOCKET_EXEC_Y = 24
SOCKET_DATA_TOP = 44
SOCKET_DATA_BOTTOM_MARGIN = 16
SOCKET_DATA_MIN_GAP = 18


def _normalize_data_type(data_type):
    if data_type is None:
        return "any"
    value = str(data_type).strip().lower()
    return value if value else "any"


def _get_data_type_color(data_type):
    normalized = _normalize_data_type(data_type)
    return QColor(DATA_TYPE_COLORS.get(normalized, DATA_TYPE_COLORS["any"]))


def _are_data_types_compatible(output_type, input_type):
    out_t = _normalize_data_type(output_type)
    in_t = _normalize_data_type(input_type)
    return out_t == "any" or in_t == "any" or out_t == in_t


def _normalize_socket_type_map(type_map):
    if not isinstance(type_map, dict):
        return {}
    return {str(k): _normalize_data_type(v) for k, v in type_map.items()}


def _normalize_dynamic_output_type_rules(config):
    if not isinstance(config, dict):
        return {}

    normalized = {}
    for socket_name, rule in config.items():
        if not isinstance(rule, dict):
            continue
        from_param = rule.get("from_param")
        if not from_param:
            continue
        source_raw = rule.get("source", "viewer_layer_type")
        source = str(source_raw).strip().lower() if source_raw is not None else "viewer_layer_type"
        if not source:
            source = "viewer_layer_type"
        fallback = _normalize_data_type(rule.get("fallback", "any"))
        normalized[str(socket_name)] = {
            "from_param": str(from_param),
            "source": source,
            "fallback": fallback,
        }
    return normalized


def _infer_data_type_from_viewer_layer(layer):
    """
    Map a napari layer instance to a flow-socket data type.
    """
    if layer is None:
        return "any"

    layer_name = layer.__class__.__name__.lower()
    by_class = {
        "image": "image",
        "labels": "labels",
        "shapes": "shapes",
        "points": "points",
        "vectors": "vectors",
    }
    if layer_name in by_class:
        return by_class[layer_name]

    layer_type = str(getattr(layer, "layer_type", "")).lower()
    by_kind = {
        "image": "image",
        "labels": "labels",
        "shapes": "shapes",
        "points": "points",
        "vectors": "vectors",
    }
    return by_kind.get(layer_type, "any")

# --- SOCKET -----------------------------------------------------
class Socket(QGraphicsEllipseItem):
    def __init__(
        self,
        node,
        socket_type,
        name,
        index,
        total_sockets,
        max_connections=None,
        data_type="any",
    ):
        super().__init__(-6, -6, 12, 12)
        self.socket_type = socket_type
        self.node = node
        self.name = name # e.g. "image_in"
        self.connected_edges = []
        self.proxy_edges = []
        self.max_connections = max_connections

        # --- Determine whether this is a logic (control-flow) socket ---
        self.is_logic = socket_type in ("logic_in", "logic_out")
        self.data_type = "exec" if self.is_logic else _normalize_data_type(data_type)

        h = node.rect().height()
        w = node.rect().width()

        if self.is_logic:
            # Exec pins always live on the sides (Unreal-style thread).
            side = "left" if socket_type == "logic_in" else "right"
            x = 0 if side == "left" else w
            # Keep exec pins in their own vertical lane (near the top) so they
            # don't overlap data pins when a node has a single data in/out.
            if total_sockets <= 1:
                y = SOCKET_EXEC_Y
            else:
                top = SOCKET_EXEC_Y
                bottom = max(top + 1, h - 20)
                step = (bottom - top) / max(total_sockets - 1, 1)
                y = top + (step * index)
            self.local_offset = QPointF(x, y)
            self.base_color = QColor("#F2F2F2")
            self.setToolTip(f"Exec: {self.name}")
        else:
            # Data sockets live in a dedicated lower lane to avoid overlap with
            # exec sockets near the header.
            top = min(max(0.0, h - SOCKET_DATA_BOTTOM_MARGIN - 1), SOCKET_DATA_TOP)
            bottom = max(top + 1, h - SOCKET_DATA_BOTTOM_MARGIN)
            if total_sockets <= 1:
                y = (top + bottom) * 0.5
            else:
                step = (bottom - top) / max(total_sockets - 1, 1)
                y = top + (step * index)
            x = 0 if socket_type == "input" else w
            self.local_offset = QPointF(x, y)
            self.base_color = _get_data_type_color(self.data_type)
            self.setToolTip(f"Data: {name} ({self.data_type})")

        self.setBrush(QBrush(self.base_color))
        self.setPen(QPen(Qt.GlobalColor.black, 1.2))
        self.setZValue(3)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.update_position()

    def _socket_path(self):
        path = QPainterPath()
        is_output = self.socket_type in ("output", "logic_out")

        if self.is_logic:
            # Exec sockets keep directional triangles.
            points = (
                [QPointF(-6, -5), QPointF(6, 0), QPointF(-6, 5)]
                if is_output
                else [QPointF(6, -5), QPointF(-6, 0), QPointF(6, 5)]
            )
            path.moveTo(points[0])
            for p in points[1:]:
                path.lineTo(p)
            path.closeSubpath()
        else:
            # Data sockets are circles (typed by color, not shape).
            path.addEllipse(QRectF(-6, -6, 12, 12))
        return path

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(10)
        path = self._socket_path()
        return path.united(stroker.createStroke(path))

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawPath(self._socket_path())

    def update_position(self):
        self.setPos(self.node.pos() + self.local_offset)
        for proxy in list(self.proxy_edges):
            if hasattr(proxy, "update_positions"):
                proxy.update_positions()
    
    # ... (keep center_pos and can_accept_connection same as before) ...
    def center_pos(self):
        return self.sceneBoundingRect().center()

    def can_accept_connection(self):
        if self.max_connections is None:
            return True
        return len(self.connected_edges) < self.max_connections

    def set_data_type(self, data_type):
        """
        Update runtime type/color for data sockets.
        """
        if self.is_logic:
            return
        self.data_type = _normalize_data_type(data_type)
        self.base_color = _get_data_type_color(self.data_type)
        self.setBrush(QBrush(self.base_color))
        self.setToolTip(f"Data: {self.name} ({self.data_type})")
        self.update()

        # Keep connected edge colors in sync with source socket type.
        for edge in list(self.connected_edges):
            if getattr(edge, "is_logic", False):
                continue
            if edge.start_socket is self:
                edge.data_type = self.data_type
            elif edge.start_socket is not None:
                edge.data_type = getattr(edge.start_socket, "data_type", "any")
            edge._apply_pen_state()


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
        self.data_type = getattr(start_socket, "data_type", "any")
        self.setZValue(1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self._apply_pen_state()

        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.update_path(self.start_socket.center_pos(), self.start_socket.center_pos())

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(20)
        return stroker.createStroke(self.path())

    def hoverEnterEvent(self, event):
        self.hovered = True
        self._apply_pen_state()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self._apply_pen_state()
        super().hoverLeaveEvent(event)

    def _apply_pen_state(self):
        selected = self.isSelected()
        if self.is_logic:
            if selected:
                color = QColor("#FFD166")
                width = 3
            else:
                color = QColor("#FFFFFF") if self.hovered else QColor("#F2F2F2")
                width = 3 if self.hovered else 2
            self.setPen(QPen(color, width))
        else:
            base = _get_data_type_color(self.data_type)
            if selected:
                color = base.lighter(140)
                width = 3
            else:
                color = base if self.hovered else base.darker(120)
                width = 3 if self.hovered else 2
            self.setPen(QPen(color, width))

    def itemChange(self, change, value):
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemSelectedChange,
        ):
            self._apply_pen_state()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if self.end_socket and (self.end_socket.center_pos() - event.scenePos()).manhattanLength() < 20:
            event.accept()
            self.detach_end()
            self.dragging = True
        else:
            super().mousePressEvent(event)

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
                # Exec edges: logic_out → logic_in, enforce acyclic exec thread
                valid = (
                    target
                    and target.socket_type == "logic_in"
                    and target.can_accept_connection()
                    and not self.scene_ref.are_already_connected(self.start_socket, target)
                    and not self.scene_ref.creates_logic_cycle(self.start_socket.node, target.node)
                )
            else:
                # Data edges: output → input, enforce DAG
                valid = (
                    target
                    and target.socket_type == "input"
                    and target.can_accept_connection()
                    and not self.scene_ref.are_already_connected(self.start_socket, target)
                    and not self.scene_ref.creates_cycle(self.start_socket.node, target.node)
                    and self.scene_ref.is_data_connection_compatible(self.start_socket, target)
                )

            if valid:
                self.scene_ref.finalize_connection(self, target)
            else:
                self.remove_from_sockets()
                self.scene_ref.removeItem(self)
            self.dragging = False
        else:
            super().mouseReleaseEvent(event)

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

    def remove_from_sockets(self):
        if self.start_socket and self in self.start_socket.connected_edges:
            self.start_socket.connected_edges.remove(self)
        if self.end_socket and self in self.end_socket.connected_edges:
            self.end_socket.connected_edges.remove(self)
        self.end_socket = None

    def update_positions(self):
        if self.start_socket and self.end_socket:
            self.update_path(
                self.start_socket.center_pos(), self.end_socket.center_pos()
            )


class MacroProxyConnection(QGraphicsPathItem):
    """
    Visual-only edge used while a macro is collapsed.

    The real executable edge stays connected to internal nodes; this item only
    mirrors that relationship on the collapsed macro shell.
    """

    def __init__(self, start_socket, end_socket, real_edge=None):
        super().__init__()
        self.start_socket = start_socket
        self.end_socket = end_socket
        self.real_edge = real_edge
        self.is_logic = bool(getattr(start_socket, "is_logic", False))
        self.data_type = getattr(start_socket, "data_type", "any")
        self.setZValue(1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(False)
        self._apply_pen_state()

        if hasattr(self.start_socket, "proxy_edges"):
            self.start_socket.proxy_edges.append(self)
        if hasattr(self.end_socket, "proxy_edges"):
            self.end_socket.proxy_edges.append(self)
        self.update_positions()

    def _apply_pen_state(self):
        selected = self.isSelected()
        if self.is_logic:
            color = QColor("#FFD166") if selected else QColor("#F2F2F2")
            width = 3 if selected else 2
        else:
            base = _get_data_type_color(self.data_type)
            color = base.lighter(140) if selected else base.darker(120)
            width = 3 if selected else 2
        self.setPen(QPen(color, width))

    def itemChange(self, change, value):
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemSelectedChange,
        ):
            self._apply_pen_state()
        return super().itemChange(change, value)

    def update_positions(self):
        if self.start_socket is None or self.end_socket is None:
            return
        start = self.start_socket.center_pos()
        end = self.end_socket.center_pos()
        path = QPainterPath()
        path.moveTo(start)
        dx = (end.x() - start.x()) * 0.5
        c1 = QPointF(start.x() + dx, start.y())
        c2 = QPointF(end.x() - dx, end.y())
        path.cubicTo(c1, c2, end)
        self.setPath(path)

    def detach(self):
        if (
            self.start_socket is not None
            and hasattr(self.start_socket, "proxy_edges")
            and self in self.start_socket.proxy_edges
        ):
            self.start_socket.proxy_edges.remove(self)
        if (
            self.end_socket is not None
            and hasattr(self.end_socket, "proxy_edges")
            and self in self.end_socket.proxy_edges
        ):
            self.end_socket.proxy_edges.remove(self)
        self.start_socket = None
        self.end_socket = None


# --- NODE -------------------------------------------------------
class Node(QGraphicsRectItem):
    def __init__(self, x, y, node_type="generic", title=None, uuid_str=None, scene=None):
        # 1. Setup Data
        self.node_type = node_type
        self.category = "Uncategorized"
        self.uid = uuid_str if uuid_str else str(uuid.uuid4())
        self.parameters = {}
        self.description = ""
        self.logic_config = _normalize_logic_config(None)
        self.dynamic_output_types = {}

        # --- State variables ---
        self.status = "gray"
        self.last_signature = None
        self.cached_results = {}
        
        # Load Definition
        inputs_data = ["in"] # Default if not found
        outputs_data = ["out"]
        input_type_map = {}
        output_type_map = {}
        
        if node_type in NODE_LIBRARY:
            definition = NODE_LIBRARY[node_type]
            self.title = title if title else definition["label"]
            self.category = definition.get("category", "Uncategorized")
            self.description = str(definition.get("description", "")).strip()
            self.logic_config = _normalize_logic_config(definition.get("logic"))
            inputs_data = definition.get("inputs", ["in"])
            outputs_data = definition.get("outputs", ["out"])
            input_type_map = _normalize_socket_type_map(definition.get("input_types"))
            output_type_map = _normalize_socket_type_map(definition.get("output_types"))
            self.dynamic_output_types = _normalize_dynamic_output_type_rules(
                definition.get("dynamic_output_types")
            )
            
            # Load Params
            for key, conf in definition["parameters"].items():
                default_val = conf.get("default")
                # Some table params are authored with "value" in params_config
                # and can end up with default=None in generated library metadata.
                if default_val is None and conf.get("type") == "table":
                    default_val = conf.get("value", [])
                self.parameters[key] = copy.deepcopy(default_val)
        else:
            self.title = title if title else node_type

        # 2. Calculate Height based on params AND sockets
        param_count = len(self.parameters)
        socket_count = max(len(inputs_data), len(outputs_data))
        h = 45 + (param_count * 20) + (socket_count * 10)
        min_h_for_data_lane = (
            SOCKET_DATA_TOP
            + SOCKET_DATA_BOTTOM_MARGIN
            + max(0, socket_count - 1) * SOCKET_DATA_MIN_GAP
        )
        h = max(h, 80, min_h_for_data_lane)
        
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
            self.inputs.append(
                Socket(
                    self,
                    "input",
                    name,
                    i,
                    len(inputs_data),
                    max_connections=1,
                    data_type=input_type_map.get(name, "any"),
                )
            )

        self.outputs = []
        for i, name in enumerate(outputs_data):
            self.outputs.append(
                Socket(
                    self,
                    "output",
                    name,
                    i,
                    len(outputs_data),
                    max_connections=None,
                    data_type=output_type_map.get(name, "any"),
                )
            )

        # 5. Generate Exec sockets (logic thread):
        # - Begin: exec_out only
        # - Loop: exec_in + (loop_body, completed) exec_outs
        # - Source Inputs (no data inputs): no exec pins
        # - Everything else: exec_in + exec_out
        is_begin = self.node_type == "begin"
        is_loop = self.node_type == "loop_control"
        is_source_input = (
            self.category in ("Inputs", "Input")
            and len(inputs_data) == 0
            and not is_begin
            and not is_loop
        )

        has_exec_in = not (is_begin or is_source_input)
        if is_begin:
            exec_output_names = ["exec_out"]
        elif is_loop:
            exec_output_names = ["loop_body", "completed"]
        elif is_source_input:
            exec_output_names = []
        else:
            exec_output_names = ["exec_out"]

        self.logic_inputs = []
        if has_exec_in:
            self.logic_inputs.append(
                Socket(self, "logic_in", "logic_in", 0, 1, max_connections=1)
            )

        self.logic_outputs = []
        for i, out_name in enumerate(exec_output_names):
            self.logic_outputs.append(
                Socket(
                    self,
                    "logic_out",
                    out_name,
                    i,
                    max(len(exec_output_names), 1),
                    max_connections=1,
                )
            )

        if scene:
            all_sockets = self.inputs + self.outputs + self.logic_inputs + self.logic_outputs
            for s in all_sockets:
                scene.addItem(s)

    # Helper to iterate ALL sockets (data + logic)
    def all_sockets(self):
        return self.inputs + self.outputs + self.logic_inputs + self.logic_outputs

    def boundingRect(self):
        """
        Expand paint bounds so off-node decorations (shadow + exec label chips)
        are always included in Qt's dirty-region repaints.
        """
        return super().boundingRect().adjusted(-70, -8, 70, 12)

    def _logic_socket_display_label(self, socket):
        if socket.socket_type not in ("logic_in", "logic_out"):
            return ""

        name = socket.name
        if socket.socket_type == "logic_in":
            return "IN"

        if self.node_type == "begin":
            if name == "exec_out":
                return "START"
            return "OUT"
        if self.node_type == "loop_control":
            mapping = {
                "logic_in": "IN",
                "loop_body": "BODY",
                "completed": "DONE",
            }
            return mapping.get(name, name.replace("_", " ").upper())

        if name in ("exec_out", "logic_out"):
            return "OUT"
        return name.replace("_", " ").upper()

    def itemChange(self, change, value):
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemSelectedChange,
        ):
            self.update()
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
            border_width = 2
        else:
            border_color = QColor("#7f9fbe") if is_control_flow else QColor("#727272")
            border_width = 2
        painter.setPen(QPen(border_color, border_width))
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

        # Exec socket labels (only when selected + socket is unconnected)
        if self.isSelected():
            logic_font = painter.font()
            logic_font.setPointSize(7)
            logic_font.setBold(True)
            painter.setFont(logic_font)
            painter.setPen(QColor("#E9E9E9"))

            for socket in self.logic_inputs + self.logic_outputs:
                label = self._logic_socket_display_label(socket)
                if not label:
                    continue
                if socket.connected_edges:
                    continue
                chip_w = max(44, painter.fontMetrics().horizontalAdvance(label) + 12)
                chip_h = 14
                y = socket.local_offset.y() - (chip_h / 2)
                if socket.socket_type == "logic_in":
                    x = -chip_w - 12
                else:
                    x = rect.width() + 12
                chip_rect = QRectF(x, y, chip_w, chip_h)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(33, 43, 55, 220))
                painter.drawRoundedRect(chip_rect, 6, 6)
                painter.setPen(QColor("#E9E9E9"))
                painter.drawText(
                    chip_rect,
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter,
                    label,
                )
        
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


class MacroGroupItem(QGraphicsRectItem):
    """
    Collapsed visual wrapper for a set of nodes.

    This is a UI-only item: the execution engine still runs the original nodes.
    """

    def __init__(self, x, y, group_id, title, node_count=0, on_open=None, scene=None):
        super().__init__(0, 0, 170, 90)
        self.group_id = group_id
        self.title = title
        self.node_count = int(node_count)
        self.on_open = on_open
        self.scene_ref = scene
        self.input_bindings = []
        self.output_bindings = []
        self.logic_input_bindings = []
        self.logic_output_bindings = []
        self.inputs = []
        self.outputs = []
        self.logic_inputs = []
        self.logic_outputs = []
        self.setPos(x, y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(2)
        self.setToolTip("Double-click to open macro")

    def all_sockets(self):
        return self.inputs + self.outputs + self.logic_inputs + self.logic_outputs

    def boundingRect(self):
        """
        Expand paint bounds so drag repaints include the macro shadow and avoid
        ghost/streak artifacts.
        """
        return super().boundingRect().adjusted(-8, -8, 12, 12)

    def set_macro_visible(self, visible):
        self.setVisible(visible)
        for socket in self.all_sockets():
            socket.setVisible(visible)
        self.update()

    def clear_proxy_edges(self):
        if self.scene_ref is None:
            return
        seen = set()
        for socket in self.all_sockets():
            for proxy in list(getattr(socket, "proxy_edges", [])):
                if id(proxy) in seen:
                    continue
                seen.add(id(proxy))
                if hasattr(proxy, "detach"):
                    proxy.detach()
                if proxy.scene() is self.scene_ref:
                    self.scene_ref.removeItem(proxy)

    def clear_sockets(self):
        if self.scene_ref is None:
            self.inputs = []
            self.outputs = []
            self.logic_inputs = []
            self.logic_outputs = []
            return
        self.clear_proxy_edges()
        for socket in self.all_sockets():
            for edge in list(getattr(socket, "connected_edges", [])):
                other = edge.end_socket if edge.start_socket is socket else edge.start_socket
                if other is not None and edge in getattr(other, "connected_edges", []):
                    other.connected_edges.remove(edge)
                if edge.scene() is self.scene_ref:
                    self.scene_ref.removeItem(edge)
            if socket.scene() is self.scene_ref:
                self.scene_ref.removeItem(socket)
        self.inputs = []
        self.outputs = []
        self.logic_inputs = []
        self.logic_outputs = []

    def set_bindings(
        self,
        *,
        input_bindings=None,
        output_bindings=None,
        logic_input_bindings=None,
        logic_output_bindings=None,
    ):
        self.input_bindings = list(input_bindings or [])
        self.output_bindings = list(output_bindings or [])
        self.logic_input_bindings = list(logic_input_bindings or [])
        self.logic_output_bindings = list(logic_output_bindings or [])
        self._rebuild_sockets()

    def _apply_binding_to_socket(self, socket, binding):
        socket.macro_group_id = self.group_id
        socket.is_macro_socket = True
        socket.macro_binding = dict(binding)

    def _rebuild_sockets(self):
        self.clear_sockets()
        if self.scene_ref is None:
            return

        total_inputs = max(len(self.input_bindings), 1)
        total_outputs = max(len(self.output_bindings), 1)
        total_logic_inputs = max(len(self.logic_input_bindings), 1)
        total_logic_outputs = max(len(self.logic_output_bindings), 1)

        for i, b in enumerate(self.input_bindings):
            sock = Socket(
                self,
                "input",
                b.get("socket_name", f"in_{i}"),
                i,
                total_inputs,
                max_connections=b.get("max_connections", 1),
                data_type=b.get("data_type", "any"),
            )
            self._apply_binding_to_socket(sock, b)
            self.inputs.append(sock)
            self.scene_ref.addItem(sock)

        for i, b in enumerate(self.output_bindings):
            sock = Socket(
                self,
                "output",
                b.get("socket_name", f"out_{i}"),
                i,
                total_outputs,
                max_connections=b.get("max_connections", None),
                data_type=b.get("data_type", "any"),
            )
            self._apply_binding_to_socket(sock, b)
            self.outputs.append(sock)
            self.scene_ref.addItem(sock)

        for i, b in enumerate(self.logic_input_bindings):
            sock = Socket(
                self,
                "logic_in",
                b.get("socket_name", "logic_in"),
                i,
                total_logic_inputs,
                max_connections=b.get("max_connections", 1),
            )
            self._apply_binding_to_socket(sock, b)
            self.logic_inputs.append(sock)
            self.scene_ref.addItem(sock)

        for i, b in enumerate(self.logic_output_bindings):
            sock = Socket(
                self,
                "logic_out",
                b.get("socket_name", f"logic_out_{i}"),
                i,
                total_logic_outputs,
                max_connections=b.get("max_connections", 1),
            )
            self._apply_binding_to_socket(sock, b)
            self.logic_outputs.append(sock)
            self.scene_ref.addItem(sock)

    def paint(self, painter, option, widget):
        rect = self.rect()

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(rect.adjusted(4, 4, 4, 4), QColor(0, 0, 0, 60))

        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor("#35513f"))
        gradient.setColorAt(1, QColor("#26392d"))
        painter.setBrush(QBrush(gradient))

        if self.isSelected():
            border_color = QColor("#ff9900")
            border_width = 2
        else:
            border_color = QColor("#6ea785")
            border_width = 2
        painter.setPen(QPen(border_color, border_width))
        painter.drawRoundedRect(rect, 12, 12)

        title_rect = QRectF(rect.x(), rect.y(), rect.width(), 25)
        title_grad = QLinearGradient(title_rect.topLeft(), title_rect.bottomRight())
        title_grad.setColorAt(0, QColor("#4e7a63"))
        title_grad.setColorAt(1, QColor("#395846"))
        painter.setBrush(QBrush(title_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(title_rect, 12, 12)
        painter.drawRect(QRectF(rect.x(), rect.y() + 15, rect.width(), 10))

        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(
            QRectF(rect.x(), rect.y(), rect.width(), 25),
            Qt.AlignmentFlag.AlignCenter,
            self.title,
        )

        painter.setPen(QColor("#d7efe0"))
        painter.drawText(
            QRectF(8, 36, rect.width() - 16, 18),
            Qt.AlignmentFlag.AlignLeft,
            f"nodes: {self.node_count}",
        )
        painter.setPen(QColor("#b4d2c1"))
        painter.drawText(
            QRectF(8, 56, rect.width() - 16, 22),
            Qt.AlignmentFlag.AlignLeft,
            "double-click to open",
        )

    def mouseDoubleClickEvent(self, event):
        if callable(self.on_open):
            self.on_open(self.group_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change, value):
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemSelectedChange,
        ):
            self.update()
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemPositionChange,
        ):
            for socket in self.all_sockets():
                socket.update_position()
                for edge in list(getattr(socket, "connected_edges", [])):
                    edge.update_positions()
                for proxy in list(getattr(socket, "proxy_edges", [])):
                    if hasattr(proxy, "update_positions"):
                        proxy.update_positions()
        return super().itemChange(change, value)


# --- SCENE ------------------------------------------------------
class FlowScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.current_connection = None
        self.editor_ref = None

    def finalize_connection(self, connection, target_socket):
        editor = getattr(self, "editor_ref", None)
        if editor is not None and hasattr(editor, "finalize_connection_with_macros"):
            handled = editor.finalize_connection_with_macros(connection, target_socket)
            if handled:
                return True
        connection.finalize(target_socket)
        return True

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

    def creates_logic_cycle(self, start_node, end_node):
        visited = set()

        def dfs(node):
            if node == start_node:
                return True
            for s in getattr(node, "logic_outputs", []):
                for e in s.connected_edges:
                    if e.end_socket:
                        next_node = e.end_socket.node
                        if next_node not in visited:
                            visited.add(next_node)
                            if dfs(next_node):
                                return True
            return False

        return dfs(end_node)

    def is_data_connection_compatible(self, start_socket, end_socket):
        out_t = getattr(start_socket, "data_type", "any")
        in_t = getattr(end_socket, "data_type", "any")
        compatible = _are_data_types_compatible(out_t, in_t)
        if not compatible:
            print(
                f"⛔ Type mismatch blocked: "
                f"{start_socket.node.title}.{start_socket.name} ({out_t}) -> "
                f"{end_socket.node.title}.{end_socket.name} ({in_t})"
            )
        return compatible

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
                # Exec edges: logic_out → logic_in, enforce acyclic exec thread
                valid = (
                    target
                    and target.socket_type == "logic_in"
                    and target.can_accept_connection()
                    and not self.are_already_connected(start_sock, target)
                    and not self.creates_logic_cycle(start_sock.node, target.node)
                )
            else:
                # Data edges: output → input, enforce DAG
                valid = (
                    target
                    and target.socket_type == "input"
                    and target.can_accept_connection()
                    and not self.are_already_connected(start_sock, target)
                    and not self.creates_cycle(start_sock.node, target.node)
                    and self.is_data_connection_compatible(start_sock, target)
                )

            if valid:
                self.finalize_connection(self.current_connection, target)
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
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.scale_factor = 1.0
        self.delete_callback = None
        self.resize_callback = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.setFocus()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if callable(self.delete_callback):
                self.delete_callback()
                event.accept()
                return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        zoom_in = 1.1
        zoom_out = 0.9
        factor = zoom_in if event.angleDelta().y() > 0 else zoom_out
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if callable(self.resize_callback):
            self.resize_callback()

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
        self._pending_interaction = None
        self._pending_param_updates = {}
        self._param_update_timer = QTimer(self)
        self._param_update_timer.setSingleShot(True)
        self._param_update_timer.timeout.connect(self._flush_pending_param_updates)
        self._last_props_selection_sig = None
        self._pending_props_refresh = False
        self._props_refresh_timer = QTimer(self)
        self._props_refresh_timer.setSingleShot(True)
        self._props_refresh_timer.timeout.connect(self._flush_deferred_props_refresh)
        self._video_props_refresh_timer = QTimer(self)
        self._video_props_refresh_timer.setSingleShot(True)
        self._video_props_refresh_timer.timeout.connect(
            self._flush_selected_make_video_properties_refresh
        )
        self._debug_ui_perf = (
            str(os.getenv("NAPARI_FLOW_DEBUG_UI", "0")).strip().lower()
            in ("1", "true", "yes", "on")
        )
        self._ui_watchdog_last = time.perf_counter()
        self._ui_watchdog_timer = QTimer(self)
        self._ui_watchdog_timer.setInterval(250)
        self._ui_watchdog_timer.timeout.connect(self._ui_watchdog_tick)
        if self._debug_ui_perf:
            self._ui_watchdog_timer.start()
        self.macro_groups = {}  # group_id -> {"title","member_uids","collapsed","item"}
        self._active_macro_id = None
        
        # 1. Main Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 2. Top Toolbar (Add, Remove, Save, Load, etc.)
        toolbar = QHBoxLayout()
        
        self.btn_add = QPushButton("Add Node")
        toolbar.addWidget(self.btn_add)
        self.menu_add_node = QMenu(self)
        self.menu_add_node.aboutToShow.connect(
            lambda: self._populate_add_menu(
                self.menu_add_node, include_control=False, control_only=False
            )
        )
        self.btn_add.setMenu(self.menu_add_node)

        self.btn_add_control = QPushButton("Add Control")
        self.btn_add_control.setStyleSheet("""
            QPushButton {
                background-color: #4f6780;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #5d7a99; }
            QPushButton:pressed { background-color: #405469; }
            QPushButton::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: bottom right;
                right: 5px;
                bottom: 3px;
            }
        """)
        toolbar.addWidget(self.btn_add_control)
        self.menu_add_control = QMenu(self)
        self.menu_add_control.aboutToShow.connect(
            lambda: self._populate_add_menu(
                self.menu_add_control, include_control=True, control_only=True
            )
        )
        self.btn_add_control.setMenu(self.menu_add_control)

        self.btn_remove = QPushButton("Remove Node")
        toolbar.addWidget(self.btn_remove)
        self.menu_remove_node = QMenu(self)
        self.menu_remove_node.aboutToShow.connect(
            lambda: self._populate_remove_menu(self.menu_remove_node)
        )
        self.btn_remove.setMenu(self.menu_remove_node)

        # File dropdown (common app pattern)
        self.btn_file_menu = QPushButton("File")
        self.file_menu = QMenu(self)
        self.action_save_pipeline = self.file_menu.addAction("Save Pipeline")
        self.action_load_pipeline = self.file_menu.addAction("Load Pipeline")
        self.file_menu.addSeparator()
        self.action_save_zarr = self.file_menu.addAction("Save to Zarr")
        self.file_menu.addSeparator()
        self.action_import_nodes = self.file_menu.addAction("Import Nodes (.py)")
        self.action_export_script = self.file_menu.addAction("Export Script")

        self.action_save_pipeline.triggered.connect(self.save_pipeline)
        self.action_load_pipeline.triggered.connect(self.load_pipeline)
        self.action_save_zarr.triggered.connect(self.save_to_zarr)
        self.action_import_nodes.triggered.connect(self.import_custom_module)
        self.action_export_script.triggered.connect(self.export_to_python)

        self.btn_file_menu.setMenu(self.file_menu)
        toolbar.addWidget(self.btn_file_menu)

        # Keep old attributes as aliases to preserve internal references.
        self.btn_save = self.action_save_pipeline
        self.btn_load = self.action_load_pipeline
        self.btn_save_zarr = self.action_save_zarr
        self.btn_import = self.action_import_nodes
        self.btn_export = self.action_export_script

        self.btn_collapse_macro = QPushButton("Collapse Macro")
        self.btn_collapse_macro.clicked.connect(self.collapse_selected_to_macro)
        toolbar.addWidget(self.btn_collapse_macro)
        self.btn_expand_macro = None
        self.btn_back_macro = None

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

        # Interaction action bar (visible only while waiting for user input)
        self.interaction_bar = QFrame()
        self.interaction_bar.setVisible(False)
        self.interaction_bar.setStyleSheet(
            "QFrame {"
            " background-color: #1f3b2f;"
            " border: 1px solid #2f6a4b;"
            " border-radius: 6px;"
            "}"
        )
        interaction_layout = QHBoxLayout(self.interaction_bar)
        interaction_layout.setContentsMargins(8, 6, 8, 6)
        interaction_layout.setSpacing(8)
        self.interaction_status_label = QLabel("Waiting for input")
        self.interaction_status_label.setStyleSheet("color: #d9ffe7; font-weight: bold;")
        self.interaction_prompt_label = QLabel("")
        self.interaction_prompt_label.setStyleSheet("color: #d6e7dc;")
        self.interaction_prompt_label.setWordWrap(True)
        self.btn_interaction_cancel = QPushButton("Cancel")
        self.btn_interaction_run = QPushButton("Run")
        self.btn_interaction_run.setStyleSheet(
            "QPushButton { background-color: #2E7D32; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #388E3C; }"
        )
        interaction_layout.addWidget(self.interaction_status_label)
        interaction_layout.addWidget(self.interaction_prompt_label, 1)
        interaction_layout.addWidget(self.btn_interaction_cancel)
        interaction_layout.addWidget(self.btn_interaction_run)
        button_layout.addWidget(self.interaction_bar)

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
        self.scene.editor_ref = self
        self.view = FlowView(self.scene)
        self.view.delete_callback = self.delete_selected_nodes
        self.view.resize_callback = self._position_floating_buttons
        self.inner_splitter.addWidget(self.view)

        # Floating in-graph button shown only while viewing inside a macro.
        self.btn_back_macro_floating = QPushButton("Back to Main", self.view.viewport())
        self.btn_back_macro_floating.clicked.connect(self.exit_macro_view)
        self.btn_back_macro_floating.setVisible(False)
        self.btn_back_macro_floating.setStyleSheet(
            "QPushButton {"
            " background-color: #4f6780;"
            " color: white;"
            " font-weight: bold;"
            " border: 1px solid #405469;"
            " border-radius: 5px;"
            " padding: 6px 10px;"
            "}"
            "QPushButton:hover { background-color: #5d7a99; }"
            "QPushButton:pressed { background-color: #405469; }"
        )
        self.btn_back_macro_floating.raise_()
        
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
        self.btn_interaction_cancel.clicked.connect(self._on_inline_interaction_cancel)
        self.btn_interaction_run.clicked.connect(self._on_inline_interaction_run)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.scene.selectionChanged.connect(self.on_selection)
        self._position_floating_buttons()
        app = QApplication.instance()
        if app is not None:
            try:
                app.focusChanged.connect(self._on_application_focus_changed)
            except Exception:
                pass

    def _position_floating_buttons(self):
        btn = getattr(self, "btn_back_macro_floating", None)
        view = getattr(self, "view", None)
        if btn is None or view is None:
            return
        viewport = view.viewport()
        hint = btn.sizeHint()
        btn.resize(hint)
        margin = 12
        x = max(margin, viewport.width() - btn.width() - margin)
        y = margin
        btn.move(x, y)

    def _set_back_macro_button_visible(self, visible):
        btn = getattr(self, "btn_back_macro_floating", None)
        if btn is None:
            return
        btn.setVisible(bool(visible))
        if visible:
            self._position_floating_buttons()
            btn.raise_()

    def _is_editing_properties_widget(self):
        focused = QApplication.focusWidget()
        return (
            focused is not None
            and self.props_group is not None
            and self.props_group.isAncestorOf(focused)
            and isinstance(
                focused,
                (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTableWidget),
            )
        )

    def _flush_deferred_props_refresh(self):
        if not self._pending_props_refresh:
            return
        if self._is_editing_properties_widget():
            return
        self._pending_props_refresh = False
        self.on_selection(force=True)

    def _schedule_selected_make_video_properties_refresh(self, delay_ms=220):
        if self._is_editing_properties_widget():
            self._pending_props_refresh = True
            return
        if self._video_props_refresh_timer.isActive():
            self._video_props_refresh_timer.stop()
        self._video_props_refresh_timer.start(max(50, int(delay_ms)))

    def _flush_selected_make_video_properties_refresh(self):
        if self._is_editing_properties_widget():
            self._pending_props_refresh = True
            return
        self._refresh_selected_make_video_properties()

    def _on_application_focus_changed(self, _old, _new):
        if self._debug_ui_perf:
            old_name = _old.__class__.__name__ if _old is not None else "None"
            new_name = _new.__class__.__name__ if _new is not None else "None"
            print(f"[FlowEditor][UI] focusChanged: {old_name} -> {new_name}")
        if not self._pending_props_refresh:
            return
        if self._is_editing_properties_widget():
            return
        self._pending_props_refresh = False
        self.on_selection(force=True)

    def _bind_viewer_refresh_events(self):
        """
        Keep make-video dynamic UI (e.g. sweep axis idx options) synchronized
        when viewer data/layout changes.
        """
        def _safe_refresh(_event=None):
            self._schedule_selected_make_video_properties_refresh()

        try:
            self.viewer.layers.events.inserted.connect(_safe_refresh)
            self.viewer.layers.events.removed.connect(_safe_refresh)
        except Exception:
            pass

    def _ui_watchdog_tick(self):
        now = time.perf_counter()
        dt_ms = (now - self._ui_watchdog_last) * 1000.0
        self._ui_watchdog_last = now
        lag_ms = dt_ms - 250.0
        if lag_ms < 500.0:
            return
        focused = QApplication.focusWidget()
        focus_name = focused.__class__.__name__ if focused is not None else "None"
        print(
            f"[FlowEditor][UI] event-loop lag detected: +{lag_ms:.1f} ms "
            f"(focus={focus_name})"
        )

        try:
            self.viewer.layers.selection.events.changed.connect(_safe_refresh)
        except Exception:
            pass

        try:
            self.viewer.dims.events.order.connect(_safe_refresh)
        except Exception:
            pass
        try:
            self.viewer.dims.events.ndisplay.connect(_safe_refresh)
        except Exception:
            pass

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

    def _next_macro_title(self):
        existing = {g.get("title", "") for g in self.macro_groups.values()}
        base = "Macro"
        if base not in existing:
            return base
        i = 1
        while True:
            name = f"{base} {i}"
            if name not in existing:
                return name
            i += 1

    def _get_all_connections(self):
        return [item for item in self.scene.items() if isinstance(item, Connection)]

    def _socket_ref_key_from_socket(self, socket):
        node = getattr(socket, "node", None)
        node_uid = getattr(node, "uid", None)
        if node_uid is None:
            return None
        return (
            str(node_uid),
            str(getattr(socket, "name", "")),
            str(getattr(socket, "socket_type", "")),
        )

    def _socket_ref_key_from_binding(self, binding):
        if not isinstance(binding, dict):
            return None
        node_uid = binding.get("node_uid")
        if not node_uid:
            return None
        return (
            str(node_uid),
            str(binding.get("socket_name", "")),
            str(binding.get("socket_type", "")),
        )

    def _socket_binding_from_socket(self, socket):
        if socket is None:
            return None
        node_uid = getattr(getattr(socket, "node", None), "uid", None)
        if not node_uid:
            return None
        binding = {
            "node_uid": str(node_uid),
            "socket_name": str(getattr(socket, "name", "")),
            "socket_type": str(getattr(socket, "socket_type", "")),
            "max_connections": getattr(socket, "max_connections", None),
        }
        if not getattr(socket, "is_logic", False):
            binding["data_type"] = _normalize_data_type(
                getattr(socket, "data_type", "any")
            )
        return binding

    def _resolve_socket_binding(self, binding):
        if not isinstance(binding, dict):
            return None
        node = self._get_node_by_uid(binding.get("node_uid"))
        if node is None:
            return None
        socket_name = binding.get("socket_name")
        socket_type = binding.get("socket_type")
        pool = []
        if socket_type == "input":
            pool = getattr(node, "inputs", [])
        elif socket_type == "output":
            pool = getattr(node, "outputs", [])
        elif socket_type == "logic_in":
            pool = getattr(node, "logic_inputs", [])
        elif socket_type == "logic_out":
            pool = getattr(node, "logic_outputs", [])
        for socket in pool:
            if socket.name == socket_name:
                return socket
        return None

    def _derive_macro_flow_endpoints(self, member_nodes):
        if not member_nodes:
            return None, None
        if len(member_nodes) == 1:
            return member_nodes[0], member_nodes[0]

        member_uids = {n.uid for n in member_nodes}
        incoming = {n.uid: 0 for n in member_nodes}
        outgoing = {n.uid: 0 for n in member_nodes}
        external_entries = set()  # outside -> inside (exec boundary in)
        external_exits = set()    # inside -> outside (exec boundary out)

        for edge in self._get_all_connections():
            if edge.start_socket is None or edge.end_socket is None:
                continue
            if not getattr(edge, "is_logic", False):
                continue
            src = getattr(edge.start_socket, "node", None)
            dst = getattr(edge.end_socket, "node", None)
            if not isinstance(src, Node) or not isinstance(dst, Node):
                continue
            src_in = src.uid in member_uids
            dst_in = dst.uid in member_uids
            if src_in and dst_in:
                outgoing[src.uid] += 1
                incoming[dst.uid] += 1
            elif (not src_in) and dst_in:
                external_entries.add(dst.uid)
            elif src_in and (not dst_in):
                external_exits.add(src.uid)

        def sort_key(n):
            p = n.pos()
            return (p.x(), p.y())

        # Prefer true exec-boundary nodes when macro is wired into the
        # external white flow thread.
        if external_entries:
            first_candidates = [
                n for n in member_nodes if n.uid in external_entries
            ]
        else:
            first_candidates = [n for n in member_nodes if incoming[n.uid] == 0]

        if external_exits:
            last_candidates = [
                n for n in member_nodes if n.uid in external_exits
            ]
        else:
            last_candidates = [n for n in member_nodes if outgoing[n.uid] == 0]

        first = min(first_candidates or member_nodes, key=sort_key)
        last = max(
            last_candidates or member_nodes,
            key=lambda n: (n.pos().x(), n.pos().y()),
        )
        return first, last

    def _sync_macro_group_bindings(self, group_id):
        group = self.macro_groups.get(group_id)
        if not group:
            return
        item = group.get("item")
        if item is None:
            return

        members = [
            self._get_node_by_uid(uid)
            for uid in group.get("member_uids", set())
        ]
        members = [n for n in members if isinstance(n, Node)]
        if not members:
            item.set_bindings()
            return

        first, last = self._derive_macro_flow_endpoints(members)
        if first is None or last is None:
            item.set_bindings()
            return

        input_bindings = [
            self._socket_binding_from_socket(s) for s in getattr(first, "inputs", [])
        ]
        output_bindings = [
            self._socket_binding_from_socket(s) for s in getattr(last, "outputs", [])
        ]
        logic_input_bindings = [
            self._socket_binding_from_socket(s)
            for s in getattr(first, "logic_inputs", [])
        ]
        logic_output_bindings = [
            self._socket_binding_from_socket(s)
            for s in getattr(last, "logic_outputs", [])
        ]

        input_bindings = [b for b in input_bindings if b]
        output_bindings = [b for b in output_bindings if b]
        logic_input_bindings = [b for b in logic_input_bindings if b]
        logic_output_bindings = [b for b in logic_output_bindings if b]

        group["input_bindings"] = input_bindings
        group["output_bindings"] = output_bindings
        group["logic_input_bindings"] = logic_input_bindings
        group["logic_output_bindings"] = logic_output_bindings

        item.set_bindings(
            input_bindings=input_bindings,
            output_bindings=output_bindings,
            logic_input_bindings=logic_input_bindings,
            logic_output_bindings=logic_output_bindings,
        )

    def _set_macro_visible(self, macro_item, visible):
        if macro_item is None:
            return
        if hasattr(macro_item, "set_macro_visible"):
            macro_item.set_macro_visible(visible)
        else:
            macro_item.setVisible(visible)

    def _clear_macro_proxy_edges(self):
        for group in self.macro_groups.values():
            item = group.get("item")
            if item is not None and hasattr(item, "clear_proxy_edges"):
                item.clear_proxy_edges()

    def _macro_socket_for_internal_endpoint(self, socket, want_start):
        if socket is None:
            return None
        if socket.isVisible():
            return socket

        node = getattr(socket, "node", None)
        if not isinstance(node, Node):
            return None

        group_id = self._group_for_node_uid(node.uid)
        if not group_id:
            return None
        group = self.macro_groups.get(group_id)
        if not group or not group.get("collapsed", True):
            return None
        item = group.get("item")
        if item is None or not item.isVisible():
            return None

        key = self._socket_ref_key_from_socket(socket)
        if key is None:
            return None

        if want_start:
            candidates = item.outputs + item.logic_outputs
        else:
            candidates = item.inputs + item.logic_inputs
        for macro_socket in candidates:
            m_key = self._socket_ref_key_from_binding(
                getattr(macro_socket, "macro_binding", None)
            )
            if m_key == key:
                return macro_socket
        return None

    def _rebuild_macro_proxy_edges(self):
        self._clear_macro_proxy_edges()
        if self._active_macro_id:
            return

        for edge in self._get_all_connections():
            if edge.start_socket is None or edge.end_socket is None:
                continue
            if getattr(edge, "dragging", False):
                continue
            s_node = getattr(edge.start_socket, "node", None)
            e_node = getattr(edge.end_socket, "node", None)
            if isinstance(s_node, Node) and isinstance(e_node, Node):
                s_group = self._group_for_node_uid(s_node.uid)
                e_group = self._group_for_node_uid(e_node.uid)
                if s_group and s_group == e_group:
                    continue

            start_proxy = self._macro_socket_for_internal_endpoint(
                edge.start_socket, want_start=True
            )
            end_proxy = self._macro_socket_for_internal_endpoint(
                edge.end_socket, want_start=False
            )

            if start_proxy is None or end_proxy is None:
                continue
            if start_proxy is edge.start_socket and end_proxy is edge.end_socket:
                continue

            proxy = MacroProxyConnection(start_proxy, end_proxy, real_edge=edge)
            self.scene.addItem(proxy)

    def _resolve_macro_socket_to_internal(self, socket):
        if socket is None:
            return None
        if not isinstance(getattr(socket, "node", None), MacroGroupItem):
            return socket
        binding = getattr(socket, "macro_binding", None)
        return self._resolve_socket_binding(binding)

    def finalize_connection_with_macros(self, temp_connection, target_socket):
        """
        Scene hook: if a drag starts/ends on macro sockets, translate it to the
        corresponding internal real sockets so execution remains unchanged.
        """
        start_socket = getattr(temp_connection, "start_socket", None)
        start_is_macro = isinstance(getattr(start_socket, "node", None), MacroGroupItem)
        target_is_macro = isinstance(getattr(target_socket, "node", None), MacroGroupItem)
        if not start_is_macro and not target_is_macro:
            return False

        real_start = self._resolve_macro_socket_to_internal(start_socket)
        real_target = self._resolve_macro_socket_to_internal(target_socket)

        if real_start is None or real_target is None:
            if temp_connection.scene() is self.scene:
                self.scene.removeItem(temp_connection)
            return True

        is_logic = bool(getattr(real_start, "is_logic", False))
        if is_logic != bool(getattr(real_target, "is_logic", False)):
            if temp_connection.scene() is self.scene:
                self.scene.removeItem(temp_connection)
            return True

        if is_logic:
            valid = (
                real_target.socket_type == "logic_in"
                and real_target.can_accept_connection()
                and not self.scene.are_already_connected(real_start, real_target)
                and not self.scene.creates_logic_cycle(real_start.node, real_target.node)
            )
        else:
            valid = (
                real_target.socket_type == "input"
                and real_target.can_accept_connection()
                and not self.scene.are_already_connected(real_start, real_target)
                and not self.scene.creates_cycle(real_start.node, real_target.node)
                and self.scene.is_data_connection_compatible(real_start, real_target)
            )

        if temp_connection.scene() is self.scene:
            self.scene.removeItem(temp_connection)
        if not valid:
            return True

        real_conn = Connection(real_start, self.scene)
        real_conn.finalize(real_target)
        self.scene.addItem(real_conn)
        self._refresh_macro_visibility()
        return True

    def _set_node_visible(self, node, visible):
        node.setVisible(visible)
        for socket in node.all_sockets():
            socket.setVisible(visible)

    def _get_node_by_uid(self, uid):
        for n in self._get_all_nodes():
            if n.uid == uid:
                return n
        return None

    def _group_for_node_uid(self, uid):
        for group_id, group in self.macro_groups.items():
            if uid in group.get("member_uids", set()):
                return group_id
        return None

    def _refresh_macro_visibility(self):
        nodes = self._get_all_nodes()

        # Keep macro card counters in sync.
        for group_id, group in self.macro_groups.items():
            members = group.get("member_uids", set())
            existing_members = [uid for uid in members if self._get_node_by_uid(uid)]
            group["member_uids"] = set(existing_members)
            item = group.get("item")
            if item:
                item.node_count = len(existing_members)
                self._sync_macro_group_bindings(group_id)
                item.update()

        if self._active_macro_id and self._active_macro_id in self.macro_groups:
            active_members = self.macro_groups[self._active_macro_id].get("member_uids", set())
            for node in nodes:
                self._set_node_visible(node, node.uid in active_members)
            for group_id, group in self.macro_groups.items():
                item = group.get("item")
                if item is not None:
                    self._set_macro_visible(item, False)
        else:
            self._active_macro_id = None
            self._set_back_macro_button_visible(False)
            collapsed_members = set()
            for group in self.macro_groups.values():
                if group.get("collapsed", True):
                    collapsed_members.update(group.get("member_uids", set()))

            for node in nodes:
                self._set_node_visible(node, node.uid not in collapsed_members)

            for group in self.macro_groups.values():
                item = group.get("item")
                if item is not None:
                    self._set_macro_visible(item, bool(group.get("collapsed", True)))

        for edge in self._get_all_connections():
            if getattr(edge, "dragging", False):
                edge.setVisible(True)
                continue
            if edge.start_socket is None:
                edge.setVisible(False)
                continue
            if edge.end_socket is None:
                edge.setVisible(edge.start_socket.isVisible())
                continue
            edge.setVisible(edge.start_socket.isVisible() and edge.end_socket.isVisible())

        self._rebuild_macro_proxy_edges()
        self.scene.update()
        self.view.viewport().update()

    def _clear_macro_groups(self):
        for group in list(self.macro_groups.values()):
            item = group.get("item")
            if item is not None and hasattr(item, "clear_sockets"):
                item.clear_sockets()
            if item is not None and item.scene() is self.scene:
                self.scene.removeItem(item)
        self.macro_groups = {}
        self._active_macro_id = None
        self._set_back_macro_button_visible(False)

    def collapse_selected_to_macro(self):
        if self._active_macro_id:
            QMessageBox.information(
                self,
                "Collapse Not Available",
                "Exit the current macro before collapsing a new selection.",
            )
            return

        selected_nodes = [
            item for item in self.scene.selectedItems()
            if isinstance(item, Node)
        ]
        if len(selected_nodes) < 2:
            QMessageBox.information(
                self,
                "Select Nodes",
                "Select at least 2 nodes to collapse into a macro.",
            )
            return

        for node in selected_nodes:
            if node.node_type == "begin":
                QMessageBox.warning(
                    self,
                    "Cannot Collapse",
                    "Begin node cannot be collapsed into a macro.",
                )
                return
            if self._group_for_node_uid(node.uid):
                QMessageBox.warning(
                    self,
                    "Cannot Collapse",
                    f"Node '{node.title}' is already in a macro group.",
                )
                return

        # Use selection bounds to place the collapsed macro card.
        bounds = selected_nodes[0].sceneBoundingRect()
        for node in selected_nodes[1:]:
            bounds = bounds.united(node.sceneBoundingRect())
        macro_pos = bounds.center() - QPointF(85, 45)

        group_id = str(uuid.uuid4())
        group_title = self._next_macro_title()
        macro_item = MacroGroupItem(
            macro_pos.x(),
            macro_pos.y(),
            group_id=group_id,
            title=group_title,
            node_count=len(selected_nodes),
            on_open=self.enter_macro,
            scene=self.scene,
        )
        self.scene.addItem(macro_item)

        self.macro_groups[group_id] = {
            "id": group_id,
            "title": group_title,
            "member_uids": {n.uid for n in selected_nodes},
            "collapsed": True,
            "item": macro_item,
        }

        for item in self.scene.selectedItems():
            item.setSelected(False)
        macro_item.setSelected(True)

        self._refresh_macro_visibility()
        self.on_selection()
        self.append_log(f"📦 Collapsed {len(selected_nodes)} nodes into '{group_title}'.")

    def enter_macro(self, group_id):
        group = self.macro_groups.get(group_id)
        if not group:
            return
        self._active_macro_id = group_id
        self._set_back_macro_button_visible(True)
        self._refresh_macro_visibility()
        self.append_log(f"🔍 Entered macro: {group.get('title', 'Macro')}")

    def exit_macro_view(self):
        if not self._active_macro_id:
            return
        self._active_macro_id = None
        self._set_back_macro_button_visible(False)
        self._refresh_macro_visibility()
        self.append_log("↩️ Returned to main graph.")

    def _expand_macro_group(self, group_id):
        group = self.macro_groups.get(group_id)
        if not group:
            return
        if self._active_macro_id == group_id:
            self._active_macro_id = None
            self._set_back_macro_button_visible(False)

        item = group.get("item")
        if item is not None and hasattr(item, "clear_sockets"):
            item.clear_sockets()
        if item is not None and item.scene() is self.scene:
            self.scene.removeItem(item)
        self.macro_groups.pop(group_id, None)
        self._refresh_macro_visibility()
        self.on_selection()

    def expand_selected_macro(self):
        selected_macros = [
            item for item in self.scene.selectedItems()
            if isinstance(item, MacroGroupItem)
        ]
        if not selected_macros:
            QMessageBox.information(
                self,
                "Select Macro",
                "Select a macro card to expand.",
            )
            return
        # V1: one-at-a-time
        macro_item = selected_macros[0]
        group = self.macro_groups.get(macro_item.group_id)
        if not group:
            return
        title = group.get("title", "Macro")
        self._expand_macro_group(macro_item.group_id)
        self.append_log(f"📂 Expanded macro: {title}")

    def _rename_macro_group(self, group_id, new_title):
        group = self.macro_groups.get(group_id)
        if not group:
            return
        title = str(new_title or "").strip()
        if not title:
            return
        if title == group.get("title", ""):
            return
        group["title"] = title
        item = group.get("item")
        if item is not None:
            item.title = title
            item.update()
        self.append_log(f"✏️ Renamed macro to '{title}'.")
        self.on_selection()
    
    # --- Node Selection Handler ---
    def on_selection(self, force=False):
        """Rebuilds the property panel based on selection."""
        t0 = time.perf_counter()
        sel = self.scene.selectedItems()
        sel_sig = tuple(
            sorted(
                (
                    f"node:{item.uid}"
                    if isinstance(item, Node)
                    else f"macro:{item.group_id}"
                    if isinstance(item, MacroGroupItem)
                    else f"other:{id(item)}"
                )
                for item in sel
            )
        )

        if not force:
            if self._is_editing_properties_widget():
                # Avoid rebuilding form widgets while user is typing.
                self._pending_props_refresh = True
                return

        self._last_props_selection_sig = sel_sig
        self._pending_props_refresh = False
        self.scene.update()
        self.view.viewport().update()
        # 1. Clear current widgets
        while self.props_layout.count():
            child = self.props_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        # 2. Get selected node
        if len(sel) == 1 and isinstance(sel[0], MacroGroupItem):
            macro_item = sel[0]
            group = self.macro_groups.get(macro_item.group_id, {})
            title = group.get("title", macro_item.title)
            member_count = len(group.get("member_uids", set()))
            self.props_layout.addRow(QLabel(f"<b>{title}</b>"))

            name_edit = QLineEdit(str(title))
            name_edit.setPlaceholderText("Macro name")
            name_edit.editingFinished.connect(
                lambda gid=macro_item.group_id, w=name_edit: self._rename_macro_group(
                    gid, w.text()
                )
            )
            self.props_layout.addRow("Name:", name_edit)

            self.props_layout.addRow(QLabel(f"Contains {member_count} node(s)."))
            self.props_layout.addRow(QLabel("<u>Nodes Inside</u>"))

            member_nodes = []
            for uid in group.get("member_uids", set()):
                node = self._get_node_by_uid(uid)
                if node is not None:
                    member_nodes.append(node)
            member_nodes.sort(key=lambda n: (n.pos().x(), n.pos().y(), n.title))

            members_list = QListWidget()
            members_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            members_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            members_list.setMaximumHeight(170)
            if not member_nodes:
                members_list.addItem("(empty)")
            else:
                for idx, node in enumerate(member_nodes, start=1):
                    members_list.addItem(QListWidgetItem(f"{idx}. {node.title}"))
            self.props_layout.addRow(members_list)

            self.props_layout.addRow(QLabel("Double-click to open internal view."))
            self.props_layout.addRow(QLabel("Select macro + Delete to ungroup."))
            return

        if len(sel) != 1 or not isinstance(sel[0], Node):
            self.props_layout.addRow(QLabel("Select a single node to edit parameters."))
            return

        node = sel[0]
        
        # --- HEADER ---
        # Display title, info tooltip and node ID.
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(6)

        title_layout.addWidget(QLabel(f"<b>{node.title}</b>"))
        if getattr(node, "description", ""):
            escaped_description = html.escape(str(node.description).strip()).replace("\n", "<br>")
            tooltip_html = (
                "<div style='max-width: 340px; white-space: pre-wrap; line-height: 1.35;'>"
                f"{escaped_description}"
                "</div>"
            )
            info_button = InfoHoverButton(tooltip_html)
            info_button.setFixedSize(16, 16)
            info_button.setCursor(Qt.CursorShape.PointingHandCursor)
            info_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            info_button.setStyleSheet(
                "QPushButton {"
                " border: 1px solid #6f7a86;"
                " border-radius: 8px;"
                " color: #dbe7f3;"
                " font-weight: bold;"
                " background: #2c3946;"
                "}"
                "QPushButton:hover { background: #395067; }"
            )
            title_layout.addWidget(info_button)
        title_layout.addStretch()

        id_label = QLabel(f"<span style='color:#888; font-size:10px;'>ID: {node.uid[:8]}...</span>")
        self.props_layout.addRow(title_widget, id_label)
        
        # Spacer
        self.props_layout.addRow(QLabel("")) 

        # --- SECTION 1: PARAMETERS ---
        self.props_layout.addRow(QLabel("<u>Parameters</u>"))

        if node.node_type not in NODE_LIBRARY:
            self.props_layout.addRow(QLabel("No parameters defined."))
        else:
            params_def = NODE_LIBRARY[node.node_type]["parameters"]
            pending = self._pending_interaction
            pending_layer_choice = (
                pending
                and pending.get("node_uid") == node.uid
                and pending.get("interaction_type") == "layer_choice"
            )
            
            for param_name, conf in params_def.items():
                default_val = conf.get("default", "")
                current_val = node.parameters.get(param_name, default_val)
                if (
                    node.node_type == "select_layer"
                    and param_name == "layer_name"
                    and not pending_layer_choice
                ):
                    placeholder = QLabel("<em>Will be chosen at runtime.</em>")
                    placeholder.setStyleSheet("color: #888;")
                    self.props_layout.addRow("Select Layer:", placeholder)
                    continue

                if (
                    node.node_type == "select_layer"
                    and param_name == "layer_type"
                    and not pending_layer_choice
                ):
                    placeholder = QLabel("<em>Auto from selected layer.</em>")
                    placeholder.setStyleSheet("color: #888;")
                    self.props_layout.addRow("Layer Type:", placeholder)
                    continue

                if pending_layer_choice and param_name == "layer_name":
                    widget = QComboBox()
                    choices = pending.get("choices", [])
                    if not choices:
                        widget.addItem("No incoming layers")
                        widget.setEnabled(False)
                    else:
                        selected = pending.get("selected_choice", {})
                        selected_index = 0
                        for i, ch in enumerate(choices):
                            label = str(
                                ch.get(
                                    "label",
                                    f"{ch.get('name', '')} ({ch.get('layer_type', '')})",
                                )
                            )
                            widget.addItem(label, ch)
                            if (
                                ch.get("name") == selected.get("name")
                                and ch.get("layer_type") == selected.get("layer_type")
                            ):
                                selected_index = i
                        widget.setCurrentIndex(selected_index)

                        def _on_choice_change(index, nuid=node.uid, combo=widget):
                            ch = combo.itemData(index)
                            if not ch:
                                return
                            self._set_pending_layer_choice(nuid, ch)
                            self.on_selection(force=True)

                        widget.currentIndexChanged.connect(_on_choice_change)
                    self.props_layout.addRow("Select Layer:", widget)
                    continue

                if pending_layer_choice and param_name == "layer_type":
                    selected = pending.get("selected_choice", {})
                    layer_type_text = str(
                        selected.get(
                            "layer_type",
                            node.parameters.get("layer_type", current_val),
                        )
                    )
                    self.props_layout.addRow("Layer Type:", QLabel(layer_type_text))
                    continue
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
                    # Avoid firing graph-invalidations on every keystroke.
                    widget.setKeyboardTracking(False)
                    widget.valueChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                    widget.editingFinished.connect(
                        lambda w=widget, n=node, k=param_name: self.update_param(
                            n, k, float(w.value())
                        )
                    )
                
                # INT
                elif conf["type"] == "int":
                    widget = QSpinBox()
                    widget.setRange(conf.get("min", -9999), conf.get("max", 9999))
                    widget.setValue(int(current_val))
                    # Make keyboard editing responsive in large graphs.
                    widget.setKeyboardTracking(False)
                    widget.valueChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                    widget.editingFinished.connect(
                        lambda w=widget, n=node, k=param_name: self.update_param(
                            n, k, int(w.value())
                        )
                    )
                
                # BOOL
                elif conf["type"] == "bool":
                    widget = QCheckBox()
                    widget.setChecked(bool(current_val))
                    widget.toggled.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                
                # ENUM (Standard static dropdowns from library)
                elif conf["type"] == "enum":
                    enum_conf = conf
                    if node.node_type == "make_video" and param_name == "axis":
                        enum_conf = self._prepare_video_axis_enum_config(conf, current_val)
                    widget = QComboBox()
                    widget.addItems(enum_conf.get("options", []))
                    widget.setCurrentText(str(current_val))
                    widget.currentTextChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                
                # PATH
                elif conf["type"] == "path":
                    widget_container = QWidget()
                    layout = QHBoxLayout(widget_container)
                    layout.setContentsMargins(0, 0, 0, 0)
                    
                    line_edit = QLineEdit(str(current_val))
                    line_edit.setClearButtonEnabled(True)
                    browse_btn = QPushButton("...")
                    browse_btn.setFixedWidth(30)
                    
                    layout.addWidget(line_edit)
                    layout.addWidget(browse_btn)
                    
                    # --- FIX 2: Pass 'conf' (c=conf) to capture the loop variable properly ---
                    def open_file_dialog(le=line_edit, n=node, k=param_name, c=conf):
                        mode = str(c.get("mode", "file")).strip().lower()
                        if mode == "directory":
                            path = QFileDialog.getExistingDirectory(
                                self, "Select Directory"
                            )
                        else:
                            file_filter = (
                                str(c.get("filter", "All Files (*)")).strip()
                                or "All Files (*)"
                            )
                            path, _ = QFileDialog.getOpenFileName(
                                self,
                                "Select File",
                                "",
                                file_filter,
                            )
                            
                        if path:
                            le.setText(path)
                            self.update_param(n, k, path)

                    browse_btn.clicked.connect(lambda _: open_file_dialog())
                    line_edit.textEdited.connect(
                        lambda val, n=node, k=param_name: self._update_param_live(
                            n, k, val
                        )
                    )
                    line_edit.editingFinished.connect(
                        lambda le=line_edit, n=node, k=param_name: self.update_param(
                            n, k, le.text()
                        )
                    )
                    line_edit.returnPressed.connect(
                        lambda le=line_edit, n=node, k=param_name: self.update_param(
                            n, k, le.text()
                        )
                    )
                    
                    widget = widget_container
                
                # TABLE
                elif conf["type"] == "table":
                    table_conf = conf
                    if node.node_type == "make_video" and param_name == "instructions":
                        table_conf = self._prepare_video_instructions_table_config(
                            conf, current_val
                        )

                    limit = table_conf.get('max_rows', None) 
                    
                    is_row_unique = table_conf.get('row_unique', False)
                    allow_list = table_conf.get('allow_duplicates', [])
                    # Pass 'limit' to the class here:
                    widget = DynamicTableWidget(
                        table_conf.get('columns', []), 
                        max_rows=limit, 
                        row_unique=is_row_unique,
                        allow_duplicates=allow_list
                    )

                    # Load Data
                    val_to_load = current_val if current_val is not None else table_conf.get('value', [])
                    widget.set_value(val_to_load)

                    # Connect Signal (debounced): table edits can emit many
                    # updates while user is still interacting.
                    widget.valueChanged.connect(
                        lambda data, n=node, k=param_name: self._schedule_param_update(
                            n, k, data, delay_ms=260
                        )
                    )
                    
                    self.props_layout.addRow(param_name.capitalize(), widget)
                    continue


                
                # STRING / OTHER
                else:
                    widget = QLineEdit(str(current_val))
                    widget.setClearButtonEnabled(True)
                    widget.textEdited.connect(
                        lambda val, n=node, k=param_name: self._update_param_live(
                            n, k, val
                        )
                    )
                    widget.editingFinished.connect(
                        lambda w=widget, n=node, k=param_name: self.update_param(
                            n, k, w.text()
                        )
                    )
                    widget.returnPressed.connect(
                        lambda w=widget, n=node, k=param_name: self.update_param(
                            n, k, w.text()
                        )
                    )

                self.props_layout.addRow(param_name.capitalize(), widget)

        # --- INTERACTIVE CARD (inline in parameters panel) ---
        pending = self._pending_interaction
        if pending and pending.get("node_uid") == node.uid:
            prompt = pending.get("config", {}).get(
                "prompt", "Draw on the layer, then click Run."
            )
            self.props_layout.addRow(QLabel(""))
            self.props_layout.addRow(QLabel("<u>Interactive</u>"))

            prompt_lbl = QLabel(prompt)
            prompt_lbl.setWordWrap(True)
            prompt_lbl.setStyleSheet(
                "font-size: 12px; padding: 6px; color: #ddd; background: #2d2d2d; border-radius: 4px;"
            )
            self.props_layout.addRow(prompt_lbl)
            self.props_layout.addRow(QLabel("<em>Use the Interaction Bar above the graph to Run or Cancel.</em>"))

        # Spacer
        self.props_layout.addRow(QLabel("")) 

        # --- SECTION 2: SOCKETS INFO ---
        self.props_layout.addRow(QLabel("<u>Connections</u>"))

        # A. Inputs
        if node.inputs:
            self.props_layout.addRow(QLabel("<b>Inputs:</b>"))
            for s in node.inputs:
                type_color = _get_data_type_color(getattr(s, "data_type", "any")).name()
                # Check connection status
                if s.connected_edges:
                    status_text = "Connected"
                    style = f"color: {type_color};"
                else:
                    status_text = "Empty"
                    style = "color: #888;"   # Gray
                
                status_lbl = QLabel(status_text)
                status_lbl.setStyleSheet(style)
                self.props_layout.addRow(
                    f"  \u25B8 {s.name} [{getattr(s, 'data_type', 'any')}]",
                    status_lbl,
                )
        
        # B. Outputs
        if node.outputs:
            self.props_layout.addRow(QLabel("<b>Outputs:</b>"))
            for s in node.outputs:
                count = len(s.connected_edges)
                status_text = f"{count} link(s)"
                type_color = _get_data_type_color(getattr(s, "data_type", "any")).name()
                style = f"color: {type_color};" if count > 0 else "color: #888;"
                
                status_lbl = QLabel(status_text)
                status_lbl.setStyleSheet(style)
                self.props_layout.addRow(
                    f"  \u25B8 {s.name} [{getattr(s, 'data_type', 'any')}]",
                    status_lbl,
                )

        # C. Exec (logic thread) connections
        self.props_layout.addRow(QLabel("<b>Exec Flow:</b>"))
        if not node.logic_inputs and not node.logic_outputs:
            self.props_layout.addRow("  (none)", QLabel("Data-only node"))
        else:
            for s in node.logic_inputs:
                count = len(s.connected_edges)
                txt = "Connected" if count else "Empty"
                style = "color: #f2f2f2;" if count else "color: #888;"
                lbl = QLabel(txt)
                lbl.setStyleSheet(style)
                self.props_layout.addRow(f"  ⟵ {s.name}", lbl)

            for s in node.logic_outputs:
                count = len(s.connected_edges)
                txt = "Connected" if count else "Empty"
                style = "color: #f2f2f2;" if count else "color: #888;"
                lbl = QLabel(txt)
                lbl.setStyleSheet(style)
                self.props_layout.addRow(f"  ⟶ {s.name}", lbl)
        
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

        if self._debug_ui_perf:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms >= 80.0:
                try:
                    target = node.title if "node" in locals() else "(none)"
                except Exception:
                    target = "(unknown)"
                print(f"[FlowEditor][UI] on_selection({target}) took {elapsed_ms:.1f} ms")

    def _remove_edge(self, edge):
        if edge.start_socket and edge in edge.start_socket.connected_edges:
            edge.start_socket.connected_edges.remove(edge)
        if edge.end_socket and edge in edge.end_socket.connected_edges:
            edge.end_socket.connected_edges.remove(edge)
        if edge.scene() is self.scene:
            self.scene.removeItem(edge)

    def _resolve_dynamic_output_type(self, node, rule):
        fallback = _normalize_data_type(rule.get("fallback", "any"))
        source = str(rule.get("source", "")).strip().lower()

        if source != "viewer_layer_type":
            return fallback

        param_name = rule.get("from_param")
        if not param_name:
            return fallback

        layer_name = str(node.parameters.get(param_name, "")).strip()
        if not layer_name or layer_name not in self.viewer.layers:
            return fallback

        layer = self.viewer.layers[layer_name]
        return _infer_data_type_from_viewer_layer(layer)

    def _refresh_dynamic_output_types(self, node, changed_param=None):
        rules = getattr(node, "dynamic_output_types", {}) or {}
        if not rules:
            return

        disconnected_messages = []
        any_socket_type_changed = False

        for socket_name, rule in rules.items():
            watched_param = rule.get("from_param")
            if changed_param is not None and watched_param and watched_param != changed_param:
                continue

            output_socket = next((s for s in node.outputs if s.name == socket_name), None)
            if output_socket is None:
                continue

            previous_type = _normalize_data_type(getattr(output_socket, "data_type", "any"))
            resolved_type = self._resolve_dynamic_output_type(node, rule)
            output_socket.set_data_type(resolved_type)
            current_type = _normalize_data_type(output_socket.data_type)

            if current_type == previous_type:
                continue
            any_socket_type_changed = True

            for edge in list(output_socket.connected_edges):
                if getattr(edge, "is_logic", False):
                    continue
                if edge.start_socket is not output_socket:
                    continue
                if edge.end_socket is None:
                    continue

                input_socket = edge.end_socket
                if _are_data_types_compatible(current_type, getattr(input_socket, "data_type", "any")):
                    continue

                target_node = input_socket.node
                disconnected_messages.append(
                    f"{node.title}.{output_socket.name} -> {target_node.title}.{input_socket.name}"
                )
                self._remove_edge(edge)
                self.set_node_status_recursive(target_node, "gray")

        if disconnected_messages:
            self.append_log(
                f"⚠️ Auto-disconnected {len(disconnected_messages)} incompatible edge(s) after type update."
            )
            for msg in disconnected_messages[:4]:
                self.append_log(f"   - {msg}")
            if len(disconnected_messages) > 4:
                self.append_log("   - ...")

        selected = self.scene.selectedItems()
        if (any_socket_type_changed or disconnected_messages) and len(selected) == 1 and selected[0] is node:
            self.on_selection()

    def _prepare_video_axis_enum_config(self, conf, current_val):
        """
        Build robust axis choices for Make Video based on current viewer dims.

        Values are plain strings so they remain JSON/pipeline-friendly:
        - none
        - t/z/y/x aliases when resolvable
        - idx:N for explicit axis addressing
        """
        if not isinstance(conf, dict):
            return conf

        enum_conf = copy.deepcopy(conf)
        options = []
        seen = set()

        def _add(value):
            v = str(value).strip()
            if not v or v in seen:
                return
            seen.add(v)
            options.append(v)

        _add("none")

        dims = getattr(self.viewer, "dims", None)
        if dims is None:
            enum_conf["options"] = options
            return enum_conf

        labels = [str(lbl).strip() for lbl in getattr(dims, "axis_labels", ())]
        labels_l = [lbl.lower() for lbl in labels]
        nsteps = tuple(int(s) for s in getattr(dims, "nsteps", ()) or ())
        ndim = int(getattr(dims, "ndim", max(len(labels), len(nsteps), 0)) or 0)
        ndim = max(ndim, len(labels), len(nsteps))

        alias_idx = {}

        def _set_alias(alias, idx):
            if alias not in alias_idx and idx is not None and 0 <= idx < ndim:
                alias_idx[alias] = int(idx)

        for i, label in enumerate(labels_l):
            if label in ("t", "time"):
                _set_alias("t", i)
            elif label in ("z", "depth"):
                _set_alias("z", i)
            elif label == "y":
                _set_alias("y", i)
            elif label == "x":
                _set_alias("x", i)

        if ndim >= 1:
            _set_alias("x", ndim - 1)
        if ndim >= 2:
            _set_alias("y", ndim - 2)
        if ndim >= 3:
            _set_alias("z", ndim - 3)

        if "t" not in alias_idx:
            for i in range(max(0, ndim - 2)):
                steps = nsteps[i] if i < len(nsteps) else 0
                if int(steps) > 1:
                    _set_alias("t", i)
                    break
            if "t" not in alias_idx and ndim >= 3:
                _set_alias("t", 0)

        for key in ("t", "z", "y", "x"):
            if key in alias_idx:
                _add(key)

        for i in range(ndim):
            label = labels[i] if i < len(labels) and labels[i] else f"axis_{i}"
            steps = nsteps[i] if i < len(nsteps) else "?"
            _add(f"idx:{i} ({label}, n={steps})")

        current = str(current_val if current_val is not None else "").strip()
        if current:
            _add(current)

        enum_conf["options"] = options
        return enum_conf

    def _video_axis_index_options(self):
        dims = getattr(self.viewer, "dims", None)
        if dims is None:
            return ["idx:0"]
        labels = [str(lbl).strip() for lbl in getattr(dims, "axis_labels", ())]
        nsteps = tuple(int(s) for s in getattr(dims, "nsteps", ()) or ())
        ndim = int(getattr(dims, "ndim", max(len(labels), len(nsteps), 0)) or 0)
        ndim = max(ndim, len(labels), len(nsteps), 1)

        out = []
        for i in range(ndim):
            label = labels[i] if i < len(labels) and labels[i] else f"axis_{i}"
            steps = nsteps[i] if i < len(nsteps) else "?"
            out.append(f"idx:{i} ({label}, n={steps})")
        return out or ["idx:0"]

    def _prepare_video_instructions_table_config(self, conf, current_val):
        if not isinstance(conf, dict):
            return conf
        table_conf = copy.deepcopy(conf)
        columns = table_conf.get("columns", [])
        axis_col = next(
            (
                c
                for c in columns
                if isinstance(c, dict) and c.get("name") == "axis"
            ),
            None,
        )
        if axis_col is None:
            return table_conf

        options = self._video_axis_index_options()
        if isinstance(current_val, list):
            # Keep rows in sync with currently valid dims options.
            default_axis = options[0] if options else "idx:0"
            for row in current_val:
                if not isinstance(row, dict):
                    continue
                op = str(row.get("op", "")).strip().lower()
                if op != "sweep":
                    continue
                axis_val = str(row.get("axis", "")).strip()
                if not axis_val or axis_val not in options:
                    row["axis"] = default_axis
        axis_col["options"] = options
        return table_conf

    def _build_video_value_sequence(self, start, end, step):
        step_abs = abs(float(step))
        if step_abs < 1e-12:
            return []
        start_f = float(start)
        end_f = float(end)
        values = []
        if end_f >= start_f:
            v = start_f
            while v <= (end_f + 1e-9):
                values.append(float(v))
                v += step_abs
        else:
            v = start_f
            while v >= (end_f - 1e-9):
                values.append(float(v))
                v -= step_abs
        if not values:
            return [end_f]
        if abs(values[-1] - end_f) > 1e-8:
            values.append(end_f)
        return values

    def _parse_video_instructions(self, instructions):
        if instructions is None:
            instructions = []
        if isinstance(instructions, str):
            text = instructions.strip()
            if text:
                try:
                    instructions = json.loads(text)
                except Exception:
                    # Keep original value for downstream type check/error.
                    pass
        if isinstance(instructions, dict):
            instructions = [instructions]
        if not isinstance(instructions, list):
            return None, "instructions must be a list."

        dims = self.viewer.dims
        nsteps = tuple(getattr(dims, "nsteps", ()))
        actions = []
        segment_info = []
        needs_3d = False
        needs_2d_rotate = False

        for row_idx, row in enumerate(instructions, start=1):
            if not isinstance(row, dict):
                continue
            op = str(row.get("op", "rotate")).strip().lower()
            if not op:
                op = "rotate"
            if op not in ("rotate", "sweep"):
                return None, f"row {row_idx}: unknown operation '{op}'."
            view_mode = str(row.get("view", "keep")).strip().lower()
            if view_mode not in ("keep", "2d", "3d"):
                view_mode = "keep"
            if view_mode == "3d":
                needs_3d = True

            try:
                start_raw = row.get("start", 0.0)
                if start_raw in ("", None):
                    start_raw = 0.0
                start = float(start_raw)

                end_default = 360.0 if op == "rotate" else start
                end_raw = row.get("end", end_default)
                if end_raw in ("", None):
                    end_raw = end_default
                end = float(end_raw)

                step_raw = row.get("step", 1.0)
                if step_raw in ("", None):
                    step_raw = 1.0
                step = float(step_raw)
            except Exception:
                return None, f"row {row_idx}: start/end/step must be numeric."
            if abs(step) < 1e-12:
                return None, f"row {row_idx}: step must be non-zero."
            values = self._build_video_value_sequence(start, end, step)
            if not values:
                return None, f"row {row_idx}: empty value sequence."

            if op == "sweep":
                axis_name = str(row.get("axis", "")).strip().lower()
                if not axis_name or axis_name in ("none", "off"):
                    axis_opts = self._video_axis_index_options()
                    axis_name = axis_opts[0] if axis_opts else "idx:0"
                axis_idx = self._resolve_video_axis_index(axis_name)
                if axis_idx is None:
                    return None, f"row {row_idx}: invalid sweep axis '{axis_name}'."
                if axis_idx >= len(nsteps) or int(nsteps[axis_idx]) <= 0:
                    return None, f"row {row_idx}: sweep axis {axis_idx} has no steps."
                axis_max = int(nsteps[axis_idx]) - 1
                for value in values:
                    actions.append(
                        {
                            "op": "sweep",
                            "row": row_idx,
                            "view_mode": view_mode,
                            "axis_idx": axis_idx,
                            "axis_max": axis_max,
                            "value": int(round(value)),
                        }
                    )
                segment_info.append(
                    f"row {row_idx}: sweep axis={axis_name} view={view_mode} start={start} end={end} step={step}"
                )
                continue

            # rotate
            direction = str(row.get("direction", "clockwise")).strip().lower()
            if direction not in ("clockwise", "counterclockwise"):
                direction = "clockwise"
            dir_sign = -1.0 if direction == "clockwise" else 1.0
            space = str(row.get("space", "3d")).strip().lower()
            if space not in ("2d", "3d"):
                space = "3d"

            if space == "2d":
                needs_2d_rotate = True
                for value in values:
                    actions.append(
                        {
                            "op": "rotate2d",
                            "row": row_idx,
                            "view_mode": view_mode,
                            "value": dir_sign * float(value),
                        }
                    )
                segment_info.append(
                    f"row {row_idx}: rotate2d view={view_mode} dir={direction} start={start} end={end} step={step}"
                )
            else:
                rot_axis = str(row.get("rot_axis", "z")).strip().lower()
                axis_map = {
                    # Preferred options in current UI.
                    "x": "x",
                    "y": "y",
                    "z": "z",
                    # Back-compat aliases from previous UI labels.
                    "pitch": "x",
                    "roll": "y",
                    "yaw": "z",
                }
                if rot_axis not in axis_map:
                    rot_axis = "z"
                needs_3d = True
                rot_axis_camera = axis_map[rot_axis]
                for value in values:
                    actions.append(
                        {
                            "op": "rotate3d",
                            "row": row_idx,
                            "view_mode": view_mode,
                            "rot_axis_camera": rot_axis_camera,
                            "value": dir_sign * float(value),
                        }
                    )
                segment_info.append(
                    f"row {row_idx}: rotate3d axis={rot_axis_camera} view={view_mode} dir={direction} start={start} end={end} step={step}"
                )

        if not actions:
            return None, "no valid instruction rows."

        return {
            "actions": actions,
            "segments": segment_info,
            "needs_3d": needs_3d,
            "needs_2d_rotate": needs_2d_rotate,
        }, None

    def _resolve_video_axis_index(self, axis_name):
        key = str(axis_name or "none").strip().lower()
        if key in ("none", "", "off"):
            return None

        m = re.search(r"idx\s*:\s*(-?\d+)", key)
        if m:
            try:
                idx = int(m.group(1))
                return idx if idx >= 0 else None
            except Exception:
                return None
        if key.lstrip("-").isdigit():
            try:
                idx = int(key)
                return idx if idx >= 0 else None
            except Exception:
                return None

        labels = [
            str(label).strip().lower()
            for label in getattr(self.viewer.dims, "axis_labels", ())
        ]
        for i, label in enumerate(labels):
            if label == key:
                return i

        alias_map = {
            "t": {"t", "time"},
            "z": {"z", "depth"},
            "y": {"y"},
            "x": {"x"},
        }
        aliases = alias_map.get(key, {key})
        for i, label in enumerate(labels):
            if label in aliases:
                return i

        ndim = int(getattr(self.viewer.dims, "ndim", len(labels) or 0))
        fallback = {
            "x": ndim - 1,
            "y": ndim - 2 if ndim >= 2 else None,
            "z": ndim - 3 if ndim >= 3 else None,
            "t": 0 if ndim >= 3 else None,
        }
        idx = fallback.get(key)
        if idx is None or idx < 0 or idx >= ndim:
            return None
        return idx

    def _snapshot_viewer_state(self):
        dims = self.viewer.dims
        camera = self.viewer.camera
        return {
            "current_step": tuple(getattr(dims, "current_step", ())),
            "ndisplay": int(getattr(dims, "ndisplay", 2)),
            "camera_angles": tuple(getattr(camera, "angles", (0.0, 0.0, 0.0))),
            "camera_center": tuple(getattr(camera, "center", (0.0, 0.0, 0.0))),
            "camera_zoom": float(getattr(camera, "zoom", 1.0)),
            "layer_visibility": {
                str(getattr(layer, "name", "")): bool(getattr(layer, "visible", True))
                for layer in list(self.viewer.layers)
            },
        }

    def _restore_viewer_state(self, state):
        if not isinstance(state, dict):
            return

        dims = self.viewer.dims
        camera = self.viewer.camera

        try:
            dims.ndisplay = int(state.get("ndisplay", getattr(dims, "ndisplay", 2)))
        except Exception:
            pass

        saved_steps = tuple(state.get("current_step", ()))
        current_steps = tuple(getattr(dims, "current_step", ()))
        if len(saved_steps) == len(current_steps):
            for axis, step in enumerate(saved_steps):
                try:
                    dims.set_current_step(axis, int(step))
                except Exception:
                    pass

        for layer in list(self.viewer.layers):
            name = str(getattr(layer, "name", ""))
            if name in state.get("layer_visibility", {}):
                try:
                    layer.visible = bool(state["layer_visibility"][name])
                except Exception:
                    pass

        try:
            camera.angles = tuple(state.get("camera_angles", camera.angles))
        except Exception:
            pass
        try:
            camera.center = tuple(state.get("camera_center", camera.center))
        except Exception:
            pass
        try:
            camera.zoom = float(state.get("camera_zoom", camera.zoom))
        except Exception:
            pass

        QApplication.processEvents()

    def _normalize_video_frame(self, frame):
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Unexpected screenshot shape: {arr.shape}")

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32, copy=False)
            max_val = float(np.nanmax(arr)) if arr.size else 0.0
            if max_val <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return arr

    def _next_available_video_path(self, folder, filename, ext):
        base = str(filename or "napari_video").strip() or "napari_video"
        ext = str(ext or ".mp4").strip().lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        if base.lower().endswith(ext):
            base = base[: -len(ext)]

        candidate = os.path.join(folder, f"{base}{ext}")
        if not os.path.exists(candidate):
            return candidate

        idx = 1
        while True:
            candidate = os.path.join(folder, f"{base}_{idx:03d}{ext}")
            if not os.path.exists(candidate):
                return candidate
            idx += 1

    def _render_video_interaction(self, pending):
        config = pending.get("config", {}) if isinstance(pending, dict) else {}
        video_params = dict(config.get("video_params", {}) or {})

        instructions = video_params.get("instructions", [])
        fps = int(video_params.get("fps", 20))
        out_format = str(video_params.get("format", ".mp4")).strip().lower()
        folder = str(video_params.get("folder", "")).strip()
        filename = str(video_params.get("filename", "napari_video")).strip()

        if fps <= 0 or fps > 120:
            self.append_log("❌ Make Video: 'fps' must be between 1 and 120.")
            return None
        if not folder:
            self.append_log("❌ Make Video: select a save folder.")
            return None

        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as exc:
            self.append_log(f"❌ Make Video: cannot create folder '{folder}': {exc}")
            return None

        if out_format not in (".mp4", ".gif"):
            self.append_log("❌ Make Video: format must be .mp4 or .gif.")
            return None

        if out_format == ".mp4" and importlib.util.find_spec("imageio_ffmpeg") is None:
            self.append_log(
                "❌ Make Video: MP4 export requires imageio-ffmpeg. "
                "Install it or switch format to .gif."
            )
            return None

        parsed, parse_error = self._parse_video_instructions(instructions)
        if parse_error:
            self.append_log(f"❌ Make Video: {parse_error}")
            return None

        actions = parsed["actions"]
        total_frames = len(actions)
        if total_frames <= 0:
            self.append_log("❌ Make Video: no frames generated from instructions.")
            return None

        if parsed["needs_3d"]:
            ndim = int(getattr(self.viewer.dims, "ndim", 2))
            if ndim < 3:
                self.append_log("❌ Make Video: 3D rotate rows require ndim >= 3.")
                return None

        output_path = self._next_available_video_path(folder, filename, out_format)
        meta_path = os.path.splitext(output_path)[0] + "_meta.json"

        import imageio.v2 as iio
        sk_rotate = None
        if parsed["needs_2d_rotate"]:
            try:
                from skimage.transform import rotate as sk_rotate
            except Exception as exc:
                self.append_log(f"❌ Make Video: 2D rotation unavailable ({exc}).")
                return None

        self.append_log(
            f"🎬 Make Video: rendering {total_frames} frames at {fps} FPS -> {output_path}"
        )
        for segment in parsed["segments"]:
            self.append_log(f"   - {segment}")
        self.append_log("   - visibility: using current viewer state")

        state = self._snapshot_viewer_state()
        run_enabled = self.btn_interaction_run.isEnabled()
        cancel_enabled = self.btn_interaction_cancel.isEnabled()
        self.btn_interaction_run.setEnabled(False)
        self.btn_interaction_cancel.setEnabled(False)
        try:
            writer_kwargs = {"fps": fps}
            if out_format == ".mp4":
                writer_kwargs.update(
                    {
                        "codec": "libx264",
                        "ffmpeg_params": ["-pix_fmt", "yuv420p"],
                    }
                )
            else:
                writer_kwargs = {"mode": "I", "duration": 1.0 / float(fps), "loop": 0}

            first_shape = None
            progress_every = max(1, total_frames // 10)
            rotate3d_row_base = {}
            current_ndisplay = int(getattr(self.viewer.dims, "ndisplay", 2))
            with iio.get_writer(output_path, **writer_kwargs) as writer:
                for i, action in enumerate(actions):
                    view_mode = str(action.get("view_mode", "keep")).strip().lower()
                    target_ndisplay = None
                    if view_mode == "2d":
                        target_ndisplay = 2
                    elif view_mode == "3d":
                        target_ndisplay = 3
                    if target_ndisplay is not None and target_ndisplay != current_ndisplay:
                        try:
                            self.viewer.dims.ndisplay = target_ndisplay
                            current_ndisplay = target_ndisplay
                            QApplication.processEvents()
                        except Exception:
                            pass
                    op = action["op"]
                    if op == "sweep":
                        target = max(0, min(int(action["axis_max"]), int(action["value"])))
                        self.viewer.dims.set_current_step(int(action["axis_idx"]), target)
                    elif op == "rotate3d":
                        try:
                            row_id = int(action.get("row", -1))
                            if row_id not in rotate3d_row_base:
                                row_angles = list(getattr(self.viewer.camera, "angles", (0.0, 0.0, 0.0)))
                                while len(row_angles) < 3:
                                    row_angles.append(0.0)
                                rotate3d_row_base[row_id] = [float(v) for v in row_angles]

                            axis_idx = {
                                "x": 0,
                                "y": 1,
                                "z": 2,
                            }.get(str(action.get("rot_axis_camera", "z")), 2)
                            theta_deg = float(action["value"])
                            base = list(rotate3d_row_base[row_id])
                            new_angles = list(base)
                            new_angles[axis_idx] = float(base[axis_idx]) + theta_deg
                            self.viewer.camera.angles = tuple(float(v) for v in new_angles[:3])
                        except Exception:
                            pass

                    QApplication.processEvents()
                    frame = self.viewer.screenshot(canvas_only=True, flash=False)
                    frame_rgb = self._normalize_video_frame(frame)
                    if op == "rotate2d":
                        frame_rgb = sk_rotate(
                            frame_rgb,
                            angle=float(action["value"]),
                            resize=False,
                            mode="edge",
                            preserve_range=True,
                        ).astype(np.uint8)

                    if first_shape is None:
                        first_shape = frame_rgb.shape
                    elif frame_rgb.shape != first_shape:
                        raise ValueError(
                            f"Frame size changed during capture: {frame_rgb.shape} != {first_shape}"
                        )

                    writer.append_data(frame_rgb)
                    if (i + 1) % progress_every == 0 or i == total_frames - 1:
                        self.append_log(f"   - frame {i + 1}/{total_frames}")

            meta_payload = {
                "created_at": datetime.datetime.now().isoformat(),
                "output_path": output_path,
                "format": out_format,
                "frames": total_frames,
                "fps": fps,
                "instructions": instructions,
                "segments": parsed["segments"],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=2)

            self.append_log(f"✅ Make Video: saved {output_path}")
            return {
                "output_path": output_path,
                "meta_path": meta_path,
                "frames": total_frames,
                "fps": fps,
                "format": out_format,
            }
        except Exception as exc:
            self.append_log(f"❌ Make Video failed: {exc}")
            return None
        finally:
            self._restore_viewer_state(state)
            self.btn_interaction_run.setEnabled(run_enabled)
            self.btn_interaction_cancel.setEnabled(cancel_enabled)

    def update_param(self, node, param_name, value):
        """Updates a parameter and invalidates the node."""
        old_value = node.parameters.get(param_name)
        try:
            if old_value == value:
                return
        except Exception:
            pass
        node.parameters[param_name] = value
        # When a parameter changes, this node and all downstream nodes become "stale"
        self.set_node_status_recursive(node, "gray")
        self._refresh_dynamic_output_types(node, changed_param=param_name)
        self.scene.update()
        if self._pending_props_refresh and not self._is_editing_properties_widget():
            self._pending_props_refresh = False
            self.on_selection(force=True)

    def _schedule_param_update(self, node, param_name, value, delay_ms=120):
        """
        Debounce text updates so typing stays responsive while still keeping
        node parameters in sync.
        """
        self._pending_param_updates[(node.uid, param_name)] = (node, param_name, value)
        self._param_update_timer.start(max(30, int(delay_ms)))

    def _update_param_live(self, node, param_name, value):
        """
        Lightweight live update for text fields while typing.
        Keeps editor state in sync without triggering expensive graph
        invalidation/repaint on every key.
        """
        try:
            node.parameters[param_name] = value
        except Exception:
            pass

    def _flush_pending_param_updates(self):
        if not self._pending_param_updates:
            return
        pending = list(self._pending_param_updates.values())
        self._pending_param_updates.clear()
        for node, param_name, value in pending:
            if isinstance(node, Node):
                self.update_param(node, param_name, value)

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

        # Also propagate across exec thread
        for exec_socket in node.logic_outputs:
            for edge in exec_socket.connected_edges:
                if edge.end_socket:
                    child_node = edge.end_socket.node
                    self.set_node_status_recursive(child_node, status)

    # --- Save Pipeline Method ---
    def save_pipeline(self):
        # 1. Build the Pipeline Dictionary (Same as before)
        pipeline = {
            "schema_version": "0.1",
            "nodes": [],
            "macro_groups": [],
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

                # Map exec connections (exec_in ← exec_out of other nodes)
                exec_connections = []
                for socket in item.logic_inputs:
                    for edge in socket.connected_edges:
                        if edge.start_socket:
                            src_node = edge.start_socket.node
                            exec_connections.append(
                                f"{src_node.uid}.{edge.start_socket.name}"
                            )

                node_data = {
                    "id": item.uid,
                    "type": item.node_type,
                    "label": item.title,
                    "position": {"x": item.pos().x(), "y": item.pos().y()},
                    "parameters": item.parameters,
                    "input_connections": input_connections,
                    "exec_connections": exec_connections,
                }
                pipeline["nodes"].append(node_data)

        for group_id, group in self.macro_groups.items():
            item = group.get("item")
            if item is None:
                continue
            pipeline["macro_groups"].append(
                {
                    "id": group_id,
                    "label": group.get("title", "Macro"),
                    "position": {"x": item.pos().x(), "y": item.pos().y()},
                    "members": sorted(list(group.get("member_uids", set()))),
                    "collapsed": bool(group.get("collapsed", True)),
                }
            )

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

        self._clear_macro_groups()
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
                    
                    if (
                        src_socket
                        and tgt_socket
                        and self.scene.is_data_connection_compatible(src_socket, tgt_socket)
                    ):
                        conn = Connection(src_socket, self.scene)
                        conn.finalize(tgt_socket)
                        self.scene.addItem(conn)

            # Reconnect Exec edges
            exec_conn_list = n_data.get("exec_connections", n_data.get("logic_connections", []))
            for src_string in exec_conn_list:
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

        # Pass 3: Restore macro groups (if present)
        for g_data in pipeline.get("macro_groups", []):
            group_id = g_data.get("id", str(uuid.uuid4()))
            title = g_data.get("label", self._next_macro_title())
            pos = g_data.get("position", {"x": 0, "y": 0})
            members = {
                uid for uid in g_data.get("members", [])
                if uid in node_map
            }
            if not members:
                continue
            item = MacroGroupItem(
                pos.get("x", 0),
                pos.get("y", 0),
                group_id=group_id,
                title=title,
                node_count=len(members),
                on_open=self.enter_macro,
                scene=self.scene,
            )
            self.scene.addItem(item)
            self.macro_groups[group_id] = {
                "id": group_id,
                "title": title,
                "member_uids": set(members),
                "collapsed": bool(g_data.get("collapsed", True)),
                "item": item,
            }

        self._active_macro_id = None
        self._set_back_macro_button_visible(False)
        self._refresh_macro_visibility()


    # --- Add Node Method ---
    # 1. The UI Logic (Dropdown)
    def open_add_menu(self):
        """Shows a categorized dropdown menu to add data-processing nodes."""
        menu = QMenu(self)
        self._populate_add_menu(menu, include_control=False, control_only=False)
        menu.exec_(QCursor.pos())

    def open_control_menu(self):
        """Shows a dropdown menu with only control-flow nodes."""
        menu = QMenu(self)
        self._populate_add_menu(menu, include_control=True, control_only=True)
        menu.exec_(QCursor.pos())

    def _populate_add_menu(self, menu, include_control=True, control_only=False):
        menu.clear()
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
        self._refresh_dynamic_output_types(node)
        if self._active_macro_id and self._active_macro_id in self.macro_groups:
            self.macro_groups[self._active_macro_id]["member_uids"].add(node.uid)
            self._refresh_macro_visibility()
        return node
    
    # --- Remove Node Method ---
    def open_remove_menu(self):
        """Shows a QMenu list of existing nodes to remove."""
        menu = QMenu(self)
        self._populate_remove_menu(menu)
        menu.exec_(QCursor.pos())

    def _populate_remove_menu(self, menu):
        menu.clear()
        # 1. Find all nodes currently in the scene
        nodes = [item for item in self.scene.items() if isinstance(item, Node)]
        
        if not nodes:
            action = menu.addAction("No nodes to remove")
            action.setEnabled(False)
            return

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

        # Remove node from any macro group membership.
        empty_groups = []
        for group_id, group in self.macro_groups.items():
            members = group.get("member_uids", set())
            if node.uid in members:
                members.discard(node.uid)
            if not members:
                empty_groups.append(group_id)
        for group_id in empty_groups:
            self._expand_macro_group(group_id)
        self._refresh_macro_visibility()

    def delete_selected_nodes(self):
        """Delete currently selected nodes (used by Delete/Backspace shortcut)."""
        selected = list(self.scene.selectedItems())
        selected_nodes = [item for item in selected if isinstance(item, Node)]
        selected_macros = [item for item in selected if isinstance(item, MacroGroupItem)]
        selected_edges = [item for item in selected if isinstance(item, Connection)]
        selected_proxy_edges = [
            item for item in selected if isinstance(item, MacroProxyConnection)
        ]

        if not selected_nodes and not selected_macros and not selected_edges and not selected_proxy_edges:
            return

        for edge in list(selected_edges):
            downstream = getattr(getattr(edge, "end_socket", None), "node", None)
            self._remove_edge(edge)
            if isinstance(downstream, Node):
                self.set_node_status_recursive(downstream, "gray")

        for proxy in list(selected_proxy_edges):
            real_edge = getattr(proxy, "real_edge", None)
            if isinstance(real_edge, Connection):
                downstream = getattr(getattr(real_edge, "end_socket", None), "node", None)
                self._remove_edge(real_edge)
                if isinstance(downstream, Node):
                    self.set_node_status_recursive(downstream, "gray")
            if hasattr(proxy, "detach"):
                proxy.detach()
            if proxy.scene() is self.scene:
                self.scene.removeItem(proxy)

        for node in list(selected_nodes):
            self.delete_node(node)
        for macro_item in list(selected_macros):
            self._expand_macro_group(macro_item.group_id)
        self._refresh_macro_visibility()
        self.on_selection()

    def _get_all_nodes(self):
        return [item for item in self.scene.items() if isinstance(item, Node)]

    def _get_exec_successors(self, node):
        successors = []
        for socket in getattr(node, "logic_outputs", []):
            for edge in socket.connected_edges:
                if edge.end_socket:
                    successors.append(edge.end_socket.node)
        return successors

    def _collect_exec_reachable(self, start_node):
        visited = set()
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node.uid in visited:
                continue
            visited.add(node.uid)
            stack.extend(self._get_exec_successors(node))
        return visited

    def _has_exec_cycle(self, nodes):
        state = {}  # 0=unseen, 1=active, 2=done

        def dfs(node):
            uid = node.uid
            mark = state.get(uid, 0)
            if mark == 1:
                return True
            if mark == 2:
                return False
            state[uid] = 1
            for nxt in self._get_exec_successors(node):
                if dfs(nxt):
                    return True
            state[uid] = 2
            return False

        for node in nodes:
            if state.get(node.uid, 0) == 0 and dfs(node):
                return True
        return False

    def validate_pipeline_graph(self):
        """
        Validate graph structure before execution.
        Returns (errors, warnings) where errors block execution.
        """
        errors = []
        warnings = []
        nodes = self._get_all_nodes()

        if not nodes:
            warnings.append("Pipeline is empty.")
            return errors, warnings

        begin_nodes = [n for n in nodes if getattr(n, "node_type", "") == "begin"]
        if not begin_nodes:
            errors.append("No Begin node found.")
            return errors, warnings
        if len(begin_nodes) > 1:
            errors.append("Multiple Begin nodes found. Keep only one Begin node.")
            return errors, warnings

        begin = begin_nodes[0]
        reachable = self._collect_exec_reachable(begin)
        reachable_nodes = [n for n in nodes if n.uid in reachable]

        if len(reachable) == 1:
            errors.append(
                "Begin node is not connected to any exec thread. "
                "Connect Begin exec_out to the first processing node."
            )
            return errors, warnings

        exec_capable = [n for n in nodes if n.logic_inputs or n.logic_outputs]
        if self._has_exec_cycle(exec_capable):
            errors.append("Exec cycle detected. Only loop nodes should express repetition.")

        disconnected = [n for n in exec_capable if n.uid not in reachable]
        if disconnected:
            preview = ", ".join(n.title for n in disconnected[:4])
            suffix = "..." if len(disconnected) > 4 else ""
            warnings.append(
                f"{len(disconnected)} exec node(s) are disconnected from Begin: {preview}{suffix}"
            )

        for node in reachable_nodes:
            node_type = getattr(node, "node_type", "")
            if node_type in ("begin", "loop_control"):
                continue
            for socket in getattr(node, "inputs", []):
                if not socket.connected_edges:
                    errors.append(
                        f"Node '{node.title}' is missing required input '{socket.name}'."
                    )

        for loop in [n for n in reachable_nodes if getattr(n, "node_type", "") == "loop_control"]:
            has_body = any(
                s.name == "loop_body" and len(s.connected_edges) > 0
                for s in getattr(loop, "logic_outputs", [])
            )
            has_completed = any(
                s.name == "completed" and len(s.connected_edges) > 0
                for s in getattr(loop, "logic_outputs", [])
            )
            if not has_body:
                warnings.append(f"Loop '{loop.title}' has no Loop Body connection.")
            if not has_completed:
                warnings.append(f"Loop '{loop.title}' has no Completed connection.")

        return errors, warnings

    # --- Run Pipeline Method ---
    def run_pipeline(self):
        # Ensure debounced parameter edits (especially table edits) are committed
        # before validation/execution starts.
        self._flush_pending_param_updates()
        # 1. Validate graph before creating worker/thread.
        self.console.clear()
        errors, warnings = self.validate_pipeline_graph()

        for warning in warnings:
            self.append_log(f"⚠️ Validation: {warning}")

        if errors:
            for error in errors:
                self.append_log(f"❌ Validation: {error}")
            QMessageBox.critical(
                self,
                "Pipeline Validation Failed",
                "\n".join(errors),
            )
            return

        # 2. Disable UI
        self.set_ui_enabled(False)
        # 3. Setup Thread
        self.thread = QThread()
        self.worker = ExecutionWorker(self.scene, self.viewer)
        self.worker.moveToThread(self.thread)
        
        # 4. Connect Signals
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
        
        # 5. Start
        self.thread.start()

    def on_stop_loop_clicked(self):
        """Signal the worker to stop the active 'Until confirm' loop."""
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.request_loop_stop()
            self.btn_stop_loop.setEnabled(False)
            self.btn_stop_loop.setText("Stopping...")
        # Also dismiss any pending inline interaction UI/layer immediately.
        self._finish_pending_interaction(data=None, notify_worker=False)

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
    # INTERACTIVE NODE UI (inline in parameter panel)
    # -----------------------------------------------------------------
    def _find_node_by_uid(self, node_uid):
        for item in self.scene.items():
            if isinstance(item, Node) and item.uid == node_uid:
                return item
        return None

    def _select_node_for_properties(self, node_uid):
        target = self._find_node_by_uid(node_uid)
        if target is None:
            return
        for item in self.scene.selectedItems():
            item.setSelected(False)
        target.setSelected(True)
        self.on_selection()

    def _cleanup_pending_interaction_layer(self):
        pending = self._pending_interaction
        if not pending:
            return
        layer_name = pending.get("layer_name")
        if not layer_name:
            return
        try:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)
        except Exception:
            pass

    def _set_pending_layer_choice(self, node_uid, choice):
        pending = self._pending_interaction
        if not pending:
            return
        if pending.get("node_uid") != node_uid:
            return
        if pending.get("interaction_type") != "layer_choice":
            return
        if not isinstance(choice, dict):
            return

        pending["selected_choice"] = dict(choice)

        node = self._find_node_by_uid(node_uid)
        if node is not None:
            node.parameters["layer_name"] = str(choice.get("name", ""))
            node.parameters["layer_type"] = str(
                choice.get("layer_type", node.parameters.get("layer_type", "image"))
            )

    def _update_interaction_bar(self):
        pending = self._pending_interaction
        if not pending:
            self.interaction_bar.setVisible(False)
            return

        node_uid = pending.get("node_uid")
        prompt = pending.get("config", {}).get(
            "prompt", "Draw on the layer, then click Run."
        )
        node = self._find_node_by_uid(node_uid)
        node_title = getattr(node, "title", "Interactive Node")
        self.interaction_status_label.setText(f"Waiting for input: {node_title}")
        self.interaction_prompt_label.setText(prompt)
        self.interaction_bar.setVisible(True)

    def _finish_pending_interaction(self, data=None, notify_worker=True):
        if not self._pending_interaction:
            return
        self._cleanup_pending_interaction_layer()
        self._pending_interaction = None
        self._update_interaction_bar()
        self.on_selection()
        if notify_worker and hasattr(self, "worker") and self.worker is not None:
            self.worker.provide_interaction_result(data)

    def _find_reference_layer_for_interaction(self, node_uid):
        """
        Best-effort lookup of the visual layer that interactive drawing should
        align to, so shape coordinates map correctly back to data indices.
        """
        node = self._find_node_by_uid(node_uid)
        if node is None:
            return getattr(self.viewer.layers.selection, "active", None)

        # 1) Prefer explicit get_layer upstream connection.
        for input_socket in getattr(node, "inputs", []):
            if not input_socket.connected_edges:
                continue
            edge = input_socket.connected_edges[0]
            source_node = edge.start_socket.node
            if getattr(source_node, "node_type", "") == "get_layer":
                layer_name = (getattr(source_node, "parameters", {}) or {}).get("layer_name")
                if layer_name and layer_name in self.viewer.layers:
                    return self.viewer.layers[layer_name]

        # 2) Fallback: infer from cached upstream metadata name.
        for input_socket in getattr(node, "inputs", []):
            if not input_socket.connected_edges:
                continue
            edge = input_socket.connected_edges[0]
            source_node = edge.start_socket.node
            source_socket_name = edge.start_socket.name
            cached = getattr(source_node, "cached_results", {}).get(source_socket_name)
            if (
                isinstance(cached, tuple)
                and len(cached) >= 2
                and isinstance(cached[1], dict)
            ):
                layer_name = cached[1].get("name")
                if layer_name and layer_name in self.viewer.layers:
                    return self.viewer.layers[layer_name]

        # 3) Last resort: current active layer.
        return getattr(self.viewer.layers.selection, "active", None)

    def _on_inline_interaction_run(self):
        pending = self._pending_interaction
        if not pending:
            return
        if pending.get("interaction_type") == "video_render":
            payload = self._render_video_interaction(pending)
            if payload is None:
                return
            self._finish_pending_interaction(data=payload, notify_worker=True)
            return
        if pending.get("interaction_type") == "layer_choice":
            selected = pending.get("selected_choice")
            if not selected:
                choices = pending.get("choices", [])
                selected = choices[0] if choices else None
            if not selected:
                self.append_log("⚠️ Select Layer: no options available.")
                return
            payload = {
                "layer_name": str(selected.get("name", "")),
                "layer_type": str(selected.get("layer_type", "image")),
                "index": int(selected.get("index", 0)),
            }
            if not payload["layer_name"]:
                self.append_log("⚠️ Select Layer: please choose a layer before Run.")
                return
            self._finish_pending_interaction(data=payload, notify_worker=True)
            return

        layer_name = pending.get("layer_name")
        data = None
        try:
            if layer_name in self.viewer.layers:
                roi_layer = self.viewer.layers[layer_name]
                if len(roi_layer.data) > 0:
                    data = [np.array(s) for s in roi_layer.data]
        except Exception:
            data = None

        if not data:
            self.append_log("⚠️ Interactive node: draw at least one shape before clicking Run.")
            self.interaction_prompt_label.setText(
                "Draw at least one shape, then click Run."
            )
            return

        self._finish_pending_interaction(data=data, notify_worker=True)

    def _on_inline_interaction_cancel(self):
        self._finish_pending_interaction(data=None, notify_worker=True)

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
          1. Create a temporary napari layer with requested settings.
          2. Store pending interaction state and show controls in Node Properties.
          3. Run/Cancel in the panel calls ``worker.provide_interaction_result``.
        """
        interaction_type = str(config.get("interaction_type", "shapes")).strip().lower()
        if interaction_type == "video_render":
            self._pending_interaction = {
                "node_uid": node_uid,
                "config": config,
                "interaction_type": "video_render",
            }
            self._update_interaction_bar()
            self._select_node_for_properties(node_uid)
            return

        if interaction_type == "layer_choice":
            choices = list(config.get("choices", []) or [])
            if not choices:
                self.worker.provide_interaction_result(None)
                return
            default_index = int(config.get("default_index", 0))
            if default_index < 0 or default_index >= len(choices):
                default_index = 0
            selected_choice = dict(choices[default_index])
            self._pending_interaction = {
                "node_uid": node_uid,
                "config": config,
                "interaction_type": "layer_choice",
                "choices": choices,
                "selected_choice": selected_choice,
            }
            self._set_pending_layer_choice(node_uid, selected_choice)
            self._update_interaction_bar()
            self._select_node_for_properties(node_uid)
            return

        layer_type = config.get("layer_type", "shapes")
        layer_name = f"__interactive_{node_uid[:8]}__"

        # --- 1. Create temporary layer ---
        try:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

            reference_layer = self._find_reference_layer_for_interaction(node_uid)

            if layer_type == "shapes":
                layer_kwargs = {"name": layer_name}
                for key in ("edge_color", "face_color", "edge_width"):
                    if key in config:
                        layer_kwargs[key] = config[key]

                # Align transforms with the source data layer to avoid
                # world/data coordinate mismatch in ROI extraction.
                if reference_layer is not None:
                    try:
                        layer_kwargs["ndim"] = int(getattr(reference_layer, "ndim"))
                    except Exception:
                        pass
                    for key in ("scale", "translate", "rotate", "shear", "affine"):
                        try:
                            value = getattr(reference_layer, key)
                            if value is not None:
                                layer_kwargs[key] = value
                        except Exception:
                            pass

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

        # --- 2. Store pending interaction and show inline controls ---
        self._pending_interaction = {
            "node_uid": node_uid,
            "config": config,
            "layer_name": layer_name,
        }
        self._update_interaction_bar()
        self._select_node_for_properties(node_uid)

    def _refresh_selected_make_video_properties(self):
        sel = self.scene.selectedItems()
        if len(sel) != 1:
            return
        node = sel[0]
        if not isinstance(node, Node):
            return
        if getattr(node, "node_type", "") != "make_video":
            return
        # Never force while user may be editing fields.
        self.on_selection()

    def handle_execution_result(self, node_title, output_name, data):
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
        plotly_figure = display_meta.get("plotly_figure") if isinstance(display_meta, dict) else None
        if is_plotly_figure(display_data):
            is_plot = True
        elif "matplotlib" in type_str and "Figure" in type_str:
            is_plot = True
        elif hasattr(display_data, "canvas"): 
            is_plot = True

        if is_plot:
            self.plot_dashboard.display(display_data, plotly_figure=plotly_figure) # <--- Pop up the widget!
            return

        # --- C. NON-VISUAL FILTER ---
        import pandas as pd
        if isinstance(display_data, (pd.DataFrame, dict, str)):
            return

        # --- D. DISPLAY HELPER ---
        def add_layer_to_viewer(layer_data, raw_meta):
            # 1. Prepare Name
            layer_name = raw_meta.get("name", f"{node_title} Output")

            # 2. Filter Metadata
            valid_napari_args = {
                "name", "opacity", "blending", "visible", "multiscale",
                "colormap", "contrast_limits", "gamma", "rgb",
                "interpolation2d", "interpolation3d",
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

            def apply_visual_kwargs(layer_obj, kwargs_obj):
                # Keep existing layer object but refresh display params so output
                # appearance stays consistent with incoming metadata.
                for key, value in kwargs_obj.items():
                    if key in {"name", "metadata", "multiscale"}:
                        continue
                    if not hasattr(layer_obj, key):
                        continue
                    try:
                        setattr(layer_obj, key, value)
                    except Exception:
                        pass

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
                                apply_visual_kwargs(layer, napari_kwargs)
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
                            apply_visual_kwargs(layer, napari_kwargs)
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
        self.btn_collapse_macro.setEnabled(enabled)
        if self.btn_expand_macro is not None:
            self.btn_expand_macro.setEnabled(enabled)
        if self.btn_back_macro is not None:
            self.btn_back_macro.setEnabled(enabled)
        if hasattr(self, "btn_back_macro_floating"):
            self.btn_back_macro_floating.setEnabled(enabled)
        if enabled:
            self.btn_stop_loop.setVisible(False)
            self.btn_stop_loop.setEnabled(False)
            self.btn_stop_loop.setText("Stop Loop")
            # Safety: if execution ended unexpectedly while waiting for interaction,
            # clear the inline interaction state/layer.
            self._finish_pending_interaction(data=None, notify_worker=False)
        
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
                    description_cfg = meta.get("description", "")
                    if description_cfg:
                        entry["description"] = description_cfg
                    # Persist interactive config if present
                    interactive_cfg = meta.get("interactive")
                    if interactive_cfg:
                        entry["interactive"] = interactive_cfg
                    logic_cfg = meta.get("logic")
                    if logic_cfg:
                        entry["logic"] = logic_cfg
                    dynamic_output_types_cfg = meta.get("dynamic_output_types")
                    if dynamic_output_types_cfg:
                        entry["dynamic_output_types"] = dynamic_output_types_cfg
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
