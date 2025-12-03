from qtpy.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsRectItem, QMainWindow,
    QGraphicsDropShadowEffect, QToolBar, QInputDialog, QFileDialog, QVBoxLayout,
    QHBoxLayout, QPushButton, QMenu, QWidget, QGroupBox, QFormLayout, QLabel,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QScrollArea, QBoxLayout,
    QFrame, QMessageBox
)

from qtpy.QtGui import (
    QBrush, QPen, QColor, QPainterPath, QPainterPathStroker, QLinearGradient, QPainter, QAction, QCursor
)
from qtpy.QtCore import Qt, QPointF, QRectF
import os, sys, json, datetime, napari, uuid, importlib.util, inspect

from napari_flow_editor import generate_library
from .execution_engine import ExecutionEngine

NODE_LIBRARY = {}

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
        # If a UUID is provided (loading), use it. Otherwise generate new one.
        self.uid = uuid_str if uuid_str else str(uuid.uuid4())
        self.parameters = {}
        
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
        # ... (Use the paint method from the previous step) ...
        # Just ensure you use the corrected paint method I gave you previously
        # that uses QRectF for drawRect
        rect = self.rect()
        painter.fillRect(rect.adjusted(4, 4, 4, 4), QColor(0, 0, 0, 60))
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor("#3F4242"))
        gradient.setColorAt(1, QColor("#2F3232")) 
        painter.setBrush(QBrush(gradient))
        
        border_color = QColor("#ff9900") if self.isSelected() else QColor("#727272")
        painter.setPen(QPen(border_color, 2))
        painter.drawRoundedRect(rect, 12, 12)

        title_rect = QRectF(rect.x(), rect.y(), rect.width(), 25)
        title_grad = QLinearGradient(title_rect.topLeft(), title_rect.bottomRight())
        title_grad.setColorAt(0, QColor("#666"))
        title_grad.setColorAt(1, QColor("#444"))
        painter.setBrush(QBrush(title_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(title_rect, 12, 12)
        
        fix_rect = QRectF(rect.x(), rect.y() + 15, rect.width(), 10)
        painter.drawRect(fix_rect)

        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, self.title)

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

        self.btn_load = QPushButton("Load Pipeline")
        self.btn_load.clicked.connect(self.load_pipeline)
        toolbar.addWidget(self.btn_load)

        self.btn_import = QPushButton("Import Nodes (.py)")
        self.btn_import.clicked.connect(self.import_custom_module)
        toolbar.addWidget(self.btn_import)        

        self.layout.addLayout(toolbar)

        # 3. Node Properties Panel (Middle)
        self.props_group = QGroupBox("Node Properties")
        self.props_group.setFixedHeight(250) # Limit height so it doesn't squash the view
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
        
        # Put scroll area into group box
        group_layout.addWidget(scroll)
        
        self.layout.addWidget(self.props_group)

        # 4. Fit Scene Button (Below Properties)
        self.btn_fit = QPushButton("Fit to Scene")
        self.layout.addWidget(self.btn_fit)

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
        self.layout.addWidget(self.btn_run)

        # 5. Graphics View (Bottom)
        self.scene = FlowScene()
        self.view = FlowView(self.scene)
        self.layout.addWidget(self.view)

        # --- CONNECTIONS ---
        # Connect the Fit Button
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
                current_val = node.parameters.get(param_name, conf["default"])
                
                # Widget creation logic (Float, Int, Bool, Enum)
                if conf["type"] == "float":
                    widget = QDoubleSpinBox()
                    widget.setRange(conf.get("min", -9999), conf.get("max", 9999))
                    widget.setSingleStep(conf.get("step", 0.1))
                    widget.setValue(float(current_val))
                    widget.valueChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                
                elif conf["type"] == "int":
                    widget = QSpinBox()
                    widget.setRange(conf.get("min", -9999), conf.get("max", 9999))
                    widget.setValue(int(current_val))
                    widget.valueChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                
                elif conf["type"] == "bool":
                    widget = QCheckBox()
                    widget.setChecked(bool(current_val))
                    widget.toggled.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                
                elif conf["type"] == "enum":
                    widget = QComboBox()
                    widget.addItems(conf.get("options", []))
                    widget.setCurrentText(str(current_val))
                    widget.currentTextChanged.connect(lambda val, n=node, k=param_name: self.update_param(n, k, val))
                
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
                self.props_layout.addRow(f"  \u25B8 {s.name}", status_lbl) # arrow symbol
        
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

    def update_param(self, node, key, value):
        node.parameters[key] = value
        # Force update of nodes
        node.update()

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
        engine = ExecutionEngine(self.scene, self.viewer)
        engine.run()

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




