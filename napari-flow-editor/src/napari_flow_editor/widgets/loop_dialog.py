from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFormLayout, QSpinBox, QAbstractItemView
)
from qtpy.QtCore import Qt


class LoopConfigDialog(QDialog):
    def __init__(self, nodes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Loop Execution")
        self.resize(400, 500)
        self.selected_nodes = []
        self.iterations = 1
        
        layout = QVBoxLayout()
        
        # Instructions
        layout.addWidget(QLabel("Select nodes to include in the loop:"))
        
        # Node List (Multi-select)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        
        # Populate List (Preserve Topological Order visually if possible, 
        # but for now, just listing them is fine)
        self.node_map = {}
        for node in nodes:
            item = QListWidgetItem(f"{node.title} (ID: {node.uid[:4]})")
            item.setData(Qt.UserRole, node.uid)
            self.list_widget.addItem(item)
            self.node_map[node.uid] = node
            
        layout.addWidget(self.list_widget)
        
        # Iteration Counter
        form = QFormLayout()
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 1000)
        self.spin_iter.setValue(5)
        form.addRow("Loop Iterations:", self.spin_iter)
        layout.addLayout(form)
        
        # Buttons
        btns = QHBoxLayout()
        btn_run = QPushButton("Run Loop")
        btn_run.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_run)
        layout.addLayout(btns)
        
        self.setLayout(layout)

    def get_config(self):
        # Return list of Node UIDs and Iteration count
        selected_items = self.list_widget.selectedItems()
        uids = [item.data(Qt.UserRole) for item in selected_items]
        return uids, self.spin_iter.value()
