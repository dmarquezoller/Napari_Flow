from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QHeaderView, QAbstractItemView, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit
)
from qtpy.QtCore import Qt, Signal


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
        self._col_index_by_name = {
            str(c.get("name", "")): i for i, c in enumerate(self.column_config)
        }
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
        self._configure_columns()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setMinimumHeight(170)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton(" + Add Rule ")
        btn_layout.addWidget(self.btn_add)
        
        self.layout.addWidget(self.table)
        self.layout.addLayout(btn_layout)
        self.setLayout(self.layout)

        self.btn_add.clicked.connect(self.add_row)
        
        self._update_ui_state()

    def _configure_columns(self):
        header = self.table.horizontalHeader()
        col_count = len(self.column_config)
        dense = col_count >= 6
        if dense:
            header.setSectionResizeMode(QHeaderView.Interactive)
        else:
            header.setSectionResizeMode(QHeaderView.Stretch)

        for i, col in enumerate(self.column_config):
            if dense:
                width = col.get("width")
                if width is None:
                    label = str(col.get("label", col.get("name", "")))
                    width = max(80, min(160, 10 * len(label) + 16))
                self.table.setColumnWidth(i, int(width))
        header.setSectionResizeMode(col_count, QHeaderView.ResizeToContents)

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
        self._apply_conditional_columns()
        
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
                self._apply_conditional_columns()
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
            w.setRange(col_def.get('min', -1e9), col_def.get('max', 1e9))
            w.setSingleStep(col_def.get('step', 0.1))
            w.setKeyboardTracking(False)
            if value is not None: w.setValue(float(value))
            w.valueChanged.connect(lambda _: self._on_value_changed())
            w.editingFinished.connect(self._on_value_changed)
            
        else:
            w = QLineEdit()
            if value: w.setText(str(value))
            w.textEdited.connect(lambda _: self._on_value_changed())
            
        self.table.setCellWidget(r, c, w)

    def _on_value_changed(self):
        """Helper to run constraints then emit signal"""
        self._update_dropdown_constraints()
        self._apply_conditional_columns()
        self.emit_change()

    def _update_dropdown_constraints(self):
        rows = self.table.rowCount()
        cols = self.table.columnCount() - 1 

        # 1. Snapshot Current State
        grid_state = {}
        for r in range(rows):
            for c in range(cols):
                w = self.table.cellWidget(r, c)
                if isinstance(w, QComboBox) and self._is_cell_active(r, c):
                    grid_state[(r, c)] = w.currentText()

        # 2. Update Every ComboBox
        for r in range(rows):
            for c in range(cols):
                w = self.table.cellWidget(r, c)
                if not isinstance(w, QComboBox): continue
                if not self._is_cell_active(r, c):
                    continue

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
        self._apply_conditional_columns()

    def _row_context(self, row_idx):
        ctx = {}
        for c, col_def in enumerate(self.column_config):
            name = str(col_def.get("name", ""))
            if not name:
                continue
            w = self.table.cellWidget(row_idx, c)
            if isinstance(w, QComboBox):
                ctx[name] = w.currentText()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                ctx[name] = w.value()
            elif isinstance(w, QLineEdit):
                ctx[name] = w.text()
            else:
                ctx[name] = None
        return ctx

    def _is_cell_active(self, row_idx, col_idx, row_ctx=None):
        col_def = self.column_config[col_idx]
        show_if = col_def.get("show_if")
        if not isinstance(show_if, dict) or not show_if:
            return True
        if row_ctx is None:
            row_ctx = self._row_context(row_idx)

        for dep_name, allowed in show_if.items():
            if not isinstance(allowed, (list, tuple, set)):
                allowed_values = [allowed]
            else:
                allowed_values = list(allowed)
            if row_ctx.get(dep_name) not in allowed_values:
                return False
        return True

    def _apply_conditional_columns(self):
        rows = self.table.rowCount()
        cols = self.table.columnCount() - 1
        for r in range(rows):
            row_ctx = self._row_context(r)
            for c in range(cols):
                w = self.table.cellWidget(r, c)
                if w is None:
                    continue
                active = self._is_cell_active(r, c, row_ctx=row_ctx)
                w.setVisible(active)
                w.setEnabled(active)

    def emit_change(self):
        self.valueChanged.emit(self.get_value())

    def get_value(self):
        data = []
        for r in range(self.table.rowCount()):
            row_data = {}
            for c, col_def in enumerate(self.column_config):
                if not self._is_cell_active(r, c):
                    row_data[col_def['name']] = None
                    continue
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
        self._apply_conditional_columns()
        self.blockSignals(False)
        self._update_ui_state()
