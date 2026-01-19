import os
from .decorator import register_node

@register_node(
    label="Save Table (CSV)",
    category="Outputs",
    outputs=[],
    params_config={
        "filename": {"type": "text", "value": "results.csv", "label": "Filename"},
        "folder": {"type": "path", "mode": "directory", "label": "Save Folder"}
    }
)
def save_table(table, folder: str = "", filename: str = "results.csv"):
    if table is None:
        print("Save Table: No data received.")
        return

    if not folder or not os.path.isdir(folder):
        raise ValueError("Please select a valid folder.")

    full_path = os.path.join(folder, filename)
    if not full_path.endswith(".csv"):
        full_path += ".csv"
        
    table.to_csv(full_path, index=False)
    print(f"Saved table to: {full_path}")

