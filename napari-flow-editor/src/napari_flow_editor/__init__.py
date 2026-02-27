"""
Package init should stay lightweight.

Tests and headless tooling may import ``napari_flow_editor`` without a full
Napari/Qt runtime; avoid hard-failing on UI import in that context.
"""

try:
    from .napari_plugin_v2 import FlowEditor
except Exception:  # pragma: no cover - optional in headless environments
    FlowEditor = None

# from .napari_plugin_test import AnalysisWidget
