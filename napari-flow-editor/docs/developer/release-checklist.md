# Release Checklist

## Goal

Keep releases consistent and low-risk.

## Checklist

## 1. Versioning and Metadata

- [ ] Update version in `pyproject.toml`.
- [ ] Verify napari manifest entry point still resolves:
  - `napari-flow-editor = "napari_flow_editor:napari.yaml"`
- [ ] Confirm any user-visible API changes are documented.

## 2. Node Library Consistency

- [ ] Regenerate node library:
  - `python src/napari_flow_editor/generate_library.py`
- [ ] Verify new/changed nodes appear correctly in `node_library.json`.
- [ ] Confirm node labels/categories/params match docs.

## 3. Quality Validation

- [ ] Run full tests:
  - `python -m pytest src/napari_flow_editor/tests -q`
- [ ] Run focused tests for recently modified subsystems.
- [ ] Perform manual smoke pass in napari:
  - basic processing pipeline
  - loop or batch workflow
  - one export/output workflow

## 4. Documentation

- [ ] Update `docs/reference/changelog.md`.
- [ ] Ensure no unresolved scaffold markers remain in published pages.
- [ ] Confirm MkDocs nav includes all intended pages.
- [ ] Local docs sanity check (`mkdocs serve` / `mkdocs build`) in docs-enabled environment.

## 5. Packaging and Distribution

- [ ] Validate editable install in a clean environment.
- [ ] Validate plugin appears in napari plugin menu.
- [ ] Create release tag and release notes.
- [ ] Attach migration notes if behavior changed significantly.

## 6. Post-Release Follow-Up

- [ ] Create short regression checklist from first user feedback.
- [ ] Log known issues in roadmap/changelog as needed.
