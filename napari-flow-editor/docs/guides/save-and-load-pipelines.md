# Save and Load Pipelines

## Goal

Document pipeline persistence and portability.

## Saving

Use `File -> Save Pipeline` to export current graph to JSON.

Saved data includes:
- schema version
- nodes (`id`, `type`, label/title, position)
- node parameters
- dynamic parameter bindings (batch mappings)
- data connections
- exec connections
- macro group metadata (when present)

## Loading

Use `File -> Load Pipeline` to restore a saved graph.

Load behavior:
1. clears current graph
2. recreates nodes with saved ids and parameters
3. reconnects data and exec edges
4. restores macro groups

Caveats:
- pipelines rely on current node library definitions
- renamed/removed nodes can break older JSON files
- incompatible socket type changes can prevent some edges from reconnecting

## Versioning Advice

Recommended workflow:
1. Commit pipeline JSON files in Git with meaningful names.
2. Keep CSV batch files next to pipeline JSON for reproducibility.
3. Store a short README in each analysis folder with:
   - plugin commit/version
   - napari version
   - dataset used

Naming suggestion:
- `pipeline_<task>_<date>.json`
