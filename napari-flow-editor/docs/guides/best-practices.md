# Best Practices

## Goal

Collect practical guidance for stable and readable pipelines.

## Design Practices

1. Keep exec flow visually linear (left-to-right) whenever possible.
2. Name nodes/macros by function, not by default labels only.
3. Use macros to group stages (`Preprocess`, `Segment`, `Export`).
4. Avoid unnecessary crossing edges; route for readability first.
5. Keep one clear output path per branch.

## Performance Practices

1. Start with a small representative dataset or ROI.
2. Add one heavy node at a time and rerun to isolate slowdowns.
3. Prefer explicit layer selection (`Select Layer`) in multi-layer inputs.
4. Validate cache behavior by changing one parameter and observing stale propagation.
5. For batch runs, test with 2-3 rows before full CSV execution.

## Reliability Practices

1. Run graph validation before long runs.
2. Save pipeline JSON before major edits.
3. Store related CSV batch files with the pipeline JSON.
4. Keep environment notes (napari/plugin versions) with analysis outputs.
5. Add at least one sanity-check output layer or plot in long pipelines.

## Collaboration Tips

1. Commit small pipeline changes frequently.
2. Use consistent node naming conventions across team members.
3. Keep documentation screenshots in sync with major UI changes.
