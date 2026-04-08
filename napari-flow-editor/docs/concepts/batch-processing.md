# Batch Processing

## Goal

Explain CSV-driven multi-run execution with dynamic parameters.

## Begin Batch

`Begin Batch` is the batch start node.

- Requires `csv_path`.
- Uses one CSV row per pipeline run.
- Reuses the same graph logic for all rows.

## Dynamic Parameters

Dynamic parameters are configured in node properties:

1. Enable `Dyn` for a parameter.
2. Select a CSV column name.
3. At runtime, that parameter is read from the current row.

`<screenshot placeholder: Dyn checkbox + column dropdown>`

Type handling:
- int parameters read integer-like CSV values
- float parameters read numeric CSV values
- bool parameters accept values like `true/false`, `1/0`, `yes/no`
- text/enum/path parameters read as strings

## Recommended CSV Layout

Keep columns explicit and stable.

Suggested columns:
- `path` for file-based input nodes.
- Processing params like `sigma`, `gamma`, `threshold`.
- Optional naming/output columns.

Example:

```csv
path,sigma,output_name
</absolute/path/img_001.tif>,1.0,<run_001>
</absolute/path/img_002.tif>,2.5,<run_002>
```

## Typical Batch Patterns

### File-based images

`Begin Batch -> Open Image File -> Select Layer -> Processing -> Save`

### Viewer-layer-driven

`Begin Batch -> Get Layer -> Processing -> Save`

In this case, bind `Get Layer.layer_name` dynamically.

## Validation Rules

- CSV path must exist.
- CSV must include a header row.
- Dynamic bindings must reference existing columns.

## Interactive Nodes in Batch

- Allowed.
- Warning shown before execution starts.
- User interaction may be required for each row.

## Common Failure Cases

- Missing column in `Dyn` binding.
- Wrong value type in numeric columns.
- Empty critical cells for required params.

## Recommended Practices

1. Start with 2-3 rows before full runs.
2. Keep one CSV per experiment configuration.
3. Store CSV together with pipeline JSON for reproducibility.
4. Include explicit output naming/path columns when runs must be separated.

## Related Pages

- `Guides -> Layers and Inputs`
- `Guides -> Save and Load Pipelines`
- `Troubleshooting -> Common Errors`
