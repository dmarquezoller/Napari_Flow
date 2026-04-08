# Plotting

## Goal

Explain available plotting outputs and when to use static vs interactive rendering.

## Plot Nodes

Current plotting nodes include:
- `Plot Histogram` (image input)
- `Colocalization Scatter` (two image inputs)
- `Table Heatmap` (table/dataframe input)

These nodes produce figure outputs handled by the plot dashboard.

## Rendering Modes

When a plot result arrives, the dashboard can show:

1. **Static embedded plot** (Matplotlib)
   - displayed directly inside the plugin panel
   - no extra dependencies beyond Matplotlib

2. **Interactive Plotly in browser**
   - user can choose interactive mode when available
   - opens a generated HTML in the default browser

3. **Static preview fallback for Plotly**
   - if interactive embed/browser path is unavailable, static preview is used when possible
   - may require `kaleido` for Plotly image conversion

## Troubleshooting

Common issues:

- Plotly embedded view unavailable:
  - expected in environments without Qt WebEngine support
  - use browser mode or static preview fallback

- Browser opens but plot file missing:
  - check write permissions on the generated plot output directory

- Static preview fails for Plotly:
  - install `kaleido` in the same environment

Practical tip:
- prefer static embedded mode for quick iteration
- use interactive browser mode for final inspection and exploration
