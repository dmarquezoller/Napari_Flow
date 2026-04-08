# Installation

Use this page to install the plugin in a clean environment and verify it loads in napari.

## Recommended Environment

- Python: `3.10` or `3.11`
- napari: latest stable version in your environment
- Qt backend: the one bundled with your napari install (`PyQt` or `PySide`)

## Option A: Install From Source (Recommended For Development)

From your plugin repo root:

```bash
cd /path/to/napari-flow-editor
pip install -e .
```

If napari is not installed yet:

```bash
pip install napari[all]
pip install -e .
```

## Option B: Conda Environment Example

```bash
conda create -n napari-flow python=3.11 -y
conda activate napari-flow
pip install napari[all]
cd /path/to/napari-flow-editor
pip install -e .
```

## Verify Installation

1. Start napari:

```bash
napari
```

2. Open the widget from:
   - `Plugins -> Add Dock Widget -> Flow Editor`

3. Confirm Flow Editor panel appears with toolbar buttons such as:
   - `Add Node`
   - `Add Control`
   - `Run Pipeline`

## Troubleshooting

- If `Flow Editor` is not visible in the plugin menu:
  - ensure `pip install -e .` was executed in the same environment used by `napari`
  - restart napari after installation
- If napari fails on startup with Qt errors:
  - verify you have one consistent Qt backend in the environment
