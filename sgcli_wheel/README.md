# SGCLI Wheel

This folder contains the `sgcli` Python wheel, which is **not** distributed on PyPI.

## Current version

`databricks_serverless_gpu_cli-0.1.0-py3-none-any.whl`

## Install

Recommended: `uv tool install` (isolated venv; adds `sgcli` to your PATH):

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repo root:
uv tool install --python 3.12 sgcli_wheel/databricks_serverless_gpu_cli-0.1.0-py3-none-any.whl
sgcli --version
```

Alternative: pip into an active venv:

```bash
pip install --force-reinstall sgcli_wheel/databricks_serverless_gpu_cli-0.1.0-py3-none-any.whl
```

> If you previously installed `sgcli` into a venv and now want to use the `uv tool install` copy, `pip uninstall sgcli` first — the venv copy takes precedence when its venv is active.

## Getting newer versions

`sgcli` is released by the Databricks SGC team. For newer versions:

- Search your org's Google Drive for `databricks_serverless_gpu_cli`
- Or ask your Databricks account team (SA/DSA) for the latest wheel — request the largest version without `dev` or `staging` qualifiers
