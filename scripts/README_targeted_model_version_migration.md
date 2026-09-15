# Targeted Model Version Migration

`test_targeted_model_version_migration.py` migrates one registered model
version, its backing MLflow run, and its model artifacts between explicitly
selected Databricks workspaces. The destination registered model is created
when missing; an existing model receives a new version.

The script creates persistent resources in the destination workspace and does
not remove them afterward.

## Local Setup

An egg build is not required. From the repository root, create an isolated
environment and install the checkout in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable installation creates local `.egg-info` metadata. Do not commit the
virtual environment or generated package metadata.

Confirm that Python imports this checkout:

```bash
python -c 'import mlflow_export_import; print(mlflow_export_import.__file__)'
```

## Authenticate Both Workspaces

Use explicitly named Databricks profiles; do not rely on `DEFAULT`:

```bash
databricks auth login --host <source-workspace-url> --profile <source-profile>
databricks auth login --host <destination-workspace-url> --profile <destination-profile>

databricks current-user me --profile <source-profile>
databricks current-user me --profile <destination-profile>
```

The source identity needs read access to the model, run, and artifacts. The
destination identity needs permission to create experiments and registered
models in the selected Unity Catalog catalog and schema.

## Run the Example

Edit the profiles, model names, version, and experiment prefix at the top of
`scripts/test_targeted_model_version_migration.py`, then run:

```bash
python scripts/test_targeted_model_version_migration.py
```

Each invocation adds a timestamp to the destination names. On success, the
script prints the destination model, version, run ID, and retained export
directory. A Feature Store warning is non-fatal: local migration registers the
model without UC feature dependency lineage.

## Use the API in Another Workflow

Install this checkout into the workflow's virtual environment:

```bash
python -m pip install -e /path/to/mlflow-export-import
```

For reproducible deployment, install a committed revision from a fork:

```bash
python -m pip install \
  "mlflow_export_import @ git+https://github.com/<owner>/mlflow-export-import.git@<commit>"
```

Call the public API from the workflow:

```python
from mlflow_export_import.model_version import migrate_model_version

result = migrate_model_version(
    source_profile="source-profile",
    destination_profile="destination-profile",
    source_model_name="source_catalog.source_schema.model",
    source_model_version="5",
    destination_model_name="destination_catalog.destination_schema.model",
    destination_experiment_name="/Users/user@example.com/model_migrations",
    output_dir="/tmp/model-migration-001",
    await_creation_for=600,
)

print(result)
```

`output_dir` must be missing or empty. Omit it to use an automatically removed
temporary directory. Use `expected_source_run_id` when the workflow should
also verify that the selected model version points to a specific run.
