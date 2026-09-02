# Model Migration Test Script

`test_model_migration.sh` exports one registered model version and its run,
imports them into another Databricks workspace, registers the destination model,
and verifies the destination run and downloadable model artifacts.

The script creates a registered model and experiment in the destination
workspace. It does not remove them afterward.

## Prerequisites

Install `uv`, Databricks CLI, and `jq`. From the repository root, create and
activate an isolated environment so the console commands use this checkout:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -e ".[tests]"
hash -r

command -v export-model
command -v import-model
python -c 'import mlflow_export_import; print(mlflow_export_import.__file__)'
```

The command paths should resolve under this repository or `.venv`.

## Authenticate Both Workspaces

Create explicitly named OAuth profiles if they do not already exist:

```bash
databricks auth login --host "https://source-workspace.example.com" --profile source-profile
databricks auth login --host "https://destination-workspace.example.com" --profile destination-profile

databricks current-user me --profile source-profile
databricks current-user me --profile destination-profile
```

The source principal needs access to the model, run, and artifacts. The
destination principal needs permission to create experiments and registered
models in the selected catalog and schema.

## Run the Migration Test

Use a unique suffix because the script refuses to overwrite an existing model:

```bash
SOURCE_PROFILE="source-profile"
DESTINATION_PROFILE="destination-profile"
SOURCE_MODEL="source_catalog.source_schema.model"
SOURCE_VERSION="1"
DESTINATION_MODEL="destination_catalog.destination_schema.model"
DESTINATION_EXPERIMENT="/Workspace/Users/user@example.com/mlflow_experiments/model"
TEST_ID="$(date +%Y%m%d_%H%M%S)"

./scripts/test_model_migration.sh \
  --source-profile "$SOURCE_PROFILE" \
  --destination-profile "$DESTINATION_PROFILE" \
  --source-model "$SOURCE_MODEL" \
  --source-version "$SOURCE_VERSION" \
  --destination-model "${DESTINATION_MODEL}_${TEST_ID}" \
  --destination-experiment "${DESTINATION_EXPERIMENT}_${TEST_ID}"
```

Use `--work-dir PATH` to retain export and verification files at a known path,
or `--await-seconds N` to change the 600-second registration timeout.

## Expected Result

The script checks authentication, exports and imports the requested version,
requires a `READY` destination version and `FINISHED` run, then downloads the
registered artifacts and confirms that they contain an `MLmodel` file.

```text
[1/6] Checking source and destination authentication
...
[6/6] Downloading registered model artifacts

Migration test passed
Destination model: destination_catalog.destination_schema.model_YYYYMMDD_HHMMSS
Destination version: 1
Destination run: <run-id>
```

A Feature Store lineage warning is non-fatal: local imports register the model
without UC feature dependency lineage. Run the import on Databricks compute if
that lineage must be preserved. Raw logs can expose workspace URLs, user names,
resource IDs, and storage locations; keep `scripts/*.log` local.
