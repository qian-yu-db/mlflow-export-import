# Targeted Model Version Migration Design

## Purpose

Provide a public Python API that migrates one registered model version between
MLflow workspaces. The selected version determines exactly one backing run;
the API exports and imports that run, its run data and artifacts, and all MLflow
3 logged-model payloads referenced by the run. It then registers one new model
version in the destination.

## Public API

Add `migrate_model_version()` under
`mlflow_export_import.model_version.migrate_model_version`:

```python
from typing import Optional

migrate_model_version(
    *,
    source_profile: str,
    destination_profile: str,
    source_model_name: str,
    source_model_version: str,
    destination_model_name: str,
    destination_experiment_name: str,
    expected_source_run_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    await_creation_for: int = 600,
    copy_aliases: bool = False,
) -> ModelVersionMigrationResult
```

Profiles are always explicit. The function constructs separate MLflow tracking
and Unity Catalog registry clients using `databricks://<profile>` and
`databricks-uc://<profile>`.

`ModelVersionMigrationResult` is an immutable dataclass containing source and
destination model names, versions, and run IDs, plus the retained export path
when `output_dir` is supplied.

## Migration Flow

1. Validate required strings, positive timeout, and the optional output path.
2. Fetch the source model version and its backing run.
3. If `expected_source_run_id` is supplied, fail before destination mutation
   when it differs from the version's recorded run ID.
4. Call the existing `export_model_version()` once. This exports only the
   selected version's run, model artifacts, logged models, and relevant model
   and experiment metadata; it does not export other experiment runs.
5. Call the existing `import_model_version()` once with `create_model=True`.
   A missing destination model is created; an existing model receives a new
   version. Version description and tags are retained. Aliases are copied only
   when `copy_aliases=True`.
6. Verify that the returned destination version is `READY`, points to a newly
   imported destination run, and can be read back from the destination client.
7. Return the structured result.

Use `TemporaryDirectory` when `output_dir` is omitted and clean it after the
operation. A caller-provided path must not contain an existing export and is
retained on success or failure for inspection and reuse.

## Existing API Adjustment

Extend `import_model_version()` with optional `await_creation_for` and pass it
to `_import_model_version()`. This keeps registration timeout handling in the
existing normal and Feature Store fallback paths, including artifact upload and
Unity Catalog finalization.

## Failure Behavior

Raise a clear exception for authentication failures, missing model versions,
missing backing runs, run-ID mismatches, unsafe output directories, import
errors, and non-`READY` registration results. Do not automatically delete a
partially imported run or model version; preserving server-side evidence is
safer for diagnosis. Repeating a successful call intentionally creates another
destination run and appends another model version.

## Testing

Add isolated pytest coverage with fake MLflow clients for validation, exact
single-version export/import calls, temporary and retained directories,
destination creation versus append behavior, alias defaults, timeout
forwarding, result construction, and failure propagation. Add a Databricks UC
integration test that migrates one version across explicitly configured source
and destination profiles and verifies one new run, a `READY` version, remapped
logged-model ID, and downloadable `MLmodel` artifact.

## Non-Goals

- Interactive model selection or a new CLI command
- Exporting an entire experiment or recursively copying parent/child runs
- Automatically selecting a Databricks profile
- Preserving Feature Store lineage when running outside Databricks compute
- Automatic rollback or deduplication
