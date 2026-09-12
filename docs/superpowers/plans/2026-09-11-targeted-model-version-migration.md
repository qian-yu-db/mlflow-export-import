# Targeted Model Version Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public Python function that migrates exactly one registered model version, its one backing run, and its model artifacts between explicitly selected Databricks workspaces.

**Architecture:** Build a small orchestrator around the existing `export_model_version()` and `import_model_version()` functions. Keep model/run serialization, logged-model ID mapping, artifact transfer, and UC finalization in those existing layers; the new module owns profile-specific clients, preflight validation, temporary/retained workspace handling, and result verification.

**Tech Stack:** Python 3.8-compatible typing and dataclasses, MLflow clients, Databricks profiles, pytest, existing Databricks UC integration fixtures.

**Spec:** `docs/superpowers/specs/2026-09-11-targeted-model-version-migration-design.md`

## Global Constraints

- Never auto-select a Databricks profile; require separate source and destination profile names.
- Export exactly the selected model version's backing run; do not export other experiment runs or recursively copy parent/child runs.
- Create the destination registered model when missing and append a version when it exists.
- Copy version description and tags; do not move aliases unless `copy_aliases=True`.
- Use a temporary local directory by default and retain a caller-provided empty directory.
- Preserve the existing Feature Store fallback that uploads artifacts and finalizes the UC version without local feature-lineage derivation.
- Do not roll back partial destination state automatically.

---

### Task 1: Forward registration timeout through model-version import

**Files:**
- Modify: `mlflow_export_import/model_version/import_model_version.py`
- Test: `tests/test_model_version_logged_models.py`

**Interfaces:**
- Consumes: existing `import_model_version(...)` and `_import_model_version(..., await_creation_for=None)`.
- Produces: `import_model_version(..., await_creation_for=None)` forwarding the value unchanged to `_import_model_version()`.

- [ ] **Step 1: Write the failing timeout-forwarding test**

Add a focused test beside the existing model-version import tests:

```python
def test_import_model_version_forwards_await_creation_timeout(tmp_path, monkeypatch):
    version = {
        "mlflow": {
            "model_version": {
                "name": "catalog.schema.source_model",
                "version": "1",
                "source": "models:/source-model",
                "run_id": "source-run",
                "tags": {},
                "aliases": [],
                "current_stage": "None",
            }
        }
    }
    (tmp_path / "version.json").write_text(json.dumps(version), encoding="utf-8")
    (tmp_path / "run").mkdir()
    destination_run = SimpleNamespace(
        info=SimpleNamespace(run_id="destination-run")
    )
    captured = {}

    def fake_import_run(**kwargs):
        kwargs["logged_model_id_map"]["source-model"] = "destination-model"
        return destination_run, None

    monkeypatch.setattr(import_model_version_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(import_model_version_module, "import_run", fake_import_run)
    monkeypatch.setattr(
        import_model_version_module,
        "_import_model_version",
        lambda mlflow_client, **kwargs: captured.update(kwargs) or kwargs,
    )

    import_model_version_module.import_model_version(
        model_name="catalog.schema.destination_model",
        experiment_name="destination-experiment",
        input_dir=str(tmp_path),
        await_creation_for=321,
        mlflow_client=_ModelVersionClient(),
    )

    assert captured["await_creation_for"] == 321
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest -q tests/test_model_version_logged_models.py::test_import_model_version_forwards_await_creation_timeout
```

Expected: failure because `import_model_version()` does not accept `await_creation_for`.

- [ ] **Step 3: Add and forward the optional argument**

Update the public function signature and final call:

```python
def import_model_version(
        model_name,
        experiment_name,
        input_dir,
        create_model=False,
        import_permissions=False,
        import_source_tags=False,
        import_stages_and_aliases=True,
        import_metadata=False,
        model_id=None,
        mlflow_client=None,
        await_creation_for=None
    ):
    ...
    dst_vr = _import_model_version(
        mlflow_client,
        model_name=model_name,
        src_vr=src_vr,
        dst_run_id=dst_run.info.run_id,
        dst_source=dst_source,
        import_stages_and_aliases=import_stages_and_aliases,
        import_source_tags=import_source_tags,
        model_id=destination_model_id,
        await_creation_for=await_creation_for,
    )
```

Document the new parameter in the function docstring.

- [ ] **Step 4: Run the focused regression tests**

Run:

```bash
python -m pytest -q tests/test_model_version_logged_models.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the timeout change**

```bash
git add mlflow_export_import/model_version/import_model_version.py tests/test_model_version_logged_models.py
git commit -m "fix: forward model version registration timeout"
```

---

### Task 2: Add the targeted migration orchestrator

**Files:**
- Create: `mlflow_export_import/model_version/migrate_model_version.py`
- Test: `tests/test_migrate_model_version.py`

**Interfaces:**
- Consumes: `mlflow.MlflowClient`, `export_model_version(...)`, and the timeout-enabled `import_model_version(...)` from Task 1.
- Produces: frozen `ModelVersionMigrationResult` and keyword-only `migrate_model_version(...)` matching the approved specification.

- [ ] **Step 1: Write failing tests for the successful retained-directory flow**

Create fake source/destination clients and patch the exporter/importer:

```python
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import importlib

migration = importlib.import_module(
    "mlflow_export_import.model_version.migrate_model_version"
)


def _version(name, version, run_id, status="READY"):
    return SimpleNamespace(
        name=name,
        version=version,
        run_id=run_id,
        status=status,
    )


def test_migrates_one_version_and_retains_requested_bundle(tmp_path, monkeypatch):
    source = MagicMock()
    destination = MagicMock()
    source.get_model_version.return_value = _version(
        "source.catalog.model", "5", "source-run"
    )
    source.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="source-run")
    )
    imported = _version("destination.catalog.model", "2", "destination-run")
    destination.get_model_version.return_value = imported
    destination.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="destination-run")
    )
    clients = iter([source, destination])
    monkeypatch.setattr(migration, "MlflowClient", lambda **kwargs: next(clients))

    calls = {}

    def fake_export(**kwargs):
        calls["export"] = kwargs
        root = Path(kwargs["output_dir"])
        (root / "run").mkdir(parents=True)
        (root / "version.json").write_text("{}", encoding="utf-8")
        (root / "run" / "run.json").write_text("{}", encoding="utf-8")

    def fake_import(**kwargs):
        calls["import"] = kwargs
        return imported

    monkeypatch.setattr(migration, "export_model_version", fake_export)
    monkeypatch.setattr(migration, "import_model_version", fake_import)
    bundle = tmp_path / "bundle"

    result = migration.migrate_model_version(
        source_profile="SRC",
        destination_profile="DST",
        source_model_name="source.catalog.model",
        source_model_version="5",
        destination_model_name="destination.catalog.model",
        destination_experiment_name="/Users/test/destination",
        expected_source_run_id="source-run",
        output_dir=str(bundle),
        await_creation_for=321,
    )

    assert calls["export"]["version"] == "5"
    assert calls["import"]["create_model"] is True
    assert calls["import"]["import_stages_and_aliases"] is False
    assert calls["import"]["await_creation_for"] == 321
    assert result.destination_run_id == "destination-run"
    assert result.output_dir == str(bundle.resolve())
    assert bundle.exists()
```

- [ ] **Step 2: Add failing validation and lifecycle tests**

Cover the remaining contract explicitly:

```python
@pytest.mark.parametrize(
    "argument",
    ["source_profile", "destination_profile", "source_model_name",
     "source_model_version", "destination_model_name",
     "destination_experiment_name"],
)
def test_rejects_blank_required_argument(argument):
    values = {
        "source_profile": "SRC",
        "destination_profile": "DST",
        "source_model_name": "source.catalog.model",
        "source_model_version": "5",
        "destination_model_name": "destination.catalog.model",
        "destination_experiment_name": "/Users/test/destination",
    }
    values[argument] = " "
    with pytest.raises(ValueError, match=argument):
        migration.migrate_model_version(**values)


def test_rejects_source_run_mismatch_before_export(monkeypatch):
    source = MagicMock()
    destination = MagicMock()
    source.get_model_version.return_value = _version(
        "source.catalog.model", "5", "actual-run"
    )
    clients = iter([source, destination])
    monkeypatch.setattr(migration, "MlflowClient", lambda **kwargs: next(clients))
    exporter = MagicMock()
    monkeypatch.setattr(migration, "export_model_version", exporter)

    with pytest.raises(ValueError, match="expected_source_run_id"):
        migration.migrate_model_version(
            source_profile="SRC",
            destination_profile="DST",
            source_model_name="source.catalog.model",
            source_model_version="5",
            destination_model_name="destination.catalog.model",
            destination_experiment_name="/Users/test/destination",
            expected_source_run_id="different-run",
        )

    exporter.assert_not_called()


def test_rejects_nonempty_output_directory(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        migration.migrate_model_version(
            source_profile="SRC",
            destination_profile="DST",
            source_model_name="source.catalog.model",
            source_model_version="5",
            destination_model_name="destination.catalog.model",
            destination_experiment_name="/Users/test/destination",
            output_dir=str(bundle),
        )
```

Add the following focused cases using the same `_version()` helper and patched
fake clients/exporter/importer from the successful test:

```python
@pytest.fixture
def successful_migration(monkeypatch):
    source = MagicMock()
    destination = MagicMock()
    source.get_model_version.return_value = _version(
        "source.catalog.model", "5", "source-run"
    )
    source.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="source-run")
    )
    imported = _version("destination.catalog.model", "2", "destination-run")
    destination.get_model_version.return_value = imported
    destination.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="destination-run")
    )
    clients = iter([source, destination])
    monkeypatch.setattr(migration, "MlflowClient", lambda **kwargs: next(clients))
    calls = {}

    def fake_export(**kwargs):
        calls["export"] = kwargs
        root = Path(kwargs["output_dir"])
        (root / "run").mkdir(parents=True)
        (root / "version.json").write_text("{}", encoding="utf-8")
        (root / "run" / "run.json").write_text("{}", encoding="utf-8")

    def fake_import(**kwargs):
        calls["import"] = kwargs
        return imported

    monkeypatch.setattr(migration, "export_model_version", fake_export)
    monkeypatch.setattr(migration, "import_model_version", fake_import)
    arguments = {
        "source_profile": "SRC",
        "destination_profile": "DST",
        "source_model_name": "source.catalog.model",
        "source_model_version": "5",
        "destination_model_name": "destination.catalog.model",
        "destination_experiment_name": "/Users/test/destination",
    }
    return SimpleNamespace(
        source_client=source,
        destination_client=destination,
        calls=calls,
        arguments=arguments,
    )


@pytest.mark.parametrize("timeout", [0, -1, 1.5, True])
def test_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="await_creation_for"):
        migration.migrate_model_version(
            source_profile="SRC",
            destination_profile="DST",
            source_model_name="source.catalog.model",
            source_model_version="5",
            destination_model_name="destination.catalog.model",
            destination_experiment_name="/Users/test/destination",
            await_creation_for=timeout,
        )


def test_rejects_model_version_without_run_id(monkeypatch):
    source = MagicMock()
    destination = MagicMock()
    source.get_model_version.return_value = _version(
        "source.catalog.model", "5", None
    )
    clients = iter([source, destination])
    monkeypatch.setattr(migration, "MlflowClient", lambda **kwargs: next(clients))

    with pytest.raises(ValueError, match="backing run_id"):
        migration.migrate_model_version(
            source_profile="SRC",
            destination_profile="DST",
            source_model_name="source.catalog.model",
            source_model_version="5",
            destination_model_name="destination.catalog.model",
            destination_experiment_name="/Users/test/destination",
        )


def test_requires_export_manifests(successful_migration, monkeypatch):
    monkeypatch.setattr(migration, "export_model_version", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="version.json"):
        migration.migrate_model_version(**successful_migration.arguments)


def test_rejects_non_ready_destination(successful_migration):
    successful_migration.destination_client.get_model_version.return_value = _version(
        "destination.catalog.model", "2", "destination-run", "PENDING_REGISTRATION"
    )
    with pytest.raises(RuntimeError, match="PENDING_REGISTRATION"):
        migration.migrate_model_version(**successful_migration.arguments)


def test_forwards_alias_opt_in(successful_migration, monkeypatch):
    calls = successful_migration.calls
    migration.migrate_model_version(
        **dict(successful_migration.arguments, copy_aliases=True)
    )
    assert calls["import"]["import_stages_and_aliases"] is True


def test_removes_default_temporary_directory(successful_migration):
    result = migration.migrate_model_version(**successful_migration.arguments)
    exported_path = Path(successful_migration.calls["export"]["output_dir"])
    assert result.output_dir is None
    assert not exported_path.exists()


def test_retains_requested_bundle_when_import_fails(
        tmp_path, successful_migration, monkeypatch):
    bundle = tmp_path / "failed-bundle"

    def fail_import(**kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(migration, "import_model_version", fail_import)
    with pytest.raises(RuntimeError, match="registration failed"):
        migration.migrate_model_version(
            **dict(successful_migration.arguments, output_dir=str(bundle))
        )

    assert (bundle / "version.json").is_file()
    assert (bundle / "run" / "run.json").is_file()
```

- [ ] **Step 3: Run the new module tests to verify they fail**

Run:

```bash
python -m pytest -q tests/test_migrate_model_version.py
```

Expected: collection failure because the migration module does not exist.

- [ ] **Step 4: Implement the result and validation helpers**

Create the module with these public and internal interfaces:

```python
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from mlflow import MlflowClient

from mlflow_export_import.model_version.export_model_version import export_model_version
from mlflow_export_import.model_version.import_model_version import import_model_version


@dataclass(frozen=True)
class ModelVersionMigrationResult:
    source_model_name: str
    source_model_version: str
    source_run_id: str
    destination_model_name: str
    destination_model_version: str
    destination_run_id: str
    output_dir: Optional[str]


def _require_text(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _prepare_output_dir(output_dir):
    path = Path(output_dir).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("output_dir must be a missing or empty directory")
    path.mkdir(parents=True, exist_ok=True)
    return path
```

- [ ] **Step 5: Implement the profile clients and orchestration**

Use explicit keyword-only arguments and a client-independent internal helper:

```python
def _profile_client(profile):
    return MlflowClient(
        tracking_uri=f"databricks://{profile}",
        registry_uri=f"databricks-uc://{profile}",
    )


def _migrate_with_clients(
        *, source_client, destination_client, source_model_name,
        source_model_version, destination_model_name,
        destination_experiment_name, expected_source_run_id,
        export_dir, retained_output_dir, await_creation_for, copy_aliases):
    source_version = source_client.get_model_version(
        source_model_name, source_model_version
    )
    source_run_id = getattr(source_version, "run_id", None)
    if not source_run_id:
        raise ValueError("The selected source model version has no backing run_id")
    if expected_source_run_id and expected_source_run_id != source_run_id:
        raise ValueError(
            "expected_source_run_id does not match the selected model version"
        )
    source_client.get_run(source_run_id)

    export_model_version(
        model_name=source_model_name,
        version=source_model_version,
        output_dir=str(export_dir),
        mlflow_client=source_client,
    )
    for relative_path in ("version.json", "run/run.json"):
        if not (export_dir / relative_path).is_file():
            raise RuntimeError(f"Export did not create {relative_path}")

    imported = import_model_version(
        model_name=destination_model_name,
        experiment_name=destination_experiment_name,
        input_dir=str(export_dir),
        create_model=True,
        import_stages_and_aliases=copy_aliases,
        await_creation_for=await_creation_for,
        mlflow_client=destination_client,
    )
    destination_version = destination_client.get_model_version(
        imported.name, imported.version
    )
    if destination_version.status != "READY":
        raise RuntimeError(
            f"Destination model version status is {destination_version.status!r}"
        )
    destination_run_id = getattr(destination_version, "run_id", None)
    if not destination_run_id:
        raise RuntimeError("Destination model version has no run_id")
    destination_client.get_run(destination_run_id)

    return ModelVersionMigrationResult(
        source_model_name=source_model_name,
        source_model_version=str(source_model_version),
        source_run_id=source_run_id,
        destination_model_name=destination_version.name,
        destination_model_version=str(destination_version.version),
        destination_run_id=destination_run_id,
        output_dir=retained_output_dir,
    )
```

The public function validates all strings and `await_creation_for` before
constructing clients. It calls `_migrate_with_clients()` inside
`TemporaryDirectory(prefix="mlflow-model-version-")` when `output_dir is None`;
otherwise it passes the resolved `_prepare_output_dir()` path and records that
path in the result.

- [ ] **Step 6: Run unit tests and correct only contract failures**

Run:

```bash
python -m pytest -q tests/test_migrate_model_version.py tests/test_model_version_logged_models.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the orchestrator**

```bash
git add mlflow_export_import/model_version/migrate_model_version.py tests/test_migrate_model_version.py
git commit -m "feat: add targeted model version migration API"
```

---

### Task 3: Publish, document, and integration-test the API

**Files:**
- Modify: `mlflow_export_import/model_version/__init__.py`
- Modify: `README_single.md`
- Create: `tests/databricks/uc/test_migrate_model_version.py`

**Interfaces:**
- Consumes: `migrate_model_version(...)` and `ModelVersionMigrationResult` from Task 2.
- Produces: package-level imports and a documented, workspace-tested Python workflow.

- [ ] **Step 1: Export the public names**

Add:

```python
from .migrate_model_version import (
    ModelVersionMigrationResult,
    migrate_model_version,
)

__all__ = ["ModelVersionMigrationResult", "migrate_model_version"]
```

- [ ] **Step 2: Document the one-version/one-run Python API**

Add a `Targeted Model Version Migration` subsection to `README_single.md` with
this example and explicitly state that aliases are opt-in and other experiment
runs are not copied:

```python
from mlflow_export_import.model_version import migrate_model_version

result = migrate_model_version(
    source_profile="source-workspace",
    destination_profile="destination-workspace",
    source_model_name="source_catalog.models.churn",
    source_model_version="5",
    destination_model_name="destination_catalog.models.churn",
    destination_experiment_name="/Users/me@example.com/model-migrations/churn",
    expected_source_run_id="run-id-recorded-on-version-5",
    output_dir="/tmp/churn-version-5",
)
print(result.destination_model_version, result.destination_run_id)
```

- [ ] **Step 3: Add a Databricks UC integration test**

Use the existing session fixture and explicit configured profile names:

```python
from pathlib import Path

import mlflow

from mlflow_export_import.model_version import migrate_model_version
from tests.databricks import local_utils
from tests.databricks.init_tests import test_context, workspace_src, workspace_dst


def _profile_name(registry_uri):
    return registry_uri.split("://", 1)[1]


def test_migrate_one_model_version_and_run(test_context, tmp_path):
    source_name = local_utils.mk_uc_model_name(workspace_src)
    source_version, _ = local_utils.create_version(
        test_context.mlflow_client_src, source_name
    )
    destination_name = local_utils.mk_uc_model_name(workspace_dst)
    destination_experiment = local_utils.mk_experiment_name(workspace_dst)

    result = migrate_model_version(
        source_profile=_profile_name(workspace_src.cfg.profile),
        destination_profile=_profile_name(workspace_dst.cfg.profile),
        source_model_name=source_name,
        source_model_version=source_version.version,
        destination_model_name=destination_name,
        destination_experiment_name=destination_experiment,
        expected_source_run_id=source_version.run_id,
        await_creation_for=600,
    )

    destination_version = test_context.mlflow_client_dst.get_model_version(
        destination_name, result.destination_model_version
    )
    assert destination_version.status == "READY"
    assert destination_version.run_id == result.destination_run_id
    assert destination_version.run_id != source_version.run_id
    assert destination_version.description == source_version.description
    for key, value in source_version.tags.items():
        assert destination_version.tags[key] == value

    destination_run = test_context.mlflow_client_dst.get_run(
        destination_version.run_id
    )
    experiment_runs = test_context.mlflow_client_dst.search_runs(
        [destination_run.info.experiment_id]
    )
    assert [run.info.run_id for run in experiment_runs] == [
        destination_version.run_id
    ]
    if getattr(source_version, "model_id", None):
        assert destination_version.model_id
        assert destination_version.model_id != source_version.model_id

    downloaded = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{destination_name}/{destination_version.version}",
        dst_path=str(tmp_path),
        tracking_uri=test_context.mlflow_client_dst.tracking_uri,
        registry_uri=test_context.mlflow_client_dst._registry_uri,
    )
    assert next(Path(downloaded).rglob("MLmodel"), None) is not None
```

- [ ] **Step 4: Run local regression suites**

Run:

```bash
python -m pytest -q tests/test_migrate_model_version.py tests/test_model_version_logged_models.py tests/test_logged_model_regressions.py
```

Expected: all tests pass.

- [ ] **Step 5: Run the Databricks integration test when configured**

From the UC test directory with `tests/databricks/config.yaml` containing
explicit source and destination profiles, run:

```bash
python -m pytest -q -s tests/databricks/uc/test_migrate_model_version.py
```

Expected: one destination run, one `READY` destination version, and a
downloaded `MLmodel` file. Do not run this test without explicit profiles.

- [ ] **Step 6: Verify packaging and diff quality**

Run:

```bash
python -m compileall -q mlflow_export_import
git diff --check
git status --short
```

Expected: compilation and whitespace checks pass; status lists only intended
files.

- [ ] **Step 7: Commit documentation and integration coverage**

```bash
git add mlflow_export_import/model_version/__init__.py README_single.md tests/databricks/uc/test_migrate_model_version.py
git commit -m "test: cover targeted workspace model migration"
```
