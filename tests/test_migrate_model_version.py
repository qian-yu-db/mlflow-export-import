import importlib
import os
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow
import pytest


migration = importlib.import_module(
    "mlflow_export_import.model_version.migrate_model_version"
)


def test_public_api_is_exported_from_model_version_package():
    from mlflow_export_import.model_version import (
        ModelVersionMigrationResult,
        migrate_model_version,
    )

    assert migrate_model_version is migration.migrate_model_version
    assert ModelVersionMigrationResult is migration.ModelVersionMigrationResult


def _version(
    name,
    version,
    run_id,
    status="READY",
    source="dbfs:/runs/source-run/artifacts/model",
):
    return SimpleNamespace(
        name=name,
        version=version,
        run_id=run_id,
        status=status,
        source=source,
    )


def _arguments(**overrides):
    values = {
        "source_profile": "SRC",
        "destination_profile": "DST",
        "source_model_name": "source.catalog.model",
        "source_model_version": "5",
        "destination_model_name": "destination.catalog.model",
        "destination_experiment_name": "/Users/test/destination",
    }
    values.update(overrides)
    return values


@pytest.fixture
def successful_migration(monkeypatch):
    source = MagicMock()
    destination = MagicMock()
    source.get_model_version.return_value = _version(
        "source.catalog.model", "5", "source-run"
    )
    source.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(
            run_id="source-run",
            artifact_uri="dbfs:/runs/source-run/artifacts",
        )
    )
    imported = _version("destination.catalog.model", "2", "destination-run")
    destination.get_model_version.return_value = imported
    destination.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="destination-run")
    )
    clients = iter([source, destination])
    client_arguments = []

    def fake_client(**kwargs):
        client_arguments.append(kwargs)
        return next(clients)

    monkeypatch.setattr(migration, "MlflowClient", fake_client)
    calls = {}

    def fake_export(**kwargs):
        calls["export"] = kwargs
        root = Path(kwargs["output_dir"])
        model_dir = root / "run" / "artifacts" / "model"
        model_dir.mkdir(parents=True)
        (root / "version.json").write_text("{}", encoding="utf-8")
        (root / "run" / "run.json").write_text("{}", encoding="utf-8")
        (model_dir / "MLmodel").write_text("flavors: {}", encoding="utf-8")

    def fake_import(**kwargs):
        calls["import"] = kwargs
        return imported

    monkeypatch.setattr(migration, "export_model_version", fake_export)
    monkeypatch.setattr(migration, "import_model_version", fake_import)
    return SimpleNamespace(
        source_client=source,
        destination_client=destination,
        client_arguments=client_arguments,
        calls=calls,
        arguments=_arguments(),
    )


def test_migrates_one_version_and_retains_requested_bundle(
    tmp_path, successful_migration
):
    bundle = tmp_path / "bundle"

    result = migration.migrate_model_version(
        **_arguments(
            expected_source_run_id="source-run",
            output_dir=str(bundle),
            await_creation_for=321,
        )
    )

    calls = successful_migration.calls
    assert successful_migration.client_arguments == [
        {
            "tracking_uri": "databricks://SRC",
            "registry_uri": "databricks-uc://SRC",
        },
        {
            "tracking_uri": "databricks://DST",
            "registry_uri": "databricks-uc://DST",
        },
    ]
    assert calls["export"]["model_name"] == "source.catalog.model"
    assert calls["export"]["version"] == "5"
    assert calls["export"]["export_version_model"] is False
    assert calls["export"]["vrm_model_artifact_path"] == ""
    assert calls["export"]["raise_exception"] is True
    assert calls["export"]["mlflow_client"] is successful_migration.source_client
    assert calls["import"]["model_name"] == "destination.catalog.model"
    assert calls["import"]["experiment_name"] == "/Users/test/destination"
    assert calls["import"]["create_model"] is True
    assert calls["import"]["import_stages_and_aliases"] is False
    assert calls["import"]["await_creation_for"] == 321
    assert calls["import"]["mlflow_client"] is successful_migration.destination_client
    assert result.source_model_name == "source.catalog.model"
    assert result.source_model_version == "5"
    assert result.source_run_id == "source-run"
    assert result.destination_model_name == "destination.catalog.model"
    assert result.destination_model_version == "2"
    assert result.destination_run_id == "destination-run"
    assert result.output_dir == str(bundle.resolve())
    assert bundle.exists()


@pytest.mark.parametrize(
    "argument",
    [
        "source_profile",
        "destination_profile",
        "source_model_name",
        "source_model_version",
        "destination_model_name",
        "destination_experiment_name",
    ],
)
def test_rejects_blank_required_argument(argument):
    values = _arguments()
    values[argument] = " "

    with pytest.raises(ValueError, match=argument):
        migration.migrate_model_version(**values)


@pytest.mark.parametrize("expected_source_run_id", ["", " ", 0])
def test_rejects_invalid_expected_source_run_id(
    expected_source_run_id, successful_migration
):
    with pytest.raises(ValueError, match="expected_source_run_id"):
        migration.migrate_model_version(
            **_arguments(expected_source_run_id=expected_source_run_id)
        )


@pytest.mark.parametrize("timeout", [0, -1, 1.5, True])
def test_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="await_creation_for"):
        migration.migrate_model_version(
            **_arguments(await_creation_for=timeout)
        )


def test_rejects_source_run_mismatch_before_export(
    successful_migration, monkeypatch
):
    successful_migration.source_client.get_model_version.return_value = _version(
        "source.catalog.model", "5", "actual-run"
    )
    exporter = MagicMock()
    monkeypatch.setattr(migration, "export_model_version", exporter)

    with pytest.raises(ValueError, match="expected_source_run_id"):
        migration.migrate_model_version(
            **_arguments(expected_source_run_id="different-run")
        )

    exporter.assert_not_called()


def test_rejects_model_version_without_run_id(successful_migration):
    successful_migration.source_client.get_model_version.return_value = _version(
        "source.catalog.model", "5", None
    )

    with pytest.raises(ValueError, match="backing run_id"):
        migration.migrate_model_version(**successful_migration.arguments)


def test_rejects_nonempty_output_directory(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        migration.migrate_model_version(
            **_arguments(output_dir=str(bundle))
        )


def test_requires_export_manifests(successful_migration, monkeypatch):
    monkeypatch.setattr(migration, "export_model_version", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="version.json"):
        migration.migrate_model_version(**successful_migration.arguments)


def test_requires_exported_model_artifact_before_import(
    successful_migration, monkeypatch
):
    def fake_export_without_model(**kwargs):
        root = Path(kwargs["output_dir"])
        (root / "run").mkdir(parents=True)
        (root / "version.json").write_text("{}", encoding="utf-8")
        (root / "run" / "run.json").write_text("{}", encoding="utf-8")
        (root / "run" / "artifacts" / "MLmodel").mkdir(parents=True)

    importer = MagicMock()
    monkeypatch.setattr(migration, "export_model_version", fake_export_without_model)
    monkeypatch.setattr(migration, "import_model_version", importer)

    with pytest.raises(RuntimeError, match="MLmodel"):
        migration.migrate_model_version(**successful_migration.arguments)

    importer.assert_not_called()


def test_exports_external_model_payload_into_backing_run_bundle(
    successful_migration, monkeypatch
):
    successful_migration.source_client.get_model_version.return_value = _version(
        "source.catalog.model",
        "5",
        "source-run",
        source="s3://external-bucket/models/customer-churn",
    )

    def fake_export(**kwargs):
        successful_migration.calls["export"] = kwargs
        root = Path(kwargs["output_dir"])
        model_dir = (
            root
            / "run"
            / "artifacts"
            / kwargs["vrm_model_artifact_path"]
        )
        assert model_dir.is_dir()
        (root / "version.json").write_text("{}", encoding="utf-8")
        (root / "run" / "run.json").write_text("{}", encoding="utf-8")
        (model_dir / "MLmodel").write_text("flavors: {}", encoding="utf-8")

    monkeypatch.setattr(migration, "export_model_version", fake_export)

    migration.migrate_model_version(**successful_migration.arguments)

    call = successful_migration.calls["export"]
    assert call["export_version_model"] is True
    assert call["vrm_model_artifact_path"] == "version_model"


def test_unrelated_mlmodel_does_not_satisfy_external_payload_export(
    successful_migration, monkeypatch
):
    successful_migration.source_client.get_model_version.return_value = _version(
        "source.catalog.model",
        "5",
        "source-run",
        source="dbfs:/external/models/customer-churn",
    )

    def fake_export(**kwargs):
        root = Path(kwargs["output_dir"])
        unrelated = root / "run" / "artifacts" / "unrelated"
        unrelated.mkdir(parents=True)
        (root / "version.json").write_text("{}", encoding="utf-8")
        (root / "run" / "run.json").write_text("{}", encoding="utf-8")
        (unrelated / "MLmodel").write_text("flavors: {}", encoding="utf-8")

    importer = MagicMock()
    monkeypatch.setattr(migration, "export_model_version", fake_export)
    monkeypatch.setattr(migration, "import_model_version", importer)

    with pytest.raises(RuntimeError, match="version_model/MLmodel"):
        migration.migrate_model_version(**successful_migration.arguments)

    importer.assert_not_called()


def test_preserves_logged_model_export_strategy(successful_migration, monkeypatch):
    successful_migration.source_client.get_model_version.return_value = _version(
        "source.catalog.model",
        "5",
        "source-run",
        source="models:/m-source-model",
    )

    def fake_export(**kwargs):
        successful_migration.calls["export"] = kwargs
        root = Path(kwargs["output_dir"])
        model_dir = root / "run" / "m-source-model"
        artifacts = model_dir / "artifacts"
        artifacts.mkdir(parents=True)
        (root / "version.json").write_text("{}", encoding="utf-8")
        (root / "run" / "run.json").write_text("{}", encoding="utf-8")
        (model_dir / "logged_model.json").write_text("{}", encoding="utf-8")
        (artifacts / "MLmodel").write_text("flavors: {}", encoding="utf-8")

    monkeypatch.setattr(migration, "export_model_version", fake_export)

    migration.migrate_model_version(**successful_migration.arguments)

    call = successful_migration.calls["export"]
    assert call["export_version_model"] is False
    assert call["vrm_model_artifact_path"] == ""


def test_scopes_profiles_and_mlflow_uris_and_restores_them_on_failure(
    successful_migration, monkeypatch
):
    original_tracking_uri = "file:///original-tracking"
    original_registry_uri = "file:///original-registry"
    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_registry_uri = mlflow.get_registry_uri()
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "ORIGINAL")
    mlflow.set_tracking_uri(original_tracking_uri)
    mlflow.set_registry_uri(original_registry_uri)

    def assert_context(profile):
        assert os.environ["DATABRICKS_CONFIG_PROFILE"] == profile
        assert os.environ["MLFLOW_TRACKING_URI"] == f"databricks://{profile}"
        assert os.environ["MLFLOW_REGISTRY_URI"] == f"databricks-uc://{profile}"
        assert mlflow.get_tracking_uri() == f"databricks://{profile}"
        assert mlflow.get_registry_uri() == f"databricks-uc://{profile}"

    def fake_export(**kwargs):
        assert_context("SRC")
        root = Path(kwargs["output_dir"])
        model_dir = root / "run" / "artifacts" / "model"
        model_dir.mkdir(parents=True)
        (root / "version.json").write_text("{}", encoding="utf-8")
        (root / "run" / "run.json").write_text("{}", encoding="utf-8")
        (model_dir / "MLmodel").write_text("flavors: {}", encoding="utf-8")

    def fail_import(**kwargs):
        assert_context("DST")
        raise RuntimeError("registration failed")

    monkeypatch.setattr(migration, "export_model_version", fake_export)
    monkeypatch.setattr(migration, "import_model_version", fail_import)

    try:
        with pytest.raises(RuntimeError, match="registration failed"):
            migration.migrate_model_version(**successful_migration.arguments)

        assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "ORIGINAL"
        assert os.environ["MLFLOW_TRACKING_URI"] == original_tracking_uri
        assert os.environ["MLFLOW_REGISTRY_URI"] == original_registry_uri
        assert mlflow.get_tracking_uri() == original_tracking_uri
        assert mlflow.get_registry_uri() == original_registry_uri
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
        mlflow.set_registry_uri(previous_registry_uri)


def test_profile_context_serializes_process_global_state():
    first_entered = Event()
    release_first = Event()
    second_started = Event()
    second_entered = Event()
    errors = []

    def first_worker():
        try:
            with migration._profile_context("FIRST"):
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise AssertionError("timed out waiting to release first context")
        except Exception as error:
            errors.append(error)

    def second_worker():
        try:
            second_started.set()
            with migration._profile_context("SECOND"):
                second_entered.set()
        except Exception as error:
            errors.append(error)

    first = Thread(target=first_worker)
    second = Thread(target=second_worker)
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    assert second_started.wait(timeout=5)

    try:
        assert not second_entered.wait(timeout=0.2)
    finally:
        release_first.set()

    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    assert errors == []


def test_rejects_non_ready_destination(successful_migration):
    successful_migration.destination_client.get_model_version.return_value = _version(
        "destination.catalog.model",
        "2",
        "destination-run",
        "PENDING_REGISTRATION",
    )

    with pytest.raises(RuntimeError, match="PENDING_REGISTRATION"):
        migration.migrate_model_version(**successful_migration.arguments)


def test_rejects_destination_version_without_run_id(successful_migration):
    successful_migration.destination_client.get_model_version.return_value = _version(
        "destination.catalog.model", "2", None
    )

    with pytest.raises(RuntimeError, match="no run_id"):
        migration.migrate_model_version(**successful_migration.arguments)


def test_forwards_alias_opt_in(successful_migration):
    migration.migrate_model_version(
        **_arguments(copy_aliases=True)
    )

    assert successful_migration.calls["import"]["import_stages_and_aliases"] is True


def test_removes_default_temporary_directory(successful_migration):
    result = migration.migrate_model_version(**successful_migration.arguments)

    exported_path = Path(successful_migration.calls["export"]["output_dir"])
    assert result.output_dir is None
    assert not exported_path.exists()


def test_retains_requested_bundle_when_import_fails(
    tmp_path, successful_migration, monkeypatch
):
    bundle = tmp_path / "failed-bundle"

    def fail_import(**kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(migration, "import_model_version", fail_import)

    with pytest.raises(RuntimeError, match="registration failed"):
        migration.migrate_model_version(
            **_arguments(output_dir=str(bundle))
        )

    assert (bundle / "version.json").is_file()
    assert (bundle / "run" / "run.json").is_file()
