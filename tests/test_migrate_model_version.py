import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def _version(name, version, run_id, status="READY"):
    return SimpleNamespace(
        name=name,
        version=version,
        run_id=run_id,
        status=status,
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
        info=SimpleNamespace(run_id="source-run")
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
        (root / "run").mkdir(parents=True)
        (root / "version.json").write_text("{}", encoding="utf-8")
        (root / "run" / "run.json").write_text("{}", encoding="utf-8")

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
