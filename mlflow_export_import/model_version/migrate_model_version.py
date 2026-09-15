"""Migrate one registered model version and its backing run."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import mlflow
from mlflow import MlflowClient
from mlflow.tracking._model_registry import utils as registry_utils
from mlflow.tracking._tracking_service import utils as tracking_utils

from mlflow_export_import.model_version.export_model_version import (
    export_model_version,
)
from mlflow_export_import.model_version.import_model_version import (
    import_model_version,
)


@dataclass(frozen=True)
class ModelVersionMigrationResult:
    """Identifiers created or resolved during a model-version migration."""

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


def _profile_client(profile):
    return MlflowClient(
        tracking_uri=f"databricks://{profile}",
        registry_uri=f"databricks-uc://{profile}",
    )


@contextmanager
def _profile_context(profile):
    environment_names = (
        "DATABRICKS_CONFIG_PROFILE",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_REGISTRY_URI",
    )
    missing = object()
    previous_environment = {
        name: os.environ.get(name, missing) for name in environment_names
    }
    previous_tracking_uri = tracking_utils._tracking_uri
    previous_registry_uri = registry_utils._registry_uri
    try:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")
        yield
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
        mlflow.set_registry_uri(previous_registry_uri)
        for name, value in previous_environment.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _migrate_with_clients(
    *,
    source_client,
    destination_client,
    source_profile,
    destination_profile,
    source_model_name,
    source_model_version,
    destination_model_name,
    destination_experiment_name,
    expected_source_run_id,
    export_dir,
    retained_output_dir,
    await_creation_for,
    copy_aliases,
):
    with _profile_context(source_profile):
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
    if not any(path.is_file() for path in export_dir.rglob("MLmodel")):
        raise RuntimeError("Export did not create an MLmodel artifact")

    with _profile_context(destination_profile):
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


def migrate_model_version(
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
) -> ModelVersionMigrationResult:
    """Migrate one registered model version and its backing run.

    The source and destination profiles are always explicit. When ``output_dir``
    is omitted, the local export bundle is removed after the call completes.
    """
    required_text = {
        "source_profile": source_profile,
        "destination_profile": destination_profile,
        "source_model_name": source_model_name,
        "source_model_version": source_model_version,
        "destination_model_name": destination_model_name,
        "destination_experiment_name": destination_experiment_name,
    }
    for name, value in required_text.items():
        _require_text(name, value)
    if (
        isinstance(await_creation_for, bool)
        or not isinstance(await_creation_for, int)
        or await_creation_for <= 0
    ):
        raise ValueError("await_creation_for must be a positive integer")

    retained_path = None
    if output_dir is not None:
        export_dir = _prepare_output_dir(output_dir)
        retained_path = str(export_dir)

    source_client = _profile_client(source_profile)
    destination_client = _profile_client(destination_profile)
    arguments = {
        "source_client": source_client,
        "destination_client": destination_client,
        "source_profile": source_profile,
        "destination_profile": destination_profile,
        "source_model_name": source_model_name,
        "source_model_version": source_model_version,
        "destination_model_name": destination_model_name,
        "destination_experiment_name": destination_experiment_name,
        "expected_source_run_id": expected_source_run_id,
        "retained_output_dir": retained_path,
        "await_creation_for": await_creation_for,
        "copy_aliases": copy_aliases,
    }

    if output_dir is not None:
        return _migrate_with_clients(export_dir=export_dir, **arguments)

    with TemporaryDirectory(prefix="mlflow-model-version-") as temporary_dir:
        return _migrate_with_clients(
            export_dir=Path(temporary_dir),
            **arguments,
        )
