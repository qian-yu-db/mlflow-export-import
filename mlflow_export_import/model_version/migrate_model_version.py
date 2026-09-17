"""Migrate one registered model version and its backing run."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Optional

import mlflow
from mlflow import MlflowClient
from mlflow.tracking._model_registry import utils as registry_utils
from mlflow.tracking._tracking_service import utils as tracking_utils

from mlflow_export_import.model.model_utils import (
    _extract_model_id,
    _is_logged_model_source,
)
from mlflow_export_import.model_version.export_model_version import (
    VERSION_MODEL_ARTIFACT_PATH,
    export_model_version,
)
from mlflow_export_import.model_version.import_model_version import (
    import_model_version,
)


# MLflow's artifact and UC registration helpers consult process-global URI state.
# Keep profile scopes from overlapping and restoring each other's values.
_PROFILE_CONTEXT_LOCK = RLock()


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
    with _PROFILE_CONTEXT_LOCK:
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


def _safe_relative_artifact_path(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        return None
    return path


def _backing_run_artifact_path(model_version, run):
    source = getattr(model_version, "source", None)
    if not isinstance(source, str) or not source.strip():
        return None
    source = source.strip().rstrip("/")

    run_id = str(getattr(run.info, "run_id", model_version.run_id))
    runs_prefix = f"runs:/{run_id}/"
    if source.startswith(runs_prefix):
        return _safe_relative_artifact_path(source[len(runs_prefix):])

    artifact_uri = getattr(run.info, "artifact_uri", None)
    if not isinstance(artifact_uri, str) or not artifact_uri.strip():
        return None
    artifact_uri = artifact_uri.strip().rstrip("/")
    if source == artifact_uri:
        return PurePosixPath(".")
    artifact_prefix = f"{artifact_uri}/"
    if source.startswith(artifact_prefix):
        return _safe_relative_artifact_path(source[len(artifact_prefix):])
    return None


def _export_strategy(model_version, run):
    source = getattr(model_version, "source", None)
    if _is_logged_model_source(source):
        return False, "", None

    artifact_path = _backing_run_artifact_path(model_version, run)
    if artifact_path is not None:
        return False, "", artifact_path

    return True, VERSION_MODEL_ARTIFACT_PATH, None


def _require_exported_model_payload(
    export_dir,
    model_version,
    backing_run_artifact_path,
    export_version_model,
):
    source = getattr(model_version, "source", None)
    if export_version_model:
        marker = (
            export_dir
            / "run"
            / "artifacts"
            / VERSION_MODEL_ARTIFACT_PATH
            / "MLmodel"
        )
        if not marker.is_file():
            relative = marker.relative_to(export_dir)
            raise RuntimeError(f"Export did not create {relative}")
        return

    if _is_logged_model_source(source):
        model_id = _extract_model_id(source)
        model_dir = export_dir / "run" / model_id
        if not (model_dir / "logged_model.json").is_file() or not any(
            path.is_file() for path in model_dir.rglob("MLmodel")
        ):
            raise RuntimeError(
                f"Export did not create the selected logged model payload {model_id}"
            )
        return

    marker = (
        export_dir
        / "run"
        / "artifacts"
        / backing_run_artifact_path
        / "MLmodel"
    )
    if not marker.is_file():
        relative = marker.relative_to(export_dir)
        raise RuntimeError(f"Export did not create {relative}")


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
        if (
            expected_source_run_id is not None
            and expected_source_run_id != source_run_id
        ):
            raise ValueError(
                "expected_source_run_id does not match the selected model version"
            )
        source_run = source_client.get_run(source_run_id)
        (
            export_version_payload,
            version_model_artifact_path,
            backing_run_artifact_path,
        ) = _export_strategy(source_version, source_run)
        if export_version_payload:
            (
                export_dir
                / "run"
                / "artifacts"
                / version_model_artifact_path
            ).mkdir(parents=True, exist_ok=True)

        export_model_version(
            model_name=source_model_name,
            version=source_model_version,
            output_dir=str(export_dir),
            export_version_model=export_version_payload,
            vrm_model_artifact_path=version_model_artifact_path,
            raise_exception=True,
            mlflow_client=source_client,
        )
    for relative_path in ("version.json", "run/run.json"):
        if not (export_dir / relative_path).is_file():
            raise RuntimeError(f"Export did not create {relative_path}")
    _require_exported_model_payload(
        export_dir,
        source_version,
        backing_run_artifact_path,
        export_version_payload,
    )

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
    if expected_source_run_id is not None:
        _require_text("expected_source_run_id", expected_source_run_id)
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
