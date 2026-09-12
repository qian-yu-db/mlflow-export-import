from pathlib import Path

import mlflow

from mlflow_export_import.model_version import migrate_model_version
from tests.databricks import local_utils
from tests.databricks.init_tests import test_context, workspace_dst, workspace_src


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
