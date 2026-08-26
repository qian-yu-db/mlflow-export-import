"""
Imports a registered model version and its run.
Optionally import registered model and experiment metadata.
"""

import os
import time
import click

from mlflow.entities.model_registry import ModelVersionTag
from mlflow.exceptions import MlflowException

from mlflow_export_import.common.click_options import (
    opt_input_dir,
    opt_model,
    opt_import_permissions,
    opt_import_source_tags
)
from . click_options import (
    opt_create_model,
    opt_experiment_name,
    opt_import_metadata,
    opt_import_stages_and_aliases
)
from mlflow_export_import.common import MlflowExportImportException
from mlflow_export_import.common import utils, io_utils, mlflow_utils, model_utils
from mlflow_export_import.common.mlflow_utils import MlflowTrackingUriTweak
from mlflow_export_import.common.source_tags import set_source_tags_for_field, fmt_timestamps
from mlflow_export_import.common.timestamp_utils import format_seconds
from mlflow_export_import.run.import_run import import_run
from mlflow_export_import.client.client_utils import create_mlflow_client, create_dbx_client
from mlflow_export_import.model.model_utils import (
    _extract_model_id,
    _get_logged_model_artifact_path,
    find_destination_logged_model_id,
)

_logger = utils.getLogger(__name__)


def import_model_version(
        model_name,
        experiment_name,
        input_dir,
        create_model = False,
        import_permissions = False,
        import_source_tags = False,
        import_stages_and_aliases = True,
        import_metadata = False,
        model_id = None,
        mlflow_client = None
    ):
    """
    Exports a model version.

    :param model_name: Registered model name.
    :param experiment_name: Destination experiment name for the version's run.
    :param input_dir: Import directory.
    :param create_model: Create registered model before creating model version.
    :param import_source_tags: Import source information for registered model and its versions and tags in destination object.
    :param import_stages_and_aliases: Import stages and aliases.
    :param import_metadata: Import registered model and experiment metadata.
    :param model_id: logged Model id if applicable. Supported from >=3.0 version
    :param mlflow_client: MlflowClient (optional).

    :return: Returns model version object.
    """

    mlflow_client = mlflow_client or create_mlflow_client()

    path = os.path.join(input_dir, "version.json")
    src_vr = io_utils.read_file_mlflow(path)["model_version"]

    dbx_client = create_dbx_client(mlflow_client)
    if import_metadata:
        path = os.path.join(input_dir, "experiment.json")
        exp = io_utils.read_file_mlflow(path)["experiment"]
        tags = utils.mk_tags_dict(exp.get("tags"))
        mlflow_utils.set_experiment(mlflow_client, dbx_client, experiment_name, tags)

    path = os.path.join(input_dir, "run")
    logged_model_id_map = {}
    dst_run, _ = import_run(
        input_dir = path,
        experiment_name = experiment_name,
        import_source_tags = import_source_tags,
        mlflow_client = mlflow_client,
        logged_model_id_map = logged_model_id_map
    )

    if create_model:
        path = os.path.join(input_dir, "model.json")
        model_dct = io_utils.read_file_mlflow(path)["registered_model"]
        created_model = model_utils.create_model(mlflow_client, model_name, model_dct, import_metadata)
        perms = model_dct.get("permissions")
        if created_model and import_permissions and perms:
            model_utils.update_model_permissions(mlflow_client, dbx_client, model_name, perms)

    destination_model_id = None
    if src_vr["source"].startswith("models:/"):
        source_model_id = _extract_model_id(src_vr["source"])
        destination_model_id = logged_model_id_map.get(source_model_id, model_id)
        if not destination_model_id:
            raise MlflowExportImportException(
                f"Cannot resolve destination logged model ID for source model '{source_model_id}'"
            )
        dst_source = _get_logged_model_artifact_path(destination_model_id, mlflow_client)
    else:
        destination_model_id = find_destination_logged_model_id(
            mlflow_client, mlflow_client.get_run(dst_run.info.run_id), src_vr["source"]
        )
        if destination_model_id:
            dst_source = _get_logged_model_artifact_path(
                destination_model_id, mlflow_client
            )
        else:
            model_path = _get_model_path(src_vr)
            dst_source = f"{dst_run.info.artifact_uri}/{model_path}"
    dst_vr = _import_model_version(
        mlflow_client,
        model_name = model_name,
        src_vr = src_vr,
        dst_run_id = dst_run.info.run_id,
        dst_source = dst_source,
        import_stages_and_aliases = import_stages_and_aliases,
        import_source_tags = import_source_tags,
        model_id = destination_model_id
    )
    return dst_vr

def _import_model_version(
        mlflow_client,
        model_name,
        src_vr,
        dst_run_id,
        dst_source,
        import_stages_and_aliases = True,
        import_source_tags = False,
        model_id = None,
        await_creation_for = None
    ):
    start_time = time.time()
    dst_source = dst_source.replace("file://","") # OSS MLflow
    if not (dst_source.startswith("dbfs:") or dst_source.startswith("s3:")) and not os.path.exists(dst_source):
        raise MlflowExportImportException(f"'source' argument for MLflowClient.create_model_version does not exist: {dst_source}", http_status_code=404)

    tags = src_vr["tags"]
    if import_source_tags:
        _set_source_tags_for_field(src_vr, tags)

    # NOTE: MLflow UC bug:
    # The client's tracking_uri is not honored. Instead MlflowClient.create_model_version()
    # seems to use mlflow.tracking_uri internally to download run artifacts for UC models.
    _logger.info(f"Importing model version with dst_source = '{dst_source}' for model '{model_name}'")

    create_model_version_params = {
        "name": model_name,
        "source": dst_source,
        "run_id": dst_run_id,
        "description": src_vr.get("description"),
        "tags": tags
    }
    if model_id:
        create_model_version_params["model_id"] = model_id
    if await_creation_for is not None:
        create_model_version_params["await_creation_for"] = await_creation_for

    dst_vr = _create_model_version_with_feature_store_fallback(
        mlflow_client,
        create_model_version_params,
    )

    if import_stages_and_aliases:
        for alias in src_vr.get("aliases",[]):
            mlflow_client.set_registered_model_alias(dst_vr.name, alias, dst_vr.version)

        if not model_utils.is_unity_catalog_model(model_name):
            src_current_stage = src_vr["current_stage"]
            if src_current_stage and src_current_stage != "None": # fails for Databricks  but not OSS
                mlflow_client.transition_model_version_stage(model_name, dst_vr.version, src_current_stage)

    dur = format_seconds(time.time()-start_time)
    _logger.info(f"Imported model version '{model_name}/{dst_vr.version}' in {dur}")
    return mlflow_client.get_model_version(dst_vr.name, dst_vr.version)


def _create_model_version_with_feature_store_fallback(
        mlflow_client,
        create_model_version_params
    ):
    try:
        with MlflowTrackingUriTweak(mlflow_client):
            return mlflow_client.create_model_version(**create_model_version_params)
    except MlflowException as e:
        feature_store_error = (
            "packaged by Databricks Feature Store and can only be registered "
            "on a Databricks cluster"
        )
        if (
            feature_store_error not in str(e)
            or not model_utils.is_unity_catalog_model(
                create_model_version_params["name"]
            )
        ):
            raise
        return _create_databricks_feature_store_model_version(
            mlflow_client=mlflow_client,
            model_name=create_model_version_params["name"],
            source=create_model_version_params["source"],
            run_id=create_model_version_params.get("run_id"),
            description=create_model_version_params.get("description"),
            tags=create_model_version_params.get("tags"),
            model_id=create_model_version_params.get("model_id"),
        )


def _create_databricks_feature_store_model_version(
        mlflow_client,
        model_name,
        source,
        run_id,
        description,
        tags,
        model_id
    ):
    """Register a Databricks Feature Store model from a local migration process.

    MLflow's Unity Catalog client intentionally rejects Feature Store-packaged
    models outside a Databricks cluster while deriving feature lineage. Migration
    still needs to perform the rest of MLflow's create, artifact-copy, and finalize
    sequence; otherwise the version remains in PENDING_REGISTRATION indefinitely.
    """
    try:
        from mlflow.protos.databricks_uc_registry_messages_pb2 import (
            CreateModelVersionRequest,
        )
        from mlflow.store._unity_catalog.registry.rest_store import (
            get_full_name_from_sc,
            get_model_version_dependencies,
            message_to_json,
            model_version_from_uc_proto,
            uc_model_version_tag_from_mlflow_tags,
        )
    except ImportError as e:
        raise MlflowExportImportException(
            "The installed MLflow version does not support local migration of "
            "Databricks Feature Store model versions. Run the import on a "
            "Databricks cluster instead."
        ) from e

    registry_client = mlflow_client._get_registry_client()
    store = registry_client.store
    required_methods = (
        "_call_endpoint",
        "_finalize_model_version",
        "_get_artifact_repo",
        "_get_run_and_headers",
        "_get_workspace_id",
        "_local_model_dir",
        "_validate_model_signature",
    )
    if any(not hasattr(store, method) for method in required_methods):
        raise MlflowExportImportException(
            "The configured registry client cannot complete local migration of "
            "a Databricks Feature Store model version. Run the import on a "
            "Databricks cluster instead."
        )

    _logger.warning(
        "MLflow cannot derive Databricks Feature Store lineage outside a "
        "Databricks cluster. Registering the migrated model without UC feature "
        "dependency lineage; run the import on a cluster when that lineage must "
        "be preserved."
    )

    full_name = get_full_name_from_sc(model_name, store.spark)
    headers, _ = store._get_run_and_headers(run_id)
    source_workspace_id = store._get_workspace_id(headers)
    mlflow_tags = [
        ModelVersionTag(key, str(value))
        for key, value in (tags or {}).items()
    ]

    with store._local_model_dir(source, None) as local_model_dir:
        store._validate_model_signature(local_model_dir)
        if hasattr(store, "_download_model_weights_if_not_saved"):
            store._download_model_weights_if_not_saved(local_model_dir)
        dependencies = get_model_version_dependencies(local_model_dir)
        request_params = dict(
            name=full_name,
            source=source,
            run_id=run_id,
            tags=uc_model_version_tag_from_mlflow_tags(mlflow_tags),
            run_tracking_server_id=source_workspace_id,
            feature_deps="",
            model_version_dependencies=dependencies,
        )
        if description is not None:
            request_params["description"] = description
        if model_id is not None:
            request_params["model_id"] = model_id
        request = CreateModelVersionRequest(**request_params)
        response = store._call_endpoint(
            CreateModelVersionRequest,
            message_to_json(request),
        )
        model_version = response.model_version
        artifact_repo = store._get_artifact_repo(model_version, full_name)
        artifact_repo.log_artifacts(local_dir=local_model_dir, artifact_path="")
        finalized = store._finalize_model_version(
            name=full_name,
            version=model_version.version,
        )
    return model_version_from_uc_proto(finalized)


def _get_model_path(src_vr):
    source = src_vr["source"]
    model_path = _extract_model_path(source)
    if not model_path:
        model_path = os.path.basename(source)
    return model_path


def _extract_model_path(source):
    """
    Extract relative path to model artifact from version source field
    :param source: 'source' field of registered model version
    :return: relative path to the model artifact
    """
    pattern = "artifacts"
    idx = source.find(pattern)
    if idx == -1:
        return None
    return source[1+idx+len(pattern):]


def _set_source_tags_for_field(dct, tags):
    set_source_tags_for_field(dct, tags)
    fmt_timestamps("creation_timestamp", dct, tags)
    fmt_timestamps("last_updated_timestamp", dct, tags)


@click.command()
@opt_model
@opt_experiment_name
@opt_input_dir
@opt_create_model
@opt_import_permissions
@opt_import_source_tags
@opt_import_stages_and_aliases
@opt_import_metadata

def main(input_dir, model, experiment_name, create_model, import_permissions, import_source_tags, import_stages_and_aliases, import_metadata):
    """
    Imports a registered model version and its run.
    """
    _logger.info("Options:")
    for k,v in locals().items():
        _logger.info(f"  {k}: {v}")
    import_model_version(
        model_name = model,
        experiment_name = experiment_name,
        input_dir = input_dir,
        create_model = create_model,
        import_permissions = import_permissions,
        import_source_tags = import_source_tags,
        import_stages_and_aliases = import_stages_and_aliases,
        import_metadata = import_metadata
    )


if __name__ == "__main__":
    main()
