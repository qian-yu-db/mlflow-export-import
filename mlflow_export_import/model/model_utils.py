
import os

import mlflow

def _extract_model_id(source):
    if source.startswith("models:/"):
        return source[len("models:/"):]

    normalized_source = source.rstrip("/")
    marker = "/models/"
    if marker in normalized_source and normalized_source.endswith("/artifacts"):
        model_id = normalized_source.rsplit(marker, 1)[1][:-len("/artifacts")]
        if model_id and "/" not in model_id:
            return model_id

    raise ValueError(f"Not a logged-model source: {source}")


def _is_logged_model_source(source):
    try:
        _extract_model_id(source)
        return True
    except (AttributeError, ValueError):
        return False

def _get_logged_model_artifact_path(model_id, mlflow_client=None):
    mlflow_client = mlflow_client or mlflow.MlflowClient()
    return mlflow_client.get_logged_model(model_id).artifact_location


def find_destination_logged_model_id(mlflow_client, dst_run, source):
    outputs = getattr(getattr(dst_run, "outputs", None), "model_outputs", []) or []
    if not outputs:
        return None
    if _is_logged_model_source(source):
        return outputs[0].model_id

    source_name = os.path.basename(source.rstrip("/"))
    matching_ids = [
        output.model_id
        for output in outputs
        if mlflow_client.get_logged_model(output.model_id).name == source_name
    ]
    return matching_ids[0] if len(matching_ids) == 1 else None
