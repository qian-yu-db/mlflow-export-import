
import os

import mlflow

def _extract_model_id(source):

    idx = source.find("models")
    if idx == 0:
        return source.split('models:/')[1]
    else:
        return source.split("models/")[1].split("/")[0]

def _get_logged_model_artifact_path(model_id, mlflow_client=None):
    mlflow_client = mlflow_client or mlflow.MlflowClient()
    return mlflow_client.get_logged_model(model_id).artifact_location


def find_destination_logged_model_id(mlflow_client, dst_run, source):
    outputs = getattr(getattr(dst_run, "outputs", None), "model_outputs", []) or []
    if not outputs:
        return None
    if source.startswith("models:/"):
        return outputs[0].model_id

    source_name = os.path.basename(source.rstrip("/"))
    matching_ids = [
        output.model_id
        for output in outputs
        if mlflow_client.get_logged_model(output.model_id).name == source_name
    ]
    return matching_ids[0] if len(matching_ids) == 1 else None
