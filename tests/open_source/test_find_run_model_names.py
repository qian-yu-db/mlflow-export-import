"""
Test find_artifacts.find_run_model_names()
"""

import os
import tempfile

import mlflow
from mlflow_export_import.common.find_artifacts import find_run_model_names
from tests.open_source.oss_utils_test import create_experiment
from tests.sklearn_utils import create_sklearn_model

client = mlflow.MlflowClient()


def _log_legacy_model_artifacts(model, artifact_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "model")
        mlflow.sklearn.save_model(model, model_dir)
        mlflow.log_artifacts(model_dir, artifact_path or None)


def test_no_model():
    create_experiment(client)
    with mlflow.start_run() as run:
        mlflow.set_tag("name","foo")
    model_paths = find_run_model_names(client, run.info.run_id)
    assert len(model_paths) == 0


def test_one_model_at_artifact_root():
    """ Test when model artifact root is '' """
    create_experiment(client)
    model = create_sklearn_model()
    with mlflow.start_run() as run:
        _log_legacy_model_artifacts(model, "")
    model_paths = find_run_model_names(client, run.info.run_id)
    assert len(model_paths) == 1
    assert model_paths[0] == ""


def test_one_model():
    create_experiment(client)
    model = create_sklearn_model()
    with mlflow.start_run() as run:
        _log_legacy_model_artifacts(model, "model")
    model_paths = find_run_model_names(client, run.info.run_id)
    assert len(model_paths) == 1
    assert model_paths[0] == "model"


def test_two_models():
    create_experiment(client)
    model = create_sklearn_model()
    with mlflow.start_run() as run:
        _log_legacy_model_artifacts(model, "model")
        _log_legacy_model_artifacts(model, "model-onnx")
    model_paths = find_run_model_names(client, run.info.run_id)
    assert len(model_paths) == 2
    assert model_paths[0] == "model"
    assert model_paths[1] == "model-onnx"


def test_two_models_nested():
    create_experiment(client)
    model = create_sklearn_model()
    with mlflow.start_run() as run:
        _log_legacy_model_artifacts(model, "model")
        _log_legacy_model_artifacts(model, "other_models/model-onnx")
    model_paths = find_run_model_names(client, run.info.run_id)
    assert len(model_paths) == 2
    assert model_paths[0] == "model"
    assert model_paths[1] == "other_models/model-onnx"
