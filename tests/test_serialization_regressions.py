import json

import pytest

from mlflow.entities import LoggedModelParameter, Metric
from mlflow.entities.model_registry import ModelVersion

from mlflow_export_import.common.model_utils import model_version_to_dict
from mlflow_export_import.common.io_utils import write_file


def test_model_version_to_dict_serializes_logged_model_metadata():
    version = ModelVersion(
        name="catalog.schema.model",
        version="1",
        creation_timestamp=1,
        source="models:/m-source",
        run_id="source-run",
        params=[LoggedModelParameter("max_depth", "4")],
        metrics=[Metric("accuracy", 0.95, 2, 3)],
    )

    serialized = json.loads(json.dumps(model_version_to_dict(version)))

    assert serialized["params"] == [{"key": "max_depth", "value": "4"}]
    assert serialized["metrics"] == [
        {
            "key": "accuracy",
            "value": 0.95,
            "timestamp": 2,
            "step": 3,
            "model_id": None,
            "dataset_name": None,
            "dataset_digest": None,
            "run_id": None,
        }
    ]


def test_write_file_preserves_existing_json_when_serialization_fails(tmp_path):
    path = tmp_path / "model.json"
    original = '{"state": "valid"}\n'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(TypeError, match="not JSON serializable"):
        write_file(str(path), {"invalid": object()})

    assert path.read_text(encoding="utf-8") == original
