import importlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

from mlflow.exceptions import MlflowException
from mlflow.entities.model_registry import ModelVersion


import_model_version_module = importlib.import_module(
    "mlflow_export_import.model_version.import_model_version"
)
export_model_version_module = importlib.import_module(
    "mlflow_export_import.model_version.export_model_version"
)
copy_model_version_module = importlib.import_module(
    "mlflow_export_import.copy.copy_model_version"
)
import_model_module = importlib.import_module("mlflow_export_import.model.import_model")


class _ModelVersionClient:
    def get_logged_model(self, model_id):
        assert model_id == "destination-model"
        return SimpleNamespace(
            name="model",
            artifact_location="dbfs:/destination/models/destination-model/artifacts"
        )


def test_import_model_version_uses_destination_logged_model_source(tmp_path, monkeypatch):
    version = {
        "mlflow": {
            "model_version": {
                "name": "catalog.schema.source_model",
                "version": "1",
                "source": "models:/source-model",
                "run_id": "source-run",
                "tags": {},
                "aliases": [],
                "current_stage": "None",
            }
        }
    }
    (tmp_path / "version.json").write_text(json.dumps(version), encoding="utf-8")
    (tmp_path / "run").mkdir()
    client = _ModelVersionClient()
    destination_run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="destination-run",
            artifact_uri="dbfs:/destination/runs/destination-run/artifacts",
        ),
        outputs=SimpleNamespace(
            model_outputs=[SimpleNamespace(model_id="destination-model")]
        ),
    )

    def import_run(**kwargs):
        model_id_map = kwargs.get("logged_model_id_map")
        if model_id_map is not None:
            model_id_map["source-model"] = "destination-model"
        return destination_run, None

    captured = {}

    def import_version(mlflow_client, **kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(import_model_version_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(import_model_version_module, "import_run", import_run)
    monkeypatch.setattr(import_model_version_module, "_import_model_version", import_version)

    import_model_version_module.import_model_version(
        model_name="catalog.schema.destination_model",
        experiment_name="destination-experiment",
        input_dir=str(tmp_path),
        mlflow_client=client,
    )

    assert captured["dst_source"] == (
        "dbfs:/destination/models/destination-model/artifacts"
    )
    assert captured["model_id"] == "destination-model"


def test_import_model_version_resolves_legacy_source_to_destination_logged_model(
    tmp_path, monkeypatch
):
    version = {
        "mlflow": {
            "model_version": {
                "name": "source-model",
                "version": "1",
                "source": "/source/source-run/artifacts/model",
                "run_id": "source-run",
                "tags": {},
                "aliases": [],
                "current_stage": "None",
            }
        }
    }
    (tmp_path / "version.json").write_text(json.dumps(version), encoding="utf-8")
    (tmp_path / "run").mkdir()
    destination_run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="destination-run",
            artifact_uri="dbfs:/destination/runs/destination-run/artifacts",
        ),
        outputs=SimpleNamespace(
            model_outputs=[SimpleNamespace(model_id="destination-model")]
        ),
    )

    class Client(_ModelVersionClient):
        def get_run(self, run_id):
            assert run_id == "destination-run"
            return destination_run

    captured = {}
    monkeypatch.setattr(import_model_version_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(
        import_model_version_module,
        "import_run",
        lambda **kwargs: (destination_run, None),
    )
    monkeypatch.setattr(
        import_model_version_module,
        "_import_model_version",
        lambda mlflow_client, **kwargs: captured.update(kwargs) or kwargs,
    )

    import_model_version_module.import_model_version(
        model_name="destination-model",
        experiment_name="destination-experiment",
        input_dir=str(tmp_path),
        mlflow_client=Client(),
    )

    assert captured["dst_source"] == (
        "dbfs:/destination/models/destination-model/artifacts"
    )
    assert captured["model_id"] == "destination-model"


class _ExportModelVersionClient:
    def __init__(self):
        self.version = ModelVersion(
            name="catalog.schema.source_model",
            version="1",
            creation_timestamp=1,
            source="models:/source-model",
            run_id="source-run",
        )

    def get_registered_model(self, model_name):
        return SimpleNamespace(name=model_name)

    def get_model_version(self, model_name, version):
        return self.version


def test_export_model_version_keeps_logged_model_inside_run_export(tmp_path, monkeypatch):
    client = _ExportModelVersionClient()
    run = SimpleNamespace(info=SimpleNamespace(experiment_id="source-experiment"))

    def export_run(**kwargs):
        model_dir = tmp_path / "run" / "source-model"
        model_dir.mkdir(parents=True)
        (model_dir / "logged_model.json").write_text("{}", encoding="utf-8")
        return run

    def export_logged_model(model_id, output_dir, mlflow_client):
        duplicate_dir = tmp_path / "source-run"
        duplicate_dir.mkdir()
        (duplicate_dir / "logged_model.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(export_model_version_module, "export_run", export_run)
    monkeypatch.setattr(
        export_model_version_module,
        "export_logged_model",
        export_logged_model,
        raising=False,
    )
    monkeypatch.setattr(
        export_model_version_module,
        "_export_registered_model",
        lambda *args: None,
    )
    monkeypatch.setattr(
        export_model_version_module,
        "_export_experiment",
        lambda *args: None,
    )

    export_model_version_module.export_model_version(
        model_name="catalog.schema.source_model",
        version="1",
        output_dir=str(tmp_path),
        mlflow_client=client,
    )

    assert (tmp_path / "run" / "source-model" / "logged_model.json").exists()
    assert not (tmp_path / "source-run" / "logged_model.json").exists()


class _CopyDestinationClient:
    def __init__(self):
        self.created = None

    def get_logged_model(self, model_id):
        assert model_id == "destination-model"
        return SimpleNamespace(
            artifact_location="dbfs:/destination/models/destination-model/artifacts"
        )

    def create_model_version(self, **kwargs):
        self.created = kwargs
        return SimpleNamespace(name=kwargs["name"], version="2")

    def get_model_version(self, name, version):
        return SimpleNamespace(name=name, version=version)


def test_copy_model_version_registers_destination_logged_model(monkeypatch):
    source_version = SimpleNamespace(
        name="catalog.schema.source_model",
        version="1",
        run_id="source-run",
        source="models:/source-model",
        tags={},
        description="source description",
    )
    destination_run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="destination-run",
            artifact_uri="dbfs:/destination/runs/destination-run/artifacts",
        )
    )
    destination_client = _CopyDestinationClient()

    def copy_run(source_run_id, experiment_name, source_client, destination, logged_model_id_map=None):
        if logged_model_id_map is not None:
            logged_model_id_map["source-model"] = "destination-model"
        return destination_run

    monkeypatch.setattr(copy_model_version_module.copy_run, "_copy", copy_run)
    monkeypatch.setattr(
        copy_model_version_module,
        "_create_model_version_with_feature_store_fallback",
        lambda client, params: client.create_model_version(**params),
    )

    copy_model_version_module._copy_model_version(
        source_version,
        "catalog.schema.destination_model",
        "destination-experiment",
        SimpleNamespace(),
        destination_client,
    )

    assert destination_client.created["source"] == (
        "dbfs:/destination/models/destination-model/artifacts"
    )
    assert destination_client.created["model_id"] == "destination-model"
    assert destination_client.created["run_id"] == "destination-run"


class _RegisteredModelImportClient:
    def get_run(self, run_id):
        return SimpleNamespace(
            info=SimpleNamespace(
                run_id=run_id,
                artifact_uri="dbfs:/destination/runs/destination-run/artifacts",
            ),
            outputs=SimpleNamespace(
                model_outputs=[SimpleNamespace(model_id="destination-model")]
            ),
        )

    def get_logged_model(self, model_id):
        assert model_id == "destination-model"
        return SimpleNamespace(
            name="model",
            artifact_location="dbfs:/destination/models/destination-model/artifacts"
        )


def _capture_registered_model_version_import(monkeypatch):
    captured = {}

    def import_version(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(import_model_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(import_model_module, "_import_model_version", import_version)
    return captured


def test_model_importer_resolves_logged_model_with_destination_client(monkeypatch):
    client = _RegisteredModelImportClient()
    captured = _capture_registered_model_version_import(monkeypatch)
    importer = import_model_module.ModelImporter(mlflow_client=client)

    importer.import_version(
        "destination-registered-model",
        {
            "source": "/source/source-run/artifacts/model",
            "run_id": "source-run",
        },
        "destination-run",
    )

    assert captured["dst_source"] == (
        "dbfs:/destination/models/destination-model/artifacts"
    )


def test_bulk_model_importer_resolves_logged_model_with_destination_client(monkeypatch):
    client = _RegisteredModelImportClient()
    captured = _capture_registered_model_version_import(monkeypatch)
    importer = import_model_module.BulkModelImporter(
        run_info_map={
            "source-run": SimpleNamespace(
                run_id="destination-run",
                artifact_uri="dbfs:/destination/runs/destination-run/artifacts",
            )
        },
        mlflow_client=client,
    )

    importer.import_version(
        "destination-registered-model",
        {
            "source": "/source/source-run/artifacts/model",
            "run_id": "source-run",
        },
        "destination-run",
    )

    assert captured["dst_source"] == (
        "dbfs:/destination/models/destination-model/artifacts"
    )


def test_import_model_version_falls_back_for_local_feature_store_registration(
    monkeypatch,
):
    class Client:
        def create_model_version(self, **kwargs):
            raise MlflowException(
                "This model was packaged by Databricks Feature Store and can only "
                "be registered on a Databricks cluster."
            )

        def get_model_version(self, name, version):
            return SimpleNamespace(name=name, version=version)

    fallback_calls = []
    expected = SimpleNamespace(
        name="catalog.schema.destination_model",
        version="1",
    )

    monkeypatch.setattr(
        import_model_version_module,
        "MlflowTrackingUriTweak",
        lambda client: nullcontext(),
    )
    monkeypatch.setattr(
        import_model_version_module,
        "_create_databricks_feature_store_model_version",
        lambda **kwargs: fallback_calls.append(kwargs) or expected,
    )

    result = import_model_version_module._import_model_version(
        mlflow_client=Client(),
        model_name="catalog.schema.destination_model",
        src_vr={
            "tags": {"source": "test"},
            "aliases": [],
            "description": "feature store model",
        },
        dst_run_id="destination-run",
        dst_source="dbfs:/destination/logged_models/destination-model/artifacts",
        model_id="destination-model",
    )

    assert result.name == expected.name
    assert result.version == expected.version
    assert len(fallback_calls) == 1
    assert fallback_calls[0]["model_name"] == "catalog.schema.destination_model"
    assert fallback_calls[0]["source"] == (
        "dbfs:/destination/logged_models/destination-model/artifacts"
    )
    assert fallback_calls[0]["run_id"] == "destination-run"
    assert fallback_calls[0]["description"] == "feature store model"
    assert fallback_calls[0]["tags"] == {"source": "test"}
    assert fallback_calls[0]["model_id"] == "destination-model"


def test_model_importer_forwards_await_creation_timeout(monkeypatch):
    client = _RegisteredModelImportClient()
    captured = _capture_registered_model_version_import(monkeypatch)
    importer = import_model_module.ModelImporter(
        mlflow_client=client,
        await_creation_for=123,
    )

    importer.import_version(
        "destination-registered-model",
        {
            "source": "/source/source-run/artifacts/model",
            "run_id": "source-run",
        },
        "destination-run",
    )

    assert captured["await_creation_for"] == 123


def test_feature_store_fallback_copies_artifacts_and_finalizes(tmp_path):
    from mlflow.protos.databricks_uc_registry_messages_pb2 import (
        ModelVersion as UcModelVersion,
    )

    (tmp_path / "MLmodel").write_text(
        "artifact_path: model\n"
        "flavors:\n"
        "  python_function:\n"
        "    loader_module: mlflow.pyfunc.model\n"
        "mlflow_version: 3.9.0\n"
        "model_uuid: destination-model\n",
        encoding="utf-8",
    )
    calls = {}

    class ArtifactRepository:
        def log_artifacts(self, local_dir, artifact_path):
            calls["artifact_upload"] = (local_dir, artifact_path)

    class Store:
        spark = None

        def _get_run_and_headers(self, run_id):
            calls["run_id"] = run_id
            return {}, SimpleNamespace()

        def _get_workspace_id(self, headers):
            return "destination-workspace"

        def _local_model_dir(self, source, local_model_path):
            calls["source"] = source
            return nullcontext(str(tmp_path))

        def _validate_model_signature(self, local_model_dir):
            calls["validated"] = local_model_dir

        def _download_model_weights_if_not_saved(self, local_model_dir):
            calls["downloaded_weights"] = local_model_dir

        def _call_endpoint(self, request_type, request_body):
            calls["request_type"] = request_type
            calls["request_body"] = json.loads(request_body)
            return SimpleNamespace(
                model_version=UcModelVersion(
                    name="catalog.schema.destination_model",
                    version="1",
                    source="dbfs:/destination/logged_models/model/artifacts",
                    run_id="destination-run",
                )
            )

        def _get_artifact_repo(self, model_version, full_name):
            calls["artifact_repo"] = (model_version.version, full_name)
            return ArtifactRepository()

        def _finalize_model_version(self, name, version):
            calls["finalized"] = (name, version)
            return UcModelVersion(
                name=name,
                version=version,
                source="dbfs:/destination/logged_models/model/artifacts",
                run_id="destination-run",
            )

    store = Store()
    client = SimpleNamespace(
        _get_registry_client=lambda: SimpleNamespace(store=store)
    )
    result = (
        import_model_version_module._create_databricks_feature_store_model_version(
            mlflow_client=client,
            model_name="catalog.schema.destination_model",
            source="dbfs:/destination/logged_models/model/artifacts",
            run_id="destination-run",
            description="feature store model",
            tags={"source": "test"},
            model_id="destination-model",
        )
    )

    assert result.name == "catalog.schema.destination_model"
    assert result.version == "1"
    assert calls["request_body"]["run_tracking_server_id"] == (
        "destination-workspace"
    )
    assert calls["request_body"]["feature_deps"] == ""
    assert calls["request_body"]["model_id"] == "destination-model"
    assert calls["artifact_upload"] == (str(tmp_path), "")
    assert calls["finalized"] == ("catalog.schema.destination_model", "1")
