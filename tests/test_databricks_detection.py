from mlflow_export_import.common import MlflowExportImportException
from mlflow_export_import.client import client_utils
from mlflow_export_import.common import utils
from mlflow_export_import.common import mlflow_utils
from mlflow_export_import.common import model_utils
from mlflow_export_import.copy import copy_model_version
from mlflow_export_import.experiment import nested_runs_utils
from mlflow_export_import.run.run_data_importer import import_run_data
from mlflow_export_import.tools import tools_utils
import importlib


import_run_module = importlib.import_module("mlflow_export_import.run.import_run")


class _DatabricksClient:
    def __init__(self, api_uri, available):
        self.api_uri = api_uri
        self.available = available

    def get_api_uri(self):
        return self.api_uri

    def get(self, resource):
        if not self.available:
            raise MlflowExportImportException("not a Databricks endpoint", http_status_code=404)
        return {"node_types": []}


def test_calling_databricks_scopes_detection_to_explicit_client(monkeypatch):
    monkeypatch.setattr(utils, "_calling_databricks", False)
    databricks_client = _DatabricksClient("https://destination/api/2.0", True)
    non_databricks_client = _DatabricksClient("https://mlflow/api/2.0", False)

    assert utils.calling_databricks(databricks_client) is True
    assert utils.calling_databricks(non_databricks_client) is False


def test_calling_databricks_treats_explicit_none_as_non_databricks(monkeypatch):
    monkeypatch.setattr(utils, "_calling_databricks", True)

    assert utils.calling_databricks(None) is False


class _RunClient:
    def __init__(self):
        self.tags = []

    def log_batch(self, run_id, params=None, metrics=None, tags=None):
        if tags:
            self.tags.extend(tags)


def test_import_run_data_filters_immutable_tags_for_destination_client(monkeypatch):
    monkeypatch.setattr(utils, "_calling_databricks", False)
    destination = _DatabricksClient("https://destination-tags/api/2.0", True)
    run_client = _RunClient()
    run_data = {
        "params": {},
        "metrics": {},
        "tags": {"mlflow.user": "source-user", "team": "ml-platform"},
    }

    import_run_data(
        run_client,
        run_data,
        "destination-run",
        import_source_tags=False,
        src_user_id="source-user",
        use_src_user_id=False,
        in_databricks=False,
        dbx_client=destination,
    )

    assert {tag.key: tag.value for tag in run_client.tags} == {"team": "ml-platform"}


def test_import_run_data_accepts_explicit_databricks_destination_override(monkeypatch):
    monkeypatch.setattr(utils, "_calling_databricks", False)
    run_client = _RunClient()
    run_data = {
        "params": {},
        "metrics": {},
        "tags": {"mlflow.user": "source-user", "team": "ml-platform"},
    }

    import_run_data(
        run_client,
        run_data,
        "destination-run",
        import_source_tags=False,
        src_user_id="source-user",
        use_src_user_id=False,
        in_databricks=False,
        is_databricks=True,
    )

    assert {tag.key: tag.value for tag in run_client.tags} == {"team": "ml-platform"}


def test_set_experiment_detects_databricks_from_destination_client(monkeypatch):
    destination = _DatabricksClient("https://destination-experiment/api/2.0", True)
    detected_clients = []

    def calling_databricks(client=None):
        detected_clients.append(client)
        return True

    monkeypatch.setattr(mlflow_utils.utils, "calling_databricks", calling_databricks)
    monkeypatch.setattr(mlflow_utils, "create_workspace_dir", lambda *args: None)
    client = type(
        "Client",
        (),
        {
            "create_experiment": lambda self, name, tags: "destination-experiment",
            "get_experiment": lambda self, experiment_id: type(
                "Experiment",
                (),
                {
                    "name": "/Shared/destination-experiment",
                    "artifact_location": "dbfs:/destination-experiment",
                },
            )(),
        },
    )()

    mlflow_utils.set_experiment(
        client,
        destination,
        "/Shared/destination-experiment",
        {"team": "ml-platform"},
    )

    assert detected_clients == [destination]


def test_import_run_notebook_detection_uses_destination_client(tmp_path, monkeypatch):
    (tmp_path / "run.json").write_text(
        '{"mlflow":{"info":{"run_id":"source-run","user_id":"source-user",'
        '"status":"FINISHED","lifecycle_stage":"active"},"params":{},'
        '"metrics":{},"tags":{},"inputs":{"dataset_inputs":[]}}}',
        encoding="utf-8",
    )
    destination = _DatabricksClient("https://destination-notebook/api/2.0", True)
    run = type("Run", (), {"info": type("Info", (), {"run_id": "destination-run"})()})()
    client = type(
        "Client",
        (),
        {
            "create_run": lambda self, experiment_id: run,
            "get_run": lambda self, run_id: run,
            "set_terminated": lambda *args: None,
        },
    )()
    detected_clients = []
    uploaded = []

    monkeypatch.setattr(import_run_module, "create_http_client", lambda client: None)
    monkeypatch.setattr(import_run_module, "create_dbx_client", lambda client: destination)
    monkeypatch.setattr(
        import_run_module.mlflow_utils,
        "set_experiment",
        lambda *args, **kwargs: type(
            "Experiment", (), {"experiment_id": "destination-experiment", "name": "destination"}
        )(),
    )
    monkeypatch.setattr(import_run_module.run_data_importer, "import_run_data", lambda *args: None)
    monkeypatch.setattr(import_run_module.run_utils, "update_mlmodel_run_id", lambda *args: None)
    monkeypatch.setattr(
        import_run_module.utils,
        "calling_databricks",
        lambda dbx_client=None: detected_clients.append(dbx_client) or True,
    )
    monkeypatch.setattr(
        import_run_module,
        "_upload_databricks_notebook",
        lambda *args: uploaded.append(args),
    )

    import_run_module.import_run(
        input_dir=str(tmp_path),
        experiment_name="destination",
        dst_notebook_dir="/Shared/notebooks",
        mlflow_client=client,
    )

    assert detected_clients == [destination]
    assert len(uploaded) == 1


def test_registered_model_permissions_detection_uses_model_client(monkeypatch):
    destination = _DatabricksClient("https://destination-model/api/2.0", True)
    requested_resources = []

    class HttpClient:
        def get(self, resource, params):
            requested_resources.append(resource)
            if resource == "databricks/registered-models/get":
                return {
                    "registered_model_databricks": {
                        "id": "destination-model-id",
                        "name": "destination-model",
                        "creation_timestamp": 1,
                        "last_updated_timestamp": 2,
                    }
                }
            return {
                "registered_model": {
                    "name": "wrong-branch",
                    "creation_timestamp": 1,
                    "last_updated_timestamp": 2,
                }
            }

    monkeypatch.setattr(model_utils, "create_http_client", lambda *args: HttpClient())
    monkeypatch.setattr(model_utils, "create_dbx_client", lambda client: destination)
    monkeypatch.setattr(
        model_utils.utils,
        "calling_databricks",
        lambda dbx_client=None: dbx_client is destination,
    )
    monkeypatch.setattr(
        model_utils.ws_permissions_utils,
        "get_model_permissions_by_id",
        lambda dbx_client, model_id: [{"group_name": "users"}],
    )

    exported = model_utils.get_registered_model(
        object(), "destination-model", get_permissions=True
    )

    assert exported["name"] == "destination-model"
    assert exported["permissions"] == [{"group_name": "users"}]
    assert requested_resources == ["databricks/registered-models/get"]


def test_copy_permissions_detection_uses_destination_client(monkeypatch):
    destination = _DatabricksClient("https://destination-copy/api/2.0", True)
    updates = []
    source_client = object()
    destination_client = object()

    monkeypatch.setattr(
        copy_model_version.copy_utils,
        "create_registered_model",
        lambda *args: False,
    )
    monkeypatch.setattr(
        client_utils, "create_dbx_client", lambda client: destination
    )
    monkeypatch.setattr(
        copy_model_version.utils,
        "calling_databricks",
        lambda dbx_client=None: dbx_client is destination,
    )
    monkeypatch.setattr(
        copy_model_version.ws_permissions_utils,
        "get_model_permissions_by_name",
        lambda *args: [{"group_name": "users"}],
    )
    monkeypatch.setattr(
        copy_model_version.model_utils,
        "update_model_permissions",
        lambda *args: updates.append(args),
    )

    copy_model_version._create_registered_model(
        source_client,
        "source-model",
        destination_client,
        "destination-model",
        copy_permissions=True,
    )

    assert updates == [
        (
            destination_client,
            destination,
            "destination-model",
            [{"group_name": "users"}],
        )
    ]


def test_nested_run_detection_uses_tracking_client(monkeypatch):
    tracking_client = object()
    destination = _DatabricksClient("https://destination-runs/api/2.0", True)
    root_run = object()

    monkeypatch.setattr(
        client_utils, "create_dbx_client", lambda client: destination
    )
    monkeypatch.setattr(
        nested_runs_utils.utils,
        "calling_databricks",
        lambda dbx_client=None: dbx_client is destination,
    )
    monkeypatch.setattr(
        nested_runs_utils,
        "get_nested_runs_by_rootRunId",
        lambda client, runs: ["databricks-descendant"],
    )

    nested = nested_runs_utils.get_nested_runs(tracking_client, [root_run])

    assert nested == ["databricks-descendant"]


def test_model_version_search_detection_uses_tracking_client(monkeypatch):
    tracking_client = object()
    destination = _DatabricksClient("https://destination-search/api/2.0", True)

    monkeypatch.setattr(
        client_utils, "create_dbx_client", lambda client: destination
    )
    monkeypatch.setattr(
        tools_utils.utils,
        "calling_databricks",
        lambda dbx_client=None: dbx_client is destination,
    )
    monkeypatch.setattr(
        tools_utils,
        "SearchRegisteredModelsIterator",
        lambda *args, **kwargs: [type("Model", (), {"name": "destination-model"})()],
    )
    monkeypatch.setattr(
        tools_utils,
        "SearchModelVersionsIterator",
        lambda client, filter: (
            ["destination-version"]
            if filter == "name='destination-model'"
            else ["wrong-branch"]
        ),
    )

    versions = tools_utils.search_model_versions(tracking_client, "tags.team = 'ml'")

    assert versions == ["destination-version"]
