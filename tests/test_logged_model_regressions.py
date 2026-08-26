import importlib
import json
from types import SimpleNamespace

import pytest
from mlflow.entities import Experiment, LoggedModelInput, LoggedModelOutput


import_run_module = importlib.import_module("mlflow_export_import.run.import_run")
export_run_module = importlib.import_module("mlflow_export_import.run.export_run")
import_experiment_module = importlib.import_module(
    "mlflow_export_import.experiment.import_experiment"
)
import_model_module = importlib.import_module("mlflow_export_import.model.import_model")
export_experiment_module = importlib.import_module(
    "mlflow_export_import.experiment.export_experiment"
)
export_model_module = importlib.import_module("mlflow_export_import.model.export_model")
export_logged_models_module = importlib.import_module(
    "mlflow_export_import.bulk.export_logged_models"
)
import_logged_model_module = importlib.import_module(
    "mlflow_export_import.logged_model.import_logged_model"
)


class _RunClient:
    def __init__(self):
        self.run = SimpleNamespace(
            info=SimpleNamespace(run_id="destination-run"),
            outputs=SimpleNamespace(model_outputs=[]),
        )

    def create_run(self, experiment_id):
        return self.run

    def get_run(self, run_id):
        return self.run

    def set_terminated(self, run_id, status):
        pass

    def delete_run(self, run_id):
        pass


def _write_run_export(
    tmp_path, *, model_inputs=None, model_outputs=None, lifecycle_stage="active"
):
    exported = {
        "mlflow": {
            "info": {
                "run_id": "source-run",
                "user_id": "source-user",
                "status": "FINISHED",
                "lifecycle_stage": lifecycle_stage,
            },
            "params": {},
            "metrics": {},
            "tags": {},
            "inputs": {
                "dataset_inputs": [],
                "model_inputs": model_inputs or [],
            },
            "outputs": {"model_outputs": model_outputs or []},
        }
    }
    (tmp_path / "run.json").write_text(json.dumps(exported), encoding="utf-8")


def _write_logged_model_payload(tmp_path, model_id):
    model_dir = tmp_path / model_id
    model_dir.mkdir()
    (model_dir / "logged_model.json").write_text("{}", encoding="utf-8")


def _configure_run_import(monkeypatch, client):
    monkeypatch.setattr(import_run_module, "create_http_client", lambda mlflow_client: None)
    monkeypatch.setattr(import_run_module, "create_dbx_client", lambda mlflow_client: None)
    monkeypatch.setattr(
        import_run_module.mlflow_utils,
        "set_experiment",
        lambda mlflow_client, dbx_client, name, **kwargs: SimpleNamespace(
            experiment_id="destination-experiment",
            name=name,
        ),
    )
    monkeypatch.setattr(import_run_module.run_data_importer, "import_run_data", lambda *args: None)
    monkeypatch.setattr(import_run_module.run_utils, "update_mlmodel_run_id", lambda *args: None)
    monkeypatch.setattr(import_run_module.utils, "calling_databricks", lambda *args: False)

    def import_logged_model(**kwargs):
        imported = SimpleNamespace(model_id=f"destination-{kwargs['input_dir'].split('/')[-1]}")
        client.run.outputs.model_outputs.append(imported)
        return imported

    monkeypatch.setattr(import_run_module, "import_logged_model", import_logged_model)


def test_import_run_imports_logged_model_outputs_by_default(tmp_path, monkeypatch):
    _write_run_export(
        tmp_path,
        model_outputs=[{"model_id": "source-model", "step": 0}],
    )
    _write_logged_model_payload(tmp_path, "source-model")
    client = _RunClient()
    _configure_run_import(monkeypatch, client)

    imported_run, _ = import_run_module.import_run(
        input_dir=str(tmp_path),
        experiment_name="destination-experiment",
        mlflow_client=client,
    )

    assert [model.model_id for model in imported_run.outputs.model_outputs] == [
        "destination-source-model"
    ]


def test_import_run_records_destination_logged_model_ids(tmp_path, monkeypatch):
    _write_run_export(
        tmp_path,
        model_outputs=[{"model_id": "source-model", "step": 0}],
    )
    _write_logged_model_payload(tmp_path, "source-model")
    client = _RunClient()
    _configure_run_import(monkeypatch, client)
    logged_model_id_map = {}

    import_run_module.import_run(
        input_dir=str(tmp_path),
        experiment_name="destination-experiment",
        mlflow_client=client,
        logged_model_id_map=logged_model_id_map,
    )

    assert logged_model_id_map == {"source-model": "destination-source-model"}


def test_import_run_uses_destination_run_id_for_logged_model_inputs(tmp_path, monkeypatch):
    _write_run_export(
        tmp_path,
        model_inputs=[{"model_id": "source-input-model"}],
    )
    _write_logged_model_payload(tmp_path, "source-input-model")
    client = _RunClient()
    _configure_run_import(monkeypatch, client)
    logged_model_id_map = {}

    import_run_module.import_run(
        input_dir=str(tmp_path),
        experiment_name="destination-experiment",
        mlflow_client=client,
        logged_model_id_map=logged_model_id_map,
    )

    assert logged_model_id_map == {
        "source-input-model": "destination-source-input-model"
    }


def test_import_run_skips_references_without_logged_model_payloads(
    tmp_path, monkeypatch
):
    _write_run_export(
        tmp_path,
        model_inputs=[{"model_id": "missing-input-model"}],
        model_outputs=[{"model_id": "missing-output-model", "step": 0}],
    )
    client = _RunClient()
    _configure_run_import(monkeypatch, client)

    def import_logged_model(*, input_dir, **kwargs):
        with open(f"{input_dir}/logged_model.json", encoding="utf-8"):
            pytest.fail("the fixture intentionally has no logged-model payload")

    monkeypatch.setattr(import_run_module, "import_logged_model", import_logged_model)
    logged_model_id_map = {}

    imported_run, _ = import_run_module.import_run(
        input_dir=str(tmp_path),
        experiment_name="destination-experiment",
        mlflow_client=client,
        logged_model_id_map=logged_model_id_map,
    )

    assert imported_run.info.run_id == "destination-run"
    assert logged_model_id_map == {}


def test_import_logged_model_uses_input_entity_without_step(tmp_path, monkeypatch):
    (tmp_path / "logged_model.json").write_text(
        json.dumps(
            {
                "mlflow": {
                    "name": "input-model",
                    "tags": {},
                    "params": [],
                    "metrics": [],
                    "model_type": None,
                    "source_run_id": "source-run",
                    "status": "READY",
                }
            }
        ),
        encoding="utf-8",
    )
    captured_inputs = []

    class Client:
        def create_logged_model(self, **kwargs):
            return SimpleNamespace(model_id="destination-model")

        def log_inputs(self, run_id, models):
            captured_inputs.extend(models)

        def log_batch(self, run_id, metrics):
            pass

        def log_outputs(self, run_id, models):
            pytest.fail("model inputs must not be logged as model outputs")

        def finalize_logged_model(self, model_id, status):
            pass

        def set_terminated(self, run_id, status):
            pass

    monkeypatch.setattr(import_logged_model_module, "has_logged_model_support", lambda: True)
    monkeypatch.setattr(import_logged_model_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(
        import_logged_model_module.mlflow_utils,
        "set_experiment",
        lambda *args: SimpleNamespace(
            experiment_id="destination-experiment", name="destination"
        ),
    )

    import_logged_model_module.import_logged_model(
        input_dir=str(tmp_path),
        experiment_name="destination",
        run_id="destination-run",
        model_type="input",
        mlflow_client=Client(),
    )

    assert captured_inputs == [LoggedModelInput("destination-model")]


class _ExportRunClient:
    def __init__(self):
        model_output = LoggedModelOutput("source-model", step=0)
        self.run = SimpleNamespace(
            info=SimpleNamespace(
                run_id="source-run",
                experiment_id="source-experiment",
                lifecycle_stage="active",
                start_time=1,
                end_time=2,
            ),
            data=SimpleNamespace(params={}, metrics={}, tags={}),
            inputs=SimpleNamespace(dataset_inputs=[], model_inputs=[]),
            outputs=SimpleNamespace(model_outputs=[model_output]),
        )

    def get_run(self, run_id):
        return self.run

    def list_artifacts(self, run_id):
        return []


def test_export_run_exports_logged_model_outputs_by_default(tmp_path, monkeypatch):
    client = _ExportRunClient()
    monkeypatch.setattr(export_run_module, "create_dbx_client", lambda mlflow_client: None)

    def export_logged_model(model_id, output_dir, mlflow_client):
        model_dir = tmp_path / model_id
        model_dir.mkdir()
        (model_dir / "logged_model.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(export_run_module, "export_logged_model", export_logged_model)

    export_run_module.export_run(
        run_id="source-run",
        output_dir=str(tmp_path),
        mlflow_client=client,
        raise_exception=True,
    )

    assert (tmp_path / "source-model" / "logged_model.json").exists()


def test_import_experiment_imports_each_logged_model_once(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "source-run"
    run_dir.mkdir(parents=True)
    (tmp_path / "experiment.json").write_text(
        json.dumps(
            {
                "info": {"failed_runs": []},
                "mlflow": {
                    "experiment": {"tags": {}},
                    "runs": ["source-run"],
                    "logged_models": ["source-model"],
                },
            }
        ),
        encoding="utf-8",
    )
    _write_run_export(
        run_dir,
        model_outputs=[{"model_id": "source-model", "step": 0}],
    )

    import_counts = {"source-model": 0}

    def import_run(*, import_logged_models=True, **kwargs):
        if import_logged_models:
            import_counts["source-model"] += 1
        return SimpleNamespace(info=SimpleNamespace(run_id="destination-run")), None

    def import_logged_model(*, input_dir, **kwargs):
        import_counts[input_dir.split("/")[-1]] += 1

    client = SimpleNamespace(set_terminated=lambda *args: None)
    monkeypatch.setattr(import_experiment_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(
        import_experiment_module.mlflow_utils,
        "set_experiment",
        lambda *args: SimpleNamespace(name="destination", experiment_id="1"),
    )
    monkeypatch.setattr(import_experiment_module, "import_run", import_run)
    monkeypatch.setattr(
        import_experiment_module, "import_logged_model", import_logged_model
    )
    monkeypatch.setattr(import_experiment_module.utils, "nested_tags", lambda *args: None)

    import_experiment_module.import_experiment(
        experiment_name="destination",
        input_dir=str(tmp_path),
        mlflow_client=client,
    )

    assert import_counts == {"source-model": 1}


def test_import_experiment_imports_model_inputs_without_step(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "consumer-run"
    run_dir.mkdir(parents=True)
    (tmp_path / "experiment.json").write_text(
        json.dumps(
            {
                "info": {"failed_runs": []},
                "mlflow": {
                    "experiment": {"tags": {}},
                    "runs": ["consumer-run"],
                    "logged_models": ["source-model"],
                },
            }
        ),
        encoding="utf-8",
    )
    _write_run_export(
        run_dir,
        model_inputs=[{"model_id": "source-model"}],
    )
    model_dir = tmp_path / "logged_models" / "source-model"
    model_dir.mkdir(parents=True)
    (model_dir / "logged_model.json").write_text("{}", encoding="utf-8")
    imports = []

    monkeypatch.setattr(import_experiment_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(
        import_experiment_module.mlflow_utils,
        "set_experiment",
        lambda *args: SimpleNamespace(name="destination", experiment_id="1"),
    )
    monkeypatch.setattr(
        import_experiment_module,
        "import_run",
        lambda **kwargs: (
            SimpleNamespace(info=SimpleNamespace(run_id="destination-run")),
            None,
        ),
    )
    monkeypatch.setattr(
        import_experiment_module,
        "import_logged_model",
        lambda **kwargs: imports.append(kwargs),
    )
    monkeypatch.setattr(import_experiment_module.utils, "nested_tags", lambda *args: None)

    import_experiment_module.import_experiment(
        experiment_name="destination",
        input_dir=str(tmp_path),
        mlflow_client=SimpleNamespace(set_terminated=lambda *args: None),
    )

    assert len(imports) == 1
    assert imports[0]["model_type"] == "input"
    assert imports[0]["input_dir"] == str(model_dir)
    assert "step" not in imports[0]


def test_import_experiment_restores_deleted_lifecycle_after_logged_models(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "runs" / "source-run"
    run_dir.mkdir(parents=True)
    (tmp_path / "experiment.json").write_text(
        json.dumps(
            {
                "info": {"failed_runs": []},
                "mlflow": {
                    "experiment": {"tags": {}},
                    "runs": ["source-run"],
                    "logged_models": ["source-model"],
                },
            }
        ),
        encoding="utf-8",
    )
    _write_run_export(
        run_dir,
        model_outputs=[{"model_id": "source-model", "step": 0}],
        lifecycle_stage="deleted",
    )

    lifecycle = "active"
    events = []

    def import_run(*, restore_run_lifecycle=True, **kwargs):
        nonlocal lifecycle
        if restore_run_lifecycle:
            lifecycle = "deleted"
        return SimpleNamespace(info=SimpleNamespace(run_id="destination-run")), None

    def import_logged_model(**kwargs):
        assert lifecycle == "active"
        events.append("logged-model")

    def delete_run(run_id):
        nonlocal lifecycle
        lifecycle = "deleted"
        events.append("delete-run")

    client = SimpleNamespace(
        set_terminated=lambda *args: None,
        delete_run=delete_run,
    )
    monkeypatch.setattr(import_experiment_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(
        import_experiment_module.mlflow_utils,
        "set_experiment",
        lambda *args: SimpleNamespace(name="destination", experiment_id="1"),
    )
    monkeypatch.setattr(import_experiment_module, "import_run", import_run)
    monkeypatch.setattr(
        import_experiment_module, "import_logged_model", import_logged_model
    )
    monkeypatch.setattr(import_experiment_module.utils, "nested_tags", lambda *args: None)

    import_experiment_module.import_experiment(
        experiment_name="destination",
        input_dir=str(tmp_path),
        mlflow_client=client,
    )

    assert events == ["logged-model", "delete-run"]
    assert lifecycle == "deleted"


@pytest.mark.parametrize(
    "source",
    ["models:/source-model", "/source/source-run/artifacts/model"],
)
def test_import_model_imports_each_logged_model_once(tmp_path, monkeypatch, source):
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    _write_run_export(run_dir)
    import_count = 0

    def import_run(*, import_logged_models=True, **kwargs):
        nonlocal import_count
        if import_logged_models:
            import_count += 1
        return SimpleNamespace(info=SimpleNamespace(run_id="destination-run")), None

    def import_logged_model(**kwargs):
        nonlocal import_count
        import_count += 1

    client = SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(
            info=SimpleNamespace(
                run_id="destination-run", artifact_uri="destination-artifacts"
            )
        )
    )
    monkeypatch.setattr(import_model_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(import_model_module, "import_run", import_run)
    monkeypatch.setattr(import_model_module, "import_logged_model", import_logged_model)
    importer = import_model_module.ModelImporter(mlflow_client=client)

    importer._import_run(
        input_dir=str(tmp_path),
        experiment_name="destination",
        vr={
            "name": "registered-model",
            "version": "1",
            "run_id": "source-run",
            "source": source,
            "current_stage": "None",
        },
    )

    assert import_count == 1


def test_import_model_recognizes_logged_model_artifact_location(tmp_path, monkeypatch):
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    _write_run_export(run_dir)
    run_imports = []
    logged_model_imports = []

    def import_run(**kwargs):
        run_imports.append(kwargs)
        return SimpleNamespace(info=SimpleNamespace(run_id="destination-run")), None

    def import_logged_model(**kwargs):
        logged_model_imports.append(kwargs)

    client = SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(
            info=SimpleNamespace(
                run_id="destination-run", artifact_uri="destination-artifacts"
            )
        )
    )
    monkeypatch.setattr(import_model_module, "import_run", import_run)
    monkeypatch.setattr(import_model_module, "import_logged_model", import_logged_model)
    importer = import_model_module.ModelImporter(mlflow_client=client)

    importer._import_run(
        input_dir=str(tmp_path),
        experiment_name="destination",
        vr={
            "name": "registered-model",
            "version": "1",
            "run_id": "source-run",
            "source": "dbfs:/source/models/source-model/artifacts",
            "current_stage": "None",
        },
    )

    assert run_imports[0]["import_logged_models"] is False
    assert logged_model_imports[0]["input_dir"] == str(tmp_path / "source-model")


def test_import_model_restores_deleted_run_after_logged_model(tmp_path, monkeypatch):
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    _write_run_export(
        run_dir,
        model_outputs=[{"model_id": "source-model", "step": 0}],
        lifecycle_stage="deleted",
    )
    lifecycle = "active"
    events = []

    def import_run(*, restore_run_lifecycle=True, **kwargs):
        nonlocal lifecycle
        if restore_run_lifecycle:
            lifecycle = "deleted"
        return SimpleNamespace(info=SimpleNamespace(run_id="destination-run")), None

    def import_logged_model(**kwargs):
        assert lifecycle == "active"
        events.append("logged-model")

    def delete_run(run_id):
        nonlocal lifecycle
        lifecycle = "deleted"
        events.append("delete-run")

    client = SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(
            info=SimpleNamespace(
                run_id="destination-run", artifact_uri="destination-artifacts"
            )
        ),
        delete_run=delete_run,
    )
    monkeypatch.setattr(import_model_module, "import_run", import_run)
    monkeypatch.setattr(import_model_module, "import_logged_model", import_logged_model)
    importer = import_model_module.ModelImporter(mlflow_client=client)

    importer._import_run(
        input_dir=str(tmp_path),
        experiment_name="destination",
        vr={
            "name": "registered-model",
            "version": "1",
            "run_id": "source-run",
            "source": "models:/source-model",
            "current_stage": "None",
        },
    )

    assert events == ["logged-model", "delete-run"]
    assert lifecycle == "deleted"


def test_export_experiment_exports_each_logged_model_once(tmp_path, monkeypatch):
    run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="source-run",
            experiment_id="source-experiment",
            lifecycle_stage="active",
            start_time=1,
        )
    )
    experiment = Experiment(
        name="source-experiment",
        experiment_id="source-experiment",
        artifact_location="source-artifacts",
        lifecycle_stage="active",
        creation_time=1,
        last_update_time=2,
        tags=[],
    )
    export_count = 0

    def export_run(*args, export_logged_models=True, **kwargs):
        nonlocal export_count
        if export_logged_models:
            export_count += 1
        return run

    def export_logged_models(**kwargs):
        nonlocal export_count
        export_count += 1
        return ["source-model"], []

    client = SimpleNamespace()
    monkeypatch.setattr(export_experiment_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(
        export_experiment_module.mlflow_utils,
        "get_experiment",
        lambda *args: experiment,
    )
    monkeypatch.setattr(
        export_experiment_module, "SearchRunsIterator", lambda *args, **kwargs: [run]
    )
    monkeypatch.setattr(export_experiment_module, "export_run", export_run)
    monkeypatch.setattr(
        export_experiment_module.export_logged_models,
        "export_logged_models",
        export_logged_models,
    )
    monkeypatch.setattr(export_experiment_module, "has_logged_model_support", lambda: True)
    monkeypatch.setattr(export_experiment_module, "has_trace_support", lambda: False)
    monkeypatch.setattr(
        export_experiment_module.io_utils, "write_export_file", lambda *args: None
    )

    export_experiment_module.export_experiment(
        experiment_id_or_name="source-experiment",
        output_dir=str(tmp_path),
        mlflow_client=client,
    )

    assert export_count == 1


def test_export_experiment_filters_logged_models_to_exported_runs(
    tmp_path, monkeypatch
):
    run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="source-run",
            experiment_id="source-experiment",
            lifecycle_stage="active",
            start_time=1,
        )
    )
    experiment = Experiment(
        name="source-experiment",
        experiment_id="source-experiment",
        artifact_location="source-artifacts",
        lifecycle_stage="active",
        creation_time=1,
        last_update_time=2,
        tags=[],
    )
    captured = {}

    monkeypatch.setattr(export_experiment_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(
        export_experiment_module.mlflow_utils,
        "get_experiment",
        lambda *args: experiment,
    )
    monkeypatch.setattr(
        export_experiment_module, "SearchRunsIterator", lambda *args, **kwargs: [run]
    )
    monkeypatch.setattr(export_experiment_module, "export_run", lambda *args, **kwargs: run)

    def export_logged_models(**kwargs):
        captured.update(kwargs)
        return ["source-model"], []

    monkeypatch.setattr(
        export_experiment_module.export_logged_models,
        "export_logged_models",
        export_logged_models,
    )
    monkeypatch.setattr(export_experiment_module, "has_logged_model_support", lambda: True)
    monkeypatch.setattr(export_experiment_module, "has_trace_support", lambda: False)
    monkeypatch.setattr(
        export_experiment_module.io_utils, "write_export_file", lambda *args: None
    )

    export_experiment_module.export_experiment(
        experiment_id_or_name="source-experiment",
        output_dir=str(tmp_path),
        mlflow_client=SimpleNamespace(),
    )

    assert captured["logged_models_filter"] == {
        "source-experiment": ["source-run"]
    }


def test_partial_experiment_export_includes_referenced_input_model(
    tmp_path, monkeypatch
):
    consumer_run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="consumer-run",
            experiment_id="source-experiment",
            lifecycle_stage="active",
            start_time=1,
        ),
        inputs=SimpleNamespace(
            model_inputs=[SimpleNamespace(model_id="referenced-model")]
        ),
        outputs=SimpleNamespace(model_outputs=[]),
    )
    experiment = Experiment(
        name="source-experiment",
        experiment_id="source-experiment",
        artifact_location="source-artifacts",
        lifecycle_stage="active",
        creation_time=1,
        last_update_time=2,
        tags=[],
    )
    referenced_model = SimpleNamespace(
        model_id="referenced-model",
        name="referenced",
        experiment_id="producer-experiment",
        source_run_id="producer-run",
    )
    unrelated_model = SimpleNamespace(
        model_id="unrelated-model",
        name="unrelated",
        experiment_id="source-experiment",
        source_run_id="other-run",
    )

    monkeypatch.setattr(export_experiment_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(
        export_experiment_module.mlflow_utils,
        "get_experiment",
        lambda *args: experiment,
    )
    monkeypatch.setattr(
        export_experiment_module,
        "SearchRunsIterator",
        lambda *args, **kwargs: [consumer_run],
    )
    monkeypatch.setattr(
        export_experiment_module,
        "export_run",
        lambda *args, **kwargs: consumer_run,
    )
    monkeypatch.setattr(export_experiment_module, "has_logged_model_support", lambda: True)
    monkeypatch.setattr(export_experiment_module, "has_trace_support", lambda: False)
    monkeypatch.setattr(export_logged_models_module, "has_logged_model_support", lambda: True)
    monkeypatch.setattr(
        export_logged_models_module,
        "get_logged_models",
        lambda *args: [unrelated_model],
    )
    monkeypatch.setattr(export_logged_models_module.utils, "show_table", lambda *args: None)

    def export_logged_model(logged_model, output_dir, client, ok, failed):
        model_dir = tmp_path / "logged_models" / logged_model.model_id
        model_dir.mkdir(parents=True)
        (model_dir / "logged_model.json").write_text("{}", encoding="utf-8")
        ok.append(logged_model.model_id)

    monkeypatch.setattr(
        export_logged_models_module, "_export_logged_model", export_logged_model
    )

    export_experiment_module.export_experiment(
        experiment_id_or_name="source-experiment",
        output_dir=str(tmp_path),
        run_ids=["consumer-run"],
        mlflow_client=SimpleNamespace(
            get_run=lambda run_id: consumer_run,
            get_logged_model=lambda model_id: referenced_model,
            get_experiment=lambda experiment_id: SimpleNamespace(
                name=str(experiment_id)
            ),
        ),
    )

    assert (
        tmp_path / "logged_models" / "referenced-model" / "logged_model.json"
    ).exists()
    assert not (tmp_path / "logged_models" / "unrelated-model").exists()


def test_run_scoped_logged_model_export_keeps_standalone_models(
    tmp_path, monkeypatch
):
    attached = SimpleNamespace(
        model_id="attached-model",
        name="attached",
        experiment_id="source-experiment",
        source_run_id="source-run",
    )
    excluded = SimpleNamespace(
        model_id="excluded-model",
        name="excluded",
        experiment_id="source-experiment",
        source_run_id="excluded-run",
    )
    standalone = SimpleNamespace(
        model_id="standalone-model",
        name="standalone",
        experiment_id="source-experiment",
        source_run_id=None,
    )
    exported = []

    monkeypatch.setattr(export_logged_models_module, "has_logged_model_support", lambda: True)
    monkeypatch.setattr(
        export_logged_models_module,
        "get_logged_models",
        lambda *args: [attached, excluded, standalone],
    )
    monkeypatch.setattr(export_logged_models_module.utils, "show_table", lambda *args: None)

    def export_logged_model(logged_model, output_dir, client, ok, failed):
        exported.append(logged_model.model_id)
        ok.append(logged_model.model_id)

    monkeypatch.setattr(
        export_logged_models_module, "_export_logged_model", export_logged_model
    )
    client = SimpleNamespace(
        get_experiment=lambda experiment_id: SimpleNamespace(name="source-experiment")
    )

    export_logged_models_module.export_logged_models(
        experiment_ids=["source-experiment"],
        output_dir=str(tmp_path),
        logged_models_filter={"source-experiment": ["source-run"]},
        mlflow_client=client,
    )

    assert exported == ["attached-model", "standalone-model"]


def test_import_standalone_logged_model_does_not_update_missing_run(
    tmp_path, monkeypatch
):
    (tmp_path / "logged_model.json").write_text(
        json.dumps(
            {
                "mlflow": {
                    "name": "standalone-model",
                    "tags": {},
                    "params": [],
                    "metrics": [],
                    "model_type": None,
                    "source_run_id": None,
                    "status": "READY",
                }
            }
        ),
        encoding="utf-8",
    )
    finalized = []

    class Client:
        def create_logged_model(self, **kwargs):
            return SimpleNamespace(model_id="destination-model")

        def finalize_logged_model(self, model_id, status):
            finalized.append((model_id, status))

        def set_terminated(self, run_id, status):
            raise AssertionError("standalone logged models do not have a run")

    monkeypatch.setattr(import_logged_model_module, "has_logged_model_support", lambda: True)
    monkeypatch.setattr(import_logged_model_module, "create_dbx_client", lambda client: None)
    monkeypatch.setattr(
        import_logged_model_module.mlflow_utils,
        "set_experiment",
        lambda *args: SimpleNamespace(
            experiment_id="destination-experiment", name="destination"
        ),
    )

    logged_model = import_logged_model_module.import_logged_model(
        input_dir=str(tmp_path),
        experiment_name="destination",
        mlflow_client=Client(),
    )

    assert logged_model.model_id == "destination-model"
    assert finalized == [("destination-model", "READY")]


@pytest.mark.parametrize(
    "source",
    ["models:/source-model", "/source/source-run/artifacts/model"],
)
def test_export_model_exports_each_logged_model_once(tmp_path, monkeypatch, source):
    export_count = 0
    version = SimpleNamespace(
        name="registered-model",
        version="1",
        current_stage="None",
        run_id="source-run",
        source=source,
    )
    run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="source-run",
            artifact_uri="source-artifacts",
            experiment_id="source-experiment",
        )
    )

    def export_run(*args, export_logged_models=True, **kwargs):
        nonlocal export_count
        if export_logged_models:
            export_count += 1
        return run

    def export_logged_model(**kwargs):
        nonlocal export_count
        export_count += 1

    monkeypatch.setattr(
        export_model_module.model_utils,
        "model_version_to_dict",
        lambda version: {"version": version.version},
    )
    monkeypatch.setattr(export_model_module, "export_run", export_run)
    monkeypatch.setattr(export_model_module, "export_logged_model", export_logged_model)
    client = SimpleNamespace(
        get_experiment=lambda experiment_id: SimpleNamespace(name="source-experiment")
    )

    exported_versions = []
    export_model_module._export_version(
        client,
        version,
        str(tmp_path),
        aliases=[],
        output_versions=exported_versions,
        failed_versions=[],
        j=0,
        num_versions=1,
        opts=SimpleNamespace(
            export_version_model=False,
            export_deleted_runs=False,
            notebook_formats=[],
        ),
    )

    assert export_count == 1
    assert len(exported_versions) == 1
