import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "test_targeted_model_version_migration.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "test_targeted_model_version_migration", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runs_timestamped_migration_with_explicit_profiles(monkeypatch, capsys):
    script = _load_script()
    profiles = []
    migration_arguments = {}

    class WorkspaceClient:
        def __init__(self, profile):
            profiles.append(profile)
            self.current_user = SimpleNamespace(
                me=lambda: SimpleNamespace(user_name=f"{profile}@example.com")
            )

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 9, 14, 12, 34, 56)

    expected_result = SimpleNamespace(
        source_model_name="fins_genai.classic_ml.advanced_mlops_churn",
        source_model_version="5",
        source_run_id="source-run",
        destination_model_name=(
            "tech_summit_qyu_catalog.classic_ml."
            "advanced_mlops_churn_20260914_123456"
        ),
        destination_model_version="1",
        destination_run_id="destination-run",
        output_dir="/tmp/model-migration-20260914_123456",
    )

    def migrate_model_version(**kwargs):
        migration_arguments.update(kwargs)
        return expected_result

    monkeypatch.setattr(script, "WorkspaceClient", WorkspaceClient)
    monkeypatch.setattr(script, "datetime", FixedDatetime)
    monkeypatch.setattr(script, "migrate_model_version", migrate_model_version)

    result = script.main()

    assert profiles == ["fevm-classic-stable", "fe-sandbox-tech-summit"]
    assert migration_arguments == {
        "source_profile": "fevm-classic-stable",
        "destination_profile": "fe-sandbox-tech-summit",
        "source_model_name": "fins_genai.classic_ml.advanced_mlops_churn",
        "source_model_version": "5",
        "destination_model_name": (
            "tech_summit_qyu_catalog.classic_ml."
            "advanced_mlops_churn_20260914_123456"
        ),
        "destination_experiment_name": (
            "/Workspace/Users/q.yu@databricks.com/mlflow_experiments/"
            "advanced_mlops_churn_20260914_123456"
        ),
        "output_dir": "/tmp/model-migration-20260914_123456",
        "await_creation_for": 600,
    }
    assert result is expected_result
    output = capsys.readouterr().out
    assert "Authenticated fevm-classic-stable" in output
    assert "Authenticated fe-sandbox-tech-summit" in output
    assert "Migration completed" in output
