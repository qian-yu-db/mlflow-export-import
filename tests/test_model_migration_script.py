import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "test_model_migration.sh"


def write_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def create_fake_toolchain(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"

    write_executable(
        bin_dir / "databricks",
        """#!/usr/bin/env bash
set -euo pipefail
echo "databricks|$*" >> "$CALL_LOG"
case "$1 $2" in
  "current-user me")
    echo '{"userName":"tester@databricks.com"}'
    ;;
  "registered-models get")
    if [[ "${DESTINATION_EXISTS:-0}" == "1" ]]; then
      echo '{"name":"destination.catalog.model"}'
    else
      echo "Routine or Model does not exist." >&2
      exit 1
    fi
    ;;
  "model-versions list")
    status="${DESTINATION_STATUS:-READY}"
    cat <<EOF
[{"version":"1","status":"$status","run_id":"destination-run","source":"dbfs:/destination/logged_models/m-destination/artifacts"}]
EOF
    ;;
  *)
    echo "unexpected databricks command: $*" >&2
    exit 9
    ;;
esac
""",
    )
    write_executable(
        bin_dir / "export-model",
        """#!/usr/bin/env bash
set -euo pipefail
echo "export-model|$*|$DATABRICKS_CONFIG_PROFILE|$MLFLOW_TRACKING_URI|$MLFLOW_REGISTRY_URI|$MLFLOW_EXPORT_IMPORT_LOG_CONFIG_FILE|${PYTHONPATH:-}" >> "$CALL_LOG"
output_dir=""
version=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) output_dir="$2"; shift 2 ;;
    --versions) version="$2"; shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$output_dir/m-source/artifacts"
printf 'model bundle\n' > "$output_dir/m-source/artifacts/MLmodel"
if [[ "${EXPORT_FAILED:-0}" == "1" ]]; then
  cat > "$output_dir/model.json" <<EOF
{"info":{"failed_versions":[{"version":{"version":"$version"}}]},"mlflow":{"registered_model":{"versions":[]}}}
EOF
else
  cat > "$output_dir/model.json" <<EOF
{"info":{"failed_versions":[]},"mlflow":{"registered_model":{"versions":[{"version":"$version","run_id":"source-run"}]}}}
EOF
fi
""",
    )
    write_executable(
        bin_dir / "import-model",
        """#!/usr/bin/env bash
set -euo pipefail
echo "import-model|$*|$DATABRICKS_CONFIG_PROFILE|$MLFLOW_TRACKING_URI|$MLFLOW_REGISTRY_URI|$MLFLOW_EXPORT_IMPORT_LOG_CONFIG_FILE|${PYTHONPATH:-}" >> "$CALL_LOG"
""",
    )
    write_executable(
        bin_dir / "mlflow",
        """#!/usr/bin/env bash
set -euo pipefail
echo "mlflow|$*|$DATABRICKS_CONFIG_PROFILE|$MLFLOW_TRACKING_URI|$MLFLOW_REGISTRY_URI|$MLFLOW_EXPORT_IMPORT_LOG_CONFIG_FILE" >> "$CALL_LOG"
if [[ "$1 $2" == "runs describe" ]]; then
  status="${RUN_STATUS:-FINISHED}"
  cat <<EOF
{"info":{"run_id":"destination-run","status":"$status"}}
EOF
  exit 0
fi
destination=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dst-path) destination="$2"; shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$destination"
if [[ "${OMIT_MLMODEL:-0}" != "1" ]]; then
  printf 'downloaded model\n' > "$destination/MLmodel"
fi
""",
    )

    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["CALL_LOG"] = str(call_log)
    return environment, call_log


def run_migration(tmp_path, environment):
    work_dir = tmp_path / "work"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--source-profile",
            "source-profile",
            "--destination-profile",
            "destination-profile",
            "--source-model",
            "source.catalog.model",
            "--source-version",
            "2",
            "--destination-model",
            "destination.catalog.model",
            "--destination-experiment",
            "/Workspace/Users/tester/mlflow_experiments/migration",
            "--work-dir",
            str(work_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, work_dir


def test_help_describes_required_migration_inputs():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--source-profile" in result.stdout
    assert "--destination-profile" in result.stdout
    assert "--source-model" in result.stdout
    assert "--source-version" in result.stdout
    assert "--destination-model" in result.stdout
    assert "--destination-experiment" in result.stdout


def test_missing_required_arguments_fail_before_running_commands():
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Missing required option: --source-profile" in result.stderr


def test_successful_migration_runs_both_clis_and_verifies_destination(tmp_path):
    environment, call_log = create_fake_toolchain(tmp_path)

    result, work_dir = run_migration(tmp_path, environment)

    assert result.returncode == 0, result.stderr
    assert "Migration test passed" in result.stdout
    assert "Destination version: 1" in result.stdout
    assert "Destination run: destination-run" in result.stdout
    assert (work_dir / "verify" / "MLmodel").is_file()

    calls = call_log.read_text(encoding="utf-8")
    assert "current-user me --profile source-profile" in calls
    assert "current-user me --profile destination-profile" in calls
    assert "export-model|--model source.catalog.model --versions 2" in calls
    assert (
        "|source-profile|databricks://source-profile|"
        "databricks-uc://source-profile|"
    ) in calls
    assert (
        "import-model|--input-dir "
        f"{work_dir}/export --model destination.catalog.model "
        "--experiment-name /Workspace/Users/tester/mlflow_experiments/migration "
        "--await-creation-for 600"
    ) in calls
    assert (
        "|destination-profile|databricks://destination-profile|"
        "databricks-uc://destination-profile|"
    ) in calls
    assert "models:/destination.catalog.model/1" in calls


def test_package_clis_load_this_checkout_before_another_editable_install(tmp_path):
    environment, call_log = create_fake_toolchain(tmp_path)
    environment["PYTHONPATH"] = "/another/checkout"

    result, _ = run_migration(tmp_path, environment)

    assert result.returncode == 0, result.stderr
    package_lines = [
        line
        for line in call_log.read_text(encoding="utf-8").splitlines()
        if line.startswith(("export-model|", "import-model|"))
    ]
    assert len(package_lines) == 2
    expected_prefix = f"{SCRIPT.parents[1]}:/another/checkout"
    assert all(line.endswith(f"|{expected_prefix}") for line in package_lines)


def test_failed_export_manifest_stops_before_import(tmp_path):
    environment, call_log = create_fake_toolchain(tmp_path)
    environment["EXPORT_FAILED"] = "1"

    result, _ = run_migration(tmp_path, environment)

    assert result.returncode == 2
    assert "failed_versions" in result.stderr
    assert "import-model|" not in call_log.read_text(encoding="utf-8")


def test_existing_destination_model_stops_before_export(tmp_path):
    environment, call_log = create_fake_toolchain(tmp_path)
    environment["DESTINATION_EXISTS"] = "1"

    result, _ = run_migration(tmp_path, environment)

    assert result.returncode == 2
    assert "Destination model already exists" in result.stderr
    assert "export-model|" not in call_log.read_text(encoding="utf-8")


def test_pending_destination_version_fails_verification(tmp_path):
    environment, call_log = create_fake_toolchain(tmp_path)
    environment["DESTINATION_STATUS"] = "PENDING_REGISTRATION"

    result, _ = run_migration(tmp_path, environment)

    assert result.returncode == 2
    assert "No READY destination model version" in result.stderr
    assert "mlflow|" not in call_log.read_text(encoding="utf-8")


def test_unfinished_destination_run_fails_verification(tmp_path):
    environment, call_log = create_fake_toolchain(tmp_path)
    environment["RUN_STATUS"] = "RUNNING"

    result, _ = run_migration(tmp_path, environment)

    assert result.returncode == 2
    assert "expected FINISHED" in result.stderr
    calls = call_log.read_text(encoding="utf-8")
    assert "mlflow|runs describe --run-id destination-run" in calls
    assert "mlflow|artifacts download" not in calls


def test_missing_downloaded_mlmodel_fails_verification(tmp_path):
    environment, _ = create_fake_toolchain(tmp_path)
    environment["OMIT_MLMODEL"] = "1"

    result, _ = run_migration(tmp_path, environment)

    assert result.returncode == 2
    assert "did not contain an MLmodel file" in result.stderr


def test_generated_logging_config_suppresses_sensitive_debug_logs(tmp_path):
    environment, _ = create_fake_toolchain(tmp_path)

    result, work_dir = run_migration(tmp_path, environment)

    assert result.returncode == 0, result.stderr
    config = (work_dir / "info-logging.yaml").read_text(encoding="utf-8")
    assert "root:\n  level: INFO" in config
    assert "botocore:\n    level: WARNING" in config
    assert (work_dir / "info-logging.yaml").stat().st_mode & 0o077 == 0
