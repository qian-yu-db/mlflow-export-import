#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: test_model_migration.sh [options]

Export one registered model version with its run, import it into another
workspace, and verify the destination registration and artifacts.

Required options:
  --source-profile NAME
  --destination-profile NAME
  --source-model CATALOG.SCHEMA.MODEL
  --source-version VERSION
  --destination-model CATALOG.SCHEMA.MODEL
  --destination-experiment PATH

Optional options:
  --work-dir PATH       Keep test files at PATH (default: a temporary directory)
  --await-seconds N     Registration timeout in seconds (default: 600)
  -h, --help            Show this help
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "$value" ]] || fail "Option $option requires a value"
}

source_profile=""
destination_profile=""
source_model=""
source_version=""
destination_model=""
destination_experiment=""
work_dir=""
await_seconds="600"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-profile)
      require_value "$1" "${2:-}"
      source_profile="$2"
      shift 2
      ;;
    --destination-profile)
      require_value "$1" "${2:-}"
      destination_profile="$2"
      shift 2
      ;;
    --source-model)
      require_value "$1" "${2:-}"
      source_model="$2"
      shift 2
      ;;
    --source-version)
      require_value "$1" "${2:-}"
      source_version="$2"
      shift 2
      ;;
    --destination-model)
      require_value "$1" "${2:-}"
      destination_model="$2"
      shift 2
      ;;
    --destination-experiment)
      require_value "$1" "${2:-}"
      destination_experiment="$2"
      shift 2
      ;;
    --work-dir)
      require_value "$1" "${2:-}"
      work_dir="$2"
      shift 2
      ;;
    --await-seconds)
      require_value "$1" "${2:-}"
      await_seconds="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

required_values=(
  "--source-profile:$source_profile"
  "--destination-profile:$destination_profile"
  "--source-model:$source_model"
  "--source-version:$source_version"
  "--destination-model:$destination_model"
  "--destination-experiment:$destination_experiment"
)
for required in "${required_values[@]}"; do
  option="${required%%:*}"
  value="${required#*:}"
  [[ -n "$value" ]] || fail "Missing required option: $option"
done

[[ "$await_seconds" =~ ^[1-9][0-9]*$ ]] || \
  fail "--await-seconds must be a positive integer"

for dependency in databricks export-model import-model mlflow jq; do
  command -v "$dependency" >/dev/null 2>&1 || \
    fail "Required command not found: $dependency"
done

if [[ -z "$work_dir" ]]; then
  work_dir="$(mktemp -d "${TMPDIR:-/tmp}/mlflow-model-migration.XXXXXX")"
else
  mkdir -p "$work_dir"
fi

export_dir="$work_dir/export"
verify_dir="$work_dir/verify"
logging_config="$work_dir/info-logging.yaml"
destination_versions_file="$work_dir/destination-versions.json"
destination_run_file="$work_dir/destination-run.json"
destination_check_file="$work_dir/destination-model-check.json"
destination_check_error="$work_dir/destination-model-check.err"

[[ ! -e "$export_dir" ]] || fail "Export directory already exists: $export_dir"
[[ ! -e "$verify_dir" ]] || fail "Verify directory already exists: $verify_dir"

umask 077
cat > "$logging_config" <<'YAML'
version: 1
disable_existing_loggers: false
formatters:
  simple:
    format: "%(asctime)s - %(levelname)s - %(message)s"
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: simple
    stream: ext://sys.stdout
root:
  level: INFO
  handlers: [console]
loggers:
  botocore:
    level: WARNING
  boto3:
    level: WARNING
  urllib3:
    level: WARNING
YAML

run_with_profile() {
  local profile="$1"
  shift
  env \
    PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
    DATABRICKS_CONFIG_PROFILE="$profile" \
    MLFLOW_TRACKING_URI="databricks://$profile" \
    MLFLOW_REGISTRY_URI="databricks-uc://$profile" \
    MLFLOW_EXPORT_IMPORT_LOG_CONFIG_FILE="$logging_config" \
    "$@"
}

echo "[1/6] Checking source and destination authentication"
databricks current-user me --profile "$source_profile" >/dev/null
databricks current-user me --profile "$destination_profile" >/dev/null

echo "[2/6] Confirming the destination model name is unused"
if databricks registered-models get "$destination_model" \
    --profile "$destination_profile" \
    --output json >"$destination_check_file" 2>"$destination_check_error"; then
  fail "Destination model already exists: $destination_model"
elif ! grep -Eiq "does not exist|RESOURCE_DOES_NOT_EXIST|NOT_FOUND" \
    "$destination_check_error"; then
  cat "$destination_check_error" >&2
  fail "Could not verify that the destination model is absent"
fi

echo "[3/6] Exporting source model version $source_version"
run_with_profile "$source_profile" export-model \
  --model "$source_model" \
  --versions "$source_version" \
  --output-dir "$export_dir"

manifest="$export_dir/model.json"
[[ -s "$manifest" ]] || fail "Export did not create a non-empty model.json"

failed_version_count="$(jq -r '(.info.failed_versions // []) | length' "$manifest")"
[[ "$failed_version_count" == "0" ]] || \
  fail "Export manifest contains $failed_version_count failed_versions entry or entries"

exported_version_count="$(
  jq -r --arg version "$source_version" \
    '[.mlflow.registered_model.versions[]? | select((.version | tostring) == $version)] | length' \
    "$manifest"
)"
[[ "$exported_version_count" == "1" ]] || \
  fail "Export manifest does not contain exactly one requested version $source_version"

echo "[4/6] Importing and registering the destination model"
run_with_profile "$destination_profile" import-model \
  --input-dir "$export_dir" \
  --model "$destination_model" \
  --experiment-name "$destination_experiment" \
  --await-creation-for "$await_seconds"

echo "[5/6] Verifying destination model version and run"
databricks model-versions list "$destination_model" \
  --profile "$destination_profile" \
  --output json > "$destination_versions_file"

destination_version_record="$(
  jq -c \
    '[.[]? | select(.status == "READY")] | sort_by(.version | tonumber) | last // empty' \
    "$destination_versions_file"
)"
[[ -n "$destination_version_record" ]] || \
  fail "No READY destination model version was found"

destination_version="$(jq -r '.version' <<<"$destination_version_record")"
destination_run="$(jq -r '.run_id // empty' <<<"$destination_version_record")"
destination_source="$(jq -r '.source // empty' <<<"$destination_version_record")"
[[ -n "$destination_run" ]] || fail "Destination model version has no run_id"
[[ -n "$destination_source" ]] || fail "Destination model version has no artifact source"

run_with_profile "$destination_profile" mlflow runs describe \
  --run-id "$destination_run" > "$destination_run_file"
run_status="$(jq -r '.info.status // empty' "$destination_run_file")"
[[ "$run_status" == "FINISHED" ]] || \
  fail "Destination run status is '$run_status', expected FINISHED"

echo "[6/6] Downloading registered model artifacts"
mkdir -p "$verify_dir"
run_with_profile "$destination_profile" mlflow artifacts download \
  --artifact-uri "models:/$destination_model/$destination_version" \
  --dst-path "$verify_dir" >/dev/null

downloaded_mlmodel="$(find "$verify_dir" -type f -name MLmodel -print -quit)"
[[ -n "$downloaded_mlmodel" ]] || \
  fail "Registered model download did not contain an MLmodel file"

echo
echo "Migration test passed"
echo "Destination model: $destination_model"
echo "Destination version: $destination_version"
echo "Destination run: $destination_run"
echo "Destination source: $destination_source"
echo "Test files: $work_dir"
