#!/usr/bin/env python3
"""Manually test migration of one model version and its backing run."""

from datetime import datetime

from databricks.sdk import WorkspaceClient

from mlflow_export_import.model_version import migrate_model_version


SOURCE_PROFILE = "fevm-classic-stable"
DESTINATION_PROFILE = "fe-sandbox-tech-summit"
SOURCE_MODEL = "fins_genai.classic_ml.advanced_mlops_churn"
SOURCE_VERSION = "5"
DESTINATION_MODEL_PREFIX = (
    "tech_summit_qyu_catalog.classic_ml.advanced_mlops_churn"
)
DESTINATION_EXPERIMENT_PREFIX = (
    "/Workspace/Users/q.yu@databricks.com/mlflow_experiments/"
    "advanced_mlops_churn"
)


def _check_authentication(profile):
    user = WorkspaceClient(profile=profile).current_user.me()
    identity = user.user_name or user.display_name or user.id
    print(f"Authenticated {profile}: {identity}")


def main():
    _check_authentication(SOURCE_PROFILE)
    _check_authentication(DESTINATION_PROFILE)

    test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination_model = f"{DESTINATION_MODEL_PREFIX}_{test_id}"
    destination_experiment = f"{DESTINATION_EXPERIMENT_PREFIX}_{test_id}"
    output_dir = f"/tmp/model-migration-{test_id}"

    result = migrate_model_version(
        source_profile=SOURCE_PROFILE,
        destination_profile=DESTINATION_PROFILE,
        source_model_name=SOURCE_MODEL,
        source_model_version=SOURCE_VERSION,
        destination_model_name=destination_model,
        destination_experiment_name=destination_experiment,
        output_dir=output_dir,
        await_creation_for=600,
    )

    print("Migration completed")
    print(f"Destination model: {result.destination_model_name}")
    print(f"Destination version: {result.destination_model_version}")
    print(f"Destination run: {result.destination_run_id}")
    print(f"Export files: {result.output_dir}")
    return result


if __name__ == "__main__":
    main()
