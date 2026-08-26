import json
from types import SimpleNamespace

import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils import rest_utils

from mlflow_export_import.common import MlflowExportImportException
from mlflow_export_import.client import client_utils
from mlflow_export_import.client import http_client as http_client_module


class _Response:
    status_code = 200
    url = "https://workspace.example/api/2.0/test"
    text = json.dumps({"result": "ok"})


def _mlflow_client_with_sdk_credentials():
    credentials = SimpleNamespace(
        host="https://workspace.example",
        token=None,
        use_databricks_sdk=True,
        use_secret_scope_token=False,
        databricks_auth_profile="oauth-profile",
    )
    store = SimpleNamespace(get_host_creds=lambda: credentials)
    tracking_client = SimpleNamespace(store=store)
    return SimpleNamespace(_tracking_client=tracking_client)


@pytest.mark.parametrize(
    ("make_client", "method", "resource", "payload", "expected_endpoint"),
    [
        (
            lambda client: client_utils.create_http_client(
                client, "catalog.schema.model"
            ),
            "get",
            "registered-models/get",
            {"name": "catalog.schema.model"},
            "/api/2.0/mlflow/unity-catalog/registered-models/get",
        ),
        (
            client_utils.create_dbx_client,
            "post",
            "workspace/mkdirs",
            {"path": "/Users/test/experiment"},
            "/api/2.0/workspace/mkdirs",
        ),
    ],
)
def test_clients_preserve_mlflow_sdk_authentication(
    monkeypatch, make_client, method, resource, payload, expected_endpoint
):
    calls = []

    def http_request(host_creds, endpoint, request_method, **kwargs):
        calls.append((host_creds, endpoint, request_method, kwargs))
        return _Response()

    monkeypatch.setattr(rest_utils, "http_request", http_request)
    monkeypatch.setattr(
        http_client_module.requests,
        method,
        lambda *args, **kwargs: pytest.fail(
            "SDK-backed credentials must not use the unauthenticated requests transport"
        ),
    )

    mlflow_client = _mlflow_client_with_sdk_credentials()
    client = make_client(mlflow_client)
    result = getattr(client, method)(resource, payload)

    assert result == {"result": "ok"}
    assert len(calls) == 1
    host_creds, endpoint, request_method, kwargs = calls[0]
    assert host_creds.databricks_auth_profile == "oauth-profile"
    assert endpoint == expected_endpoint
    assert request_method == method.upper()
    assert "extra_headers" not in kwargs
    if method == "get":
        assert kwargs["params"] == payload
    else:
        assert kwargs["json"] == payload


def test_sdk_http_errors_use_package_exception_contract(monkeypatch):
    response = _Response()
    response.status_code = 404
    response.text = json.dumps({"error_code": "RESOURCE_DOES_NOT_EXIST"})

    def http_request(
        host_creds, endpoint, request_method, raise_on_status=True, **kwargs
    ):
        if raise_on_status:
            raise MlflowException(
                "not found", error_code="RESOURCE_DOES_NOT_EXIST"
            )
        return response

    monkeypatch.setattr(rest_utils, "http_request", http_request)
    client = http_client_module.DatabricksHttpClient(
        host_creds=_mlflow_client_with_sdk_credentials()
        ._tracking_client.store.get_host_creds()
    )

    with pytest.raises(MlflowExportImportException) as exc_info:
        client.get("clusters/list-node-types")

    assert exc_info.value.http_status_code == 404


def test_delete_preserves_mlflow_sdk_authentication(monkeypatch):
    calls = []

    def http_request(host_creds, endpoint, request_method, **kwargs):
        calls.append((host_creds, endpoint, request_method, kwargs))
        return _Response()

    monkeypatch.setattr(rest_utils, "http_request", http_request)
    monkeypatch.setattr(
        http_client_module.requests,
        "delete",
        lambda *args, **kwargs: pytest.fail(
            "SDK-backed credentials must not use the unauthenticated requests transport"
        ),
    )
    client = client_utils.create_dbx_client(_mlflow_client_with_sdk_credentials())

    result = client.delete("workspace/delete")

    assert result == {"result": "ok"}
    assert len(calls) == 1
    host_creds, endpoint, request_method, kwargs = calls[0]
    assert host_creds.databricks_auth_profile == "oauth-profile"
    assert endpoint == "/api/2.0/workspace/delete"
    assert request_method == "DELETE"
    assert "params" not in kwargs
    assert "json" not in kwargs
