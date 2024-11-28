"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import json
import logging
import sys

import pytest
import requests
from mlflow import MlflowClient

from tests.tracking.integration_test_utils import (
    _init_server,
    _send_rest_tracking_post_request,
)

_logger = logging.getLogger(__name__)


@pytest.fixture(params=["file", "sqlalchemy"])
def mlflow_client(request, tmp_path):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    if request.param == "file":
        backend_uri = tmp_path.joinpath("file").as_uri()
    elif request.param == "sqlalchemy":
        path = tmp_path.joinpath("sqlalchemy.db").as_uri()
        backend_uri = ("sqlite://" if sys.platform == "win32" else "sqlite:////") + path[
            len("file://") :
        ]

    with _init_server(backend_uri, root_artifact_uri=tmp_path.as_uri()) as url:
        yield MlflowClient(url)


@pytest.fixture
def cli_env(mlflow_client):
    """Provides an environment for the MLflow CLI pointed at the local tracking server."""
    return {
        "LC_ALL": "en_US.UTF-8",
        "LANG": "en_US.UTF-8",
        "MLFLOW_TRACKING_URI": mlflow_client.tracking_uri,
    }


def create_experiments(client, names):
    return [client.create_experiment(n) for n in names]


"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import json
import logging
import sys

import pytest
import requests
from mlflow import MlflowClient
from mlflow.entities import (
    Dataset,
    DatasetInput,
    InputTag,
)
from mlflow.utils.proto_json_utils import message_to_json

from tests.tracking.integration_test_utils import (
    _init_server,
    _send_rest_tracking_post_request,
)

_logger = logging.getLogger(__name__)


@pytest.fixture(params=["file", "sqlalchemy"])
def mlflow_client(request, tmp_path):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    if request.param == "file":
        backend_uri = tmp_path.joinpath("file").as_uri()
    elif request.param == "sqlalchemy":
        path = tmp_path.joinpath("sqlalchemy.db").as_uri()
        backend_uri = ("sqlite://" if sys.platform == "win32" else "sqlite:////") + path[
            len("file://") :
        ]

    with _init_server(backend_uri, root_artifact_uri=tmp_path.as_uri()) as url:
        yield MlflowClient(url)


@pytest.fixture
def cli_env(mlflow_client):
    """Provides an environment for the MLflow CLI pointed at the local tracking server."""
    return {
        "LC_ALL": "en_US.UTF-8",
        "LANG": "en_US.UTF-8",
        "MLFLOW_TRACKING_URI": mlflow_client.tracking_uri,
    }


def create_experiments(client, names):
    return [client.create_experiment(n) for n in names]


def test_log_inputs_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("log inputs validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/runs/log-inputs",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    dataset = Dataset(
        name="name1",
        digest="digest1",
        source_type="source_type1",
        source="source1",
    )
    tags = [InputTag(key="tag1", value="value1")]
    dataset_inputs = [
        json.loads(message_to_json(DatasetInput(dataset=dataset, tags=tags).to_proto()))
    ]
    assert_bad_request(
        {
            "datasets": dataset_inputs,
        },
        "Missing value for required parameter 'run_id'",
    )
    assert_bad_request(
        {
            "run_id": run_id,
        },
        "Missing value for required parameter 'datasets'",
    )


def test_update_run_name_without_changing_status(mlflow_client):
    experiment_id = mlflow_client.create_experiment("update run name")
    created_run = mlflow_client.create_run(experiment_id)
    mlflow_client.set_terminated(created_run.info.run_id, "FINISHED")

    mlflow_client.update_run(created_run.info.run_id, name="name_abc")
    updated_run_info = mlflow_client.get_run(created_run.info.run_id).info
    assert updated_run_info.run_name == "name_abc"
    assert updated_run_info.status == "FINISHED"


def test_create_promptlab_run_handler_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={},
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify experiment_id.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={"experiment_id": "123"},
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify prompt_template.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={"experiment_id": "123", "prompt_template": "my_prompt_template"},
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify prompt_parameters.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={
            "experiment_id": "123",
            "prompt_template": "my_prompt_template",
            "prompt_parameters": [{"key": "my_key", "value": "my_value"}],
        },
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify model_route.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={
            "experiment_id": "123",
            "prompt_template": "my_prompt_template",
            "prompt_parameters": [{"key": "my_key", "value": "my_value"}],
            "model_route": "my_route",
        },
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify model_input.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={
            "experiment_id": "123",
            "prompt_template": "my_prompt_template",
            "prompt_parameters": [{"key": "my_key", "value": "my_value"}],
            "model_route": "my_route",
            "model_input": "my_input",
        },
    )
    assert_response(
        response,
        "CreatePromptlabRun request must specify mlflow_version.",
    )

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={
            "experiment_id": "123",
            "prompt_template": "my_prompt_template",
            "prompt_parameters": [{"key": "my_key", "value": "my_value"}],
            "model_route": "my_route",
            "model_input": "my_input",
            "mlflow_version": "1.0.0",
        },
    )
