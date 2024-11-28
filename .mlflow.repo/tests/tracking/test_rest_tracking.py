"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import json
import logging
import sys

import pytest
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
