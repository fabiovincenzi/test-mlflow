"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import json
import logging
import sys
import time
from unittest import mock

import mlflow.experiments
import mlflow.pyfunc
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
import time
from unittest import mock

import mlflow.experiments
import mlflow.pyfunc
import pandas as pd
import pytest
import requests
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities import (
    Dataset,
    DatasetInput,
    InputTag,
    RunInputs,
)
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import RestException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.server.handlers import _get_sampled_steps_from_steps
from mlflow.tracing.constant import TraceTagKey
from mlflow.utils import mlflow_tags
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATASET_CONTEXT,
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


def test_get_metric_history_bulk_interval_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    url = f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval"

    assert_response(
        requests.get(url, params={"metric_key": "key"}),
        "Missing value for required parameter 'run_ids'.",
    )

    assert_response(
        requests.get(url, params={"run_ids": [], "metric_key": "key"}),
        "Missing value for required parameter 'run_ids'.",
    )

    assert_response(
        requests.get(
            url, params={"run_ids": [f"id_{i}" for i in range(1000)], "metric_key": "key"}
        ),
        "GetMetricHistoryBulkInterval request must specify at most 100 run_ids.",
    )

    assert_response(
        requests.get(url, params={"run_ids": ["123"], "metric_key": "key", "max_results": 0}),
        "max_results must be between 1 and 2500",
    )

    assert_response(
        requests.get(url, params={"run_ids": ["123"], "metric_key": ""}),
        "Missing value for required parameter 'metric_key'",
    )

    assert_response(
        requests.get(url, params={"run_ids": ["123"], "max_results": 5}),
        "Missing value for required parameter 'metric_key'",
    )

    assert_response(
        requests.get(
            url,
            params={
                "run_ids": ["123"],
                "metric_key": "key",
                "start_step": 1,
                "end_step": 0,
                "max_results": 5,
            },
        ),
        "end_step must be greater than start_step. ",
    )

    assert_response(
        requests.get(
            url, params={"run_ids": ["123"], "metric_key": "key", "start_step": 1, "max_results": 5}
        ),
        "If either start step or end step are specified, both must be specified.",
    )


def test_get_metric_history_bulk_interval_respects_max_results(mlflow_client):
    experiment_id = mlflow_client.create_experiment("get metric history bulk")
    run_id1 = mlflow_client.create_run(experiment_id).info.run_id
    metric_history = [
        {"key": "metricA", "timestamp": 1, "step": i, "value": 10.0} for i in range(10)
    ]
    for metric in metric_history:
        mlflow_client.log_metric(run_id1, **metric)

    url = f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval"
    response_limited = requests.get(
        url,
        params={"run_ids": [run_id1], "metric_key": "metricA", "max_results": 5},
    )
    assert response_limited.status_code == 200
    expected_steps = [0, 2, 4, 6, 8, 9]
    expected_metrics = [
        {**metric, "run_id": run_id1}
        for metric in metric_history
        if metric["step"] in expected_steps
    ]
    assert response_limited.json().get("metrics") == expected_metrics

    # with start_step and end_step
    response_limited = requests.get(
        url,
        params={
            "run_ids": [run_id1],
            "metric_key": "metricA",
            "start_step": 0,
            "end_step": 4,
            "max_results": 5,
        },
    )
    assert response_limited.status_code == 200
    assert response_limited.json().get("metrics") == [
        {**metric, "run_id": run_id1} for metric in metric_history[:5]
    ]

    # multiple runs
    run_id2 = mlflow_client.create_run(experiment_id).info.run_id
    metric_history2 = [
        {"key": "metricA", "timestamp": 1, "step": i, "value": 10.0} for i in range(20)
    ]
    for metric in metric_history2:
        mlflow_client.log_metric(run_id2, **metric)
    response_limited = requests.get(
        url,
        params={"run_ids": [run_id1, run_id2], "metric_key": "metricA", "max_results": 5},
    )
    expected_steps = [0, 4, 8, 9, 12, 16, 19]
    expected_metrics = []
    for run_id, metric_history in [(run_id1, metric_history), (run_id2, metric_history2)]:
        expected_metrics.extend(
            [
                {**metric, "run_id": run_id}
                for metric in metric_history
                if metric["step"] in expected_steps
            ]
        )
    assert response_limited.json().get("metrics") == expected_metrics

    # test metrics with same steps
    metric_history_timestamp2 = [
        {"key": "metricA", "timestamp": 2, "step": i, "value": 10.0} for i in range(10)
    ]
    for metric in metric_history_timestamp2:
        mlflow_client.log_metric(run_id1, **metric)

    response_limited = requests.get(
        url,
        params={"run_ids": [run_id1], "metric_key": "metricA", "max_results": 5},
    )
    assert response_limited.status_code == 200
    expected_steps = [0, 2, 4, 6, 8, 9]
    expected_metrics = [
        {"key": "metricA", "timestamp": j, "step": i, "value": 10.0, "run_id": run_id1}
        for i in expected_steps
        for j in [1, 2]
    ]
    assert response_limited.json().get("metrics") == expected_metrics


@pytest.mark.parametrize(
    ("min_step", "max_step", "max_results", "nums", "expected"),
    [
        # should be evenly spaced and include the beginning and
        # end despite sometimes making it go above max_results
        (0, 10, 5, list(range(10)), {0, 2, 4, 6, 8, 9}),
        # if the clipped list is shorter than max_results,
        # then everything will be returned
        (4, 8, 5, list(range(10)), {4, 5, 6, 7, 8}),
        # works if steps are logged in intervals
        (0, 100, 5, list(range(0, 101, 20)), {0, 20, 40, 60, 80, 100}),
        (0, 1000, 5, list(range(0, 1001, 10)), {0, 200, 400, 600, 800, 1000}),
    ],
)
def test_get_sampled_steps_from_steps(min_step, max_step, max_results, nums, expected):
    assert _get_sampled_steps_from_steps(min_step, max_step, max_results, nums) == expected


def test_search_dataset_handler_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    response_no_experiment_id_field = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/experiments/search-datasets",
        json={},
    )
    assert_response(
        response_no_experiment_id_field,
        "SearchDatasets request must specify at least one experiment_id.",
    )

    response_empty_experiment_id_field = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/experiments/search-datasets",
        json={"experiment_ids": []},
    )
    assert_response(
        response_empty_experiment_id_field,
        "SearchDatasets request must specify at least one experiment_id.",
    )

    response_too_many_experiment_ids = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/experiments/search-datasets",
        json={"experiment_ids": [f"id_{i}" for i in range(1000)]},
    )
    assert_response(
        response_too_many_experiment_ids,
        "SearchDatasets request cannot specify more than",
    )


def test_search_dataset_handler_returns_expected_results(mlflow_client):
    experiment_id = mlflow_client.create_experiment("log inputs test")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    dataset1 = Dataset(
        name="name1",
        digest="digest1",
        source_type="source_type1",
        source="source1",
    )
    dataset_inputs1 = [
        DatasetInput(
            dataset=dataset1, tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value="training")]
        )
    ]
    mlflow_client.log_inputs(run_id, dataset_inputs1)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/experiments/search-datasets",
        json={"experiment_ids": [experiment_id]},
    )
    expected = {
        "experiment_id": experiment_id,
        "name": "name1",
        "digest": "digest1",
        "context": "training",
    }

    assert response.status_code == 200
    assert response.json().get("dataset_summaries") == [expected]


def test_create_model_version_with_path_source(mlflow_client):
    name = "model"
    mlflow_client.create_registered_model(name)
    exp_id = mlflow_client.create_experiment("test")
    run = mlflow_client.create_run(experiment_id=exp_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri[len("file://") :],
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # run_id is not specified
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri[len("file://") :],
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]

    # run_id is specified but source is not in the run's artifact directory
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "/tmp",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]


def test_create_model_version_with_non_local_source(mlflow_client):
    name = "model"
    mlflow_client.create_registered_model(name)
    exp_id = mlflow_client.create_experiment("test")
    run = mlflow_client.create_run(experiment_id=exp_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri[len("file://") :],
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # Test that remote uri's supplied as a source with absolute paths work fine
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts:/models",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # A single trailing slash
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts:/models/",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # Multiple trailing slashes
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts:/models///",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # Multiple slashes
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts:/models/foo///bar",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts://host:9000/models",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # Multiple dots
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts://host:9000/models/artifact/..../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # Test that invalid remote uri's cannot be created
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts://host:9000/models/../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "http://host:9000/models/../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "https://host/api/2.0/mlflow-artifacts/artifacts/../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "s3a://my_bucket/api/2.0/mlflow-artifacts/artifacts/../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "ftp://host:8888/api/2.0/mlflow-artifacts/artifacts/../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts://host:9000/models/..%2f..%2fartifacts",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "mlflow-artifacts://host:9000/models/artifact%00",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "If supplying a source as an http, https," in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"dbfs:/{run.info.run_id}/artifacts/a%3f/../../../../../../../../../../",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "Invalid model version source" in response.json()["message"]


def test_create_model_version_with_file_uri(mlflow_client):
    name = "test"
    mlflow_client.create_registered_model(name)
    exp_id = mlflow_client.create_experiment("test")
    run = mlflow_client.create_run(experiment_id=exp_id)
    assert run.info.artifact_uri.startswith("file://")
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri,
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"{run.info.artifact_uri}/model",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"{run.info.artifact_uri}/.",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"{run.info.artifact_uri}/model/..",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # run_id is not specified
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri,
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]

    # run_id is specified but source is not in the run's artifact directory
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "file:///tmp",
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "file://123.456.789.123/path/to/source",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 500, response.json()
    assert "is not a valid remote uri" in response.json()["message"]


def test_logging_model_with_local_artifact_uri(mlflow_client):
    from sklearn.linear_model import LogisticRegression

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    with mlflow.start_run() as run:
        assert run.info.artifact_uri.startswith("file://")
        mlflow.sklearn.log_model(LogisticRegression(), "model", registered_model_name="rmn")
        mlflow.pyfunc.load_model("models:/rmn/1")


def test_log_input(mlflow_client, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    path = tmp_path / "temp.csv"
    df.to_csv(path)
    dataset = from_pandas(df, source=path)

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)

    with mlflow.start_run() as run:
        mlflow.log_input(dataset, "train", {"foo": "baz"})

    dataset_inputs = mlflow_client.get_run(run.info.run_id).inputs.dataset_inputs

    assert len(dataset_inputs) == 1
    assert dataset_inputs[0].dataset.name == "dataset"
    assert dataset_inputs[0].dataset.digest == "f0f3e026"
    assert dataset_inputs[0].dataset.source_type == "local"
    assert json.loads(dataset_inputs[0].dataset.source) == {"uri": str(path)}
    assert json.loads(dataset_inputs[0].dataset.schema) == {
        "mlflow_colspec": [
            {"name": "a", "type": "long", "required": True},
            {"name": "b", "type": "long", "required": True},
            {"name": "c", "type": "long", "required": True},
        ]
    }
    assert json.loads(dataset_inputs[0].dataset.profile) == {"num_rows": 2, "num_elements": 6}

    assert len(dataset_inputs[0].tags) == 2
    assert dataset_inputs[0].tags[0].key == "foo"
    assert dataset_inputs[0].tags[0].value == "baz"
    assert dataset_inputs[0].tags[1].key == mlflow_tags.MLFLOW_DATASET_CONTEXT
    assert dataset_inputs[0].tags[1].value == "train"


def test_log_inputs(mlflow_client):
    experiment_id = mlflow_client.create_experiment("log inputs test")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    dataset1 = Dataset(
        name="name1",
        digest="digest1",
        source_type="source_type1",
        source="source1",
    )
    dataset_inputs1 = [DatasetInput(dataset=dataset1, tags=[InputTag(key="tag1", value="value1")])]

    mlflow_client.log_inputs(run_id, dataset_inputs1)
    run = mlflow_client.get_run(run_id)
    assert len(run.inputs.dataset_inputs) == 1

    assert isinstance(run.inputs, RunInputs)
    assert isinstance(run.inputs.dataset_inputs[0], DatasetInput)
    assert isinstance(run.inputs.dataset_inputs[0].dataset, Dataset)
    assert run.inputs.dataset_inputs[0].dataset.name == "name1"
    assert run.inputs.dataset_inputs[0].dataset.digest == "digest1"
    assert run.inputs.dataset_inputs[0].dataset.source_type == "source_type1"
    assert run.inputs.dataset_inputs[0].dataset.source == "source1"
    assert len(run.inputs.dataset_inputs[0].tags) == 1
    assert run.inputs.dataset_inputs[0].tags[0].key == "tag1"
    assert run.inputs.dataset_inputs[0].tags[0].value == "value1"


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


def test_create_promptlab_run_handler_returns_expected_results(mlflow_client):
    experiment_id = mlflow_client.create_experiment("log inputs test")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/runs/create-promptlab-run",
        json={
            "experiment_id": experiment_id,
            "run_name": "my_run_name",
            "prompt_template": "my_prompt_template",
            "prompt_parameters": [{"key": "my_key", "value": "my_value"}],
            "model_route": "my_route",
            "model_parameters": [{"key": "temperature", "value": "0.1"}],
            "model_input": "my_input",
            "model_output": "my_output",
            "model_output_parameters": [{"key": "latency", "value": "100"}],
            "mlflow_version": "1.0.0",
            "user_id": "username",
            "start_time": 456,
        },
    )
    assert response.status_code == 200
    run_json = response.json()
    assert run_json["run"]["info"]["run_name"] == "my_run_name"
    assert run_json["run"]["info"]["experiment_id"] == experiment_id
    assert run_json["run"]["info"]["user_id"] == "username"
    assert run_json["run"]["info"]["status"] == "FINISHED"
    assert run_json["run"]["info"]["start_time"] == 456

    assert {"key": "model_route", "value": "my_route"} in run_json["run"]["data"]["params"]
    assert {"key": "prompt_template", "value": "my_prompt_template"} in run_json["run"]["data"][
        "params"
    ]
    assert {"key": "temperature", "value": "0.1"} in run_json["run"]["data"]["params"]

    assert {
        "key": "mlflow.loggedArtifacts",
        "value": '[{"path": "eval_results_table.json", "type": "table"}]',
    } in run_json["run"]["data"]["tags"]
    assert {"key": "mlflow.runSourceType", "value": "PROMPT_ENGINEERING"} in run_json["run"][
        "data"
    ]["tags"]


def test_gateway_proxy_handler_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    with _init_server(
        backend_uri=mlflow_client.tracking_uri,
        root_artifact_uri=mlflow_client.tracking_uri,
        extra_env={"MLFLOW_DEPLOYMENTS_TARGET": "http://localhost:5001"},
    ) as url:
        patched_client = MlflowClient(url)

        response = requests.post(
            f"{patched_client.tracking_uri}/ajax-api/2.0/mlflow/gateway-proxy",
            json={},
        )
        assert_response(
            response,
            "Deployments proxy request must specify a gateway_path.",
        )


def test_upload_artifact_handler_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    experiment_id = mlflow_client.create_experiment("upload_artifacts_test")
    created_run = mlflow_client.create_run(experiment_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact", params={}
    )
    assert_response(response, "Request must specify run_uuid.")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact",
        params={
            "run_uuid": created_run.info.run_id,
        },
    )
    assert_response(response, "Request must specify path.")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact",
        params={"run_uuid": created_run.info.run_id, "path": ""},
    )
    assert_response(response, "Request must specify path.")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact",
        params={"run_uuid": created_run.info.run_id, "path": "../test.txt"},
    )
    assert_response(response, "Invalid path")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact",
        params={
            "run_uuid": created_run.info.run_id,
            "path": "test.txt",
        },
    )
    assert_response(response, "Request must specify data.")


def test_upload_artifact_handler(mlflow_client):
    experiment_id = mlflow_client.create_experiment("upload_artifacts_test")
    created_run = mlflow_client.create_run(experiment_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/upload-artifact",
        params={
            "run_uuid": created_run.info.run_id,
            "path": "test.txt",
        },
        data="hello world",
    )
    assert response.status_code == 200

    response = requests.get(
        f"{mlflow_client.tracking_uri}/get-artifact",
        params={
            "run_uuid": created_run.info.run_id,
            "path": "test.txt",
        },
    )
    assert response.status_code == 200
    assert response.text == "hello world"


def test_graphql_handler(mlflow_client):
    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": 'query testQuery {test(inputString: "abc") { output }}',
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200


def test_get_experiment_graphql(mlflow_client):
    experiment_id = mlflow_client.create_experiment("GraphqlTest")
    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": 'query testQuery {mlflowGetExperiment(input: {experimentId: "'
            + experiment_id
            + '"}) { experiment { name } }}',
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200
    json = response.json()
    assert json["data"]["mlflowGetExperiment"]["experiment"]["name"] == "GraphqlTest"


def test_get_run_and_experiment_graphql(mlflow_client):
    name = "GraphqlTest"
    mlflow_client.create_registered_model(name)
    experiment_id = mlflow_client.create_experiment(name)
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    mlflow_client.create_model_version("GraphqlTest", "runs:/graphql_test/model", run_id)
    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                query testQuery @component(name: "Test") {{
                    mlflowGetRun(input: {{runId: "{run_id}"}}) {{
                        run {{
                            info {{
                                status
                            }}
                            experiment {{
                                name
                            }}
                            modelVersions {{
                                name
                            }}
                        }}
                    }}
                }}
            """,
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200
    json = response.json()
    assert json["errors"] is None
    assert json["data"]["mlflowGetRun"]["run"]["info"]["status"] == created_run.info.status
    assert json["data"]["mlflowGetRun"]["run"]["experiment"]["name"] == name
    assert json["data"]["mlflowGetRun"]["run"]["modelVersions"][0]["name"] == name


def test_start_and_end_trace(mlflow_client):
    experiment_id = mlflow_client.create_experiment("start end trace")

    # Trace CRUD APIs are not directly exposed as public API of MlflowClient,
    # so we use the underlying tracking client to test them.
    client = mlflow_client._tracking_client

    # Helper function to remove auto-added system tags (mlflow.xxx) from testing
    def _exclude_system_tags(tags: dict[str, str]):
        return {k: v for k, v in tags.items() if not k.startswith("mlflow.")}

    trace_info = client.start_trace(
        experiment_id=experiment_id,
        timestamp_ms=1000,
        request_metadata={
            "meta1": "apple",
            "meta2": "grape",
        },
        tags={
            "tag1": "football",
            "tag2": "basketball",
        },
    )
    assert trace_info.request_id is not None
    assert trace_info.experiment_id == experiment_id
    assert trace_info.timestamp_ms == 1000
    assert trace_info.execution_time_ms == 0
    assert trace_info.status == TraceStatus.IN_PROGRESS
    assert trace_info.request_metadata == {
        "meta1": "apple",
        "meta2": "grape",
    }
    assert _exclude_system_tags(trace_info.tags) == {
        "tag1": "football",
        "tag2": "basketball",
    }

    trace_info = client.end_trace(
        request_id=trace_info.request_id,
        timestamp_ms=3000,
        status=TraceStatus.OK,
        request_metadata={
            "meta1": "orange",
            "meta3": "banana",
        },
        tags={
            "tag1": "soccer",
            "tag3": "tennis",
        },
    )
    assert trace_info.request_id is not None
    assert trace_info.experiment_id == experiment_id
    assert trace_info.timestamp_ms == 1000
    assert trace_info.execution_time_ms == 2000
    assert trace_info.status == TraceStatus.OK
    assert trace_info.request_metadata == {
        "meta1": "orange",
        "meta2": "grape",
        "meta3": "banana",
    }
    assert _exclude_system_tags(trace_info.tags) == {
        "tag1": "soccer",
        "tag2": "basketball",
        "tag3": "tennis",
    }

    assert trace_info == client.get_trace_info(trace_info.request_id)


def test_start_and_end_trace_non_string_name(mlflow_client):
    # OpenTelemetry span can accept non-string name like 1234. However, it is problematic
    # when we use it as a trace name (which is set from a root span name) and log it to
    # remote tracking server. Trace name is stored as mlflow.traceName tag and tag value
    # can only be string, otherwise protobuf serialization will fail. Therefore, this test
    # verifies that non-string span name is correctly handled before sending to the server.
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    exp_id = mlflow_client.create_experiment("non-string trace")

    span = mlflow_client.start_trace(name=1234, experiment_id=exp_id)
    child_span = mlflow_client.start_span(
        name=None, request_id=span.request_id, parent_id=span.span_id
    )
    mlflow_client.end_span(
        request_id=child_span.request_id, span_id=child_span.span_id, status="OK"
    )
    mlflow_client.end_trace(request_id=span.request_id, status="OK")

    traces = mlflow_client.search_traces(experiment_ids=[exp_id])
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.tags[TraceTagKey.TRACE_NAME] == "1234"
    assert trace.info.status == TraceStatus.OK
    assert len(trace.data.spans) == 2
    assert trace.data.spans[0].name == 1234
    assert trace.data.spans[0].status.status_code == "OK"
    assert trace.data.spans[1].name is None
    assert trace.data.spans[1].status.status_code == "OK"


def test_search_traces(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow_client.create_experiment("search traces")

    # Create test traces
    def _create_trace(name, status):
        span = mlflow_client.start_trace(name=name, experiment_id=experiment_id)
        mlflow_client.end_trace(request_id=span.request_id, status=status)
        return span.request_id

    request_id_1 = _create_trace(name="trace1", status=TraceStatus.OK)
    request_id_2 = _create_trace(name="trace2", status=TraceStatus.OK)
    request_id_3 = _create_trace(name="trace3", status=TraceStatus.ERROR)

    def _get_request_ids(traces):
        return [t.info.request_id for t in traces]

    # Validate search
    traces = mlflow_client.search_traces(experiment_ids=[experiment_id])
    assert _get_request_ids(traces) == [request_id_3, request_id_2, request_id_1]
    assert traces.token is None

    traces = mlflow_client.search_traces(
        experiment_ids=[experiment_id],
        filter_string="status = 'OK'",
        order_by=["timestamp ASC"],
    )
    assert _get_request_ids(traces) == [request_id_1, request_id_2]
    assert traces.token is None

    traces = mlflow_client.search_traces(
        experiment_ids=[experiment_id],
        max_results=2,
    )
    assert _get_request_ids(traces) == [request_id_3, request_id_2]
    assert traces.token is not None
    traces = mlflow_client.search_traces(
        experiment_ids=[experiment_id],
        page_token=traces.token,
    )
    assert _get_request_ids(traces) == [request_id_1]
    assert traces.token is None


def test_delete_traces(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow_client.create_experiment("delete traces")

    def _create_trace(name, status):
        span = mlflow_client.start_trace(name=name, experiment_id=experiment_id)
        mlflow_client.end_trace(request_id=span.request_id, status=status)
        return span.request_id

    def _is_trace_exists(request_id):
        try:
            trace_info = mlflow_client._tracking_client.get_trace_info(request_id)
            return trace_info is not None
        except RestException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                return False
            raise

    # Case 1: Delete all traces under experiment ID
    request_id_1 = _create_trace(name="trace1", status=TraceStatus.OK)
    request_id_2 = _create_trace(name="trace2", status=TraceStatus.OK)
    assert _is_trace_exists(request_id_1)
    assert _is_trace_exists(request_id_2)

    deleted_count = mlflow_client.delete_traces(experiment_id, max_timestamp_millis=int(1e15))
    assert deleted_count == 2
    assert not _is_trace_exists(request_id_1)
    assert not _is_trace_exists(request_id_2)

    # Case 2: Delete with max_traces limit
    request_id_1 = _create_trace(name="trace1", status=TraceStatus.OK)
    time.sleep(0.1)  # Add some time gap to avoid timestamp collision
    request_id_2 = _create_trace(name="trace2", status=TraceStatus.OK)

    deleted_count = mlflow_client.delete_traces(
        experiment_id, max_traces=1, max_timestamp_millis=int(1e15)
    )
    assert deleted_count == 1
    # TODO: Currently the deletion order in the file store is random (based on
    # the order of the trace files in the directory), so we don't validate which
    # one is deleted. Uncomment the following lines once the deletion order is fixed.
    # assert not _is_trace_exists(request_id_1)  # Old created trace should be deleted
    # assert _is_trace_exists(request_id_2)

    # Case 3: Delete with explicit request ID
    request_id_1 = _create_trace(name="trace1", status=TraceStatus.OK)
    request_id_2 = _create_trace(name="trace2", status=TraceStatus.OK)

    deleted_count = mlflow_client.delete_traces(experiment_id, request_ids=[request_id_1])
    assert deleted_count == 1
    assert not _is_trace_exists(request_id_1)
    assert _is_trace_exists(request_id_2)


def test_set_and_delete_trace_tag(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    experiment_id = mlflow_client.create_experiment("set delete tag")

    # Create test trace
    trace_info = mlflow_client._tracking_client.start_trace(
        experiment_id=experiment_id,
        timestamp_ms=1000,
        request_metadata={},
        tags={
            "tag1": "red",
            "tag2": "blue",
        },
    )

    # Validate set tag
    mlflow_client.set_trace_tag(trace_info.request_id, "tag1", "green")
    trace_info = mlflow_client._tracking_client.get_trace_info(trace_info.request_id)
    assert trace_info.tags["tag1"] == "green"

    # Validate delete tag
    mlflow_client.delete_trace_tag(trace_info.request_id, "tag2")
    trace_info = mlflow_client._tracking_client.get_trace_info(trace_info.request_id)
    assert "tag2" not in trace_info.tags


def test_get_trace_artifact_handler(mlflow_client):
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)

    experiment_id = mlflow_client.create_experiment("get trace artifact")

    span = mlflow_client.start_trace(name="test", experiment_id=experiment_id)
    request_id = span.request_id
    span.set_attributes({"fruit": "apple"})
    mlflow_client.end_trace(request_id=request_id)

    response = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/get-trace-artifact",
        params={"request_id": request_id},
    )
    assert response.status_code == 200
    assert response.headers["Content-Disposition"] == "attachment; filename=traces.json"

    # Validate content
    trace_data = TraceData.from_dict(json.loads(response.text))
    assert trace_data.spans[0].to_dict() == span.to_dict()


def test_get_metric_history_bulk_interval_graphql(mlflow_client):
    name = "GraphqlTest"
    mlflow_client.create_registered_model(name)
    experiment_id = mlflow_client.create_experiment(name)
    created_run = mlflow_client.create_run(experiment_id)

    metric_name = "metric_0"
    for i in range(10):
        mlflow_client.log_metric(created_run.info.run_id, metric_name, i, step=i)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                query testQuery {{
                    mlflowGetMetricHistoryBulkInterval(input: {{
                        runIds: ["{created_run.info.run_id}"],
                        metricKey: "{metric_name}",
                    }}) {{
                        metrics {{
                            key
                            timestamp
                            value
                        }}
                    }}
                }}
            """,
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    json = response.json()
    expected = [{"key": metric_name, "timestamp": mock.ANY, "value": i} for i in range(10)]
    assert json["data"]["mlflowGetMetricHistoryBulkInterval"]["metrics"] == expected


def test_search_runs_graphql(mlflow_client):
    name = "GraphqlTest"
    mlflow_client.create_registered_model(name)
    experiment_id = mlflow_client.create_experiment(name)
    created_run_1 = mlflow_client.create_run(experiment_id)
    created_run_2 = mlflow_client.create_run(experiment_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                mutation testMutation {{
                    mlflowSearchRuns(input: {{ experimentIds: ["{experiment_id}"] }}) {{
                        runs {{
                            info {{
                                runId
                            }}
                        }}
                    }}
                }}
            """,
            "operationName": "testMutation",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    json = response.json()
    expected = [
        {"info": {"runId": created_run_2.info.run_id}},
        {"info": {"runId": created_run_1.info.run_id}},
    ]
    assert json["data"]["mlflowSearchRuns"]["runs"] == expected


def test_list_artifacts_graphql(mlflow_client, tmp_path):
    name = "GraphqlTest"
    experiment_id = mlflow_client.create_experiment(name)
    created_run_id = mlflow_client.create_run(experiment_id).info.run_id
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")
    mlflow_client.log_artifact(created_run_id, file_path.absolute().as_posix())
    mlflow_client.log_artifact(created_run_id, file_path.absolute().as_posix(), "testDir")

    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                fragment FilesFragment on MlflowListArtifactsResponse {{
                    files {{
                        path
                        isDir
                        fileSize
                    }}
                }}

                query testQuery {{
                    file: mlflowListArtifacts(input: {{ runId: "{created_run_id}" }}) {{
                        ...FilesFragment
                    }}
                    subdir: mlflowListArtifacts(input: {{
                        runId: "{created_run_id}",
                        path: "testDir",
                    }}) {{
                        ...FilesFragment
                    }}
                }}
            """,
            "operationName": "testQuery",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    json = response.json()
    file_expected = [
        {"path": "test.txt", "isDir": False, "fileSize": "11"},
        {"path": "testDir", "isDir": True, "fileSize": "0"},
    ]
    assert json["data"]["file"]["files"] == file_expected
    subdir_expected = [
        {"path": "testDir/test.txt", "isDir": False, "fileSize": "11"},
    ]
    assert json["data"]["subdir"]["files"] == subdir_expected


def test_search_datasets_graphql(mlflow_client):
    name = "GraphqlTest"
    experiment_id = mlflow_client.create_experiment(name)
    created_run_id = mlflow_client.create_run(experiment_id).info.run_id
    dataset1 = Dataset(
        name="test-dataset-1",
        digest="12345",
        source_type="script",
        source="test",
    )
    dataset_input1 = DatasetInput(dataset=dataset1, tags=[])
    dataset2 = Dataset(
        name="test-dataset-2",
        digest="12346",
        source_type="script",
        source="test",
    )
    dataset_input2 = DatasetInput(
        dataset=dataset2, tags=[InputTag(key=MLFLOW_DATASET_CONTEXT, value="training")]
    )
    mlflow_client.log_inputs(created_run_id, [dataset_input1, dataset_input2])

    response = requests.post(
        f"{mlflow_client.tracking_uri}/graphql",
        json={
            "query": f"""
                mutation testMutation {{
                    mlflowSearchDatasets(input:{{experimentIds: ["{experiment_id}"]}}) {{
                        datasetSummaries {{
                            experimentId
                            name
                            digest
                            context
                        }}
                    }}
                }}
            """,
            "operationName": "testMutation",
        },
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    json = response.json()

    def sort_dataset_summaries(l1):
        return sorted(l1, key=lambda x: x["digest"])

    expected = sort_dataset_summaries(
        [
            {
                "experimentId": experiment_id,
                "name": "test-dataset-2",
                "digest": "12346",
                "context": "training",
            },
            {
                "experimentId": experiment_id,
                "name": "test-dataset-1",
                "digest": "12345",
                "context": "",
            },
        ]
    )
    assert (
        sort_dataset_summaries(json["data"]["mlflowSearchDatasets"]["datasetSummaries"]) == expected
    )
