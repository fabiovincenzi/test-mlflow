"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import json
import logging
import sys

import mlflow.experiments
import mlflow.pyfunc
import pytest
import requests
from mlflow import MlflowClient

from tests.tracking.integration_test_utils import (
    _init_server,
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
from mlflow.server.handlers import _get_sampled_steps_from_steps
from mlflow.utils import mlflow_tags
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATASET_CONTEXT,
)

from tests.tracking.integration_test_utils import (
    _init_server,
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
