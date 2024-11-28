"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import logging
import pathlib
import sys

import pytest
from mlflow import MlflowClient
from mlflow.entities import (
    ViewType,
)
from mlflow.utils.os import is_windows

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


def test_create_get_search_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        "My Experiment", artifact_location="my_location", tags={"key1": "val1", "key2": "val2"}
    )
    exp = mlflow_client.get_experiment(experiment_id)
    assert exp.name == "My Experiment"
    if is_windows():
        assert exp.artifact_location == pathlib.Path.cwd().joinpath("my_location").as_uri()
    else:
        assert exp.artifact_location == str(pathlib.Path.cwd().joinpath("my_location"))
    assert len(exp.tags) == 2
    assert exp.tags["key1"] == "val1"
    assert exp.tags["key2"] == "val2"

    experiments = mlflow_client.search_experiments()
    assert {e.name for e in experiments} == {"My Experiment", "Default"}
    mlflow_client.delete_experiment(experiment_id)
    assert {e.name for e in mlflow_client.search_experiments()} == {"Default"}
    assert {e.name for e in mlflow_client.search_experiments(view_type=ViewType.ACTIVE_ONLY)} == {
        "Default"
    }
    assert {e.name for e in mlflow_client.search_experiments(view_type=ViewType.DELETED_ONLY)} == {
        "My Experiment"
    }
    assert {e.name for e in mlflow_client.search_experiments(view_type=ViewType.ALL)} == {
        "My Experiment",
        "Default",
    }
    active_exps_paginated = mlflow_client.search_experiments(max_results=1)
    assert {e.name for e in active_exps_paginated} == {"Default"}
    assert active_exps_paginated.token is None

    all_exps_paginated = mlflow_client.search_experiments(max_results=1, view_type=ViewType.ALL)
    first_page_names = {e.name for e in all_exps_paginated}
    all_exps_second_page = mlflow_client.search_experiments(
        max_results=1, view_type=ViewType.ALL, page_token=all_exps_paginated.token
    )
    second_page_names = {e.name for e in all_exps_second_page}
    assert len(first_page_names) == 1
    assert len(second_page_names) == 1
    assert first_page_names.union(second_page_names) == {"Default", "My Experiment"}
