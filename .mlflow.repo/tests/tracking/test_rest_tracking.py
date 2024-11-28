"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""

import logging
import sys

import pytest
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

import logging
import sys

import pytest
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
