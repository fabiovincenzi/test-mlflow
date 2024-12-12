import json
import os
import posixpath
import sys
import threading
import traceback

import click
import pytest
from mlflow.environment_variables import _MLFLOW_TESTING, MLFLOW_TRACKING_URI
from mlflow.version import VERSION


def pytest_addoption(parser):
    parser.addoption(
        "--requires-ssh",
        action="store_true",
        dest="requires_ssh",
        default=False,
        help="Run tests decorated with 'requires_ssh' annotation. "
        "These tests require keys to be configured locally "
        "for SSH authentication.",
    )
    parser.addoption(
        "--ignore-flavors",
        action="store_true",
        dest="ignore_flavors",
        default=False,
        help="Ignore tests for model flavors.",
    )
    parser.addoption(
        "--splits",
        default=None,
        type=int,
        help="The number of groups to split tests into.",
    )
    parser.addoption(
        "--group",
        default=None,
        type=int,
        help="The group of tests to run.",
    )
    parser.addoption(
        "--serve-wheel",
        action="store_true",
        default=os.getenv("CI", "false").lower() == "true",
        help="Serve a wheel for the dev version of MLflow. True by default in CI, False otherwise.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_ssh")
    config.addinivalue_line("markers", "notrackingurimock")
    config.addinivalue_line("markers", "allow_infer_pip_requirements_fallback")
    config.addinivalue_line(
        "markers", "do_not_disable_new_import_hook_firing_if_module_already_exists"
    )
    config.addinivalue_line("markers", "classification")

    labels = fetch_pr_labels() or []
    if "fail-fast" in labels:
        config.option.maxfail = 1


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    group = config.getoption("group")
    splits = config.getoption("splits")

    if splits is None and group is None:
        return None

    if splits and group is None:
        raise pytest.UsageError("`--group` is required")

    if group and splits is None:
        raise pytest.UsageError("`--splits` is required")

    if splits < 0:
        raise pytest.UsageError("`--splits` must be >= 1")

    if group < 1 or group > splits:
        raise pytest.UsageError("`--group` must be between 1 and {splits}")

    return None


def pytest_sessionstart(session):
    if uri := MLFLOW_TRACKING_URI.get():
        click.echo(
            click.style(
                (
                    f"Environment variable {MLFLOW_TRACKING_URI} is set to {uri!r}, "
                    "which may interfere with tests."
                ),
                fg="red",
            )
        )


def pytest_runtest_setup(item):
    markers = [mark.name for mark in item.iter_markers()]
    if "requires_ssh" in markers and not item.config.getoption("--requires-ssh"):
        pytest.skip("use `--requires-ssh` to run this test")


def fetch_pr_labels():
    """
    Returns the labels associated with the current pull request.
    """
    if "GITHUB_ACTIONS" not in os.environ:
        return None

    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return None

    with open(os.environ["GITHUB_EVENT_PATH"]) as f:
        pr_data = json.load(f)
        return [label["name"] for label in pr_data["pull_request"]["labels"]]


