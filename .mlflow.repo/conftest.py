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


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    outcome = yield
    if report.when == "call":
        try:
            import psutil
        except ImportError:
            return

        (*rest, result) = outcome.get_result()
        mem = psutil.virtual_memory()
        mem_used = mem.used / 1024**3
        mem_total = mem.total / 1024**3

        disk = psutil.disk_usage("/")
        disk_used = disk.used / 1024**3
        disk_total = disk.total / 1024**3
        outcome.force_result(
            (
                *rest,
                (
                    f"{result} | "
                    f"MEM {mem_used:.1f}/{mem_total:.1f} GB | "
                    f"DISK {disk_used:.1f}/{disk_total:.1f} GB"
                ),
            )
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_ignore_collect(collection_path, config):
    outcome = yield
    if not outcome.get_result() and config.getoption("ignore_flavors"):
        # If not ignored by the default hook and `--ignore-flavors` specified

        # Ignored files and directories must be included in dev/run-python-flavor-tests.sh
        model_flavors = [
            # Tests of flavor modules.
            "tests/autogen",
            "tests/azureml",
            "tests/catboost",
            "tests/diviner",
            "tests/dspy",
            "tests/fastai",
            "tests/gemini",
            "tests/gluon",
            "tests/h2o",
            "tests/johnsnowlabs",
            "tests/keras",
            "tests/keras_core",
            "tests/llama_index",
            "tests/langchain",
            "tests/lightgbm",
            "tests/litellm",
            "tests/mleap",
            "tests/models",
            "tests/onnx",
            "tests/openai",
            "tests/paddle",
            "tests/pmdarima",
            "tests/promptflow",
            "tests/prophet",
            "tests/pyfunc",
            "tests/pytorch",
            "tests/sagemaker",
            "tests/sentence_transformers",
            "tests/shap",
            "tests/sklearn",
            "tests/spacy",
            "tests/spark",
            "tests/statsmodels",
            "tests/tensorflow",
            "tests/transformers",
            "tests/xgboost",
            # Lazy loading test.
            "tests/test_mlflow_lazily_imports_ml_packages.py",
            # Tests of utils.
            "tests/utils/test_model_utils.py",
            # This test is included here because it imports many big libraries like tf, keras, etc.
            "tests/tracking/fluent/test_fluent_autolog.py",
            # Cross flavor autologging related tests.
            "tests/autologging/test_autologging_safety_unit.py",
            "tests/autologging/test_autologging_behaviors_unit.py",
            "tests/autologging/test_autologging_behaviors_integration.py",
            "tests/autologging/test_autologging_utils.py",
            "tests/autologging/test_training_session.py",
            # Opt in authentication feature.
            "tests/server/auth",
            "tests/gateway",
        ]

        relpath = os.path.relpath(str(collection_path))
        relpath = relpath.replace(os.sep, posixpath.sep)  # for Windows

        if relpath in model_flavors:
            outcome.force_result(True)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    # Executing `tests.server.test_prometheus_exporter` after `tests.server.test_handlers`
    # results in an error because Flask >= 2.2.0 doesn't allow calling setup method such as
    # `before_request` on the application after the first request. To avoid this issue,
    # execute `tests.server.test_prometheus_exporter` first by reordering the test items.
    items.sort(key=lambda item: item.module.__name__ != "tests.server.test_prometheus_exporter")

    # Select the tests to run based on the group and splits
    if (splits := config.getoption("--splits")) and (group := config.getoption("--group")):
        items[:] = items[(group - 1) :: splits]


@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    yield
    failed_test_reports = terminalreporter.stats.get("failed", [])
    if failed_test_reports:
        if len(failed_test_reports) <= 30:
            terminalreporter.section("command to run failed test cases")
            ids = [repr(report.nodeid) for report in failed_test_reports]
        else:
            terminalreporter.section("command to run failed test suites")
            # Use dict.fromkeys to preserve the order
            ids = list(dict.fromkeys(report.fspath for report in failed_test_reports))
        terminalreporter.write(" ".join(["pytest"] + ids))
        terminalreporter.write("\n" * 2)

        # If some tests failed at installing mlflow, we suggest using `--serve-wheel` flag.
        # Some test cases try to install mlflow via pip e.g. model loading. They pins
        # mlflow version to install based on local environment i.e. dev version ahead of
        # the latest release, hence it's not found on PyPI. `--serve-wheel` flag was
        # introduced to resolve this issue, which starts local PyPI server and serve
        # an mlflow wheel based on local source code.
        # Ref: https://github.com/mlflow/mlflow/pull/10247
        msg = f"No matching distribution found for mlflow=={VERSION}"
        for rep in failed_test_reports:
            if any(msg in t for t in (rep.longreprtext, rep.capstdout, rep.capstderr)):
                terminalreporter.section("HINTS", yellow=True)
                terminalreporter.write(
                    f"Found test(s) that failed with {msg!r}. Adding"
                    " --serve-wheel` flag to your pytest command may help.\n\n",
                    yellow=True,
                )
                break

    main_thread = threading.main_thread()
    if threads := [t for t in threading.enumerate() if t is not main_thread]:
        terminalreporter.section("Remaining threads", yellow=True)
        for idx, thread in enumerate(threads, start=1):
            terminalreporter.write(f"{idx}: {thread}\n")

        if non_daemon_threads := [t for t in threads if not t.daemon]:
            frames = sys._current_frames()
            terminalreporter.section("Tracebacks of non-daemon threads", yellow=True)
            for thread in non_daemon_threads:
                thread.join(timeout=1)
                if thread.is_alive() and (frame := frames.get(thread.ident)):
                    terminalreporter.section(repr(thread), sep="~")
                    terminalreporter.write("".join(traceback.format_stack(frame)))

    try:
        import psutil
    except ImportError:
        pass
    else:
        current_process = psutil.Process()
        if children := current_process.children(recursive=True):
            terminalreporter.section("Remaining child processes", yellow=True)
            for idx, child in enumerate(children, start=1):
                terminalreporter.write(f"{idx}: {child}\n")


@pytest.fixture(scope="session", autouse=True)
def enable_mlflow_testing():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_MLFLOW_TESTING.name, "TRUE")
        yield
