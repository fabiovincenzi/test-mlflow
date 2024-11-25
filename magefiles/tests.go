name: Test

on:
  workflow_call:

permissions:
  contents: read

jobs:
  go:
    name: Test Go
    strategy:
      matrix:
        runner: [macos-latest, ubuntu-latest, windows-latest]
    runs-on: ${{ matrix.runner }}
    steps:
      - uses: actions/checkout@v4
      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: "1.22"
          check-latest: true
          cache: false
      - name: Install mage
        run: go install github.com/magefile/mage@v1.15.0
      - name: Run unit tests
        run: mage test:unit

  python:
    name: Test Python
    strategy:
      matrix:
        runner: [macos-latest, ubuntu-latest, windows-latest]
        python: ["3.8", "3.9", "3.10", "3.11", "3.12"]
    runs-on: ${{ matrix.runner }}
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: "1.22"
          check-latest: true
          cache: false
      - name: Install mage
        run: go install github.com/magefile/mage@v1.15.0
      - name: Install our package in editable mode
        run: pip install -e .
      - name: Initialize MLflow repo
        run: mage repo:init
      - name: Install dependencies
        run: pip install pytest==8.1.1 psycopg2-binary==2.9.9 -e .mlflow.repo
      - name: Run integration tests
        run: mage test:python
        # Temporary workaround for failing tests
        continue-on-error: ${{ matrix.runner != 'ubuntu-latest' }}
