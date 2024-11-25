//go:build mage

//nolint:wrapcheck
package main

import (
	"runtime"

	"github.com/magefile/mage/mg"
	"github.com/magefile/mage/sh"
)

type Test mg.Namespace

func runPythonTests(pytestArgs []string) error {
	// Prepara gli argomenti per pytest
	args := []string{
		"pytest",
		"-p", "no:warnings",
		"--confcutdir=.",
		"-k", "not [file",
	}
	args = append(args, pytestArgs...)

	// Variabili di ambiente da esportare per i test
	environmentVariables := map[string]string{
		"MLFLOW_GO_LIBRARY_PATH": ".", // Modifica se necessario
	}

	if runtime.GOOS == "windows" {
		environmentVariables["MLFLOW_SQLALCHEMYSTORE_POOLCLASS"] = "NullPool"
	}

	// Esegui pytest direttamente
	if err := sh.RunWithV(environmentVariables, "python3", args...); err != nil {
		return err
	}

	return nil
}

// Run mlflow Python tests against the Go backend.
func (Test) Python() error {
	return runPythonTests([]string{
		".mlflow.repo/tests/tracking/test_rest_tracking.py",
		".mlflow.repo/tests/tracking/test_model_registry.py",
		".mlflow.repo/tests/store/tracking/test_sqlalchemy_store.py",
		".mlflow.repo/tests/store/model_registry/test_sqlalchemy_store.py",
	})
}

// Run specific Python test against the Go backend.
func (Test) PythonSpecific(testName string) error {
	return runPythonTests([]string{
		testName,
		"-vv",
	})
}

// Run the Go unit tests.
func (Test) Unit() error {
	return sh.RunV("go", "test", "./pkg/...")
}

// Run all tests.
func (Test) All() {
	mg.Deps(Test.Unit, Test.Python)
}
