//go:build mage

//nolint:wrapcheck
package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/magefile/mage/mg"
	"github.com/magefile/mage/sh"
)

type Test mg.Namespace

// cleanUpMemoryFile rimuove eventuali file :memory: residui.
func cleanUpMemoryFile() error {
	filename := ":memory:"
	_, err := os.Stat(filename)
	if err == nil {
		// Il file esiste, rimuovilo
		log.Printf("Cleaning up memory file: %s", filename)
		err = os.Remove(filename)
		if err != nil {
			return fmt.Errorf("failed to clean up memory file: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("error checking memory file: %w", err)
	}
	return nil
}

// logMemoryUsage stampa l'utilizzo della memoria.
func logMemoryUsage(stage string) {
	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("vm_stat")
	case "linux":
		cmd = exec.Command("free", "-m")
	default:
		log.Printf("Memory usage logging is not supported on %s", runtime.GOOS)
		return
	}

	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Failed to get memory usage at stage '%s': %v", stage, err)
		return
	}
	log.Printf("Memory usage at stage '%s':\n%s", stage, string(output))
}

// Python esegue i test Python contro il backend Go.
func (Test) Python() error {
	log.Println("Starting Python tests...")

	// Traccia utilizzo della memoria prima dei test
	logMemoryUsage("before tests")

	libpath, err := os.MkdirTemp("", "mlflow_go_*")
	if err != nil {
		return fmt.Errorf("failed to create temporary directory: %w", err)
	}
	log.Printf("Temporary directory created at: %s", libpath)

	defer func() {
		log.Println("Cleaning up temporary resources...")
		if err := os.RemoveAll(libpath); err != nil {
			log.Printf("Failed to clean up temporary directory: %v", err)
		}
		if err := cleanUpMemoryFile(); err != nil {
			log.Printf("Failed to clean up memory file: %v", err)
		}
	}()

	// Build Go binary in a temporary directory
	log.Println("Building Go binary...")
	if err := sh.RunV("python", "-m", "mlflow_go.lib", ".", libpath); err != nil {
		return fmt.Errorf("failed to build Go binary: %w", err)
	}

	// Run the tests
	log.Println("Running Python tests...")
	testEnv := map[string]string{
		"MLFLOW_GO_LIBRARY_PATH": libpath,
	}
	if err := sh.RunWithV(testEnv, "pytest",
		"--confcutdir=.",
		".mlflow.repo/tests/tracking/test_rest_tracking.py",
		"-k", "not [file",
		"-vv",
	); err != nil {
		return fmt.Errorf("Python tests failed: %w", err)
	}

	// Traccia utilizzo della memoria dopo i test
	logMemoryUsage("after tests")

	log.Println("Python tests completed successfully.")
	return nil
}

// Unit esegue i test unitari Go.
func (Test) Unit() error {
	log.Println("Starting Go unit tests...")
	if err := sh.RunV("go", "test", "./pkg/..."); err != nil {
		return fmt.Errorf("Go unit tests failed: %w", err)
	}
	log.Println("Go unit tests completed successfully.")
	return nil
}

// All esegue tutti i test.
func (Test) All() {
	mg.Deps(Test.Unit, Test.Python)
}
