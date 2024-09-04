// Code generated by mlflow/go/cmd/generate/main.go. DO NOT EDIT.

package main

import "C"
import (
	"unsafe"
	"github.com/mlflow/mlflow-go/pkg/protos"
)
//export TrackingServiceGetExperimentByName
func TrackingServiceGetExperimentByName(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.GetExperimentByName, new(protos.GetExperimentByName), requestData, requestSize, responseSize)
}
//export TrackingServiceCreateExperiment
func TrackingServiceCreateExperiment(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.CreateExperiment, new(protos.CreateExperiment), requestData, requestSize, responseSize)
}
//export TrackingServiceGetExperiment
func TrackingServiceGetExperiment(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.GetExperiment, new(protos.GetExperiment), requestData, requestSize, responseSize)
}
//export TrackingServiceDeleteExperiment
func TrackingServiceDeleteExperiment(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.DeleteExperiment, new(protos.DeleteExperiment), requestData, requestSize, responseSize)
}
//export TrackingServiceRestoreExperiment
func TrackingServiceRestoreExperiment(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.RestoreExperiment, new(protos.RestoreExperiment), requestData, requestSize, responseSize)
}
//export TrackingServiceUpdateExperiment
func TrackingServiceUpdateExperiment(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.UpdateExperiment, new(protos.UpdateExperiment), requestData, requestSize, responseSize)
}
//export TrackingServiceCreateRun
func TrackingServiceCreateRun(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.CreateRun, new(protos.CreateRun), requestData, requestSize, responseSize)
}
//export TrackingServiceUpdateRun
func TrackingServiceUpdateRun(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.UpdateRun, new(protos.UpdateRun), requestData, requestSize, responseSize)
}
//export TrackingServiceLogMetric
func TrackingServiceLogMetric(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.LogMetric, new(protos.LogMetric), requestData, requestSize, responseSize)
}
//export TrackingServiceSearchRuns
func TrackingServiceSearchRuns(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.SearchRuns, new(protos.SearchRuns), requestData, requestSize, responseSize)
}
//export TrackingServiceLogBatch
func TrackingServiceLogBatch(serviceID int64, requestData unsafe.Pointer, requestSize C.int, responseSize *C.int) unsafe.Pointer {
	service, err := trackingServices.Get(serviceID)
	if err != nil {
		return makePointerFromError(err, responseSize)
	}
	return invokeServiceMethod(service.LogBatch, new(protos.LogBatch), requestData, requestSize, responseSize)
}
