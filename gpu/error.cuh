#pragma once
#include <cstdio>
#include "cublas_v2.h"

#define CHECK(call)                                                   \
  do {                                                                \
    const cudaError_t error_code = call;                              \
    if (error_code != cudaSuccess) {                                  \
      printf("CUDA Error:\n");                                        \
      printf("    File:       %s\n", __FILE__);                       \
      printf("    Line:       %d\n", __LINE__);                       \
      printf("    Error code: %d\n", error_code);                     \
      printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

// Macro for checking cuBLAS return codes
#define CUBLAS_CHECK(call)                                                                              \
  do {                                                                                                  \
    cublasStatus_t status = (call);                                                                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                              \
      fprintf(stderr, "cuBLAS Error: %s at %s:%d\n", cublasGetErrorString(status), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                               \
    }                                                                                                   \
  } while (false)

// Function to get a string representation of cuBLAS status
char const*
cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "Unknown cuBLAS error";
  }
}
