#pragma once
#include <cublas_v2.h>
#include "error.cuh"
#include "unit.cuh"

struct MatMul {
  cublasHandle_t handle;
  MatMul() { CUBLAS_CHECK(cublasCreate(&handle)); }
  ~MatMul() { CUBLAS_CHECK(cublasDestroy(handle)); }

  void
  operator()(int n_ch, int n_e_obs, int n_phase_bins, float* rsp_gpu, float* flux_gpu, float* output_gpu, float alpha) {
    constexpr float beta = 0.0;
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,

        n_phase_bins,
        n_ch,
        n_e_obs,

        &alpha,

        flux_gpu,
        n_phase_bins,

        rsp_gpu,
        n_e_obs,

        &beta,

        output_gpu,
        n_phase_bins
    ));
  }
} matmul;
