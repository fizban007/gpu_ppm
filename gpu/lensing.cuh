#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hunt.cuh"
#include "matrix.cuh"
#include "read_file.cuh"

struct Lens {
  float u_min, u_max;
  int N_u;

  float cos_psi_min, cos_psi_max;
  int N_cos_psi;

  float cos_alpha_min, cos_alpha_max;
  int N_cos_alpha;

  float* cos_alpha_of_u_cos_psi_gpu;
  float* lf_of_u_cos_psi_gpu;
  float* cdt_over_R_of_u_cos_alpha_gpu;

  Lens(std::string const& setting) {
    read3(u_min, u_max, N_u, "tables/lensing/" + setting + "/u.txt");
    read3(cos_psi_min, cos_psi_max, N_cos_psi, "tables/lensing/" + setting + "/cos_psi.txt");
    read3(cos_alpha_min, cos_alpha_max, N_cos_alpha, "tables/lensing/" + setting + "/cos_alpha.txt");

    Matrix<float> cos_alpha_of_u_cos_psi, lf_of_u_cos_psi, cdt_over_R_of_u_cos_alpha;
    read_matrix(cos_alpha_of_u_cos_psi, "tables/lensing/" + setting + "/cos_alpha_of_u_cos_psi.txt", N_u, N_cos_psi);
    read_matrix(lf_of_u_cos_psi, "tables/lensing/" + setting + "/lf_of_u_cos_psi.txt", N_u, N_cos_psi);
    read_matrix(cdt_over_R_of_u_cos_alpha, "tables/lensing/" + setting + "/cdt_over_R_of_u_cos_alpha.txt", N_u, N_cos_alpha);

    cudaMalloc(&cos_alpha_of_u_cos_psi_gpu, N_u * N_cos_psi * sizeof(float));
    cudaMalloc(&lf_of_u_cos_psi_gpu, N_u * N_cos_psi * sizeof(float));
    cudaMalloc(&cdt_over_R_of_u_cos_alpha_gpu, N_u * N_cos_alpha * sizeof(float));

    cudaMemcpy(
        cos_alpha_of_u_cos_psi_gpu,
        cos_alpha_of_u_cos_psi.data.data(),
        N_u * N_cos_psi * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(lf_of_u_cos_psi_gpu, lf_of_u_cos_psi.data.data(), N_u * N_cos_psi * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(
        cdt_over_R_of_u_cos_alpha_gpu,
        cdt_over_R_of_u_cos_alpha.data.data(),
        N_u * N_cos_alpha * sizeof(float),
        cudaMemcpyHostToDevice
    );
  }

  void
  free_gpu_memory() {
    cudaFree(cos_alpha_of_u_cos_psi_gpu);
    cudaFree(lf_of_u_cos_psi_gpu);
    cudaFree(cdt_over_R_of_u_cos_alpha_gpu);
  }

  __device__ float
  at2d(float* data, int n_row, int n_col, int row, int col) {
    return data[row * n_col + col];
  }

  __device__ void
  ca_lf_of_u_cos_psi(float u, float cos_psi, float& ca, float& lf) {
    int i_u, i_cp;
    float a_u, a_cp;
    UniformHunt(u_min, u_max, N_u)(u, i_u, a_u);
    UniformHunt(cos_psi_min, cos_psi_max, N_cos_psi)(cos_psi, i_cp, a_cp);
    float b_u = 1 - a_u;
    float b_cp = 1 - a_cp;

    ca = b_u * b_cp * cos_alpha_of_u_cos_psi_gpu[i_u * N_cos_psi + i_cp]
       + a_u * b_cp * cos_alpha_of_u_cos_psi_gpu[(i_u + 1) * N_cos_psi + i_cp]
       + b_u * a_cp * cos_alpha_of_u_cos_psi_gpu[i_u * N_cos_psi + (i_cp + 1)]
       + a_u * a_cp * cos_alpha_of_u_cos_psi_gpu[(i_u + 1) * N_cos_psi + (i_cp + 1)];

    lf = b_u * b_cp * lf_of_u_cos_psi_gpu[i_u * N_cos_psi + i_cp]
       + a_u * b_cp * lf_of_u_cos_psi_gpu[(i_u + 1) * N_cos_psi + i_cp]
       + b_u * a_cp * lf_of_u_cos_psi_gpu[i_u * N_cos_psi + (i_cp + 1)]
       + a_u * a_cp * lf_of_u_cos_psi_gpu[(i_u + 1) * N_cos_psi + (i_cp + 1)];
  }

  __device__ float
  cdt_over_R_of_u_ca(float u, float cos_alpha) {
    int i_u, i_ca;
    float a_u, a_ca;
    UniformHunt(u_min, u_max, N_u)(u, i_u, a_u);
    UniformHunt(cos_alpha_min, cos_alpha_max, N_cos_alpha)(cos_alpha, i_ca, a_ca);
    float b_u = 1 - a_u;
    float b_ca = 1 - a_ca;
    return b_u * b_ca * cdt_over_R_of_u_cos_alpha_gpu[i_u * N_cos_alpha + i_ca]
         + a_u * b_ca * cdt_over_R_of_u_cos_alpha_gpu[(i_u + 1) * N_cos_alpha + i_ca]
         + b_u * a_ca * cdt_over_R_of_u_cos_alpha_gpu[i_u * N_cos_alpha + (i_ca + 1)]
         + a_u * a_ca * cdt_over_R_of_u_cos_alpha_gpu[(i_u + 1) * N_cos_alpha + (i_ca + 1)];
  }
};
