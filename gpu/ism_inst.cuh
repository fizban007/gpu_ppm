#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "matrix.cuh"
#include "read_file.cuh"

struct IsmInst {
  int N_CH;
  int N_E_obs;
  float* E_obs_gpu;
  float* rsp_gpu;

  IsmInst(std::string const& setting) {
    read2(N_CH, N_E_obs, "tables/ism_inst/" + setting + "/num.txt");

    std::vector<float> E_obs_cpu;
    Matrix<float> rsp_cpu;

    read_vector(E_obs_cpu, "tables/ism_inst/" + setting + "/E_obs.txt", N_E_obs);
    read_matrix(rsp_cpu, "tables/ism_inst/" + setting + "/rsp.txt", N_CH, N_E_obs);

    cudaMalloc(&E_obs_gpu, N_E_obs * sizeof(float));
    cudaMalloc(&rsp_gpu, N_CH * N_E_obs * sizeof(float));

    cudaMemcpy(E_obs_gpu, E_obs_cpu.data(), N_E_obs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rsp_gpu, rsp_cpu.data.data(), N_CH * N_E_obs * sizeof(float), cudaMemcpyHostToDevice);
  }

  void
  free_gpu_memory() {
    cudaFree(E_obs_gpu);
    cudaFree(rsp_gpu);
  }
};
