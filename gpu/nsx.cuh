#pragma once

#include <cstdio>
#include <fstream>
#include <vector>

struct NsxConfig {
  char const* nsx_fname;
  int N_logT;
  int N_logg;
  int N_logEkT;
  int N_mu;
  float logT_min;
  float logT_max;
  float logg_min;
  float logg_max;
  float logEkT_min;
  float logEkT_max;
};

constexpr NsxConfig nsx_H_v200804 = {
    .nsx_fname = "ext/model_data/nsx_H_v200804.out",
    .N_logT = 35,
    .N_logg = 14,
    .N_logEkT = 166,
    .N_mu = 67,
    .logT_min = 5.10,
    .logT_max = 6.80,
    .logg_min = 13.7,
    .logg_max = 15.0,
    .logEkT_min = -1.30,
    .logEkT_max = 2.00
};

constexpr NsxConfig nsx_H_v171019 = {
    .nsx_fname = "ext/model_data/nsx_H_v171019.out",
    .N_logT = 35,
    .N_logg = 11,
    .N_logEkT = 166,
    .N_mu = 67,
    .logT_min = 5.10,
    .logT_max = 6.80,
    .logg_min = 13.7,
    .logg_max = 14.7,
    .logEkT_min = -1.30,
    .logEkT_max = 2.00
};

constexpr NsxConfig c = nsx_H_v200804;

struct NSX {
  float* mu_vec_gpu;
  float* data_gpu;
  float mu_min;

  __device__ __host__ int
  flat_id(int i_logT, int i_logg, int i_logEkT, int i_mu) {
    return i_logT * c.N_logg * c.N_logEkT * c.N_mu  //
         + i_logg * c.N_logEkT * c.N_mu             //
         + i_logEkT * c.N_mu                        //
         + i_mu;
  }

  __device__ float&
  operator()(int i_logT, int i_logg, int i_logEkT, int i_mu) {
#ifndef NDEBUG
    if (i_logT < 0
        || i_logT >= c.N_logT
        || i_logg < 0
        || i_logg >= c.N_logg
        || i_logEkT < 0
        || i_logEkT >= c.N_logEkT
        || i_mu < 0
        || i_mu >= c.N_mu) {
      printf("NSX_Gpu::operator() : Index out of bounds\n");
      asm("trap;");
    }
#endif
    int index = flat_id(i_logT, i_logg, i_logEkT, i_mu);
    return data_gpu[index];
  }

  NSX() {
    constexpr int data_size = c.N_logT * c.N_logg * c.N_logEkT * c.N_mu;
    std::vector<float> mu_vec_cpu(c.N_mu, 0.0f);
    std::vector<float> data_cpu(data_size, 0.0f);

    std::ifstream in_file(c.nsx_fname);
    if (!in_file) {
      printf("Could not open NSX file: %s\n", c.nsx_fname);
      std::exit(EXIT_FAILURE);
    }

    printf("Loading NSX data from %s\n", c.nsx_fname);

    for (int i_logT = 0; i_logT < c.N_logT; ++i_logT) {
      for (int i_logg = 0; i_logg < c.N_logg; ++i_logg) {
        for (int i_logEkT = 0; i_logEkT < c.N_logEkT; ++i_logEkT) {
          for (int i_mu = 0; i_mu < c.N_mu; ++i_mu) {
            double logEkT, mu, logInuT3, logT, logg;
            in_file >> logEkT >> mu >> logInuT3 >> logT >> logg;

            if (!in_file) {
              printf("Error reading NSX data at indices: %d, %d, %d, %d\n", i_logT, i_logg, i_logEkT, i_mu);
              std::exit(EXIT_FAILURE);
            }

            if (mu_vec_cpu[i_mu] == 0.0f) {
              mu_vec_cpu[i_mu] = mu;
            }

            int index = flat_id(i_logT, i_logg, i_logEkT, i_mu);
            data_cpu[index] = std::pow(10, logInuT3);
          }
        }
      }
    }
    in_file.close();
    printf("NSX data loaded successfully from %s\n", c.nsx_fname);

    cudaMalloc(&mu_vec_gpu, c.N_mu * sizeof(float));
    cudaMemcpy(mu_vec_gpu, mu_vec_cpu.data(), c.N_mu * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&data_gpu, data_size * sizeof(float));
    cudaMemcpy(data_gpu, data_cpu.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);

    mu_min = mu_vec_cpu[c.N_mu - 1];
  }

  void
  free_gpu_memory() {
    cudaFree(mu_vec_gpu);
    cudaFree(data_gpu);
  }
};

// cubic lagrange interpolation
struct CubicLagrange {
  float w[4];

  __device__ void
  operator()(float x, float x0, float x1, float x2, float x3) {
    w[0] = (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3));
    w[1] = (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3));
    w[2] = (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3));
    w[3] = (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2));
  }
};
