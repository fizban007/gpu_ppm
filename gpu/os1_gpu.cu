#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "error.cuh"
#include "healpix.cuh"
#include "hunt.cuh"
#include "lensing.cuh"
#include "unit.cuh"

#include <cub/block/block_reduce.cuh>

constexpr int BLOCK_DIM_X = 256;
constexpr int N_side = 256;

constexpr float M = 1.4;      // M_sun;
constexpr float Re = 12;      // km
constexpr float kT = 0.35;    // keV
constexpr float E_obs = 1.0;  // keV
constexpr int N_output_phase = 128;
auto output_phase_grid = linspace(0, 0.992187500, N_output_phase);
constexpr int N_phase_multiplier = 3;
constexpr int N_fine_phase = N_output_phase * N_phase_multiplier;

constexpr float Rs = M * schwarzschild_radius_of_sun_in_km;
constexpr float x = 0.5 * Rs / Re;

constexpr float D = 0.2;  // kpc
constexpr float D2 = D * D;
constexpr float Ibb_constant = 2. / (h_in_keV_s * h_in_keV_s * h_in_keV_s * c_in_cm_s * c_in_cm_s * kpc_in_km * kpc_in_km);
constexpr float Ibb_div_D2 = Ibb_constant / D2;

// blackbody spectrum
__device__ inline float
Ibb(float E, float kT) {
  return (E * E * E) / expm1(E / kT);
}

struct StarData {
  float nu;
  float spot_center_theta;
  float inc;
  float angular_radius;
  char ss;
  char beaming;
};

// clang-format off
StarData os1a = {.nu = 600, .spot_center_theta = 90 * degree, .inc = 90 * degree, .angular_radius = 0.01, .ss = 'a', .beaming = 'i'};
StarData os1b = {.nu = 600, .spot_center_theta = 90 * degree, .inc = 90 * degree, .angular_radius =    1, .ss = 'b', .beaming = 'i'};
StarData os1c = {.nu = 200, .spot_center_theta = 90 * degree, .inc = 90 * degree, .angular_radius = 0.01, .ss = 'c', .beaming = 'i'};
StarData os1d = {.nu =   1, .spot_center_theta = 90 * degree, .inc = 90 * degree, .angular_radius =    1, .ss = 'd', .beaming = 'i'};
StarData os1e = {.nu = 600, .spot_center_theta = 60 * degree, .inc = 30 * degree, .angular_radius =    1, .ss = 'e', .beaming = 'i'};
StarData os1f = {.nu = 600, .spot_center_theta = 20 * degree, .inc = 80 * degree, .angular_radius =    1, .ss = 'f', .beaming = 'i'};
StarData os1g = {.nu = 600, .spot_center_theta = 60 * degree, .inc = 30 * degree, .angular_radius =    1, .ss = 'g', .beaming = 'c'};
StarData os1h = {.nu = 600, .spot_center_theta = 60 * degree, .inc = 30 * degree, .angular_radius =    1, .ss = 'h', .beaming = 's'};
StarData os1i = {.nu = 600, .spot_center_theta = 20 * degree, .inc = 80 * degree, .angular_radius =    1, .ss = 'i', .beaming = 'c'};
StarData os1j = {.nu = 600, .spot_center_theta = 20 * degree, .inc = 80 * degree, .angular_radius =    1, .ss = 'j', .beaming = 's'};
// clang-format on

__device__ float
cal_dt2(float RR) {
  float lg = log1p((Re - RR) / (RR - Rs));
  float cdt = Re - RR + Rs * lg;
  return cdt / float(c_in_km_s);
};

__global__ void
KernelFunc(StarData sd, Lens lens, float* output_phase_grid_gpu, float* total_flux_gpu) {
  __shared__ int healpix_i;
  __shared__ float cos_theta;
  __shared__ float sin_theta;
  __shared__ float cos_center_theta;
  __shared__ float sin_center_theta;
  __shared__ float cos_angular_radius;

  __shared__ float R;
  __shared__ float u;
  __shared__ float uu;
  __shared__ float cos_eta;
  __shared__ float sin_eta;

  __shared__ float dS;

  __shared__ float cos_i;
  __shared__ float sin_i;

  __shared__ float cc;
  __shared__ float ss;

  __shared__ float active_phi[HEALPix::max_ring_size(N_side)];
  __shared__ int N_active_phi;

  if (threadIdx.x == 0) {
    healpix_i = blockIdx.x + HEALPix::i_min();
    cos_theta = (gridDim.x == 1) ? cos(sd.spot_center_theta) : HEALPix::cos_theta(N_side, healpix_i);
    sin_theta = sqrt(1.f - cos_theta * cos_theta);
    cos_center_theta = cos(sd.spot_center_theta);
    sin_center_theta = sin(sd.spot_center_theta);
    cos_angular_radius = cos(sd.angular_radius);

    float Omega_bar = float(two_pi) * sd.nu * sqrt(Re * Re * Re / M) * float(sqrt_km3_over_s2_GMsun);
    float o2 = Omega_bar * Omega_bar * (-0.788f + 1.030f * x);
    R = float(Re) * (1.f + o2 * cos_theta * cos_theta);
    float dRdtheta = -2.f * float(Re) * o2 * cos_theta * sin_theta;
    u = float(Rs) / R;
    uu = sqrt(1.f - u);

    float f = dRdtheta / (R * uu);
    float ff = sqrt(1.f + f * f);
    cos_eta = 1.f / ff;
    sin_eta = f / ff;

    float sin_half_ar = sin(sd.angular_radius * 0.5f);
    float dOmega = (gridDim.x == 1) ? float(two_pi) * (2.f * sin_half_ar * sin_half_ar) : HEALPix::dOmega(N_side);
    dS = R * R * dOmega * ff;

    cos_i = cos(sd.inc);
    sin_i = sin(sd.inc);

    cc = cos_i * cos_theta;
    ss = sin_i * sin_theta;

    N_active_phi = 0;
  }
  __syncthreads();

  if (gridDim.x == 1) {
    if (threadIdx.x == 0) {
      active_phi[0] = 0;
      N_active_phi = 1;
    }
  } else {
    int healpix_j = threadIdx.x + HEALPix::j_min();
    int jmax = HEALPix::j_max(N_side, healpix_i);
    for (; healpix_j <= jmax; healpix_j += blockDim.x) {
      float phi = HEALPix::phi(N_side, healpix_i, healpix_j);
      float cos_rho = sin_theta * sin_center_theta * std::cos(phi) + cos_theta * cos_center_theta;
      if (cos_rho > cos_angular_radius) {
        int idx = atomicAdd_block(&N_active_phi, 1);
        active_phi[idx] = phi;
      }
    }
  }
  __syncthreads();

  if (N_active_phi == 0) return;  // no patch in emission region

  __shared__ float fluxes_over_I[N_fine_phase + 1];
  __shared__ float redshift_factors[N_fine_phase + 1];
  __shared__ float phase_o[N_fine_phase + 1];

  int invisible_i_phase_min = N_fine_phase + 1;
  int invisible_i_phase_max = -1;
  for (int i_phase = threadIdx.x; i_phase < N_fine_phase; i_phase += blockDim.x) {
    bool is_visible = true;
    do {
      float phase = 1.f * i_phase / N_fine_phase;
      float phi = float(two_pi) * phase;
      float cos_phi = cos(phi);
      float sin_phi = sin(phi);
      float cos_psi = cc + ss * cos_phi;

      if (cos_psi < lens.cos_psi_min) {
        is_visible = false;
        break;
      }

      float sin_psi = sqrt(1.f - cos_psi * cos_psi);
      float cos_alpha, lf;
      lens.ca_lf_of_u_cos_psi(u, cos_psi, cos_alpha, lf);
      float sin_alpha = sqrt(1.f - cos_alpha * cos_alpha);
      float sin_alpha_over_sin_psi = cos_psi == 1.f ? sqrt(lf) : sin_alpha / sin_psi;
      float cos_sigma =
          cos_alpha * cos_eta + sin_alpha_over_sin_psi * sin_eta * (cos_i * sin_theta - sin_i * cos_theta * cos_phi);

      if (cos_sigma <= 0.f) {
        is_visible = false;
        break;
      }

      float beta = float(two_pi) * sd.nu * R * sin_theta / uu / float(c_in_km_s);
      float gamma = 1.f / sqrt(1.f - beta * beta);
      float cos_xi = -sin_alpha_over_sin_psi * sin_i * sin_phi;
      float delta = 1.f / (gamma * (1.f - beta * cos_xi));
      float delta3 = delta * delta * delta;

      float cdt_over_R = lens.cdt_over_R_of_u_ca(u, cos_alpha);
      float dt1 = cdt_over_R * R / float(c_in_km_s);

      float dt2 = 0;
      if (cos_alpha >= 0) {
        dt2 = cal_dt2(R);
      } else {
        float tmp = (2.f * sin_alpha) / sqrt(3.f * (1.f - u));
        float p_over_R = -tmp * cos((acos(3.f * u / tmp) + float(two_pi)) / 3.f);
        float p = p_over_R * R;
        dt2 = 2.f * cal_dt2(p) - cal_dt2(R);
      }
      float delta_phase = (dt1 + dt2) * sd.nu;

      float beaming_factor;
      float cos_sigma_prime = cos_sigma * delta;
      if (sd.beaming == 'i') {
        beaming_factor = 1.f;
      } else if (sd.beaming == 's') {
        beaming_factor = 1.f - cos_sigma_prime * cos_sigma_prime;
      } else if (sd.beaming == 'c') {
        beaming_factor = cos_sigma_prime * cos_sigma_prime;
      }

      fluxes_over_I[i_phase] = uu * delta3 * cos_sigma_prime * lf * gamma * dS * Ibb_div_D2 * beaming_factor;
      redshift_factors[i_phase] = 1.f / (delta * uu);
      phase_o[i_phase] = phase + delta_phase;
    } while (0);
    if (is_visible == false) {
      fluxes_over_I[i_phase] = 0.f;
      redshift_factors[i_phase] = -1.f;
      phase_o[i_phase] = -1.f;
      invisible_i_phase_min = min(invisible_i_phase_min, i_phase);
      invisible_i_phase_max = max(invisible_i_phase_max, i_phase);
    }
  }
  __syncthreads();

  using BlockReduce = cub::BlockReduce<int, BLOCK_DIM_X>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  BlockReduce block_reduce(temp_storage);
  static_assert(sizeof(temp_storage) >= 0);
  __shared__ int s_invisible_i_phase_min;
  __shared__ int s_invisible_i_phase_max;

  invisible_i_phase_min = block_reduce.Reduce(invisible_i_phase_min, cuda::minimum<int>{});
  __syncthreads();
  invisible_i_phase_max = block_reduce.Reduce(invisible_i_phase_max, cuda::maximum<int>{});
  __syncthreads();

  if (threadIdx.x == 0) {
    s_invisible_i_phase_min = invisible_i_phase_min;
    s_invisible_i_phase_max = invisible_i_phase_max;
  }
  __syncthreads();

  if (s_invisible_i_phase_min == 0 || s_invisible_i_phase_max == N_fine_phase - 1) return;  // all phases are invisible
  // fill phase_o and redshift_factors for invisible phases
  if (s_invisible_i_phase_min <= s_invisible_i_phase_max) {
    int beg = s_invisible_i_phase_min - 1;
    int end = s_invisible_i_phase_max + 1;
    float phase0 = phase_o[beg];
    float phase1 = phase_o[end];
    float rf0 = redshift_factors[beg];
    float rf1 = redshift_factors[end];
    for (int i_phase = threadIdx.x + s_invisible_i_phase_min; i_phase <= s_invisible_i_phase_max; i_phase += blockDim.x) {
      phase_o[i_phase] = phase0 + (phase1 - phase0) * (float(i_phase - beg) / (end - beg));
      redshift_factors[i_phase] = rf0 + (rf1 - rf0) * (float(i_phase - beg) / (end - beg));
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    fluxes_over_I[N_fine_phase] = fluxes_over_I[0];
    redshift_factors[N_fine_phase] = redshift_factors[0];
    phase_o[N_fine_phase] = 1.f + phase_o[0];
  }
  __syncthreads();

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  VecHunt phase_o_hunt(phase_o, N_fine_phase + 1);
  __shared__ float total_flux_shared[N_output_phase];
  for (int i = threadIdx.x; i < N_output_phase; i += blockDim.x) {
    total_flux_shared[i] = 0;
  }
  __syncthreads();

  int N_iter = N_active_phi * N_output_phase;
  for (int i_iter = threadIdx.x; i_iter < N_iter; i_iter += blockDim.x) {
    int i_active_phi = i_iter / N_output_phase;
    int i_output_phase = i_iter % N_output_phase;
    float phase_shift = active_phi[i_active_phi] / float(two_pi);
    float patch_phase = fmod(output_phase_grid_gpu[i_output_phase] + phase_shift + 1.f, 1.f);
    if (patch_phase < phase_o[0]) patch_phase += 1.f;
    int i;
    float a;
    phase_o_hunt(patch_phase, i, a);
    float flux_over_I = (1.f - a) * fluxes_over_I[i] + a * fluxes_over_I[i + 1];
    float redshift_factor = (1.f - a) * redshift_factors[i] + a * redshift_factors[i + 1];
    float E_emit = E_obs * redshift_factor;
    float I = flux_over_I * Ibb(E_emit, kT);
    atomicAdd_block(&total_flux_shared[i_output_phase], I);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < N_output_phase; i += blockDim.x) {
    atomicAdd(&total_flux_gpu[i], total_flux_shared[i]);
  }
}

void
calc_one_star(StarData const& sd, Lens const& lens, float* output_phase_grid_gpu) {
  float* total_flux_gpu;
  cudaMalloc(&total_flux_gpu, N_output_phase * sizeof(float));
  cudaMemset(total_flux_gpu, 0, N_output_phase * sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaEventQuery(start);

  int n_rings = (sd.angular_radius < 0.1) ? 1 : HEALPix::n_rings(N_side);
  KernelFunc<<<n_rings, BLOCK_DIM_X>>>(sd, lens, output_phase_grid_gpu, total_flux_gpu);

  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t_ms = 0;
  cudaEventElapsedTime(&t_ms, start, stop);
  std::cout << "Time taken: " << t_ms << " ms\n";

  std::vector<float> total_flux_cpu(N_output_phase);
  cudaMemcpy(total_flux_cpu.data(), total_flux_gpu, N_output_phase * sizeof(float), cudaMemcpyDeviceToHost);

  {
    std::string output_file_name = std::string("output/os1") + sd.ss + "_gpu.txt";
    std::ofstream out_file(output_file_name);
    if (!out_file.is_open()) {
      std::cerr << "Error opening output file " << output_file_name << "\n";
      exit(1);
    }
    out_file << std::setprecision(16);

    for (int i_output_phase = 0; i_output_phase < output_phase_grid.size(); ++i_output_phase) {
      out_file << total_flux_cpu[i_output_phase] << ' ';
    }
    out_file.close();
    std::cout << "Output written to file ./" << output_file_name << "\n";
  }

  cudaFree(total_flux_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int
main(int argc, char* argv[]) {
  float* output_phase_grid_gpu;
  cudaMalloc(&output_phase_grid_gpu, N_output_phase * sizeof(float));
  cudaMemcpy(output_phase_grid_gpu, output_phase_grid.data(), N_output_phase * sizeof(float), cudaMemcpyHostToDevice);

  Lens lens("std");

  calc_one_star(os1a, lens, output_phase_grid_gpu);
  calc_one_star(os1b, lens, output_phase_grid_gpu);
  calc_one_star(os1c, lens, output_phase_grid_gpu);
  calc_one_star(os1d, lens, output_phase_grid_gpu);
  calc_one_star(os1e, lens, output_phase_grid_gpu);
  calc_one_star(os1f, lens, output_phase_grid_gpu);
  calc_one_star(os1g, lens, output_phase_grid_gpu);
  calc_one_star(os1h, lens, output_phase_grid_gpu);
  calc_one_star(os1i, lens, output_phase_grid_gpu);
  calc_one_star(os1j, lens, output_phase_grid_gpu);

  cudaFree(output_phase_grid_gpu);
  lens.free_gpu_memory();
}
