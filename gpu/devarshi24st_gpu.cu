#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "error.cuh"
#include "gl.cuh"
#include "healpix.cuh"
#include "ism_inst.cuh"
#include "lensing.cuh"
#include "matmul.cuh"
#include "nsx.cuh"
#include "unit.cuh"

#include <cublas_v2.h>
#include <cub/block/block_reduce.cuh>

constexpr int BLOCK_SIZE = 1024;

constexpr double M = 1.4;                // M_sun;
constexpr double Re = 12;                // km
constexpr double nu = 300;               // Hz
constexpr double inc = 2.391;            // rad
constexpr int N_output_phase_bins = 32;  // number of phase bins
constexpr double phase_bin_width = 1.0 / N_output_phase_bins;

constexpr int s_nsx_size = c.N_logEkT * c.N_mu;

constexpr double Re2 = Re * Re;
constexpr double Re3 = Re2 * Re;
constexpr double Rs = M * schwarzschild_radius_of_sun_in_km;
constexpr double x = 0.5 * Rs / Re;

constexpr double logT = 6.0;
constexpr double T_in_kelvin = 1e6;
constexpr double T3_in_kelvin = T_in_kelvin * T_in_kelvin * T_in_kelvin;
constexpr double kT_in_keV = T_in_kelvin * boltzmann_constant_in_keV_over_K;

constexpr double D = 0.15 * kpc_in_km;
constexpr double D2 = D * D;

constexpr double exposure_time = 1e6;

__device__ void
calc_oblate(float cos_theta, float sin_theta, float& R, float& dRdtheta, float& logg) {
  float Omega_bar = float(two_pi) * float(nu) * sqrt(float(Re3) / float(M)) * float(sqrt_km3_over_s2_GMsun);
  float Ob2 = Omega_bar * Omega_bar;
  float Ob4 = Ob2 * Ob2;
  float Ob6 = Ob4 * Ob2;

  float o2 = Ob2 * (-0.788f + 1.030f * float(x));
  dRdtheta = -2.0f * float(Re) * o2 * cos_theta * sin_theta;
  R = float(Re) * (1.0f + o2 * cos_theta * cos_theta);

  float ue = float(Rs) / float(Re);
  float uue = sqrt(1.0f - ue);
  float logg0 = log10(float(M) / (float(Re2) * uue));

  float ce = -0.791f + 0.776f * float(x);
  float cp = 1.138f + -1.431f * float(x);
  float de = (-1.315f + 2.431f * float(x)) * float(x);
  float fe = -1.172f * float(x);
  float dp = (0.653f - 2.864f * float(x)) * float(x);
  float fp = 0.975f * float(x);
  float d60 = (13.47f - 27.13f * float(x)) * float(x);

  float g_g0 = 1.0f
             + sin_theta * sin_theta * (ce * Ob2 + de * Ob4 + fe * Ob6)
             + cos_theta * cos_theta * (cp * Ob2 + dp * Ob4 + fp * Ob6 - d60 * Ob4)
             + abs(cos_theta) * d60 * Ob4;
  float logg_g0 = log10(g_g0);

  constexpr float logGMsun_s2_km2cm = 16.12291163415351764;
  logg = logg_g0 + logg0 + logGMsun_s2_km2cm;
}

struct StarData {
  double theta1;
  double phi1;
  double angrad1;
  double theta2;
  double phi2;
  double angrad2;
  std::string name;
};

struct StarParams {
  float sin_theta1, cos_theta1;
  float sin_theta2, cos_theta2;
  float phi1, phi2;
  float cos_angrad1, cos_angrad2;
  float sin_i, cos_i;

  int N_side;
  int N_fine_phase;
  int N_points_per_phase_bin;

  StarParams(StarData const& star_data, int N_side_, int N_fine_phase_, int N_points_per_phase_bin_) {
    sin_theta1 = std::sin(star_data.theta1);
    cos_theta1 = std::cos(star_data.theta1);
    phi1 = star_data.phi1;
    cos_angrad1 = std::cos(star_data.angrad1);
    sin_theta2 = std::sin(star_data.theta2);
    cos_theta2 = std::cos(star_data.theta2);
    phi2 = star_data.phi2;
    cos_angrad2 = std::cos(star_data.angrad2);
    sin_i = std::sin(inc);
    cos_i = std::cos(inc);

    N_side = N_side_;
    N_fine_phase = N_fine_phase_;
    N_points_per_phase_bin = N_points_per_phase_bin_;
  }

  __device__ bool
  IsInDisk(
      float sin_theta,
      float cos_theta,
      float sin_disk_center_theta,
      float cos_disk_center_theta,
      float phi_diff,
      float cos_disk_angular_radius
  ) {
    float cos_rho = sin_disk_center_theta * sin_theta * cos(phi_diff) + cos_disk_center_theta * cos_theta;
    return cos_rho > cos_disk_angular_radius;
  }

  __device__ bool
  IsInDiskDiff(float sin_theta, float cos_theta, float phi) {
    return IsInDisk(sin_theta, cos_theta, sin_theta1, cos_theta1, phi - phi1, cos_angrad1)
        && !IsInDisk(sin_theta, cos_theta, sin_theta2, cos_theta2, phi - phi2, cos_angrad2);
  }
};

// clang-format off
StarData         warmup = {.theta1 =   0.5 * pi, .phi1 =  -0.2, .angrad1 =  0.5, .theta2 =   0.5 * pi, .phi2 = 0.0, .angrad2 =             0.25, .name = "warmup"};
StarData        ring_eq = {.theta1 =   0.5 * pi, .phi1 =  -0.2, .angrad1 =  0.5, .theta2 =   0.5 * pi, .phi2 = 0.0, .angrad2 =             0.25, .name = "Ring_Eq"};
StarData     ring_polar = {.theta1 = pi - 0.001, .phi1 = -0.02, .angrad1 = 0.05, .theta2 = pi - 0.001, .phi2 = 0.0, .angrad2 =            0.025, .name = "Ring_Polar"};
StarData    crescent_eq = {.theta1 =        1.0, .phi1 =  -1.0, .angrad1 =  1.4, .theta2 =        0.5, .phi2 = 0.0, .angrad2 = 0.5 * pi - 0.001, .name = "Crescent_Eq"};
StarData crescent_polar = {.theta1 =       1.15, .phi1 =  -0.1, .angrad1 =  1.3, .theta2 =       1.75, .phi2 = 0.0, .angrad2 = 0.5 * pi - 0.001, .name = "Crescent_Polar"};
// clang-format on

__device__ float
cal_dt2(float RR) {
  float lg = log1p((float(Re) - RR) / (RR - float(Rs)));
  float cdt = float(Re) - RR + float(Rs) * lg;
  return cdt / float(c_in_km_s);
};

__global__ void
KernelFunc(StarParams s, Lens lens, NSX nsx, IsmInst ism_inst, float* flux_gpu) {
  extern __shared__ float dynamic_shared_memory[];
  float* s_nsx = dynamic_shared_memory;
  float* fluxes_over_I = s_nsx + s_nsx_size;
  float* redshift_factors = fluxes_over_I + (s.N_fine_phase + 1);
  float* phase_o = redshift_factors + (s.N_fine_phase + 1);
  float* cos_sigma_primes = phase_o + (s.N_fine_phase + 1);
  float* active_phi = cos_sigma_primes + (s.N_fine_phase + 1);

  __shared__ int healpix_i;
  __shared__ float cos_theta;
  __shared__ float sin_theta;

  __shared__ float R;
  __shared__ float logg;
  __shared__ float u;
  __shared__ float uu;
  __shared__ float cos_eta;
  __shared__ float sin_eta;

  __shared__ float dS;

  if (threadIdx.x == 0) {
    healpix_i = blockIdx.x + HEALPix::i_min();
    cos_theta = HEALPix::cos_theta(s.N_side, healpix_i);
    sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    float dRdtheta;
    calc_oblate(cos_theta, sin_theta, R, dRdtheta, logg);

    u = float(Rs) / R;
    uu = sqrt(1.0f - u);

    float f = dRdtheta / (R * uu);
    float ff = sqrt(1.0f + f * f);
    cos_eta = 1.0f / ff;
    sin_eta = f / ff;

    float dOmega = HEALPix::dOmega(s.N_side);
    dS = R * R * dOmega * ff;
  }
  __syncthreads();

  __shared__ int N_active_phi;

  if (threadIdx.x == 0) N_active_phi = 0;
  __syncthreads();

  int jmax = HEALPix::j_max(s.N_side, healpix_i);
  for (int healpix_j = threadIdx.x + HEALPix::j_min(); healpix_j <= jmax; healpix_j += blockDim.x) {
    float phi = HEALPix::phi(s.N_side, healpix_i, healpix_j);
    if (s.IsInDiskDiff(sin_theta, cos_theta, phi)) {
      int idx = atomicAdd_block(&N_active_phi, 1);
      active_phi[idx] = phi;
    }
  }
  __syncthreads();

  // no patch in emission region
  if (N_active_phi == 0) return;

  int invisible_i_phase_min = s.N_fine_phase + 1;
  int invisible_i_phase_max = -1;
  for (int i_phase = threadIdx.x; i_phase < s.N_fine_phase; i_phase += blockDim.x) {
    bool is_visible = true;
    do {
      float phase = float(i_phase) / s.N_fine_phase;
      float phi = float(two_pi) * phase;
      float cos_phi = cos(phi);
      float sin_phi = sin(phi);
      float cos_psi = s.cos_i * cos_theta + s.sin_i * sin_theta * cos_phi;

      if (cos_psi < lens.cos_psi_min) {
        is_visible = false;
        break;
      }

      float sin_psi = sqrt(1.0f - cos_psi * cos_psi);
      float cos_alpha, lf;
      lens.ca_lf_of_u_cos_psi(u, cos_psi, cos_alpha, lf);
      float sin_alpha = sqrt(1.0f - cos_alpha * cos_alpha);
      float sin_alpha_over_sin_psi = cos_psi == 1.0f ? sqrt(lf) : sin_alpha / sin_psi;
      float cos_sigma =
          cos_alpha * cos_eta + sin_alpha_over_sin_psi * sin_eta * (s.cos_i * sin_theta - s.sin_i * cos_theta * cos_phi);

      if (cos_sigma <= float(0)) {
        is_visible = false;
        break;
      }

      float beta = float(two_pi) * float(nu) * R * sin_theta / uu / float(c_in_km_s);
      float gamma = 1.0f / sqrt(1.0f - beta * beta);
      float cos_xi = -sin_alpha_over_sin_psi * s.sin_i * sin_phi;
      float delta = 1.0f / (gamma * (1.0f - beta * cos_xi));
      float delta3 = delta * delta * delta;

      float cdt_over_R = lens.cdt_over_R_of_u_ca(u, cos_alpha);
      float dt1 = cdt_over_R * R / float(c_in_km_s);

      float dt2 = 0;
      if (cos_alpha >= 0) {
        dt2 = cal_dt2(R);
      } else {
        float tmp = (2.0f * sin_alpha) / sqrt(3.0f * (1.0f - u));
        float p_over_R = -tmp * cos((acos(3.0f * u / tmp) + float(two_pi)) / 3.0f);
        float p = p_over_R * R;
        dt2 = 2.0f * cal_dt2(p) - cal_dt2(R);
      }
      float delta_phase = (dt1 + dt2) * float(nu);

      float cos_sigma_prime = cos_sigma * delta;
      fluxes_over_I[i_phase] = uu * delta3 * cos_sigma_prime * lf * gamma * dS;
      redshift_factors[i_phase] = 1.0f / (delta * uu);
      phase_o[i_phase] = phase + delta_phase;
      cos_sigma_primes[i_phase] = cos_sigma_prime;
    } while (0);
    if (is_visible == false) {
      fluxes_over_I[i_phase] = 0.0f;
      redshift_factors[i_phase] = -1.0f;
      phase_o[i_phase] = -1.0f;
      cos_sigma_primes[i_phase] = 0.0f;
      invisible_i_phase_min = min(invisible_i_phase_min, i_phase);
      invisible_i_phase_max = max(invisible_i_phase_max, i_phase);
    }
  }
  __syncthreads();

  using BlockReduce = cub::BlockReduce<int, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  BlockReduce block_reduce(temp_storage);
  __shared__ int s_invisible_i_phase_min;
  __shared__ int s_invisible_i_phase_max;

  invisible_i_phase_min = block_reduce.Reduce(invisible_i_phase_min, int_min{});
  __syncthreads();
  invisible_i_phase_max = block_reduce.Reduce(invisible_i_phase_max, int_max{});
  __syncthreads();

  if (threadIdx.x == 0) {
    s_invisible_i_phase_min = invisible_i_phase_min;
    s_invisible_i_phase_max = invisible_i_phase_max;
  }
  __syncthreads();

  // all phases are invisible
  if (s_invisible_i_phase_min == 0 || s_invisible_i_phase_max == s.N_fine_phase - 1) return;
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
    fluxes_over_I[s.N_fine_phase] = fluxes_over_I[0];
    redshift_factors[s.N_fine_phase] = redshift_factors[0];
    phase_o[s.N_fine_phase] = 1.0f + phase_o[0];
    cos_sigma_primes[s.N_fine_phase] = cos_sigma_primes[0];
  }
  __syncthreads();

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  for (int i = threadIdx.x; i < s_nsx_size; i += blockDim.x) {
    s_nsx[i] = 0;
  }
  __syncthreads();

  int i_logT, i_logg;
  float a_logT, a_logg;
  UniformHunt logT_hunt(c.logT_min, c.logT_max, c.N_logT);
  UniformHunt logg_hunt(c.logg_min, c.logg_max, c.N_logg);
  UniformHunt logEkT_hunt(c.logEkT_min, c.logEkT_max, c.N_logEkT);
  CubicLagrange logEkT_cl, mu_cl;

  logT_hunt(logT, i_logT, a_logT);
  logg_hunt(logg, i_logg, a_logg);

  for (int j_logT = 0; j_logT <= 1; ++j_logT) {
    for (int j_logg = 0; j_logg <= 1; ++j_logg) {
      float weight = (j_logT * a_logT + (1 - j_logT) * (1 - a_logT)) * (j_logg * a_logg + (1 - j_logg) * (1 - a_logg));
      int i_logT_ = i_logT + j_logT;
      int i_logg_ = i_logg + j_logg;
      for (int i = threadIdx.x; i < s_nsx_size; i += blockDim.x) {
        int i_logEkT = i / c.N_mu;
        int i_mu = i % c.N_mu;

        s_nsx[i] += nsx(i_logT_, i_logg_, i_logEkT, i_mu) * weight;
      }
      __syncthreads();
    }
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  __shared__ float s_mu_vec[c.N_mu];
  for (int i = threadIdx.x; i < c.N_mu; i += blockDim.x) {
    s_mu_vec[i] = nsx.mu_vec_gpu[i];
  }
  __syncthreads();
  VecHunt s_mu_hunt(s_mu_vec, c.N_mu);
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  VecHunt phase_o_hunt(phase_o, s.N_fine_phase + 1);
  int N_iter = N_active_phi * N_output_phase_bins * ism_inst.N_E_obs * s.N_points_per_phase_bin;
  for (int i_iter = threadIdx.x; i_iter < N_iter; i_iter += blockDim.x) {
    int i_active_phi = i_iter / (N_output_phase_bins * ism_inst.N_E_obs * s.N_points_per_phase_bin);
    int temp = i_iter % (N_output_phase_bins * ism_inst.N_E_obs * s.N_points_per_phase_bin);
    int i_E_obs = temp / (N_output_phase_bins * s.N_points_per_phase_bin);
    temp = temp % (N_output_phase_bins * s.N_points_per_phase_bin);
    int i_phase_bin = temp / s.N_points_per_phase_bin;
    int j_in_bin = temp % s.N_points_per_phase_bin;
    int i_total_flux = i_E_obs * N_output_phase_bins + i_phase_bin;

    float phase_shift = active_phi[i_active_phi] / float(two_pi);
    float phase_lb = float(i_phase_bin) * float(phase_bin_width);
    float phase_ub = float(i_phase_bin + 1) * float(phase_bin_width);
    float phase_point = GLNodes(s.N_points_per_phase_bin, j_in_bin, phase_lb, phase_ub);
    float patch_phase = fmod(phase_point + phase_shift + 1.0f, 1.0f);
    if (patch_phase < phase_o[0]) patch_phase += 1.0f;
    int i;
    float a;
    phase_o_hunt(patch_phase, i, a);
    float flux_over_I = (1.0f - a) * fluxes_over_I[i] + a * fluxes_over_I[i + 1];
    float redshift_factor = (1.0f - a) * redshift_factors[i] + a * redshift_factors[i + 1];
    float mu = (1.0f - a) * cos_sigma_primes[i] + a * cos_sigma_primes[i + 1];
    if (mu < nsx.mu_min) mu = nsx.mu_min;

    float E_obs = ism_inst.E_obs_gpu[i_E_obs];
    float E_emit = E_obs * redshift_factor;
    float logEkT = log10(E_emit / float(kT_in_keV));

    /////
    int i_logEkT, i_mu;
    float a_logEkT, a_mu;
    logEkT_hunt(logEkT, i_logEkT, a_logEkT);
    s_mu_hunt(mu, i_mu, a_mu);

    i_logEkT = clamp_int(i_logEkT - 1, 0, c.N_logEkT - 4);
    logEkT_cl(logEkT, logEkT_hunt[i_logEkT], logEkT_hunt[i_logEkT + 1], logEkT_hunt[i_logEkT + 2], logEkT_hunt[i_logEkT + 3]);

    int j_mu_max;
    if (i_mu >= c.N_mu - 5) {
      // linear at edge
      j_mu_max = 1;
      mu_cl.w[0] = 1 - a_mu;
      mu_cl.w[1] = a_mu;
    } else {
      // cubic in the middle
      j_mu_max = 3;
      i_mu = clamp_int(i_mu - 1, 0, c.N_mu - 4);
      mu_cl(mu, s_mu_vec[i_mu], s_mu_vec[i_mu + 1], s_mu_vec[i_mu + 2], s_mu_vec[i_mu + 3]);
    }

    float IT3 = 0;
    for (int j_logEkT = 0; j_logEkT <= 3; ++j_logEkT) {
      for (int j_mu = 0; j_mu <= j_mu_max; ++j_mu) {
        int i_logEkT_ = i_logEkT + j_logEkT;
        int i_mu_ = i_mu + j_mu;
        int idx = i_logEkT_ * c.N_mu + i_mu_;
        float v = s_nsx[idx];
        float weight = logEkT_cl.w[j_logEkT] * mu_cl.w[j_mu];
        IT3 += v * weight;
      }
    }
    /////

    float II = flux_over_I * IT3 / E_obs * GLWeights(s.N_points_per_phase_bin, j_in_bin, 1);
    atomicAdd(&flux_gpu[i_total_flux], II);
  }
  __syncthreads();
}

void
calc_one_star(
    StarData const& star_data,
    Lens const& lens,
    NSX const& nsx,
    IsmInst const& ism_inst,
    int N_side,
    int N_fine_phase,
    int N_points_per_phase_bin,
    std::string const& setting
) {
  float* flux_gpu;
  int total_flux_gpu_size = N_output_phase_bins * ism_inst.N_E_obs;
  cudaMalloc(&flux_gpu, total_flux_gpu_size * sizeof(float));
  cudaMemset(flux_gpu, 0, total_flux_gpu_size * sizeof(float));

  int counts_size = ism_inst.N_CH * N_output_phase_bins;
  float* counts_gpu;
  cudaMalloc(&counts_gpu, counts_size * sizeof(float));
  cudaMemset(counts_gpu, 0, counts_size * sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaEventQuery(start);

  StarParams star_params(star_data, N_side, N_fine_phase, N_points_per_phase_bin);

  int n_rings = HEALPix::n_rings(N_side);
  int dsmem_bytes = 90 * 1024;
  cudaFuncSetAttribute(KernelFunc, cudaFuncAttributeMaxDynamicSharedMemorySize, dsmem_bytes);
  KernelFunc<<<n_rings, BLOCK_SIZE, dsmem_bytes>>>(star_params, lens, nsx, ism_inst, flux_gpu);

  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  constexpr float alpha = exposure_time * phase_bin_width * T3_in_kelvin / planck / D2;
  matmul(
      ism_inst.N_CH,        // 270
      ism_inst.N_E_obs,     // 256
      N_output_phase_bins,  // 32
      ism_inst.rsp_gpu,     // (270, 256)
      flux_gpu,             // (256, 32)
      counts_gpu,           // (270, 32)
      alpha
  );

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float t_ms = 0;
  cudaEventElapsedTime(&t_ms, start, stop);
  std::cout << "Time taken: " << t_ms << " ms\n";

  {
    std::ofstream ofs("output/" + star_data.name + "_" + setting + "_time_gpu.txt");
    ofs << std::setprecision(16) << t_ms << '\n';
    ofs.close();
  }

  std::vector<float> counts_cpu(counts_size);
  cudaMemcpy(counts_cpu.data(), counts_gpu, counts_size * sizeof(float), cudaMemcpyDeviceToHost);
  std::string ofname = "output/" + star_data.name + "_" + setting + "_counts_gpu.txt";
  std::ofstream out_file(ofname);
  if (!out_file.is_open()) {
    std::cerr << "Error opening output file.\n";
    exit(1);
  }
  out_file << std::setprecision(16);
  for (int i = 0; i < counts_size; ++i) {
    out_file << counts_cpu[i] << ' ';
    if ((i + 1) % N_output_phase_bins == 0) out_file << '\n';
  }
  out_file.close();
  std::cout << "Output written to file ./" << ofname << '\n';

  cudaFree(counts_gpu);
  cudaFree(flux_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int
main(int argc, char* argv[]) {
  NSX nsx;
  Lens lens("std");
  IsmInst ism_inst("std");
  int N_points_per_phase_bin = 2;
  int N_fine_phase = 32 * 6;

  calc_one_star(warmup, lens, nsx, ism_inst, 173, N_fine_phase, N_points_per_phase_bin, "std");
  calc_one_star(ring_eq, lens, nsx, ism_inst, 173, N_fine_phase, N_points_per_phase_bin, "std");
  calc_one_star(ring_polar, lens, nsx, ism_inst, 1708, N_fine_phase, N_points_per_phase_bin, "std");
  calc_one_star(crescent_eq, lens, nsx, ism_inst, 128, N_fine_phase, N_points_per_phase_bin, "std");
  calc_one_star(crescent_polar, lens, nsx, ism_inst, 189, N_fine_phase, N_points_per_phase_bin, "std");

  calc_one_star(ring_eq, lens, nsx, ism_inst, 345, N_fine_phase, N_points_per_phase_bin, "high");
  calc_one_star(ring_polar, lens, nsx, ism_inst, 3413, N_fine_phase, N_points_per_phase_bin, "high");
  calc_one_star(crescent_eq, lens, nsx, ism_inst, 247, N_fine_phase, N_points_per_phase_bin, "high");
  calc_one_star(crescent_polar, lens, nsx, ism_inst, 382, N_fine_phase, N_points_per_phase_bin, "high");

  nsx.free_gpu_memory();
  lens.free_gpu_memory();
  ism_inst.free_gpu_memory();
}
