#include <fstream>
#include "gl.hpp"
#include "grid.hpp"
#include "ism_inst.hpp"
#include "lensing.hpp"
#include "matrix.hpp"
#include "nsx.hpp"
#include "timer.hpp"
#include "unit.hpp"
#include "ut.hpp"

struct StarData {
  double theta1, phi1, angrad1, theta2, phi2, angrad2;
  std::string name;
};

// clang-format off
StarData        ring_eq = {.theta1 =   0.5 * pi, .phi1 =  -0.2, .angrad1 =  0.5, .theta2 =   0.5 * pi, .phi2 = 0.0, .angrad2 =             0.25, .name = "Ring_Eq"};
StarData     ring_polar = {.theta1 = pi - 0.001, .phi1 = -0.02, .angrad1 = 0.05, .theta2 = pi - 0.001, .phi2 = 0.0, .angrad2 =            0.025, .name = "Ring_Polar"};
StarData    crescent_eq = {.theta1 =        1.0, .phi1 =  -1.0, .angrad1 =  1.4, .theta2 =        0.5, .phi2 = 0.0, .angrad2 = 0.5 * pi - 0.001, .name = "Crescent_Eq"};
StarData crescent_polar = {.theta1 =       1.15, .phi1 =  -0.1, .angrad1 =  1.3, .theta2 =       1.75, .phi2 = 0.0, .angrad2 = 0.5 * pi - 0.001, .name = "Crescent_Polar"};
// clang-format on

void
calc_one_star(
    StarData s,
    Lensing& lt,
    NSX& nsx,
    IsmInst& ism_inst,
    int N_side,
    int N_fine_phase,
    int N_points_per_phase_bin,
    std::string const& setting
) {
  Grid grid;
  auto disk = DiskDifference{
      .theta1 = s.theta1,
      .phi1 = s.phi1,
      .angrad1 = s.angrad1,
      .theta2 = s.theta2,
      .phi2 = s.phi2,
      .angrad2 = s.angrad2
  };

  int N_active = GenerateHealpixGrid(N_side, disk, grid);
  printf(">>> N_side: %d, N_active: %d\n", N_side, N_active);

  constexpr double M = 1.4;             // M_sun;
  constexpr double Re = 12;             // km
  constexpr double frequency_nu = 300;  // Hz
  constexpr double obs_theta = 2.391;   // rad
  constexpr double D = 0.15 * kpc_in_km;
  constexpr double exposure_time = 1e6;
  constexpr int N_output_phase_bins = 32;  // number of phase bins
  constexpr double phase_bin_width = 1. / N_output_phase_bins;

  double logT = 6.0;
  double T_in_kelvin = std::pow(10, logT);
  double kT_in_keV = T_in_kelvin * 8.617333262E-8;

  ////////////////////////////////////////////////////////////////////////////////////////////////

  double Rs = M * schwarzschild_radius_of_sun_in_km;
  double ue = Rs / Re;
  double uue = std::sqrt(1. - ue);
  double logg0 = std::log10(M / (Re * Re * uue));
  // https://ssd.jpl.nasa.gov/astro_par.html
  double logGMsun_s2_km2cm = std::log10(1.32712440041279419E16);
  double Omega_bar = two_pi * frequency_nu * std::sqrt(Re * Re * Re / M) * sqrt_km3_over_s2_GMsun;
  double x = 0.5 * Rs / Re;
  double o2 = Omega_bar * Omega_bar * (-0.788 + 1.030 * x);
  double D2 = D * D;
  double cos_obs_theta = std::cos(obs_theta);
  double sin_obs_theta = std::sin(obs_theta);

  std::vector<double> fluxes_over_I(N_fine_phase + 1);
  std::vector<double> redshift_factors(N_fine_phase + 1);
  std::vector<double> phase_o(N_fine_phase + 1);
  std::vector<double> cos_sigma_primes(N_fine_phase + 1);

  VecHunt phase_o_hunt{phase_o};

  Matrix total_flux(N_output_phase_bins, ism_inst.E_obs.size());
  total_flux.fill(0);

  Timer tm;
  tm.start();

  for (auto const& ring : grid) {
    double cos_spot_theta = ring.cos_theta;
    double sin_spot_theta = std::sqrt(1. - cos_spot_theta * cos_spot_theta);

    double R = Re * (1. + o2 * cos_spot_theta * cos_spot_theta);
    double dRdtheta = -2. * Re * o2 * cos_spot_theta * sin_spot_theta;
    double u = Rs / R;
    double uu = std::sqrt(1. - u);
    double f = dRdtheta / (R * uu);
    double ff = std::sqrt(1. + f * f);
    double dS = R * R * ring.dOmega * ff;
    double cos_tau = 1. / ff;
    double sin_tau = f / ff;

    constexpr double c0e = -0.791, c1e = 0.776, c0p = 1.138, c1p = -1.431, d1e = -1.315, d2e = 2.431, f1e = -1.172, d1p = 0.653,
                     d2p = -2.864, f1p = 0.975, d160 = 13.47, d260 = -27.13, f060 = 1.69;
    double ce = c0e + c1e * x;
    double cp = c0p + c1p * x;
    double de = d1e * x + d2e * x * x;
    double fe = f1e * x;
    double dp = d1p * x + d2p * x * x;
    double fp = f1p * x;
    double d60 = d160 * x + d260 * x * x;
    double f60 = f060;
    double Ob2 = Omega_bar * Omega_bar;
    double Ob4 = Ob2 * Ob2;
    double Ob6 = Ob4 * Ob2;
    double sc = cos_spot_theta;
    double sc2 = cos_spot_theta * cos_spot_theta;
    double ss2 = sin_spot_theta * sin_spot_theta;
    double g_g0 = 1.
                + ss2 * (ce * Ob2 + de * Ob4 + fe * Ob6)
                + sc2 * (cp * Ob2 + dp * Ob4 + fp * Ob6 - d60 * Ob4)
                + std::abs(sc) * d60 * Ob4;
    double logg_g0 = std::log10(g_g0);
    double logg = logg_g0 + logg0 + logGMsun_s2_km2cm;

    int invisible_i_phase_min = N_fine_phase + 1;
    int invisible_i_phase_max = -1;
    for (int i_phase = 0; i_phase < N_fine_phase; ++i_phase) {
      bool is_visible = true;
      do {
        double phase = 1. * i_phase / N_fine_phase;
        double spot_phi = two_pi * phase;
        double cos_spot_phi = std::cos(spot_phi);
        double sin_spot_phi = std::sin(spot_phi);
        double cos_psi = cos_obs_theta * cos_spot_theta + sin_obs_theta * sin_spot_theta * cos_spot_phi;
        if (cos_psi < lt.cos_psi.x_min) {
          is_visible = false;
          break;
        }
        double sin_psi = std::sqrt(1. - cos_psi * cos_psi);
        auto [cos_alpha, lf] = lt.cal_cos_alpha_lf_of_u_cos_psi(u, cos_psi);
        double sin_alpha = std::sqrt(1. - cos_alpha * cos_alpha);
        double sin_alpha_over_sin_psi = cos_psi == 1. ? std::sqrt(lf) : sin_alpha / sin_psi;

        double cos_sigma =
            cos_alpha * cos_tau
            + sin_alpha_over_sin_psi * sin_tau * (cos_obs_theta * sin_spot_theta - sin_obs_theta * cos_spot_theta * cos_spot_phi);
        if (cos_sigma <= 0) {
          is_visible = false;
          break;
        }

        double beta = two_pi * frequency_nu * R * sin_spot_theta / uu / c_in_km_s;
        double gamma = 1. / std::sqrt(1. - beta * beta);
        double cos_xi = -sin_alpha_over_sin_psi * sin_obs_theta * sin_spot_phi;
        double delta = 1. / (gamma * (1. - beta * cos_xi));
        double delta3 = delta * delta * delta;

        double cdt_over_R = lt.cal_cdt_over_R_of_u_cos_alpha(u, cos_alpha);
        double dt1 = cdt_over_R * R / c_in_km_s;

        auto cal_dt2 = [Re, Rs](double RR) {
          double lg = std::log1p((Re - RR) / (RR - Rs));
          double cdt = Re - RR + Rs * lg;
          return cdt / c_in_km_s;
        };

        double dt2 = 0;
        if (cos_alpha >= 0) {
          dt2 = cal_dt2(R);
        } else {
          double tmp = (2. * sin_alpha) / std::sqrt(3. * (1. - u));
          double p_over_R = -tmp * std::cos((std::acos(3. * u / tmp) + 2. * pi) / 3.);
          double p = p_over_R * R;
          dt2 = 2. * cal_dt2(p) - cal_dt2(R);
        }
        double delta_phase = (dt1 + dt2) * frequency_nu;

        double cos_sigma_prime = cos_sigma * delta;
        fluxes_over_I[i_phase] = uu * delta3 * cos_sigma_prime * lf * (dS * gamma) / D2;
        redshift_factors[i_phase] = 1. / (delta * uu);
        phase_o[i_phase] = phase + delta_phase;
        cos_sigma_primes[i_phase] = cos_sigma_prime;
      } while (0);
      if (is_visible == false) {
        fluxes_over_I[i_phase] = 0;
        redshift_factors[i_phase] = -1;
        phase_o[i_phase] = -1;
        cos_sigma_primes[i_phase] = 0;
        invisible_i_phase_min = std::min(invisible_i_phase_min, i_phase);
        invisible_i_phase_max = std::max(invisible_i_phase_max, i_phase);
      }
    }

    if (invisible_i_phase_min == 0 || invisible_i_phase_max == N_fine_phase - 1) continue;
    if (invisible_i_phase_min <= invisible_i_phase_max) {
      int beg = invisible_i_phase_min - 1;
      int end = invisible_i_phase_max + 1;
      double phase0 = phase_o[beg];
      double phase1 = phase_o[end];
      double rf0 = redshift_factors[beg];
      double rf1 = redshift_factors[end];
      for (int i_phase = invisible_i_phase_min; i_phase <= invisible_i_phase_max; ++i_phase) {
        phase_o[i_phase] = phase0 + (phase1 - phase0) * (double(i_phase - beg) / (end - beg));
        redshift_factors[i_phase] = rf0 + (rf1 - rf0) * (double(i_phase - beg) / (end - beg));
      }
    }

    fluxes_over_I[N_fine_phase] = fluxes_over_I[0];
    redshift_factors[N_fine_phase] = redshift_factors[0];
    phase_o[N_fine_phase] = 1.0 + phase_o[0];
    cos_sigma_primes[N_fine_phase] = cos_sigma_primes[0];

    for (int i_phase_points = 0; i_phase_points < N_output_phase_bins * N_points_per_phase_bin; ++i_phase_points) {
      int i_phase_bin = i_phase_points / N_points_per_phase_bin;
      int j_in_bin = i_phase_points % N_points_per_phase_bin;
      for (auto const& phi : ring.active_phi) {
        double phase_shift = phi / two_pi;
        double phase_lb = i_phase_bin * phase_bin_width;
        double phase_ub = (i_phase_bin + 1) * phase_bin_width;
        double phase_point = GLNodes(N_points_per_phase_bin, j_in_bin, phase_lb, phase_ub);
        double patch_phase = std::fmod(phase_point + phase_shift + 1., 1.);
        if (patch_phase < phase_o[0]) patch_phase += 1.;
        auto [i, a] = phase_o_hunt(patch_phase);
        double flux_over_I = (1 - a) * fluxes_over_I[i] + a * fluxes_over_I[i + 1];
        if (flux_over_I == 0) continue;

        double redshift_factor = (1 - a) * redshift_factors[i] + a * redshift_factors[i + 1];
        double mu = (1 - a) * cos_sigma_primes[i] + a * cos_sigma_primes[i + 1];
        if (mu < nsx.mu_vec.back()) mu = nsx.mu_vec.back();
        for (int i_E = 0; i_E < ism_inst.E_obs.size(); ++i_E) {
          double E_obs_in_keV = ism_inst.E_obs[i_E];
          double E_emit_in_keV = E_obs_in_keV * redshift_factor;

          double logEkT = std::log10(E_emit_in_keV / kT_in_keV);
          double IT3 = nsx.Interp_IT3_2l2c_le(logT, logg, logEkT, mu);
          double I = IT3 * std::pow(10, 3 * logT) / planck;
          total_flux(i_phase_bin, i_E) += flux_over_I * I / E_obs_in_keV * GLWeights(N_points_per_phase_bin, j_in_bin, 1);
        }
      }
    }
  }

  tm.record();

  std::string output_filename = "output/" + s.name + "_" + setting + "_counts_cpu.txt";
  std::ofstream total_counts_file(output_filename);
  total_counts_file.precision(10);
  for (int i_ch = 0; i_ch < ism_inst.N_CH; ++i_ch) {
    for (int i_output_phase = 0; i_output_phase < N_output_phase_bins; ++i_output_phase) {
      double sum = 0;
      for (int i_E = 0; i_E < ism_inst.E_obs.size(); ++i_E) {
        sum += total_flux(i_output_phase, i_E) * ism_inst.rsp(i_ch, i_E);
      }
      total_counts_file << sum * exposure_time * phase_bin_width << ' ';
    }
    total_counts_file << '\n';
  }
  total_counts_file.close();
  printf(">>> Total counts written to %s\n", output_filename.c_str());

  tm.record();
  tm.print();
  tm.save("output/" + s.name + "_" + setting + "_time_cpu.txt");
}

int
main(int argc, char* argv[]) {
  NSX nsx;
  Lensing lt("std");
  IsmInst ism_inst("std");
  int N_fine_phase = 32 * 6;
  int N_points_per_phase_bin = 2;

  // int N_active_patches = 128 * 128;
  calc_one_star(ring_eq, lt, nsx, ism_inst, 173, N_fine_phase, N_points_per_phase_bin, "std");
  calc_one_star(ring_polar, lt, nsx, ism_inst, 1708, N_fine_phase, N_points_per_phase_bin, "std");
  calc_one_star(crescent_eq, lt, nsx, ism_inst, 128, N_fine_phase, N_points_per_phase_bin, "std");
  calc_one_star(crescent_polar, lt, nsx, ism_inst, 189, N_fine_phase, N_points_per_phase_bin, "std");

  // int N_active_patches = 256 * 256;
  calc_one_star(ring_eq, lt, nsx, ism_inst, 345, N_fine_phase, N_points_per_phase_bin, "high");
  calc_one_star(ring_polar, lt, nsx, ism_inst, 3413, N_fine_phase, N_points_per_phase_bin, "high");
  calc_one_star(crescent_eq, lt, nsx, ism_inst, 247, N_fine_phase, N_points_per_phase_bin, "high");
  calc_one_star(crescent_polar, lt, nsx, ism_inst, 382, N_fine_phase, N_points_per_phase_bin, "high");
}
