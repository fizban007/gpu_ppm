#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include "grid.hpp"
#include "lensing.hpp"
#include "unit.hpp"
#include "ut.hpp"

struct StarData {
  double frequency_nu, spot_center_theta, obs_theta, angular_radius;
  std::string name;
  char beaming;
};

// clang-format off
StarData os1a{.frequency_nu = 600, .spot_center_theta = 90 * degree, .obs_theta = 90 * degree, .angular_radius = 0.01, .name = "os1a", .beaming = 'i'};
StarData os1b{.frequency_nu = 600, .spot_center_theta = 90 * degree, .obs_theta = 90 * degree, .angular_radius =    1, .name = "os1b", .beaming = 'i'};
StarData os1c{.frequency_nu = 200, .spot_center_theta = 90 * degree, .obs_theta = 90 * degree, .angular_radius = 0.01, .name = "os1c", .beaming = 'i'};
StarData os1d{.frequency_nu =   1, .spot_center_theta = 90 * degree, .obs_theta = 90 * degree, .angular_radius =    1, .name = "os1d", .beaming = 'i'};
StarData os1e{.frequency_nu = 600, .spot_center_theta = 60 * degree, .obs_theta = 30 * degree, .angular_radius =    1, .name = "os1e", .beaming = 'i'};
StarData os1f{.frequency_nu = 600, .spot_center_theta = 20 * degree, .obs_theta = 80 * degree, .angular_radius =    1, .name = "os1f", .beaming = 'i'};
StarData os1g{.frequency_nu = 600, .spot_center_theta = 60 * degree, .obs_theta = 30 * degree, .angular_radius =    1, .name = "os1g", .beaming = 'c'};
StarData os1h{.frequency_nu = 600, .spot_center_theta = 60 * degree, .obs_theta = 30 * degree, .angular_radius =    1, .name = "os1h", .beaming = 's'};
StarData os1i{.frequency_nu = 600, .spot_center_theta = 20 * degree, .obs_theta = 80 * degree, .angular_radius =    1, .name = "os1i", .beaming = 'c'};
StarData os1j{.frequency_nu = 600, .spot_center_theta = 20 * degree, .obs_theta = 80 * degree, .angular_radius =    1, .name = "os1j", .beaming = 's'};
// clang-format on

constexpr double M = 1.4;      // M_sun;
constexpr double Re = 12;      // km
constexpr double kT = 0.35;    // keV
constexpr double E_obs = 1.0;  // keV
constexpr double D = 0.2 * kpc_in_km;
constexpr int N_side = 256;
std::vector<double> output_phase_grid = linspace(0, 0.992187500, 128);

void
calc_one_star(StarData s, Lensing& lt) {
  Grid source = (s.angular_radius < 0.1)  //
                  ? PointSource(s.spot_center_theta, s.angular_radius, kT)
                  : CapSourceHP(s.spot_center_theta, s.angular_radius, kT, N_side);

  double Rs = M * schwarzschild_radius_of_sun_in_km;
  double Omega_bar = two_pi * s.frequency_nu * std::sqrt(Re * Re * Re / M) * sqrt_km3_over_s2_GMsun;
  double x = 0.5 * Rs / Re;
  double o2 = Omega_bar * Omega_bar * (-0.788 + 1.030 * x);
  double D2 = D * D;
  double cos_obs_theta = std::cos(s.obs_theta);
  double sin_obs_theta = std::sin(s.obs_theta);

  int N_fine_phase = output_phase_grid.size() * 3;
  std::vector<double> fluxes_over_I(N_fine_phase + 1);
  std::vector<double> redshift_factors(N_fine_phase + 1);
  std::vector<double> phase_o(N_fine_phase + 1);

  VecHunt phase_o_hunt{phase_o};

  std::vector<double> total_flux(output_phase_grid.size());
  std::fill(total_flux.begin(), total_flux.end(), 0);

  for (auto const& ring : source) {
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
            cos_alpha * cos_tau + sin_alpha_over_sin_psi * sin_tau * (cos_obs_theta - cos_spot_theta * cos_psi) / sin_spot_theta;
        if (cos_sigma <= 0) {
          is_visible = false;
          break;
        }

        double beta = two_pi * s.frequency_nu * R * sin_spot_theta / uu / c_in_km_s;
        double gamma = 1. / std::sqrt(1. - beta * beta);
        double cos_xi = -sin_alpha_over_sin_psi * sin_obs_theta * sin_spot_phi;
        double delta = 1. / (gamma * (1. - beta * cos_xi));
        double delta3 = delta * delta * delta;

        double cdt_over_R = lt.cal_cdt_over_R_of_u_cos_alpha(u, cos_alpha);
        double dt1 = cdt_over_R * R / c_in_km_s;

        auto cal_dt2 = [Rs](double RR) {
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
        double delta_phase = (dt1 + dt2) * s.frequency_nu;

        double beaming_factor;
        double cos_sigma_prime = cos_sigma * delta;
        if (s.beaming == 'i') {
          beaming_factor = 1.0;
        } else if (s.beaming == 's') {
          beaming_factor = 1. - cos_sigma_prime * cos_sigma_prime;
        } else if (s.beaming == 'c') {
          beaming_factor = cos_sigma_prime * cos_sigma_prime;
        }

        fluxes_over_I[i_phase] = uu * delta3 * cos_sigma_prime * lf * (dS * gamma) / D2 * beaming_factor;
        redshift_factors[i_phase] = 1. / (delta * uu);
        phase_o[i_phase] = phase + delta_phase;
      } while (0);
      if (is_visible == false) {
        fluxes_over_I[i_phase] = 0;
        redshift_factors[i_phase] = -1;
        phase_o[i_phase] = -1;
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

    for (int i_phase_points = 0; i_phase_points < output_phase_grid.size(); ++i_phase_points) {
      for (auto const& phi : ring.active_phi) {
        double phase_shift = phi / two_pi;
        double patch_phase = std::fmod(output_phase_grid[i_phase_points] + phase_shift + 1., 1.);
        if (patch_phase < phase_o[0]) patch_phase += 1.;
        auto [i, a] = phase_o_hunt(patch_phase);
        double flux_over_I = (1 - a) * fluxes_over_I[i] + a * fluxes_over_I[i + 1];
        if (flux_over_I == 0) continue;
        double redshift_factor = (1 - a) * redshift_factors[i] + a * redshift_factors[i + 1];
        double E_emit = E_obs * redshift_factor;
        double I = Ibb(E_emit, kT);
        total_flux[i_phase_points] += flux_over_I * I / E_obs;
      }
    }
  }

  std::string output_filename = "output/" + s.name + "_cpu.txt";
  std::ofstream output_file(output_filename);
  output_file.precision(10);
  for (int i_output_phase = 0; i_output_phase < output_phase_grid.size(); ++i_output_phase) {
    output_file << total_flux[i_output_phase] << ' ';
  }
  output_file << '\n';
  output_file.close();
  printf(">>> Total counts written to %s\n", output_filename.c_str());
}

int
main(int argc, char* argv[]) {
  Lensing lt("std");
  calc_one_star(os1a, lt);
  calc_one_star(os1b, lt);
  calc_one_star(os1c, lt);
  calc_one_star(os1d, lt);
  calc_one_star(os1e, lt);
  calc_one_star(os1f, lt);
  calc_one_star(os1g, lt);
  calc_one_star(os1h, lt);
  calc_one_star(os1i, lt);
  calc_one_star(os1j, lt);
}
