#pragma once

// https://pdg.lbl.gov/2024/reviews/rpp2024-rev-astrophysical-constants.pdf
constexpr double schwarzschild_radius_of_sun_in_km = 2.9532501;

// https://en.wikipedia.org/wiki/Solar_mass
constexpr double sqrt_km3_over_s2_GMsun = 2.745011592867327e-6;

// https://physics.nist.gov/cgi-bin/cuu/Value?Rkev
constexpr double boltzmann_constant_in_keV_over_K = 8.617333262e-8;

constexpr double kpc_in_km = 3.08567758e+16;
constexpr double planck = 6.62607015E-27;
constexpr double h_in_keV_s = 4.135668e-18;

constexpr double pi = 3.14159265358979323846;
constexpr double two_pi = 2. * pi;

constexpr double c_in_km_s = 299792458. * 1e-3;
constexpr double c_in_cm_s = 299792458. * 1e2;

constexpr double degree = pi / 180.;

std::vector<float>
linspace(float beg, float end, int num_points) {
  if (num_points == 1) {
    return {0.5f * (beg + end)};
  }
  std::vector<float> vec(num_points);
  float step = (end - beg) / (num_points - 1);
  for (int i = 0; i < num_points; ++i) {
    vec[i] = beg + i * step;
  }
  return vec;
}
