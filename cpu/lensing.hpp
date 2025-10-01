#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include "hunt.hpp"
#include "matrix.hpp"

struct Lensing {
  UniformHunt cos_psi, u, cos_alpha;
  Matrix cos_alpha_of_u_cos_psi, lf_of_u_cos_psi, cdt_over_R_of_u_cos_alpha;

  void
  read_matrix(Matrix& mat, std::string const& fname, int n_rows, int n_cols) {
    std::ifstream in_file(fname);
    if (!in_file) {
      std::cerr << "Could not open lensing table file: " << fname << "\n";
      std::exit(EXIT_FAILURE);
    }
    mat.reset(n_rows, n_cols);
    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < n_cols; ++j) {
        if (!(in_file >> mat(i, j))) {
          std::cerr << "Error reading matrix from file: " << fname << "\n";
          std::exit(EXIT_FAILURE);
        }
      }
    }
    in_file.close();
  }

  Lensing(std::string const& setting) :
      cos_psi("tables/lensing/" + setting + "/cos_psi.txt"),
      u("tables/lensing/" + setting + "/u.txt"),
      cos_alpha("tables/lensing/" + setting + "/cos_alpha.txt") {
    read_matrix(cos_alpha_of_u_cos_psi, "tables/lensing/" + setting + "/cos_alpha_of_u_cos_psi.txt", u.n, cos_psi.n);
    read_matrix(lf_of_u_cos_psi, "tables/lensing/" + setting + "/lf_of_u_cos_psi.txt", u.n, cos_psi.n);
    read_matrix(cdt_over_R_of_u_cos_alpha, "tables/lensing/" + setting + "/cdt_over_R_of_u_cos_alpha.txt", u.n, cos_alpha.n);
    printf("Lensing table loaded: u[%d], cos_psi[%d], cos_alpha[%d]\n", u.n, cos_psi.n, cos_alpha.n);
  }

  auto
  cal_cos_alpha_lf_of_u_cos_psi(double u_, double cos_psi_) {
    auto [i_u, a_u] = u(u_);
    auto [i_cp, a_cp] = cos_psi(cos_psi_);
    double b_u = 1. - a_u, b_cp = 1. - a_cp;

    double cos_alpha_ = b_u * b_cp * cos_alpha_of_u_cos_psi(i_u, i_cp)
                      + a_u * b_cp * cos_alpha_of_u_cos_psi(i_u + 1, i_cp)
                      + b_u * a_cp * cos_alpha_of_u_cos_psi(i_u, i_cp + 1)
                      + a_u * a_cp * cos_alpha_of_u_cos_psi(i_u + 1, i_cp + 1);

    double lf_ = b_u * b_cp * lf_of_u_cos_psi(i_u, i_cp)
               + a_u * b_cp * lf_of_u_cos_psi(i_u + 1, i_cp)
               + b_u * a_cp * lf_of_u_cos_psi(i_u, i_cp + 1)
               + a_u * a_cp * lf_of_u_cos_psi(i_u + 1, i_cp + 1);

    return std::make_tuple(cos_alpha_, lf_);
  }

  double
  cal_cdt_over_R_of_u_cos_alpha(double u_, double cos_alpha_) {
    auto [i_u, a_u] = u(u_);
    auto [i_ca, a_ca] = cos_alpha(cos_alpha_);
    double b_u = 1. - a_u, b_ca = 1. - a_ca;

    return b_u * b_ca * cdt_over_R_of_u_cos_alpha(i_u, i_ca)
         + a_u * b_ca * cdt_over_R_of_u_cos_alpha(i_u + 1, i_ca)
         + b_u * a_ca * cdt_over_R_of_u_cos_alpha(i_u, i_ca + 1)
         + a_u * a_ca * cdt_over_R_of_u_cos_alpha(i_u + 1, i_ca + 1);
  }
};
