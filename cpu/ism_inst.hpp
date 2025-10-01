#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>

#include "matrix.hpp"

inline void
read2(int& v1, int& v2, std::string const& fname) {
  std::ifstream in_file(fname);
  if (!in_file) {
    std::cerr << "Could not open file: " << fname << "\n";
    std::exit(EXIT_FAILURE);
  }
  if (!(in_file >> v1 >> v2)) {
    std::cerr << "Error reading 2 values from file\n";
    std::exit(EXIT_FAILURE);
  }
  in_file.close();
}

inline void
read_vector(std::vector<double>& vec, std::string const& fname, int n) {
  std::ifstream in_file(fname);
  if (!in_file) {
    std::cerr << "Could not open file: " << fname << "\n";
    std::exit(EXIT_FAILURE);
  }
  vec.resize(n);
  for (int i = 0; i < n; ++i) {
    if (!(in_file >> vec[i])) {
      std::cerr << "Error reading vector from file: " << fname << "\n";
      std::exit(EXIT_FAILURE);
    }
  }
  in_file.close();
}

inline void
read_matrix(Matrix& mat, std::string const& fname, int n_rows, int n_cols) {
  std::ifstream in_file(fname);
  if (!in_file) {
    std::cerr << "Could not open file: " << fname << "\n";
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

struct IsmInst {
  int N_CH;
  int N_E_obs;
  std::vector<double> E_obs;
  Matrix rsp;

  IsmInst(std::string const& setting) {
    read2(N_CH, N_E_obs, "tables/ism_inst/" + setting + "/num.txt");
    read_vector(E_obs, "tables/ism_inst/" + setting + "/E_obs.txt", N_E_obs);
    read_matrix(rsp, "tables/ism_inst/" + setting + "/rsp.txt", N_CH, N_E_obs);
  }
};
