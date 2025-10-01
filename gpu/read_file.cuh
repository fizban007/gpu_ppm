#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include "matrix.cuh"
#include "unit.cuh"

inline void
read2(int& v1, int& v2, std::string const& fname) {
  std::ifstream in_file(fname);
  if (!in_file) {
    std::cerr << "Could not open file: " << fname << "\n";
    std::exit(EXIT_FAILURE);
  }
  if (!(in_file >> v1 >> v2)) {
    std::cerr << "Error reading 2 values from file: " << fname << "\n";
    std::exit(EXIT_FAILURE);
  }
  in_file.close();
}

inline void
read3(float& v1, float& v2, int& v3, std::string const& fname) {
  std::ifstream in_file(fname);
  if (!in_file) {
    std::cerr << "Could not open file: " << fname << "\n";
    std::exit(EXIT_FAILURE);
  }
  if (!(in_file >> v1 >> v2 >> v3)) {
    std::cerr << "Error reading 3 values from file: " << fname << "\n";
    std::exit(EXIT_FAILURE);
  }
  in_file.close();
}

inline void
read_vector(std::vector<float>& vec, std::string const& fname, int n) {
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
read_matrix(Matrix<float>& mat, std::string const& fname, int n_rows, int n_cols) {
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
