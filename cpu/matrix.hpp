#pragma once

#include <fstream>
#include <vector>

struct Matrix {
  int n_row = 0;
  int n_col = 0;
  std::vector<double> data_;

  Matrix() = default;
  Matrix(int n_row, int n_col) : n_row(n_row), n_col(n_col), data_(n_row * n_col) {}

  double&
  operator()(int row, int col) {
    if (row < 0 || row >= n_row || col < 0 || col >= n_col) {
      printf("Matrix index out of bounds: (%d, %d) for size (%d, %d)\n", row, col, n_row, n_col);
      std::exit(EXIT_FAILURE);
    }

    return data_[row * n_col + col];
  }

  void
  fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
  }
  void
  reset(int n_row_, int n_col_) {
    n_row = n_row_;
    n_col = n_col_;
    data_.assign(n_row * n_col, -123);
  }

  void
  dump(std::string const& filename) {
    std::ofstream file(filename);
    if (!file) {
      printf("Error opening file %s for writing\n", filename.c_str());
      std::exit(EXIT_FAILURE);
    }
    file.precision(10);
    for (int i = 0; i < n_row; ++i) {
      for (int j = 0; j < n_col; ++j) {
        file << (*this)(i, j) << " ";
      }
      file << "\n";
    }
  }
};
