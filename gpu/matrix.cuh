#pragma once

#include <vector>

template <typename T>
struct Matrix {
  int n_row = 0;
  int n_col = 0;
  std::vector<T> data;

  Matrix() = default;
  Matrix(int n_row, int n_col) : n_row(n_row), n_col(n_col), data(n_row * n_col) {}

  T&
  operator()(int row, int col) {
    return data[row * n_col + col];
  }

  void
  fill(T value) {
    std::fill(data.begin(), data.end(), value);
  }

  void
  reset(int n_row_, int n_col_) {
    n_row = n_row_;
    n_col = n_col_;
    data.resize(n_row * n_col);
  }
};
