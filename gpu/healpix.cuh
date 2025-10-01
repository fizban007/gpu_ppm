#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "unit.cuh"

namespace HEALPix {
inline __device__ float
dOmega(int N_side) {
  return float(pi) / (3.0f * N_side * N_side);
}

inline __device__ int
i_min() {
  return 1;
}

inline __device__ int
i_max(int N_side) {
  return 4 * N_side - 1;
}

inline int
n_rings(int N_side) {
  return 4 * N_side - 1;
}

inline __device__ int
j_min() {
  return 1;
}

inline __device__ int
j_max(int N_side, int i) {
  int temp = N_side;
  if (temp > i) temp = i;
  if (temp > 4 * N_side - i) temp = 4 * N_side - i;
  return 4 * temp;
}

inline __device__ __host__ constexpr int
max_ring_size(int N_side) {
  return 4 * N_side;
}

inline __device__ float
cos_theta(int N_side, int i) {
#ifndef NDEBUG
  if (i < i_min() || i > i_max(N_side)) {
    printf("In HEALPix::cos_theta, N_side: %d, i: %d is out of range.\n", N_side, i);
    asm("trap;");
  }
#endif

  bool minus_flag;
  if (i > 2 * N_side) {
    i = 4 * N_side - i;
    minus_flag = true;
  } else {
    minus_flag = false;
  }

  float result = (i < N_side) ? (1.0f - 1.0f * i * i / (3.0f * N_side * N_side)) : (4.0f / 3.0f - 2.0f * i / (3.0f * N_side));

  if (minus_flag) result = -result;

  return result;
}

inline __device__ float
phi(int N_side, int i, int j) {
#ifndef NDEBUG
  if (i < i_min() || i > i_max(N_side) || j < j_min() || j > j_max(N_side, i)) {
    printf("In HEALPix::phi, N_side: %d, i: %d, j: %d is out of range.\n", N_side, i, j);
    asm("trap;");
  }
#endif

  if (i > 4 * N_side - i) i = 4 * N_side - i;

  if (i < N_side) {
    return float(pi) / (2.0f * i) * (j - 0.5f);
  } else {
    int s = (i - N_side + 1) % 2;
    return float(pi) / (2.0f * N_side) * (j - 0.5f * s);
  }
}

}  // namespace HEALPix
