#pragma once

#include <cstdio>
#include "unit.cuh"

__device__ int
clamp_int(int i, int i_min, int i_max) {
  if (i < i_min) {
    return i_min;
  }
  if (i > i_max) {
    return i_max;
  }
  return i;
}

struct VecHunt {
  float const* xx;
  int n;
  int jsav;

  __device__
  VecHunt(float const* x, int size) :
      xx(x), n(size), jsav(0) {}

  __device__ void
  operator()(float x, int& i, float& a) {
#ifndef NDEBUG
    if ((x - xx[0]) * (x - xx[n - 1]) > 0) {
      printf(
          "[b%d,t%d]VecHunt::operator(): x = %g is out of bounds [%g, %g]: %d\n",
          blockIdx.x,
          threadIdx.x,
          x,
          xx[0],
          xx[n - 1],
          n
      );
      asm("trap;");
    }
#endif
    int jl = jsav, jm, ju, inc = 1;
    bool ascnd = (xx[n - 1] >= xx[0]);
    if (jl < 0 || jl > n - 1) {
      jl = 0;
      ju = n - 1;
    } else {
      if (x >= xx[jl] == ascnd) {
        for (;;) {
          ju = jl + inc;
          if (ju >= n - 1) {
            ju = n - 1;
            break;
          } else if (x < xx[ju] == ascnd)
            break;
          else {
            jl = ju;
            inc += inc;
          }
        }
      } else {
        ju = jl;
        for (;;) {
          jl = jl - inc;
          if (jl <= 0) {
            jl = 0;
            break;
          } else if (x >= xx[jl] == ascnd)
            break;
          else {
            ju = jl;
            inc += inc;
          }
        }
      }
    }
    while (ju - jl > 1) {
      jm = (ju + jl) >> 1;
      if (x >= xx[jm] == ascnd)
        jl = jm;
      else
        ju = jm;
    }
    jsav = jl;
    i = clamp_int(jl, 0, n - 2);
    a = (x - xx[i]) / (xx[i + 1] - xx[i]);
  }
};

struct UniformHunt {
  float x_min, x_max;
  int n;

  __device__
  UniformHunt(float x_min, float x_max, int n) :
      x_min(x_min), x_max(x_max), n(n) {}

  __device__ void
  operator()(float x, int& i, float& a) {
#ifndef NDEBUG
    if (x < x_min) {
      printf(
          "[b%d,t%d]UniformHunt::operator(): x = %g is out of bounds [%g, %g]: %d\n",
          blockIdx.x,
          threadIdx.x,
          x,
          x_min,
          x_max,
          n
      );
      asm("trap;");
    }
    if (x > x_max) {
      printf(
          "[b%d,t%d]UniformHunt::operator(): x = %g is out of bounds [%g, %g]: %d\n",
          blockIdx.x,
          threadIdx.x,
          x,
          x_min,
          x_max,
          n
      );
      asm("trap;");
    }
#endif
    float t = (x - x_min) / (x_max - x_min) * (n - 1);
    i = int(t);
    a = t - i;
  }

  __device__ float
  operator[](int i) {
#ifndef NDEBUG
    if (i < 0 || i >= n) {
      printf("UniformHunt::operator[]: Index %d out of bounds [0, %d)\n", i, n);
      asm("trap;");
    }
#endif
    return x_min + (x_max - x_min) * i / (n - 1);
  }
};
