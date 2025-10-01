#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "hunt.hpp"

struct NsxConfig {
  char const* nsx_fname;
  int N_logT;
  int N_logg;
  int N_logEkT;
  int N_mu;
  double logT_min;
  double logT_max;
  double logg_min;
  double logg_max;
  double logEkT_min;
  double logEkT_max;
};

constexpr NsxConfig nsx_H_v200804 = {
    .nsx_fname = "./ext/model_data/nsx_H_v200804.out",
    .N_logT = 35,
    .N_logg = 14,
    .N_logEkT = 166,
    .N_mu = 67,
    .logT_min = 5.10,
    .logT_max = 6.80,
    .logg_min = 13.7,
    .logg_max = 15.0,
    .logEkT_min = -1.30,
    .logEkT_max = 2.00
};

constexpr NsxConfig nsx_H_v171019 = {
    .nsx_fname = "./ext/model_data/nsx_H_v171019.out",
    .N_logT = 35,
    .N_logg = 11,
    .N_logEkT = 166,
    .N_mu = 67,
    .logT_min = 5.10,
    .logT_max = 6.80,
    .logg_min = 13.7,
    .logg_max = 14.7,
    .logEkT_min = -1.30,
    .logEkT_max = 2.00
};

constexpr NsxConfig c = nsx_H_v200804;

struct NSX {
  std::vector<double> mu_vec;
  VecHunt mu_hunt;
  UniformHunt logEkT_hunt, logT_hunt, logg_hunt;
  std::vector<double> data;

  double&
  operator()(int i_logT, int i_logg, int i_logEkT, int i_mu) {
#if 1
    if (i_logT < 0
        || i_logT >= c.N_logT
        || i_logg < 0
        || i_logg >= c.N_logg
        || i_logEkT < 0
        || i_logEkT >= c.N_logEkT
        || i_mu < 0
        || i_mu >= c.N_mu) {
      printf("NSX::operator() : Index out of bounds\n");
      std::exit(EXIT_FAILURE);
    }
#endif
    int index = i_logT * c.N_logg * c.N_logEkT * c.N_mu  //
              + i_logg * c.N_logEkT * c.N_mu             //
              + i_logEkT * c.N_mu                        //
              + i_mu;
    return data[index];
  }

  NSX() :
      mu_vec(c.N_mu),
      mu_hunt(mu_vec),
      logEkT_hunt(c.logEkT_min, c.logEkT_max, c.N_logEkT),
      logT_hunt(c.logT_min, c.logT_max, c.N_logT),
      logg_hunt(c.logg_min, c.logg_max, c.N_logg),
      data(c.N_logT * c.N_logg * c.N_logEkT * c.N_mu, 0.0) {
    std::ifstream in_file(c.nsx_fname);
    if (!in_file) {
      printf("Could not open NSX file: %s\n", c.nsx_fname);
      std::exit(EXIT_FAILURE);
    }

    printf("Loading NSX data from %s\n", c.nsx_fname);

    for (int i_logT = 0; i_logT < c.N_logT; ++i_logT) {
      for (int i_logg = 0; i_logg < c.N_logg; ++i_logg) {
        for (int i_logEkT = 0; i_logEkT < c.N_logEkT; ++i_logEkT) {
          for (int i_mu = 0; i_mu < c.N_mu; ++i_mu) {
            double logEkT, mu, logInuT3, logT, logg;
            in_file >> logEkT >> mu >> logInuT3 >> logT >> logg;

            if (!in_file) {
              printf("Error reading NSX data at indices: %d, %d, %d, %d\n", i_logT, i_logg, i_logEkT, i_mu);
              std::exit(EXIT_FAILURE);
            }

            if (mu_hunt[i_mu] == 0) {
              mu_vec[i_mu] = mu;
            } else {
              if (!IsClose(mu_hunt[i_mu], mu)) {
                printf("Inconsistent mu value at index %d: %g vs %g\n", i_mu, mu_hunt[i_mu], mu);
                std::exit(EXIT_FAILURE);
              }
            }

            if (!IsClose(logEkT_hunt[i_logEkT], logEkT)) {
              printf("Inconsistent logEkT value at index %d: %g vs %g\n", i_logEkT, logEkT_hunt[i_logEkT], logEkT);
              std::exit(EXIT_FAILURE);
            }

            if (!IsClose(logT_hunt[i_logT], logT)) {
              printf("Inconsistent logT value at index %d: %g vs %g\n", i_logT, logT_hunt[i_logT], logT);
              std::exit(EXIT_FAILURE);
            }

            if (!IsClose(logg_hunt[i_logg], logg)) {
              printf("Inconsistent logg value at index %d: %g vs %g\n", i_logg, logg_hunt[i_logg], logg);
              std::exit(EXIT_FAILURE);
            }

            this->operator()(i_logT, i_logg, i_logEkT, i_mu) = std::pow(10, logInuT3);
          }
        }
      }
    }
    in_file.close();
    printf("NSX data loaded successfully from %s\n", c.nsx_fname);

    for (int i = 0; i < c.N_mu - 1; ++i) {
      if (mu_hunt[i] < mu_hunt[i + 1]) {
        printf("mu_hunt is not monotonically decreasing at index %d: %g > %g\n", i, mu_hunt[i], mu_hunt[i + 1]);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  bool
  IsClose(double a, double b) {
    constexpr double atol = 10 * std::numeric_limits<double>::epsilon();
    constexpr double rtol = 1e-8;
    double bigger = std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= atol + rtol * bigger;
  }

  // cubic lagrange interpolation
  struct CubicLagrange {
    double w[4];
    void
    operator()(double x, double x0, double x1, double x2, double x3) {
      w[0] = (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3));
      w[1] = (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3));
      w[2] = (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3));
      w[3] = (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2));
    }
  };

  CubicLagrange logT_cl, logg_cl, logEkT_cl, mu_cl;

  double
  Interp_IT3_4c(double logT, double logg, double logEkT, double mu, bool use_sg) {
    auto [i_logT, a_logT] = logT_hunt(logT);
    auto [i_logg, a_logg] = logg_hunt(logg);
    auto [i_logEkT, a_logEkT] = logEkT_hunt(logEkT);
    auto [i_mu, a_mu] = mu_hunt(mu);

    i_logT = std::clamp(i_logT - 1, 0, c.N_logT - 4);
    i_logg = std::clamp(i_logg - 1, 0, c.N_logg - 4);
    i_logEkT = std::clamp(i_logEkT - 1, 0, c.N_logEkT - 4);
    i_mu = std::clamp(i_mu - 1, 0, c.N_mu - 4);

    logT_cl(logT, logT_hunt[i_logT], logT_hunt[i_logT + 1], logT_hunt[i_logT + 2], logT_hunt[i_logT + 3]);
    logg_cl(logg, logg_hunt[i_logg], logg_hunt[i_logg + 1], logg_hunt[i_logg + 2], logg_hunt[i_logg + 3]);
    logEkT_cl(logEkT, logEkT_hunt[i_logEkT], logEkT_hunt[i_logEkT + 1], logEkT_hunt[i_logEkT + 2], logEkT_hunt[i_logEkT + 3]);
    mu_cl(mu, mu_hunt[i_mu], mu_hunt[i_mu + 1], mu_hunt[i_mu + 2], mu_hunt[i_mu + 3]);

    double ret = 0;
    for (int j_logT = 0; j_logT <= 3; ++j_logT) {
      for (int j_logg = 0; j_logg <= 3; ++j_logg) {
        for (int j_logEkT = 0; j_logEkT <= 3; ++j_logEkT) {
          for (int j_mu = 0; j_mu <= 3; ++j_mu) {
            double v = this->operator()(i_logT + j_logT, i_logg + j_logg, i_logEkT + j_logEkT, i_mu + j_mu);
            double weight = logT_cl.w[j_logT] * logg_cl.w[j_logg] * logEkT_cl.w[j_logEkT] * mu_cl.w[j_mu];
            ret += v * weight;
          }
        }
      }
    }
    if (use_sg && ret < 0) ret = 0;
    return ret;
  }

  double
  Interp_IT3_4c_le(double logT, double logg, double logEkT, double mu) {
    auto [i_logT, a_logT] = logT_hunt(logT);
    auto [i_logg, a_logg] = logg_hunt(logg);
    auto [i_logEkT, a_logEkT] = logEkT_hunt(logEkT);
    auto [i_mu, a_mu] = mu_hunt(mu);

    i_logT = std::clamp(i_logT - 1, 0, c.N_logT - 4);
    i_logg = std::clamp(i_logg - 1, 0, c.N_logg - 4);
    i_logEkT = std::clamp(i_logEkT - 1, 0, c.N_logEkT - 4);

    logT_cl(logT, logT_hunt[i_logT], logT_hunt[i_logT + 1], logT_hunt[i_logT + 2], logT_hunt[i_logT + 3]);
    logg_cl(logg, logg_hunt[i_logg], logg_hunt[i_logg + 1], logg_hunt[i_logg + 2], logg_hunt[i_logg + 3]);
    logEkT_cl(logEkT, logEkT_hunt[i_logEkT], logEkT_hunt[i_logEkT + 1], logEkT_hunt[i_logEkT + 2], logEkT_hunt[i_logEkT + 3]);

    int j_mu_max;
    if (i_mu >= c.N_mu - 5) {
      // linear at edge
      j_mu_max = 1;
      mu_cl.w[0] = 1 - a_mu;
      mu_cl.w[1] = a_mu;
    } else {
      // cubic in the middle
      j_mu_max = 3;
      i_mu = std::clamp(i_mu - 1, 0, c.N_mu - 4);
      mu_cl(mu, mu_hunt[i_mu], mu_hunt[i_mu + 1], mu_hunt[i_mu + 2], mu_hunt[i_mu + 3]);
    }

    double ret = 0;
    for (int j_logT = 0; j_logT <= 3; ++j_logT) {
      for (int j_logg = 0; j_logg <= 3; ++j_logg) {
        for (int j_logEkT = 0; j_logEkT <= 3; ++j_logEkT) {
          for (int j_mu = 0; j_mu <= j_mu_max; ++j_mu) {
            double v = this->operator()(i_logT + j_logT, i_logg + j_logg, i_logEkT + j_logEkT, i_mu + j_mu);
            double weight = logT_cl.w[j_logT] * logg_cl.w[j_logg] * logEkT_cl.w[j_logEkT] * mu_cl.w[j_mu];
            ret += v * weight;
          }
        }
      }
    }
    return ret;
  }

  double
  Interp_IT3_2l2c_le(double logT, double logg, double logEkT, double mu) {
    auto [i_logT, a_logT] = logT_hunt(logT);
    auto [i_logg, a_logg] = logg_hunt(logg);
    auto [i_logEkT, a_logEkT] = logEkT_hunt(logEkT);
    auto [i_mu, a_mu] = mu_hunt(mu);

    i_logEkT = std::clamp(i_logEkT - 1, 0, c.N_logEkT - 4);
    logEkT_cl(logEkT, logEkT_hunt[i_logEkT], logEkT_hunt[i_logEkT + 1], logEkT_hunt[i_logEkT + 2], logEkT_hunt[i_logEkT + 3]);

    int j_mu_max;
    if (i_mu >= c.N_mu - 5) {
      // linear at edge
      j_mu_max = 1;
      mu_cl.w[0] = 1 - a_mu;
      mu_cl.w[1] = a_mu;
    } else {
      // cubic in the middle
      j_mu_max = 3;
      i_mu = std::clamp(i_mu - 1, 0, c.N_mu - 4);
      mu_cl(mu, mu_hunt[i_mu], mu_hunt[i_mu + 1], mu_hunt[i_mu + 2], mu_hunt[i_mu + 3]);
    }

    double ret = 0;
    for (int j_logT = 0; j_logT <= 1; ++j_logT) {
      for (int j_logg = 0; j_logg <= 1; ++j_logg) {
        for (int j_logEkT = 0; j_logEkT <= 3; ++j_logEkT) {
          for (int j_mu = 0; j_mu <= j_mu_max; ++j_mu) {
            double v = this->operator()(i_logT + j_logT, i_logg + j_logg, i_logEkT + j_logEkT, i_mu + j_mu);
            double weight = (j_logT * a_logT + (1 - j_logT) * (1 - a_logT))
                          * (j_logg * a_logg + (1 - j_logg) * (1 - a_logg))
                          * logEkT_cl.w[j_logEkT]
                          * mu_cl.w[j_mu];
            ret += v * weight;
          }
        }
      }
    }
    // if (ret < 0) {
    //   printf("Warning: Interp_IT3_2l2c_le returned negative value %g, setting to 0\n", ret);
    //   printf("  logT = %g, logg = %g, logEkT = %g, mu = %g\n", logT, logg, logEkT, mu);
    //   printf("  i_logT = %d, a_logT = %g\n", i_logT, a_logT);
    //   printf("  i_logg = %d, a_logg = %g\n", i_logg, a_logg);
    //   printf("  i_logEkT = %d, a_logEkT = %g\n", i_logEkT, a_logEkT);
    //   printf("  i_mu = %d, a_mu = %g\n", i_mu, a_mu);
    //   ret = 0;
    // }
    return ret;
  }

  double
  Interp_IT3_4l(double logT, double logg, double logEkT, double mu) {
    auto [i_logT, a_logT] = logT_hunt(logT);
    auto [i_logg, a_logg] = logg_hunt(logg);
    auto [i_logEkT, a_logEkT] = logEkT_hunt(logEkT);
    auto [i_mu, a_mu] = mu_hunt(mu);

    double ret = 0;
    for (int j_logT = 0; j_logT <= 1; ++j_logT) {
      for (int j_logg = 0; j_logg <= 1; ++j_logg) {
        for (int j_logEkT = 0; j_logEkT <= 1; ++j_logEkT) {
          for (int j_mu = 0; j_mu <= 1; ++j_mu) {
            double v = this->operator()(i_logT + j_logT, i_logg + j_logg, i_logEkT + j_logEkT, i_mu + j_mu);
            double weight = (j_logT * a_logT + (1 - j_logT) * (1 - a_logT))
                          * (j_logg * a_logg + (1 - j_logg) * (1 - a_logg))
                          * (j_logEkT * a_logEkT + (1 - j_logEkT) * (1 - a_logEkT))
                          * (j_mu * a_mu + (1 - j_mu) * (1 - a_mu));
            ret += v * weight;
          }
        }
      }
    }
    return ret;
  }
};
