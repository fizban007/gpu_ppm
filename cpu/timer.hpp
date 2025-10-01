#pragma once

#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

struct Timer {
  std::chrono::high_resolution_clock::time_point last_t;
  std::vector<double> dt;

  void
  start() {
    dt.clear();
    dt.reserve(100);
    last_t = std::chrono::high_resolution_clock::now();
  }

  void
  record() {
    auto now_t = std::chrono::high_resolution_clock::now();
    double t_ms = std::chrono::duration<double, std::milli>(now_t - last_t).count();
    dt.push_back(t_ms * 1e-3);
    last_t = now_t;
  }

  void
  print() const {
    double sum = std::accumulate(dt.begin(), dt.end(), 0.0);
    printf("Time taken: %.4g (", sum);
    for (int i = 0; i < dt.size(); ++i) {
      if (i != 0) printf(", ");
      printf("%.4g", dt[i]);
    }
    printf(") s \n");
  }

  void
  save(std::string const& fname) const {
    std::ofstream out_file(fname);
    if (!out_file) {
      std::cerr << "Could not open file to save timer data: " << fname << "\n";
      return;
    }
    double sum = std::accumulate(dt.begin(), dt.end(), 0.0);
    out_file << sum << "\n";
    out_file.close();
    printf("Timer data saved to %s\n", fname.c_str());
  }
};
