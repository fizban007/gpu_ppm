NVCC := nvcc
gpu_executables := devarshi24st_gpu os1_gpu
gpu_src_files := $(wildcard gpu/*)

GPUFLAGS := -arch=sm_89 -std=c++20 -O3 -Xcompiler "-std=c++20,-O3,-march=native"
GPUFLAGS += -lcublas
GPUFLAGS += --use_fast_math
# GPUFLAGS += --resource-usage
GPUFLAGS += --maxrregcount=64
# GPUFLAGS += -DNDEBUG


cpu_executables := devarshi24st_cpu t_interp os1_cpu
cpu_headers := $(wildcard cpu/*)
CXXFLAGS := -std=c++20 -O3 -march=native -ffast-math


ifneq (command line,$(origin CXX))
  CXX := clang++
#   CXX := g++
endif


$(gpu_executables): % : gpu/%.cu $(gpu_src_files) build output
	$(NVCC) $(GPUFLAGS) $< -o build/$@

$(cpu_executables): % : cpu/%.cpp $(cpu_headers) build output
	$(CXX) $(CXXFLAGS) $< -o build/$@

build:
	mkdir -p build
output:
	mkdir -p output
