NVCC := nvcc
gpu_executables := os1_gpu devarshi24st_gpu
gpu_src_files := $(wildcard gpu/*)

GPUARCH := -arch=native
GPUFLAGS := -std=c++20 -O3 -Xcompiler "-std=c++20,-O3,-march=native"
GPUFLAGS += -lcublas
GPUFLAGS += --use_fast_math
GPUFLAGS += --maxrregcount=64
# GPUFLAGS += --resource-usage
# GPUFLAGS += -DNDEBUG

cpu_executables := os1_cpu t_interp devarshi24st_cpu
cpu_headers := $(wildcard cpu/*)
CXXFLAGS := -std=c++20 -O3 -march=native -ffast-math

ifneq (command line,$(origin CXX))
  CXX := clang++
endif


$(gpu_executables): % : gpu/%.cu $(gpu_src_files) build output
	$(NVCC) $(GPUARCH) $(GPUFLAGS) $< -o build/$@

$(cpu_executables): % : cpu/%.cpp $(cpu_headers) build output
	$(CXX) $(CXXFLAGS) $< -o build/$@

build:
	mkdir -p build
output:
	mkdir -p output
