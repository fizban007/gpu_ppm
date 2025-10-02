gpu_ppm: GPU-Accelerated Pulsar / Neutron Star Pulse Profile Modeling
=====================================================================

High‑performance C++20 / CUDA code for generating phase‑resolved pulse profiles and energy–phase counts for rotating neutron star surface emission with relativistic light bending, time delays, Doppler / aberration effects, oblate star shape, atmosphere beaming (NSX tables), and instrument response folding. Includes CPU reference implementations for validation and comparison.

This repository is intended to reproduce the results of our work (arXiv:xxxx.xxxx). Please consider citing the paper if you use this code in academic work.

## Key Features

* CUDA implementations (`devarshi24st_gpu`, `os1_gpu`) with heavy shared memory use and warp‑efficient interpolation / reduction.
* CPU counterparts (`devarshi24st_cpu`, `os1_cpu`) for correctness cross‑checks and baseline timing.
* General relativistic lensing pre‑tabulated (tables/lensing/*) with bilinear interpolation on the device.
* NSX hydrogen atmosphere model (ext/model_data) loaded once and cached in GPU global memory; cubic Lagrange and linear edge interpolation in energy / angle.
* HEALPix‑like spherical tiling on the GPU for surface integration with dynamic active patch selection per ring.
* Time‑of‑flight corrections and rotational phase shifts (including oblate geometry corrections) handled per surface element.
* Instrument response folding (tables/ism_inst/*) to produce channel counts vs phase (simulating detector observation) on the GPU via a matrix multiply.
* Multiple spot morphologies: rings, crescents, single circular spots, difference of disks.
* Blackbody and atmosphere spectral synthesis with relativistic redshift and Doppler boosting.

## Repository Layout

```
Makefile                # Build targets for CPU & GPU binaries
cpu/                    # C++20 reference implementations and utilities
gpu/                    # CUDA (.cu / .cuh) kernels & GPU data structures
ext/                    # External large data (NSX atmosphere, pulse profile comparisons, etc.)
tables/                 # Precomputed lensing & instrument response tables
output/                 # Generated output text files (counts, fluxes, timings)
*.py                    # Plotting / analysis scripts (e.g. generate tables, result comparisons)
build/                  # Created by Makefile for compiled executables
```

## Build Instructions

### Prerequisites

* A C++20 capable host compiler (Clang++ default; switch to g++ by defining `CXX=g++`).
* CUDA toolkit (nvcc) with support for your GPU architecture (Makefile currently sets `-arch=sm_89`; adjust if needed, e.g. `sm_80`, `sm_90`).
* Standard Unix build utilities (make, bash). No external package manager dependencies.

### External Data Requirements
All required external data files are packaged in the `ext.tar.xz` archive. 
Extract it in the repository root to populate the `ext/` directory.
Use the following command to extract the archive:
```bash
tar -xJvf ext.tar.xz   # run this in repository root folder
```
This will create the `ext/` directory with the necessary subdirectories and files.

### Generate Precomputed Tables
Both CPU and GPU program executables depend on precomputed lensing and instrument response tables located in the `tables/` directory. If these tables are not already present, you can generate them using the provided Python scripts:
```bash
python3 c_lensing.py   # generates lensing tables in ./tables/lensing
python3 c_ism_inst.py  # generates ism and instrument response tables in ./tables/ism_inst
```
Ensure you have Python3 installed along with all required libraries (`numpy` and `scipy`). The generated tables will be stored in the appropriate subdirectories under `tables/`.

### Compile CPU binaries
Select a C++20 capable compiler (default is Clang++) and ensure it's available in your PATH.
Use the following command to build CPU executables:
```bash
# builds all CPU executables into ./build using default compiler(clang++)
make os1_cpu t_interp devarshi24st_cpu 
# or use user defined compiler(g++) instead of clang++
make CXX=g++ os1_cpu t_interp devarshi24st_cpu   
``` 

Targets produced:
* `build/t_interp`
* `build/os1_cpu`
* `build/devarshi24st_cpu`

### Compile GPU binaries
Ensure you have the CUDA toolkit installed and your GPU is supported. (Check https://developer.nvidia.com/cuda-downloads for installation instructions.)
Use the following command to build GPU executables:
```bash
# builds all GPU executables into ./build 
make os1_gpu devarshi24st_gpu
```
If your cuda compiler(`nvcc`) does not support `-arch=native`, you could either upgrade your cuda compiler to a newer version, or check:

https://developer.nvidia.com/cuda-gpus

https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-steering-gpu-code-generation

https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-feature-list

https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list

and manually set the `GPUARCH` variable in the `Makefile` to match your GPU's compute capability. 
For example, for an GeForce RTX 4080, you would set `-arch=sm_89`.


Targets produced:
* `build/os1_gpu`
* `build/devarshi24st_gpu`


## Data Requirements

You must have the external data directories populated:

* `ext/model_data/nsx_H_v200804.out` (or alternative NSX grid) – atmosphere intensities.
* `tables/lensing/<setting>/` with files:
	* `u.txt`, `cos_psi.txt`, `cos_alpha.txt`
	* `cos_alpha_of_u_cos_psi.txt`, `lf_of_u_cos_psi.txt`, `cdt_over_R_of_u_cos_alpha.txt`
* `tables/ism_inst/<setting>/` with files:
	* `num.txt` (N_CH N_E_obs)
	* `E_obs.txt`
	* `rsp.txt` (response matrix, shape N_CH × N_E_obs)

Two example settings referenced in code: `std` and `high`.

## Running the Programs

### Oblate Schwarzschild (OS1) Test case
Use the following commands to run the OS1 scenario simulations:
```bash
./build/os1_gpu
./build/os1_cpu
```
It will produces per‑scenario flux vs phase files:

* `output/os1a_gpu.txt` ... `output/os1j_gpu.txt`
* CPU equivalents: `output/os1a_cpu.txt` ... etc.

Run data visualization scripts to generate comparison plots:
```bash
python3 plot_os1.py
```


### Atmosphere Interpolation Test case
Use the following commands to run the atmosphere interpolation test:
```bash
./build/t_interp
```
Run data visualization scripts to generate comparison plots:
```bash
python3 plot_t_interp.py
```


### Test case in [Devarshi et al. (2024)](https://iopscience.iop.org/article/10.3847/1538-4357/ad7255) 
Use the following commands to run:
```bash
./build/devarshi24st_cpu   # CPU reference 
./build/devarshi24st_gpu   # GPU version 
```
Run data visualization scripts:
```bash
python3 p_devarshi24st.py a
python3 p_devarshi24st.py b
python3 p_devarshi24st.py c
python3 p_devarshi24st.py d
```

### Plot Oblate Star Shape Demonstration Figure
```bash
python3 p_oblate_geom.py
```
