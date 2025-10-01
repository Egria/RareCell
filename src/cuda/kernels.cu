#include <cuda_runtime.h>

// Place CUDA kernels here later (e.g., per-gene histograms, bit packing, kNN).
__global__ void noop_kernel() {}

extern "C" void rarecell_cuda_warmup() {
  noop_kernel<<<1,1>>>();
  cudaDeviceSynchronize();
}