#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

#define CUDA_CHECK_NOEXCEPT(call)                                        \
  do {                                                                   \
    cudaError_t err__ = (call);                                          \
    if (err__ != cudaSuccess) {                                          \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
             cudaGetErrorString(err__));                                 \
    }                                                                    \
  } while (0)

// --------------------- device helpers (NO extern "C") ---------------------

template <typename IndexT>
__device__ __forceinline__ IndexT grid_stride_total(IndexT /*n*/) {
  // (n) is unused here; kept for clarity if you later use it
  return (IndexT)blockDim.x * (IndexT)gridDim.x;
}

// Atomic max for non-negative floats via CAS (portable)
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
  int* addr_as_i = reinterpret_cast<int*>(addr);
  int old = *addr_as_i, assumed;
  if (__int_as_float(old) >= val) return;
  do {
    assumed = old;
    if (__int_as_float(assumed) >= val) break;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
  } while (assumed != old);
}

// --------------------- kernels (NO extern "C") ----------------------------

__global__
void k_count_cells_per_gene(
  long long nnz,
  const int32_t* __restrict__ col_idx,  // length nnz
  const float*   __restrict__ data,     // length nnz
  float cutoff,
  int32_t*       __restrict__ out_counts  // length n_cols (must be zero-initialized)
){
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
       i < nnz;
       i += grid_stride_total(nnz)) {
    float v = data[i];
    if (v > cutoff) {
      atomicAdd(&out_counts[col_idx[i]], 1);
    }
  }
}

__global__
void k_ngene_per_cell(
  long long n_rows,
  const long long* __restrict__ indptr,   // length n_rows+1
  const int32_t*  __restrict__ indices,   // length nnz
  const float*    __restrict__ data,
  const uint8_t*  __restrict__ keep_gene, // length n_cols, 0/1
  int32_t*        __restrict__ out_counts // length n_rows
){
  long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (r >= n_rows) return;
  long long start = indptr[r];
  long long end   = indptr[r+1];
  int32_t c = 0;
  for (long long p = start; p < end; ++p) {
    int32_t col = indices[p];
    if (keep_gene[col] && data[p] > 0.0f) ++c;
  }
  out_counts[r] = c;
}

__global__
void k_max_per_gene_masked(
  long long n_rows,
  const long long* __restrict__ indptr,
  const int32_t*   __restrict__ indices,
  const float*     __restrict__ data,
  const uint8_t*   __restrict__ keep_cell, // length n_rows
  const uint8_t*   __restrict__ keep_gene, // length n_cols
  float*           __restrict__ gene_max   // length n_cols, zero-initialized
){
  long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (r >= n_rows) return;
  if (!keep_cell[r]) return;

  long long start = indptr[r];
  long long end   = indptr[r+1];
  for (long long p = start; p < end; ++p) {
    int32_t g = indices[p];
    if (keep_gene[g]) {
      float v = data[p];
      if (v > 0.0f) atomicMaxFloat(&gene_max[g], v);
    }
  }
}

// --------------------- C-linkage host wrappers (callable from C++) --------
// These wrappers compute grid/block and launch the kernels. Keep ONLY these
// in extern "C" so you can declare/call them from a normal .cpp file.

extern "C" {

void launch_count_cells_per_gene(
  long long nnz,
  const int32_t* col_idx,
  const float*   data,
  float          cutoff,
  int32_t*       out_counts
){
  int block = 256;
  int grid  = (int)((nnz + block - 1) / block);
  if (grid > 65535) grid = 65535;
  k_count_cells_per_gene<<<grid, block>>>(nnz, col_idx, data, cutoff, out_counts);
  CUDA_CHECK_NOEXCEPT(cudaGetLastError());
}

void launch_ngene_per_cell(
  long long n_rows,
  const long long* indptr,
  const int32_t*   indices,
  const float*     data,
  const uint8_t*   keep_gene,
  int32_t*         out_counts
){
  int block = 256;
  int grid  = (int)((n_rows + block - 1) / block);
  k_ngene_per_cell<<<grid, block>>>(n_rows, indptr, indices, data, keep_gene, out_counts);
  CUDA_CHECK_NOEXCEPT(cudaGetLastError());
}

void launch_max_per_gene_masked(
  long long n_rows,
  const long long* indptr,
  const int32_t*   indices,
  const float*     data,
  const uint8_t*   keep_cell,
  const uint8_t*   keep_gene,
  float*           gene_max
){
  int block = 256;
  int grid  = (int)((n_rows + block - 1) / block);
  k_max_per_gene_masked<<<grid, block>>>(n_rows, indptr, indices, data, keep_cell, keep_gene, gene_max);
  CUDA_CHECK_NOEXCEPT(cudaGetLastError());
}

} // extern "C"
