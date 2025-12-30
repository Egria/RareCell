#include "cuda_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>
#include "cuda/metrics_cuda.hpp"

#define CUDA_CHECK(expr) do {                                     \
  cudaError_t _e = (expr);                                        \
  if (_e != cudaSuccess) {                                        \
    char msg[512];                                                \
    snprintf(msg, sizeof(msg), "CUDA error: %s at %s:%d",         \
             cudaGetErrorString(_e), __FILE__, __LINE__);         \
    throw std::runtime_error(msg);                                \
  }                                                               \
} while(0)

namespace rarecell {

// ----------------------------- GPU kernels -----------------------------

// FANO: sum and sum of squares per gene over FILTERED local CSR
__global__ void k_fano_sums(const int32_t* __restrict__ d_indices,
                            const float*   __restrict__ d_data,
                            int64_t nnz,
                            double* d_S1, double* d_S2)
{
  for (int64_t p = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       p < nnz;
       p += (int64_t)blockDim.x * gridDim.x) {
    int32_t g = d_indices[p];
    float v = d_data[p];
    if (v != 0.f) {
      double vd = (double)v;
      atomicAdd(&d_S1[g], vd);
      atomicAdd(&d_S2[g], vd*vd);
    }
  }
}

// RAW: pass A — scalars & overflow count, restricted to kept rows & kept genes
__global__ void k_raw_passA_scalars_overflow(const int64_t* __restrict__ d_indptr,
                                             const int32_t* __restrict__ d_indices,
                                             const float*   __restrict__ d_data,
                                             const int64_t* __restrict__ d_kept_rows,
                                             int64_t n_kept_rows,
                                             const int32_t* __restrict__ d_remap, // old->new col (>=0 else skip)
                                             int hist_L_cap,
                                             double* d_S, unsigned long long* d_nnz, int32_t* d_vmax,
                                             unsigned long long* d_overflow_counter)
{
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       i < n_kept_rows;
       i += (int64_t)blockDim.x * gridDim.x) {
    int64_t r = d_kept_rows[i];
    int64_t s = d_indptr[r];
    int64_t e = d_indptr[r+1];
    for (int64_t p = s; p < e; ++p) {
      int32_t oldc = d_indices[p];
      int32_t g = d_remap[oldc];
      if (g < 0) continue;
      int v = (int)llround((double)d_data[p]);
      if (v <= 0) continue;
      atomicAdd(&d_S[g], (double)v);
      atomicAdd(&d_nnz[g], 1ULL);
      atomicMax(&d_vmax[g], (int32_t)v);
      if (v >= hist_L_cap) atomicAdd(d_overflow_counter, 1ULL);
    }
  }
}

// RAW: pass B — fill low histogram and materialize overflow (gene, value) pairs
__global__ void k_raw_passB_hist_overflow(const int64_t* __restrict__ d_indptr,
                                          const int32_t* __restrict__ d_indices,
                                          const float*   __restrict__ d_data,
                                          const int64_t* __restrict__ d_kept_rows,
                                          int64_t n_kept_rows,
                                          const int32_t* __restrict__ d_remap,
                                          int hist_L_cap,
                                          unsigned long long* d_hist_low, // [G * L_cap]
                                          int32_t* d_overflow_gene, int32_t* d_overflow_val,
                                          unsigned long long* d_overflow_write)
{
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       i < n_kept_rows;
       i += (int64_t)blockDim.x * gridDim.x) {
    int64_t r = d_kept_rows[i];
    int64_t s = d_indptr[r];
    int64_t e = d_indptr[r+1];
    for (int64_t p = s; p < e; ++p) {
      int32_t oldc = d_indices[p];
      int32_t g = d_remap[oldc];
      if (g < 0) continue;
      int v = (int)llround((double)d_data[p]);
      if (v <= 0) continue;
      if (v < hist_L_cap) {
        atomicAdd(&d_hist_low[(int64_t)g * hist_L_cap + v], 1ULL);
      } else {
        unsigned long long idx = atomicAdd(d_overflow_write, 1ULL);
        d_overflow_gene[idx] = g;
        d_overflow_val[idx]  = v;
      }
    }
  }
}

// ----------------------------- helpers -----------------------------

static int pick_device_for_rank() {
  int ndev = 0;
  cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e != cudaSuccess || ndev <= 0) {
    // no device — let caller handle fallback if desired
    throw std::runtime_error("No CUDA device available");
  }
  int dev = 0;
  // simple round-robin by process env if needed; here choose device 0
  // You can change to rank%ndev if you pass rank into these functions.
  return dev;
}

// Group/compress overflow (gene,value) pairs into per-gene runs
struct OverflowCompressed {
  // For each gene g:
  //   start[g] = starting index into vals/cnts for gene g (or -1 if none)
  //   len[g]   = number of unique values for gene g (0 if none)
  std::vector<int64_t> start;
  std::vector<int32_t> len;
  std::vector<int32_t> vals;     // concatenated unique values per gene
  std::vector<uint64_t> cnts;    // matching counts
};

static OverflowCompressed compress_overflow_pairs(const std::vector<int32_t>& gene,
                                                  const std::vector<int32_t>& val,
                                                  int64_t G)
{
  OverflowCompressed oc;
  const size_t O = gene.size();
  oc.start.assign((size_t)G, -1);
  oc.len.assign((size_t)G, 0);
  oc.vals.clear(); oc.cnts.clear();
  if (O == 0) return oc;

  std::vector<size_t> idx(O);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](size_t a, size_t b){
              if (gene[a] != gene[b]) return gene[a] < gene[b];
              return val[a] < val[b];
            });

  int32_t curG = -1, curV = -1;
  uint64_t run = 0;
  int64_t base = 0;

  for (size_t ii = 0; ii < O; ++ii) {
    int32_t g = gene[idx[ii]];
    int32_t v = val[idx[ii]];
    if (g != curG || v != curV) {
      // flush previous
      if (run > 0) {
        oc.vals.push_back(curV);
        oc.cnts.push_back(run);
        oc.len[(size_t)curG] += 1;
      }
      // new run
      if (g != curG) {
        curG = g;
        curV = v;
        run  = 1;
        base = (int64_t)oc.vals.size();
        if (oc.start[(size_t)g] == -1) oc.start[(size_t)g] = base;
        else oc.start[(size_t)g] = base; // continue; subsequent entries will extend
      } else {
        curV = v;
        run  = 1;
      }
    } else {
      run += 1;
    }
  }
  if (run > 0) {
    oc.vals.push_back(curV);
    oc.cnts.push_back(run);
    oc.len[(size_t)curG] += 1;
  }
  return oc;
}

// Fill a dense count vector for a single gene using low histogram + compressed overflow
static void materialize_dense_counts_for_gene(int32_t g,
                                              int L, int hist_L_cap,
                                              const std::vector<unsigned long long>& hist_low, // G*hist_L_cap
                                              const OverflowCompressed& oc,
                                              std::vector<long long>& out_cnt) // length L, zeros on entry
{
  const int64_t G = (int64_t)hist_low.size() / hist_L_cap;
  (void)G;

  // low part
  const int upto = std::min(hist_L_cap, L);
  const unsigned long long* row = &hist_low[(int64_t)g * hist_L_cap];
  for (int v = 1; v < upto; ++v) { // skip v=0 (zeros handled analytically)
    out_cnt[(size_t)v] += (long long)row[v];
  }

  // overflow part
  int64_t st = oc.start[(size_t)g];
  int32_t ln = oc.len[(size_t)g];
  if (st >= 0 && ln > 0) {
    for (int32_t k = 0; k < ln; ++k) {
      int32_t v = oc.vals[(size_t)(st + k)];
      if (v < L) out_cnt[(size_t)v] += (long long)oc.cnts[(size_t)(st + k)];
      // else impossible: L = vmax+1 >= v+1 by construction (v <= vmax)
    }
  }
}

// ------------------------------- Fano (CUDA) -------------------------------

FanoResult compute_fano_all_genes_cuda(MPI_Comm comm, const FilterOutputs& outs)
{
  FanoResult R;

  int rank=0, world=1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world);

  const auto& X = outs.X_local_filtered;
  const int64_t n_rows_local = X.row1 - X.row0;
  const int64_t nnz          = (int64_t)X.data.size();
  const int64_t n_genes      = X.n_cols;

  // Global N = total kept cells
  long long N_local = (long long)n_rows_local, N_total=0;
  MPI_Allreduce(&N_local, &N_total, 1, MPI_LONG_LONG, MPI_SUM, comm);
  R.N_total = (int64_t)N_total;
  if (n_genes == 0 || N_total == 0) { R.fano.clear(); return R; }

  // Copy filtered CSR to device
  int dev = pick_device_for_rank();
  CUDA_CHECK(cudaSetDevice(dev));

  int32_t* d_idx = nullptr; float* d_val = nullptr;
  double *d_S1 = nullptr, *d_S2 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_idx, nnz * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_val, nnz * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_idx, X.indices.data(), nnz*sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_val, X.data.data(),    nnz*sizeof(float),   cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_S1, n_genes * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_S2, n_genes * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_S1, 0, n_genes*sizeof(double)));
  CUDA_CHECK(cudaMemset(d_S2, 0, n_genes*sizeof(double)));

  const int BLK = 256;
  int grid = (int)std::min<int64_t>( (nnz + BLK - 1) / BLK, 65535 );
  k_fano_sums<<<grid, BLK>>>(d_idx, d_val, nnz, d_S1, d_S2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back S1,S2
  std::vector<double> S1_local((size_t)n_genes), S2_local((size_t)n_genes);
  CUDA_CHECK(cudaMemcpy(S1_local.data(), d_S1, n_genes*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(S2_local.data(), d_S2, n_genes*sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_idx)); CUDA_CHECK(cudaFree(d_val));
  CUDA_CHECK(cudaFree(d_S1));  CUDA_CHECK(cudaFree(d_S2));

  // Allreduce to global
  std::vector<double> S1_gl((size_t)n_genes, 0.0), S2_gl((size_t)n_genes, 0.0);
  MPI_Allreduce(S1_local.data(), S1_gl.data(), (int)n_genes, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(S2_local.data(), S2_gl.data(), (int)n_genes, MPI_DOUBLE, MPI_SUM, comm);

  // Compute Fano
  R.fano.resize((size_t)n_genes);
  const double N = (double)R.N_total;
  for (int64_t g = 0; g < n_genes; ++g) {
    const double mu  = S1_gl[(size_t)g] / N;
    if (mu <= 0.0) R.fano[(size_t)g] = 0.0f;
    else {
      double ex2 = S2_gl[(size_t)g] / N;
      double var = ex2 - mu*mu;
      if (var < 0.0) var = 0.0;
      R.fano[(size_t)g] = (float)(var / mu);
    }
  }
  return R;
}

// -------- shared RAW histogram builder on GPU (used by Gini & Palma) --------

struct RawGPUHistLocal {
  int G = 0;               // kept genes
  int Lcap = 128;          // low histogram width
  std::vector<unsigned long long> hist_low; // size G*Lcap
  std::vector<double> S;                // per-gene sum (double)
  std::vector<long long> nnz;          // per-gene nnz
  std::vector<int32_t> vmax;           // per-gene vmax
  std::vector<int32_t> ov_gene;        // overflow pairs
  std::vector<int32_t> ov_val;         // overflow pairs
};

static RawGPUHistLocal build_raw_hist_local_on_gpu(const H5ADReadResult& R,
                                                   const FilterOutputs& outs,
                                                   int hist_L_cap,
                                                   long long& N_total_out) // global N
{
  RawGPUHistLocal H;
  H.G = (int)outs.gene_names_filtered.size();
  H.Lcap = hist_L_cap;
  if (H.G == 0) { N_total_out = 0; return H; }

  // Global N: kept cells
  long long N_local = (long long)outs.kept_local_rows.size();
  long long N_total = 0;
  MPI_Allreduce(&N_local, &N_total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  N_total_out = N_total;

  const auto& X = R.X_local; // RAW CSR
  const int64_t n_kept_rows = (int64_t)outs.kept_local_rows.size();

  // Device select & alloc / copy
  int dev = pick_device_for_rank();
  CUDA_CHECK(cudaSetDevice(dev));

  // CSR arrays
  int64_t* d_indptr=nullptr; int32_t* d_indices=nullptr; float* d_data=nullptr;
  const int64_t n_rows_local_csr = (int64_t)X.indptr.size() - 1;
  CUDA_CHECK(cudaMalloc(&d_indptr, (n_rows_local_csr + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_indices, (int64_t)X.indices.size() * (int64_t)sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_data,    (int64_t)X.data.size()    * (int64_t)sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_indptr, X.indptr.data(), (n_rows_local_csr + 1) * sizeof(int64_t),
                      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_indices, X.indices.data(), X.indices.size()*sizeof(int32_t),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data,    X.data.data(),    X.data.size()*sizeof(float),         cudaMemcpyHostToDevice));

  // kept rows
  int64_t* d_kept_rows=nullptr;
  CUDA_CHECK(cudaMalloc(&d_kept_rows, n_kept_rows * (int64_t)sizeof(int64_t)));
  CUDA_CHECK(cudaMemcpy(d_kept_rows, outs.kept_local_rows.data(),
                        n_kept_rows*sizeof(int64_t), cudaMemcpyHostToDevice));

  // remap
  int32_t* d_remap=nullptr;
  CUDA_CHECK(cudaMalloc(&d_remap, outs.col_remap_final.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(d_remap, outs.col_remap_final.data(),
                        outs.col_remap_final.size()*sizeof(int32_t), cudaMemcpyHostToDevice));

  // outputs on device
  double* d_S=nullptr; unsigned long long* d_nnz=nullptr; int32_t* d_vmax=nullptr;
  CUDA_CHECK(cudaMalloc(&d_S,    H.G * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_nnz,  H.G * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_vmax, H.G * sizeof(int32_t)));
  CUDA_CHECK(cudaMemset(d_S,    0, H.G*sizeof(double)));
  CUDA_CHECK(cudaMemset(d_nnz,  0, H.G*sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_vmax, 0, H.G*sizeof(int32_t)));

  unsigned long long* d_overflow_counter=nullptr;
  CUDA_CHECK(cudaMalloc(&d_overflow_counter, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_overflow_counter, 0, sizeof(unsigned long long)));

  // pass A
  const int BLK=256;
  int grid = (int)std::min<int64_t>( (n_kept_rows + BLK - 1) / BLK, 65535 );
  k_raw_passA_scalars_overflow<<<grid, BLK>>>(
      d_indptr, d_indices, d_data, d_kept_rows, n_kept_rows,
      d_remap, H.Lcap, d_S, d_nnz, d_vmax, d_overflow_counter);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // fetch overflow count
  unsigned long long overflow_count = 0ULL;
  CUDA_CHECK(cudaMemcpy(&overflow_count, d_overflow_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

  // low hist + overflow arrays
  unsigned long long* d_hist_low=nullptr;
  CUDA_CHECK(cudaMalloc(&d_hist_low, (int64_t)H.G * H.Lcap * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_hist_low, 0, (int64_t)H.G * H.Lcap * sizeof(unsigned long long)));

  int32_t* d_ov_gene=nullptr; int32_t* d_ov_val=nullptr; unsigned long long* d_ov_write=nullptr;
  if (overflow_count > 0) {
    CUDA_CHECK(cudaMalloc(&d_ov_gene, overflow_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ov_val,  overflow_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ov_write, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_ov_write, 0, sizeof(unsigned long long)));
  }

  // pass B
  k_raw_passB_hist_overflow<<<grid, BLK>>>(
      d_indptr, d_indices, d_data, d_kept_rows, n_kept_rows,
      d_remap, H.Lcap, d_hist_low, d_ov_gene, d_ov_val, d_ov_write);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // copy results back
  H.hist_low.resize((size_t)H.G * H.Lcap);
  H.S.resize((size_t)H.G);
  H.nnz.resize((size_t)H.G);
  H.vmax.resize((size_t)H.G);

  CUDA_CHECK(cudaMemcpy(H.hist_low.data(), d_hist_low, (int64_t)H.G*H.Lcap*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(H.S.data(),        d_S,        H.G*sizeof(double),                              cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(H.nnz.data(),      d_nnz,      H.G*sizeof(unsigned long long),                  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(H.vmax.data(),     d_vmax,     H.G*sizeof(int32_t),                             cudaMemcpyDeviceToHost));

  if (overflow_count > 0) {
    H.ov_gene.resize((size_t)overflow_count);
    H.ov_val.resize((size_t)overflow_count);
    CUDA_CHECK(cudaMemcpy(H.ov_gene.data(), d_ov_gene, overflow_count*sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(H.ov_val.data(),  d_ov_val,  overflow_count*sizeof(int32_t), cudaMemcpyDeviceToHost));
  }

  // free
  if (d_ov_write) CUDA_CHECK(cudaFree(d_ov_write));
  if (d_ov_val)   CUDA_CHECK(cudaFree(d_ov_val));
  if (d_ov_gene)  CUDA_CHECK(cudaFree(d_ov_gene));
  CUDA_CHECK(cudaFree(d_hist_low));
  CUDA_CHECK(cudaFree(d_overflow_counter));
  CUDA_CHECK(cudaFree(d_vmax));
  CUDA_CHECK(cudaFree(d_nnz));
  CUDA_CHECK(cudaFree(d_S));
  CUDA_CHECK(cudaFree(d_remap));
  CUDA_CHECK(cudaFree(d_kept_rows));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_indices));
  CUDA_CHECK(cudaFree(d_indptr));

  return H;
}

// --------------------------------- Gini (CUDA) ---------------------------------

GiniResult compute_gini_raw_int_cuda_rank0(MPI_Comm comm,
                                           const H5ADReadResult& R,
                                           const FilterOutputs& outs,
                                           int hist_L_cap)
{
  GiniResult res;
  int rank=0; MPI_Comm_rank(comm, &rank);

  long long N_total=0;
  RawGPUHistLocal H = build_raw_hist_local_on_gpu(R, outs, hist_L_cap, N_total);
  res.N_total = (int64_t)N_total;

  const int64_t G = (int64_t)H.G;
  if (G == 0 || N_total == 0) {
    if (rank == 0) res.gini.clear();
    return res;
  }

  // Global scalars per gene (nnz, S, vmax)
  std::vector<long long> nnz_gl((size_t)G, 0), nnz_l = H.nnz;
  std::vector<double>    S_gl((size_t)G, 0.0),       S_l  = H.S;
  std::vector<int32_t>   vmax_gl((size_t)G, 0),      vmax_l= H.vmax;

  MPI_Allreduce(nnz_l.data(), nnz_gl.data(), (int)G, MPI_LONG_LONG, MPI_SUM, comm);
  MPI_Allreduce(S_l.data(),   S_gl.data(),   (int)G, MPI_DOUBLE,    MPI_SUM, comm);
  MPI_Allreduce(vmax_l.data(),vmax_gl.data(),(int)G, MPI_INT,       MPI_MAX, comm);

  // Compress overflow into (gene -> list of (v,count))
  OverflowCompressed oc = compress_overflow_pairs(H.ov_gene, H.ov_val, G);

  if (rank == 0) res.gini.assign((size_t)G, 0.0f);

  // Reduce per gene histogram to rank 0, compute exact Gini
  for (int64_t g = 0; g < G; ++g) {
    const long long N = N_total;
    const double    S = S_gl[(size_t)g];
    if (N <= 0 || S <= 0.0) {
      if (rank == 0) res.gini[(size_t)g] = 0.0f;
      continue;
    }

    const int L = vmax_gl[(size_t)g] + 1;
    if (L <= 1) { // only zeros
      if (rank == 0) res.gini[(size_t)g] = 0.0f;
      continue;
    }

    // Build local dense counts [0..L-1] (skip 0)
    std::vector<long long> cnt_local((size_t)L, 0);
    materialize_dense_counts_for_gene((int32_t)g, L, H.Lcap, H.hist_low, oc, cnt_local);

    // Reduce to rank 0
    std::vector<long long> cnt_glob;
    if (rank == 0) cnt_glob.resize((size_t)L, 0);
    MPI_Reduce(cnt_local.data(), rank==0 ? cnt_glob.data() : nullptr,
               L, MPI_LONG_LONG, MPI_SUM, 0, comm);

    if (rank != 0) continue;

    const long long nnz = nnz_gl[(size_t)g];
    const long long zeros = N - nnz;

    long long pos_before = zeros;
    long double weighted = 0.0L;
    for (int v = 1; v < L; ++v) {
      long long c = cnt_glob[(size_t)v];
      if (c == 0) continue;
      long long sum_i = pos_before * c + (c * (c + 1)) / 2;
      long double term = (long double)v * ( (long double)2 * (long double)sum_i
                           - ((long double)N + 1.0L) * (long double)c );
      weighted += term;
      pos_before += c;
    }
    long double denom = (long double)N * (long double)S;
    long double Gini = (denom > 0.0L) ? (weighted / denom) : 0.0L;
    if (Gini < 0.0L) Gini = 0.0L;
    if (Gini > 1.0L) Gini = 1.0L;

    res.gini[(size_t)g] = (float)Gini;
  }

  return res;
}

// --------------------------------- Palma (CUDA) ---------------------------------

PalmaResult compute_palma_raw_int_cuda_distributed_reduce_rank0(MPI_Comm comm,
                                                                const H5ADReadResult& R,
                                                                const FilterOutputs& outs,
                                                                const FilterConfig& cfg,
                                                                int hist_L_cap)
{
  PalmaResult res;
  int rank=0, world=1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world);

  long long N_total=0;
  RawGPUHistLocal H = build_raw_hist_local_on_gpu(R, outs, hist_L_cap, N_total);
  res.N_total = (int64_t)N_total;

  const int64_t G = (int64_t)H.G;
  if (G == 0 || N_total == 0) {
    if (rank == 0) res.palma.clear();
    return res;
  }

  // Global scalars
  std::vector<long long> nnz_gl((size_t)G, 0), nnz_l = H.nnz;
  std::vector<double>    S_gl((size_t)G, 0.0),       S_l  = H.S;
  std::vector<int32_t>   vmax_gl((size_t)G, 0),      vmax_l= H.vmax;

  MPI_Allreduce(nnz_l.data(), nnz_gl.data(), (int)G, MPI_LONG_LONG, MPI_SUM, comm);
  MPI_Allreduce(S_l.data(),   S_gl.data(),   (int)G, MPI_DOUBLE,    MPI_SUM, comm);
  MPI_Allreduce(vmax_l.data(),vmax_gl.data(),(int)G, MPI_INT,       MPI_MAX, comm);

  // overflow compression
  OverflowCompressed oc = compress_overflow_pairs(H.ov_gene, H.ov_val, G);

  const long long need_bottom = (long long)std::ceil((double)cfg.palma_lower * (double)N_total);
  const long long need_top    = (long long)std::ceil((double)cfg.palma_upper * (double)N_total);
  const double alpha = (double)cfg.palma_alpha;
  const double winsor_p = (double)cfg.palma_winsor;

  std::vector<float> palma_local((size_t)G, 0.0f);

  auto winsorize_upper = [&](long long zeros, const long long nnz,
                             std::vector<long long>& cnt, int L, double& S_used) {
    if (winsor_p <= 0.0) return;
    long long q_high = (long long)std::ceil((1.0 - winsor_p) * (double)N_total);
    long long cum = zeros;
    int v_high = 0;
    for (int v = 1; v < L; ++v) {
      cum += cnt[(size_t)v];
      if (cum >= q_high) { v_high = v; break; }
    }
    if (v_high == 0 && q_high <= zeros) {
      for (int v = 1; v < L; ++v) cnt[(size_t)v] = 0;
      S_used = 0.0;
    } else if (v_high > 0) {
      long long move = 0;
      double delta_sum = 0.0;
      for (int v = v_high+1; v < L; ++v) {
        long long c = cnt[(size_t)v];
        if (!c) continue;
        move += c;
        delta_sum += (double)c * (double)(v_high - v);
        cnt[(size_t)v] = 0;
      }
      if (move > 0) {
        cnt[(size_t)v_high] += move;
        S_used += delta_sum;
      }
    }
  };

  auto sum_bottom = [&](long long zeros, const std::vector<long long>& cnt, int L)->double {
    if (need_bottom <= 0) return 0.0;
    if (zeros >= need_bottom) return 0.0;
    long long rem = need_bottom - zeros;
    double acc = 0.0;
    for (int v = 1; v < L && rem > 0; ++v) {
      long long c = cnt[(size_t)v];
      if (!c) continue;
      if (c <= rem) { acc += (double)c * (double)v; rem -= c; }
      else          { acc += (double)rem * (double)v; rem = 0; }
    }
    return acc;
  };

  auto sum_top = [&](const std::vector<long long>& cnt, int L, long long nnz, double S_used)->double {
    if (need_top <= 0) return 0.0;
    if (nnz <= need_top) return S_used;
    long long rem = need_top;
    double acc = 0.0;
    for (int v = L-1; v >= 1 && rem > 0; --v) {
      long long c = cnt[(size_t)v];
      if (!c) continue;
      if (c <= rem) { acc += (double)c * (double)v; rem -= c; }
      else          { acc += (double)rem * (double)v; rem = 0; }
    }
    return acc;
  };

  // Each gene reduces to its owner, owner computes Palma, then final reduce (SUM) to rank 0
  for (int64_t g = 0; g < G; ++g) {
    const int owner = (int)(g % world);
    const int L = vmax_gl[(size_t)g] + 1;

    // Materialize dense local counts
    std::vector<long long> cnt_local((size_t)L, 0);
    materialize_dense_counts_for_gene((int32_t)g, L, H.Lcap, H.hist_low, oc, cnt_local);

    // Reduce to owner
    std::vector<long long> cnt_owner;
    if (rank == owner) cnt_owner.resize((size_t)L, 0);
    MPI_Reduce(cnt_local.data(),
               rank==owner ? cnt_owner.data() : nullptr,
               L, MPI_LONG_LONG, MPI_SUM, owner, comm);

    if (rank != owner) continue;

    const double S_raw = S_gl[(size_t)g];
    const long long nnz = nnz_gl[(size_t)g];
    const long long zeros = N_total - nnz;

    double Palma = 1.0;
    if (S_raw > 0.0 && N_total > 0) {
      double S_used = S_raw;
      winsorize_upper(zeros, nnz, cnt_owner, L, S_used);
      const double S_lower = sum_bottom(zeros, cnt_owner, L);
      const double S_upper = sum_top(cnt_owner, L, nnz, S_used);
      if (S_used > 0.0) {
        const double num = (S_upper / S_used) + alpha;
        const double den = (S_lower / S_used) + alpha;
        Palma = (den > 0.0) ? (num / den) : 1.0;
      }
      if (!std::isfinite(Palma)) Palma = 1.0;
    }
    palma_local[(size_t)g] = (float)Palma;
  }

  // Final gather to rank 0
  std::vector<float> palma_rank0;
  if (rank == 0) palma_rank0.resize((size_t)G, 0.0f);
  MPI_Reduce(palma_local.data(),
             rank==0 ? palma_rank0.data() : nullptr,
             (int)G, MPI_FLOAT, MPI_SUM, 0, comm);

  if (rank == 0) res.palma = std::move(palma_rank0);
  return res;
}

int rarecell::compute_gamma_cutoff_selected_cuda(MPI_Comm comm,
                                                 const H5ADReadResult& R,
                                                 const FilterOutputs& outs,
                                                 const std::vector<int>& selected_gene_idx,
                                                 double gamma,
                                                 int hist_L_cap)
{
  int rank=0; MPI_Comm_rank(comm, &rank);

  const int64_t K = (int64_t)selected_gene_idx.size();
  if (K == 0) {
    int cutoff = 1; // harmless default
    MPI_Bcast(&cutoff, 1, MPI_INT, 0, comm);
    return cutoff;
  }

  // 1) Build local raw histograms on GPU for *all kept genes* (fast single scan),
  //    then we'll only reduce hist for the selected ones.
  long long N_total = 0;
  RawGPUHistLocal H = build_raw_hist_local_on_gpu(R, outs, hist_L_cap, N_total);

  const int64_t G = (int64_t)H.G;

  // 2) Global scalars (S, nnz, vmax) across ranks for all kept genes
  std::vector<long long> nnz_gl((size_t)G, 0), nnz_l = H.nnz;
  std::vector<double>    S_gl((size_t)G, 0.0),        S_l  = H.S;
  std::vector<int32_t>   vmax_gl((size_t)G, 0),       vmax_l= H.vmax;

  MPI_Allreduce(nnz_l.data(), nnz_gl.data(), (int)G, MPI_LONG_LONG, MPI_SUM, comm);
  MPI_Allreduce(S_l.data(),   S_gl.data(),   (int)G, MPI_DOUBLE,    MPI_SUM, comm);
  MPI_Allreduce(vmax_l.data(),vmax_gl.data(),(int)G, MPI_INT,       MPI_MAX, comm);

  // Overflow compression (local) to let us materialize dense rows easily
  OverflowCompressed oc = compress_overflow_pairs(H.ov_gene, H.ov_val, G);

  // 3) For each selected gene (in the given order), reduce its dense histogram to rank 0,
  //    then compute bc_high/low -> bc_med.
  std::vector<double> bc_med; bc_med.reserve((size_t)K);

  for (int64_t t = 0; t < K; ++t) {
    const int g = selected_gene_idx[(size_t)t];
    if (g < 0 || g >= H.G) {
      // out of range (should not happen)
      if (rank == 0) bc_med.push_back(0.0);
      continue;
    }

    const int L = vmax_gl[(size_t)g] + 1;
    // Build local dense counts [0..L-1] for this gene (skip 0, handled via zeros)
    std::vector<long long> cnt_local((size_t)L, 0);
    materialize_dense_counts_for_gene(g, L, H.Lcap, H.hist_low, oc, cnt_local);

    std::vector<long long> cnt_glob;
    if (rank == 0) cnt_glob.resize((size_t)L, 0);

    MPI_Reduce(cnt_local.data(),
               rank==0 ? cnt_glob.data() : nullptr,
               L, MPI_LONG_LONG, MPI_SUM, 0, comm);

    if (rank != 0) continue;

    // Replicate your Python logic exactly
    const long long nnz  = nnz_gl[(size_t)g];
    const long long zeros= N_total - nnz;
    const double    S    = S_gl[(size_t)g]; // denom

    // Build c (values) and f (frequencies), including zero
    std::vector<long long> c;
    std::vector<long long> f;
    c.reserve((size_t)L); f.reserve((size_t)L);
    c.push_back(0); f.push_back(zeros);
    long long denom_check = 0;
    for (int v = 1; v < L; ++v) {
      long long cnt = cnt_glob[(size_t)v];
      if (cnt > 0) {
        c.push_back((long long)v);
        f.push_back(cnt);
        denom_check += (long long)v * cnt;
      }
    }

    double denom = S; // exact global sum of counts
    std::vector<double> csum(c.size(), 0.0);
    if (denom <= 0.0) {
      // warnings.warn in python; here just set csum all zeros
      std::fill(csum.begin(), csum.end(), 0.0);
    } else {
      // tail_ge = cumsum((c*f)[::-1])[::-1]
      const size_t M = c.size();
      std::vector<long double> contrib(M, 0.0L), tail(M, 0.0L);
      for (size_t i = 0; i < M; ++i) contrib[i] = (long double)c[i] * (long double)f[i];
      long double acc = 0.0L;
      for (size_t k = 0; k < M; ++k) {
        size_t j = M - 1 - k; // reverse index
        acc += contrib[j];
        tail[j] = acc;
      }
      for (size_t i = 0; i < M; ++i) csum[i] = (double)(tail[i] / denom);
    }

    // Descending order by c (since c is ascending, reverse is "descending")
    const size_t M = c.size();
    double hi = 0.0, lo = 0.0;
    if (M == 0) {
      hi = lo = 0.0;
    } else {
      // Find first index in descending where csum_desc > gamma
      size_t n_idx = 0;
      bool found = false;
      for (size_t k = 0; k < M; ++k) {
        size_t j = M - 1 - k; // position in ascending arrays
        double csum_desc_k = csum[j];
        if (csum_desc_k > gamma) { n_idx = k; found = true; break; }
      }
      if (!found) n_idx = 0;
      // Clamp per your Python
      n_idx = std::max<size_t>(2, n_idx);
      if (n_idx >= M - 1) n_idx = (M >= 2 ? M - 2 : 0);

      // c_desc[n_idx] = c[M-1 - n_idx]
      size_t j_hi = M - 1 - n_idx;
      size_t j_lo = (M - 1 - (n_idx + 1));
      hi = (double)c[j_hi];
      lo = (double)c[(M >= 2 ? j_lo : j_hi)];
    }

    bc_med.push_back(0.5 * (hi + lo));
  }

  int cutoff = 1;
  if (rank == 0) {
    // cutoff = floor(mean(bc_med[:top_n_gene])), where top_n_gene = max(int(len*0.10), 10)
    size_t top_n_gene = (size_t)std::max( (int)std::floor((double)bc_med.size() * 0.10), 10 );
    if (top_n_gene > bc_med.size()) top_n_gene = bc_med.size();
    double sum = 0.0;
    for (size_t i = 0; i < top_n_gene; ++i) sum += bc_med[i];
    double mean = (top_n_gene > 0 ? sum / (double)top_n_gene : 0.0);
    cutoff = (int)std::floor(mean);
    if (cutoff < 1) cutoff = 1; // keep it positive
  }

  MPI_Bcast(&cutoff, 1, MPI_INT, 0, comm);
  return cutoff;
}

} // namespace rarecell
