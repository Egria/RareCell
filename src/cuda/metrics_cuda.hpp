#pragma once
#include <mpi.h>
#include <cstdint>
#include <vector>
#include <string>
#include "io/h5ad_reader.hpp"
#include "filter/filter.hpp"
#include "metrics/fano.hpp"   // FanoResult
#include "metrics/gini.hpp"   // GiniResult
#include "metrics/palma.hpp"  // PalmaResult
#include "config/filter_config.hpp"

namespace rarecell {

    // GPU-accelerated Fano on the FILTERED local CSR (outs.X_local_filtered).
    // Exact: Fano = Var/Mean with zeros handled through global N; per-gene S1,S2 come from GPU.
    // Returns per-gene fano on ALL ranks (like your CPU Fano), plus N_total.
    FanoResult compute_fano_all_genes_cuda(MPI_Comm comm,
        const FilterOutputs& outs);

    // GPU-accelerated Gini on RAW integer counts (R.X_local restricted to kept rows/genes).
    // Exact via per-gene histograms with overflow capture (no approximation).
    // Reduces per-gene histograms to rank 0 and computes exact Gini there.
    // Rank 0 returns gini; other ranks return {}.
    GiniResult compute_gini_raw_int_cuda_rank0(MPI_Comm comm,
        const H5ADReadResult& R,
        const FilterOutputs& outs,
        int hist_L_cap = 128);

    // GPU-accelerated Palma on RAW integer counts (distributed owners).
    // Exact via per-gene histograms with overflow capture (no approximation).
    // Owners compute Palma using cfg.{palma_alpha, palma_upper, palma_lower, palma_winsor}.
    // Rank 0 receives the final Palma vector; other ranks return {}.
    PalmaResult compute_palma_raw_int_cuda_distributed_reduce_rank0(MPI_Comm comm,
        const H5ADReadResult& R,
        const FilterOutputs& outs,
        const FilterConfig& cfg,
        int hist_L_cap = 128);

    int compute_gamma_cutoff_selected_cuda(MPI_Comm comm,
        const H5ADReadResult& R,
        const FilterOutputs& outs,
        const std::vector<int>& selected_gene_idx,
        double gamma,               // e.g., 0.90
        int hist_L_cap = 128);

} // namespace rarecell
