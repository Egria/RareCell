#pragma once
#include <mpi.h>
#include <cstdint>
#include <vector>
#include <string>
#include "io/h5ad_reader.hpp"
#include "filter/filter.hpp"

namespace rarecell {

    // Two-pass LOWESS detrending (rank 0) for ANY per-gene metric vector.
    //
    // Inputs:
    //   - comm: MPI communicator
    //   - R:    original RAW-count CSR (used to compute log2(max+0.1) on kept rows/genes)
    //   - outs: FilterOutputs (kept rows & genes mapping)
    //   - metric_rank0: per-gene values on rank 0 (size == #kept genes). Other ranks may pass {}.
    //   - outlier_q: quantile in (0,1) over positive residuals after pass-1 (default 0.75)
    //   - span: LOWESS fraction in (0,1] (default 0.90)
    // Returns:
    //   - residuals on rank 0 (size = #kept genes); other ranks return empty {}.
    //
    // Notes:
    //   • Uses RAW (non-CPM) data to compute x = log2(max + 0.1) exactly as in your Python.
    //   • The LOWESS is a 1-D local linear fit with tricube kernel (no robustness iterations).
    std::vector<float> lowess_twopass_detrend_metric_rank0(MPI_Comm comm,
        const H5ADReadResult& R,
        const FilterOutputs& outs,
        const std::vector<float>& metric_rank0,
        double outlier_q = 0.75,
        double span = 0.90);

    // Convenience wrappers if you prefer semantic names (both call the generic function)
    inline std::vector<float> lowess_twopass_detrend_gini_rank0(
        MPI_Comm comm, const H5ADReadResult& R, const FilterOutputs& outs,
        const std::vector<float>& gini_rank0, double outlier_q = 0.75, double span = 0.90) {
        return lowess_twopass_detrend_metric_rank0(comm, R, outs, gini_rank0, outlier_q, span);
    }

    inline std::vector<float> lowess_twopass_detrend_palma_rank0(
        MPI_Comm comm, const H5ADReadResult& R, const FilterOutputs& outs,
        const std::vector<float>& palma_rank0, double outlier_q = 0.75, double span = 0.90) {
        return lowess_twopass_detrend_metric_rank0(comm, R, outs, palma_rank0, outlier_q, span);
    }

} // namespace rarecell
