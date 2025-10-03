#pragma once
#include <mpi.h>
#include <cstdint>
#include <vector>
#include <string>
#include "io/h5ad_reader.hpp"  // H5ADReadResult
#include "filter/filter.hpp"   // FilterOutputs

namespace rarecell {

    struct GiniResult {
        // On rank 0: Gini per kept gene (same order as outs.gene_names_filtered)
        // On other ranks: empty vector
        std::vector<float> gini;
        // Global number of kept cells (after filtering)
        int64_t N_total = 0;
    };

    // Compute exact Gini for all kept genes using RAW INTEGER COUNTS.
    // Implementation note: builds per-gene integer histograms locally,
    // reduces them to rank 0, and computes exact Gini there.
    // Only rank 0 returns the gini vector; other ranks return empty.
    GiniResult compute_gini_raw_int_rank0(MPI_Comm comm,
        const H5ADReadResult& R,
        const FilterOutputs& outs);

    // Utility: return top-k (name, value) pairs, sorted descending.
    std::vector<std::pair<std::string, float>>
        top_k_gini(const std::vector<std::string>& gene_names,
            const std::vector<float>& gini,
            std::size_t k);

} // namespace rarecell