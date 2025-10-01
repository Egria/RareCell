#pragma once
#include <mpi.h>
#include <cstdint>
#include <vector>
#include <string>
#include "common/types.hpp"
#include "filter/filter.hpp"  // for FilterOutputs

namespace rarecell {

    struct FanoResult {
        // Fano per kept gene (same order as outs.gene_names_filtered)
        std::vector<float> fano;
        // Global number of kept cells (after filtering)
        int64_t N_total = 0;
    };

    // Compute exact Fano factor for all filtered genes.
    // Uses: E[X] = S1/N, Var[X] = E[X^2] - E[X]^2; Fano = Var / E[X] (0 if mean==0).
    FanoResult compute_fano_all_genes(MPI_Comm comm, const FilterOutputs& outs);

    // Return top-k (name, fano) pairs, sorted descending by fano.
    std::vector<std::pair<std::string, float>>
        top_k_fano(const std::vector<std::string>& gene_names,
            const std::vector<float>& fano, std::size_t k);

} // namespace rarecell