#pragma once
#include <mpi.h>
#include <cstdint>
#include <vector>
#include <string>
#include "io/h5ad_reader.hpp"
#include "filter/filter.hpp"
#include "config/filter_config.hpp"

namespace rarecell {

    struct PalmaResult {
        // On rank 0: Palma_alpha per kept gene (same order as outs.gene_names_filtered)
        // On other ranks: empty.
        std::vector<float> palma;
        int64_t N_total = 0;
    };

    PalmaResult compute_palma_raw_int_distributed_batched_rank0(MPI_Comm comm,
        const H5ADReadResult& R,
        const FilterOutputs& outs,
        const FilterConfig& cfg);

    std::vector<std::pair<std::string, float>>
        top_k_palma(const std::vector<std::string>& gene_names,
            const std::vector<float>& palma,
            std::size_t k);

} // namespace rarecell
