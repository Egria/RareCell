#pragma once
#include <mpi.h>
#include <vector>
#include <string>
#include "filter/filter.hpp"
#include "config/filter_config.hpp"

namespace rarecell {

    // Output: indices are positions in outs.gene_names_filtered
    struct FeaturePanels {
        std::vector<int> gini_idx;   // size = cfg.gini_nfeatures  (after clamp)
        std::vector<int> fano_idx;   // size = cfg.fano_nfeatures
        std::vector<int> palma_idx;  // size = cfg.palma_nfeatures

        std::vector<std::string> gini_genes;   // names (filled on all ranks)
        std::vector<std::string> fano_genes;
        std::vector<std::string> palma_genes;
    };

    // Select top?K by: gini_detrended (rank0), fano (available on all ranks),
    // palma_detrended (rank0). Rank 0 sorts and writes files, then broadcasts
    // chosen indices to all ranks; names are populated everywhere.
    FeaturePanels select_feature_panels_rank0_bcast(
        MPI_Comm comm,
        const FilterOutputs& outs,
        const std::vector<float>& gini_detrended_rank0,
        const std::vector<float>& fano_allranks,
        const std::vector<float>& palma_detrended_rank0,
        const FilterConfig& cfg);

} // namespace rarecell