#pragma once
#include <mpi.h>
#include <cstdint>
#include <vector>
#include <string>
#include "graph/binarize.hpp"  // BinaryPanel

namespace rarecell {

    // Local directed kNN for this rank's rows [row0,row1) with Jaccard weights.
    struct KNNGraphLocal {
        int64_t n_nodes = 0;   // global #cells
        int64_t row0 = 0;   // local start row
        int64_t row1 = 0;   // local end row (exclusive)
        std::vector<int64_t> indptr;   // size = local_rows + 1
        std::vector<int64_t> indices;  // neighbor global ids
        std::vector<float>   weights;  // Jaccard similarities in [0,1]
    };

    // Build Jaccard kNN for a binary panel (cells_local x features), distributed over 'comm'.
    // Fast & robust on large datasets by:
    //  - skipping self rows and zero rows,
    //  - ring streaming of peer CSC,
    //  - OPTIONAL candidate-generation cap on ultra-common features (df_cap_frac).
    //
    // Parameters
    //  - k: neighbors per cell (directed)
    //  - block_size: local rows processed per block
    //  - df_cap_frac: ignore features whose global DF > df_cap_frac * n_nodes when generating candidates.
    //                 The feature is still counted in |A| and |B| (degrees) via deg vectors; we only
    //                 avoid iterating its massive posting lists. Set to 1.0 to disable.
    //  - verbose: if true, print per-block progress with timings.
    //
    // Returns
    //  - Directed top-k per local row. You can symmetrize later if needed.
    //
    KNNGraphLocal build_knn_graph_jaccard_mpi(
        MPI_Comm comm,
        const BinaryPanel& panel,
        int k = 30,
        int block_size = 4096,
        double df_cap_frac = 0.02,  // 2% default; tune 0.5–5% as needed. Use 1.0 to disable.
        bool verbose = true
    );

} // namespace rarecell
