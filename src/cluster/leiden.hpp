#pragma once
#include <mpi.h>
#include <cstdint>
#include <vector>

#include "graph/knn.hpp" // rarecell::KNNGraphLocal

namespace rarecell {

    struct LeidenParams {
        // True if G.weights store distances in [0,1]; convert to similarity = 1-d.
        // False if G.weights already store similarity in [0,1] (this is your current Jaccard kNN output).
        bool is_distance = false;

        // If true, we assume G already represents an undirected graph (both directions present),
        // so we only ship the upper triangle (j > i) to rank0.
        //
        // If your kNN builder produces directed top-k (very common), set this to false.
        bool assume_symmetric = false;

        // If assume_symmetric=false:
        //   force_symmetrize=true merges duplicate undirected edges by MAX weight (union-max).
        //   force_symmetrize=false merges duplicates by SUM weight.
        bool force_symmetrize = true;

        // Leiden parameters
        double resolution = 1.0;
        double beta = 0.01;
        int n_iterations = -1;
        int seed = 0;

        bool verbose = true;
    };

    // Input: distributed KNN graph (CSR rows local to each rank).
    // Output: labels for local rows [row0,row1), same order as local rows.
    std::vector<int32_t> leiden_cluster_mpi(const KNNGraphLocal& G_local,
        MPI_Comm comm,
        const LeidenParams& p);

} // namespace rarecell
