#pragma once
#include <mpi.h>
#include <cstdint>
#include <string>
#include <vector>

#include "common/types.hpp"       // rarecell::CSRMatrixF32
#include "graph/knn.hpp"   // rarecell::KNNGraphLocal

namespace rarecell {

    struct RefineParams {
        // Data transform
        bool use_arctan = false;

        // PCA
        int n_pcs = 20;
        int power_iters = 60;

        // Cosine kNN
        int k_knn = 20;                 // local refinement k
        double mix_alpha = 0.7;         // A_loc = alpha*A_cos + (1-alpha)*A_globalCand
        bool use_global_candidates = true;

        // Child acceptance (minimal)
        int min_child_size_abs = 10;
        double min_child_size_frac_parent = 0.005; // e.g. 0.5%

        // Leiden
        double resolution = 1.5;
        double beta = 0.01;
        int n_iterations = -1;
        int seed = 12277;

        bool verbose = true;
    };

    // Refine major clusters using PCA+cosine inside each parent cluster.
    //
    // Inputs:
    //  - X_local: cells x genes CSR slice on this rank (global row0/row1 are required to be correct).
    //  - gene_names: size = n_genes (replicated)
    //  - labels_major_local: size = local_rows, int32 Leiden labels from previous step
    //  - refine_gene_set: list of gene names used for refinement (e.g., Palma genes)
    //  - G_global: global mixed graph (optional but recommended). Used for candidate edges and mixing.
    //
    // Output:
    //  - refined labels local, size = local_rows (keeps largest child as parent label; other children get new ids)
    std::vector<int32_t> refine_pca_cosine_mpi(
        MPI_Comm comm,
        const CSRMatrixF32& X_local,
        const std::vector<std::string>& gene_names,
        const std::vector<int32_t>& labels_major_local,
        const std::vector<std::string>& refine_gene_set,
        const KNNGraphLocal* G_global,   // pass &G_mix
        const RefineParams& p);

} // namespace rarecell
