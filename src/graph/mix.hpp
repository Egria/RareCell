#pragma once
#include <cstdint>
#include <vector>
#include <utility>
#include "graph/knn.hpp"   // KNNGraphLocal

namespace rarecell {

    // Mix three kNN graphs (directed, local-row CSR) by weighted sum.
    // - graphs_are_distance=false (recommended): treat input weights as similarities already.
    // - graphs_are_distance=true : treat input weights as distances in [0,1] and convert sim=(1-d).
    // - prune_topk: if >0, keep only top-k edges per row after mixing (recommended: same k=30).
    KNNGraphLocal mix_three_knn_graphs_local(
        const KNNGraphLocal& G_gini,
        const KNNGraphLocal& G_fano,
        const KNNGraphLocal& G_palma,
        double w_gini,
        double w_fano,
        double w_palma,
        bool graphs_are_distance = false,
        int prune_topk = -1,
        bool drop_self_loops = true);

} // namespace rarecell
