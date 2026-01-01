#include "graph/mix.hpp"
#include <algorithm>
#include <stdexcept>

namespace rarecell {

    static inline void append_scaled_row_edges(
        const KNNGraphLocal& G,
        int64_t i_local,
        double w_graph,
        bool graphs_are_distance,
        std::vector<std::pair<int64_t, double>>& out)
    {
        if (w_graph == 0.0) return;
        const int64_t s = G.indptr[(size_t)i_local];
        const int64_t e = G.indptr[(size_t)i_local + 1];
        out.reserve(out.size() + (size_t)(e - s));

        for (int64_t p = s; p < e; ++p) {
            int64_t j = G.indices[(size_t)p];
            double  w = (double)G.weights[(size_t)p];

            // Convert distance->similarity if requested
            if (graphs_are_distance) {
                // distance assumed in [0,1]; similarity = 1 - d
                w = 1.0 - w;
                if (w < 0.0) w = 0.0;
                if (w > 1.0) w = 1.0;
            }

            if (w <= 0.0) continue;
            out.emplace_back(j, w * w_graph);
        }
    }

    KNNGraphLocal mix_three_knn_graphs_local(
        const KNNGraphLocal& G_gini,
        const KNNGraphLocal& G_fano,
        const KNNGraphLocal& G_palma,
        double w_gini,
        double w_fano,
        double w_palma,
        bool graphs_are_distance,
        int prune_topk,
        bool drop_self_loops)
    {
        // Basic consistency checks
        if (G_gini.n_nodes != G_fano.n_nodes || G_gini.n_nodes != G_palma.n_nodes)
            throw std::runtime_error("mix_three_knn_graphs_local: n_nodes mismatch");

        if (G_gini.row0 != G_fano.row0 || G_gini.row0 != G_palma.row0 ||
            G_gini.row1 != G_fano.row1 || G_gini.row1 != G_palma.row1)
            throw std::runtime_error("mix_three_knn_graphs_local: row ranges mismatch");

        const int64_t local_rows = (int64_t)G_gini.indptr.size() - 1;
        if ((int64_t)G_fano.indptr.size() - 1 != local_rows ||
            (int64_t)G_palma.indptr.size() - 1 != local_rows)
            throw std::runtime_error("mix_three_knn_graphs_local: indptr size mismatch");

        KNNGraphLocal out;
        out.n_nodes = G_gini.n_nodes;
        out.row0 = G_gini.row0;
        out.row1 = G_gini.row1;
        out.indptr.assign((size_t)local_rows + 1, 0);

        // Reserve rough: <= (deg_gini+deg_fano+deg_palma) per row
        std::vector<int64_t> out_idx;
        std::vector<float>   out_w;
        out_idx.reserve((size_t)local_rows * 64);
        out_w.reserve((size_t)local_rows * 64);

        std::vector<std::pair<int64_t, double>> edges;
        edges.reserve(128);

        for (int64_t i = 0; i < local_rows; ++i) {
            const int64_t i_global = out.row0 + i;
            edges.clear();

            // Gather edges from three graphs
            append_scaled_row_edges(G_gini, i, w_gini, graphs_are_distance, edges);
            append_scaled_row_edges(G_fano, i, w_fano, graphs_are_distance, edges);
            append_scaled_row_edges(G_palma, i, w_palma, graphs_are_distance, edges);

            if (edges.empty()) {
                out.indptr[(size_t)i + 1] = out.indptr[(size_t)i];
                continue;
            }

            // Drop self loops (Python: W.setdiag(0.0))
            if (drop_self_loops) {
                edges.erase(std::remove_if(edges.begin(), edges.end(),
                    [&](const auto& pr) { return pr.first == i_global; }),
                    edges.end());
                if (edges.empty()) {
                    out.indptr[(size_t)i + 1] = out.indptr[(size_t)i];
                    continue;
                }
            }

            // Combine duplicates by summing (Python: COO->CSR merges duplicates)
            // Sort by neighbor id
            std::sort(edges.begin(), edges.end(),
                [](const auto& a, const auto& b) {
                    return (a.first < b.first);
                });

            std::vector<std::pair<int64_t, double>> merged;
            merged.reserve(edges.size());

            for (size_t t = 0; t < edges.size(); ) {
                int64_t j = edges[t].first;
                double  s = edges[t].second;
                size_t u = t + 1;
                while (u < edges.size() && edges[u].first == j) {
                    s += edges[u].second;
                    ++u;
                }
                if (s > 0.0) merged.emplace_back(j, s);
                t = u;
            }

            if (merged.empty()) {
                out.indptr[(size_t)i + 1] = out.indptr[(size_t)i];
                continue;
            }

            // Optional top-k prune per row (Python: _prune_topk_csr)
            if (prune_topk > 0 && (int)merged.size() > prune_topk) {
                const int K = prune_topk;

                // partial sort by weight desc, tie by neighbor asc for determinism
                std::partial_sort(merged.begin(), merged.begin() + K, merged.end(),
                    [](const auto& a, const auto& b) {
                        if (a.second != b.second) return a.second > b.second;
                        return a.first < b.first;
                    });
                merged.resize((size_t)K);

                // After pruning, sort indices for CSR stability
                std::sort(merged.begin(), merged.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }

            // Append to output CSR
            const int64_t base = (int64_t)out_idx.size();
            for (const auto& pr : merged) {
                out_idx.push_back(pr.first);
                out_w.push_back((float)pr.second);
            }
            out.indptr[(size_t)i + 1] = (int64_t)out_idx.size();
            (void)base;
        }

        out.indices.swap(out_idx);
        out.weights.swap(out_w);
        return out;
    }

} // namespace rarecell
