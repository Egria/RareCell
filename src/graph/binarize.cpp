#include "graph/binarize.hpp"
#include <algorithm>
#include <numeric>

namespace rarecell {

    BinaryPanel build_binary_panel_cpu(const FilterOutputs& outs,
        const std::vector<int>& panel_gene_idx,
        int cutoff)
    {
        BinaryPanel P;
        const auto& X = outs.X_local_filtered;

        const int64_t local_rows = (int64_t)X.indptr.size() - 1;
        const int64_t panel_cols = (int64_t)panel_gene_idx.size();

        // Map kept-gene index -> panel column (or -1 if not selected)
        std::vector<int32_t> g2p((size_t)X.n_cols, -1);
        for (int32_t j = 0; j < (int32_t)panel_cols; ++j) {
            int32_t g = (int32_t)panel_gene_idx[(size_t)j];
            if (g >= 0 && g < (int32_t)X.n_cols) g2p[(size_t)g] = j;
        }

        // Prepare CSR
        P.B_local.n_rows = X.n_rows;     // global rows count unchanged
        P.B_local.n_cols = panel_cols;   // features = panel size
        P.B_local.row0 = X.row0;
        P.B_local.row1 = X.row1;

        P.B_local.indptr.assign((size_t)local_rows + 1, 0);
        P.zero_cells_local.assign((size_t)local_rows, 0);

        // First pass: count kept entries per row
        int64_t nnz_est = 0;
        for (int64_t r = 0; r < local_rows; ++r) {
            int64_t s = X.indptr[(size_t)r];
            int64_t e = X.indptr[(size_t)r + 1];
            int64_t cnt = 0;
            for (int64_t p = s; p < e; ++p) {
                int32_t g = X.indices[(size_t)p];
                int32_t col = g2p[(size_t)g];
                if (col >= 0 && X.data[(size_t)p] >= (float)cutoff) ++cnt;
            }
            P.B_local.indptr[(size_t)r + 1] = P.B_local.indptr[(size_t)r] + cnt;
            P.zero_cells_local[(size_t)r] = (cnt == 0) ? 1u : 0u;
            nnz_est += cnt;
        }

        P.B_local.indices.resize((size_t)nnz_est);
        P.B_local.data.resize((size_t)nnz_est, (uint8_t)1);

        // Second pass: write indices
        for (int64_t r = 0; r < local_rows; ++r) {
            int64_t s = X.indptr[(size_t)r];
            int64_t e = X.indptr[(size_t)r + 1];
            int64_t w = P.B_local.indptr[(size_t)r];
            for (int64_t p = s; p < e; ++p) {
                int32_t g = X.indices[(size_t)p];
                int32_t col = g2p[(size_t)g];
                float   v = X.data[(size_t)p];
                if (col >= 0 && v >= (float)cutoff) {
                    P.B_local.indices[(size_t)w++] = col;
                    // data is already 1
                }
            }
        }

        // Record feature names in panel order
        P.feature_names.clear();
        P.feature_names.reserve(panel_gene_idx.size());
        for (int idx : panel_gene_idx) {
            P.feature_names.push_back(outs.gene_names_filtered[(size_t)idx]);
        }

        P.cutoff = cutoff;
        return P;
    }

} // namespace rarecell
