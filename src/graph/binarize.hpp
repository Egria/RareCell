#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "filter/filter.hpp"

namespace rarecell {

    // Compact binary CSR for local rows: data elements are 0/1.
    struct CSRMatrixU8 {
        int64_t n_rows = 0;  // global rows (cells)
        int64_t n_cols = 0;  // global cols (panel features)
        int64_t row0 = 0;  // local range [row0,row1)
        int64_t row1 = 0;
        std::vector<int64_t> indptr;   // size = local_rows + 1
        std::vector<int32_t> indices;  // col ids in [0, n_cols)
        std::vector<uint8_t> data;     // all ones
    };

    struct BinaryPanel {
        CSRMatrixU8 B_local;                  // (cells_local × panel_size) binary
        std::vector<uint8_t> zero_cells_local;// 1 if a row has 0 ones after binarization
        int cutoff = 1;                       // global cutoff used
        std::vector<std::string> feature_names; // names in panel order
    };

    // Build B for a selected-gene panel using a global cutoff (the same for all
    // genes in the panel). Uses outs.X_local_filtered (cells × kept-genes, CSR).
    BinaryPanel build_binary_panel_cpu(const FilterOutputs& outs,
        const std::vector<int>& panel_gene_idx,
        int cutoff);

} // namespace rarecell