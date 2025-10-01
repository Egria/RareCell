#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rarecell {

	struct CSRMatrixF32 {
		// Global shape of X (cells x genes). This local object holds only rows [row0, row1).
		int64_t n_rows = 0;     // global
		int64_t n_cols = 0;     // global
		int64_t row0 = 0;     // global start row owned by this rank
		int64_t row1 = 0;     // global end row (exclusive)

		// Local CSR slice for rows [row0, row1).
		// indptr.size() == (local_rows + 1); indices.size() == data.size() == local_nnz
		std::vector<int64_t> indptr;   // 64-bit row pointers (safe for large nnz)
		std::vector<int32_t> indices;  // column indices
		std::vector<float>   data;     // values
	};

	struct H5ADReadResult {
		CSRMatrixF32 X_local;                      // local rows on this rank
		std::vector<std::string> gene_names;       // from /var/Gene (replicated)
		std::vector<std::string> cell_ids_local;   // from /obs/_index (local slice)
		std::vector<std::string> cell_type_local;  // from /obs/cell_type (local slice)
	};

} // namespace rarecell