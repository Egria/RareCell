#pragma once
#include <mpi.h>
#include <cstdint>
#include <string>
#include <vector>
#include "common/types.hpp"
#include "config/filter_config.hpp"

namespace rarecell {

	struct FilterOutputs {
		// Final masks / remaps
		std::vector<uint8_t> keep_gene_final;   // length = n_cols (global mask)
		std::vector<int32_t> col_remap_final;   // length = n_cols (-1 = dropped, else new idx)
		std::vector<uint8_t> keep_cell_local;   // length = local_rows
		std::vector<int64_t> kept_local_rows;   // local row indices kept

		// Filtered CSR (local rows only; columns remapped to [0..n_keep_genes-1])
		CSRMatrixF32 X_local_filtered;

		// Names after filtering
		std::vector<std::string> gene_names_filtered;     // global, replicated
		std::vector<std::string> cell_ids_local_filtered; // local slice (kept rows)
	};

	class FilterEngine {
	public:
		FilterEngine(MPI_Comm comm, const H5ADReadResult& R, const FilterConfig& cfg);

		// Run the entire filtering pipeline and return outputs (also writes CSVs if configured)
		FilterOutputs run();

	private:
		// MPI
		MPI_Comm comm_;
		int rank_ = 0;
		int world_ = 1;

		// Inputs
		const H5ADReadResult& R_;
		const FilterConfig& cfg_;

		// Selected CUDA device for this rank
		int device_id_ = -1;

		// Dimensions
		int64_t n_rows_local_ = 0;  // local rows for this rank
		int64_t nnz_local_ = 0;  // local nnz for this rank
		int64_t n_cols_ = 0;  // global number of genes

		// Host-side working buffers
		std::vector<int32_t> ncells_per_gene_global_; // length n_cols
		std::vector<uint8_t> keep_gene_mask_;         // length n_cols
		std::vector<uint8_t> keep_cell_mask_;         // length n_rows_local
		std::vector<float>   gene_max_global_;        // length n_cols

		// Device buffers
		int64_t* d_indptr_ = nullptr;
		int32_t* d_indices_ = nullptr;
		float* d_data_ = nullptr;

		int32_t* d_counts_gene_ = nullptr;
		uint8_t* d_keep_gene_ = nullptr;

		int32_t* d_ngenes_cell_ = nullptr;
		uint8_t* d_keep_cell_ = nullptr;

		float* d_gene_max_ = nullptr;

		// Orchestration
		void init_cuda_device();  // choose GPU per node-local rank, warm-up, print mem/estimate
		void device_alloc();      // allocate & copy CSR + scratch
		void device_free();       // free all device buffers

		// Steps (match your Python)
		void step_count_cells_per_gene(); // > expression_cutoff, reduce over ranks
		void step_build_initial_keep_gene(); // remove MIR/Mir + apply min_cells
		void step_ngene_per_cell(); // per-cell counts among kept genes, apply min_genes
		void step_gene_max_on_kept(); // per-gene max on kept cells+genes, reduce MAX
		void step_apply_log2_band();  // keep genes in [log2_cutoffl, log2_cutoffh]
		FilterOutputs step_materialize_filtered(); // compact CSR & names; optional CPM
		void step_cpm_normalize(CSRMatrixF32& X);  // per-row (cell) CPM
	};

} // namespace rarecell
