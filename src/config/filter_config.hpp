#pragma once
#include <string>

namespace rarecell {

	struct FilterConfig {
		// Thresholding parameters
		float expression_cutoff = 0.0f;   // (raw_counts > cutoff) when counting cells per gene
		int   min_cells = 1;              // keep genes seen in at least this many cells
		int   min_genes = 1;              // keep cells with at least this many detected genes (>0)
		float log2_cutoffh = 1e9f;        // upper bound for log2(max+0.1)
		float log2_cutoffl = -1e9f;       // lower bound for log2(max+0.1)

		// Preprocess
		std::string preprocess_method = "none"; // "cpm" or "none"

		// I/O
		std::string output_folder = ".";  // folder to write outputs
		bool write_dense_csv = false;     // (use with small data only)
		bool write_coo_csv = true;        // write triplets: cell_id,gene,value (per rank)

		// Schema hints (reader already uses these)
		std::string gene_name_column = "Gene";
		std::string cell_type_column = "cell_type";

		// Gene name screen
		bool remove_mir = true;           // drop genes whose names contain "mir" (case-insensitive)
	};

	// Load from a JSON file on disk.
	FilterConfig load_filter_config(const std::string& json_path);

} // namespace rarecell
