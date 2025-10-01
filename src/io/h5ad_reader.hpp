#pragma once
#include <mpi.h>
#include <string>
#include "common/types.hpp"

namespace rarecell {

	class H5ADReader {
	public:
		explicit H5ADReader(MPI_Comm comm);
		H5ADReadResult read(const std::string& h5ad_path);

	private:
		MPI_Comm comm_;
		int rank_ = 0;
		int world_ = 1;

		// helpers
		static void split_rows(int64_t n_rows, int rank, int world, int64_t& row0, int64_t& row1);
	};

} // namespace rarecell