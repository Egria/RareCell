#include <mpi.h>
#include <iostream>
#include <string>
#include "io/h5ad_reader.hpp"
#include "config/filter_config.hpp"
#include "filter/filter.hpp"

using rarecell::H5ADReader;
using rarecell::H5ADReadResult;
using rarecell::FilterConfig;
using rarecell::FilterEngine;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: rarecell_filter <file.h5ad> <config.json>\n";
        }
        MPI_Finalize();
        return 1;
    }

    try {
        const std::string h5ad_path = argv[1];
        const std::string cfg_path = argv[2];

        // Load config
        FilterConfig cfg = rarecell::load_filter_config(cfg_path);
        if (rank == 0) {
            std::cout << "Output folder: " << cfg.output_folder
                << " | preprocess: " << cfg.preprocess_method << "\n";
        }

        // Read
        H5ADReader reader(MPI_COMM_WORLD);
        H5ADReadResult R = reader.read(h5ad_path);

        // Filter
        FilterEngine engine(MPI_COMM_WORLD, R, cfg);
        auto outs = engine.run();

        // Summary
        int64_t n_genes_keep = (int64_t)outs.gene_names_filtered.size();
        int64_t n_cells_local_keep = (int64_t)outs.cell_ids_local_filtered.size();
        std::cout << "[rank " << rank << "/" << world << "] kept "
            << n_genes_keep << " genes, "
            << n_cells_local_keep << " cells, "
            << (int64_t)outs.X_local_filtered.data.size() << " nnz\n";

    }
    catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] Error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Finalize();
    return 0;
}
