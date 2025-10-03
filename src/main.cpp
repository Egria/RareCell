#include <mpi.h>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <iomanip>                 // NEW: for formatting
#include "io/h5ad_reader.hpp"
#include "config/filter_config.hpp"
#include "filter/filter.hpp"
#include "metrics/fano.hpp"        // NEW
#include "metrics/gini.hpp"

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

        // -------- Compute Fano factors (exact) and print top 10 on rank 0 --------
        auto fano_res = rarecell::compute_fano_all_genes(MPI_COMM_WORLD, outs);

        if (rank == 0) {
            auto top10 = rarecell::top_k_fano(outs.gene_names_filtered, fano_res.fano, 10);
            std::cout << "\nTop 10 genes by Fano factor (N=" << fano_res.N_total << " cells):\n";
            std::cout << std::left << std::setw(6) << "Rank"
                << std::setw(24) << "Gene"
                << "Fano\n";
            for (std::size_t i = 0; i < top10.size(); ++i) {
                std::cout << std::left << std::setw(6) << (i + 1)
                    << std::setw(24) << top10[i].first
                    << std::fixed << std::setprecision(6) << top10[i].second << "\n";
            }

            // Optional: write all Fano values to CSV
            try {
                std::ofstream csv(std::filesystem::path(cfg.output_folder) / "fano_all_genes.csv");
                csv << "gene,fano\n";
                for (std::size_t g = 0; g < outs.gene_names_filtered.size(); ++g) {
                    csv << outs.gene_names_filtered[g] << "," << fano_res.fano[g] << "\n";
                }
            }
            catch (...) {  } 
        }

        // -------- Compute exact Gini (raw integer counts) and print top 10 on rank 0 --------
        auto gini_res = rarecell::compute_gini_raw_int_rank0(MPI_COMM_WORLD, R, outs);

        if (rank == 0) {
            auto top10g = rarecell::top_k_gini(outs.gene_names_filtered, gini_res.gini, 10);
            std::cout << "\nTop 10 genes by Gini (N=" << gini_res.N_total << " cells):\n";
            std::cout << std::left << std::setw(6) << "Rank"
                << std::setw(24) << "Gene"
                << "Gini\n";
            for (std::size_t i = 0; i < top10g.size(); ++i) {
                std::cout << std::left << std::setw(6) << (i + 1)
                    << std::setw(24) << top10g[i].first
                    << std::fixed << std::setprecision(6) << top10g[i].second << "\n";
            }

            // Optional: write all Gini values to CSV
            try {
                namespace fs = std::filesystem;
                fs::path outdir = cfg.output_folder.empty() ? fs::path(".") : fs::path(cfg.output_folder);
                std::error_code ec; fs::create_directories(outdir, ec);
                fs::path csv_path = outdir / "gini_all_genes_rawint.csv";
                std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
                if (!csv) {
                    std::cerr << "[rank 0] Warning: cannot open " << csv_path.string() << " for writing.\n";
                }
                else {
                    csv << "gene,gini\n";
                    for (std::size_t g = 0; g < gini_res.gini.size(); ++g) {
                        csv << outs.gene_names_filtered[g] << "," << gini_res.gini[g] << "\n";
                    }
                }
            }
            catch (...) {
                std::cerr << "[rank 0] Warning: exception while writing gini_all_genes_rawint.csv\n";
            }
        }


    }
    catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] Error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Finalize();
    return 0;
}
