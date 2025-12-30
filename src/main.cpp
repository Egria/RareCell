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
#include "metrics/palma.hpp"
#include "metrics/lowess.hpp"
#include "graph/selection.hpp"
#include "cuda/metrics_cuda.hpp"
#include "graph/binarize.hpp"
#include "graph/knn.hpp"

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

        /*
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

        // -------- Compute EXACT Palma (raw integer counts) with batching --------
        auto palma_res = rarecell::compute_palma_raw_int_distributed_batched_rank0(
            MPI_COMM_WORLD, R, outs, cfg);

        if (rank == 0) {
            auto top10p = rarecell::top_k_palma(outs.gene_names_filtered, palma_res.palma, 10);
            std::cout << "\nTop 10 genes by Palma_alpha (N=" << palma_res.N_total
                << ", upper=" << cfg.palma_upper << ", lower=" << cfg.palma_lower
                << ", alpha=" << cfg.palma_alpha << ", winsor=" << cfg.palma_winsor << "):\n";
            std::cout << std::left << std::setw(6) << "Rank"
                << std::setw(24) << "Gene"
                << "Palma\n";
            for (std::size_t i = 0; i < top10p.size(); ++i) {
                std::cout << std::left << std::setw(6) << (i + 1)
                    << std::setw(24) << top10p[i].first
                    << std::fixed << std::setprecision(6) << top10p[i].second << "\n";
            }

            // CSV write (unchanged)
            try {
                namespace fs = std::filesystem;
                fs::path outdir = cfg.output_folder.empty() ? fs::path(".") : fs::path(cfg.output_folder);
                std::error_code ec; fs::create_directories(outdir, ec);
                fs::path csv_path = outdir / "palma_all_genes_rawint.csv";
                std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
                if (csv) {
                    csv << "gene,palma\n";
                    for (std::size_t g = 0; g < palma_res.palma.size(); ++g) {
                        csv << outs.gene_names_filtered[g] << "," << palma_res.palma[g] << "\n";
                    }
                }
                else {
                    std::cerr << "[rank 0] Warning: cannot open " << csv_path.string() << " for writing.\n";
                }
            }
            catch (...) {
                std::cerr << "[rank 0] Warning: exception while writing palma_all_genes_rawint.csv\n";
            }
        }*/

        // Fano (CUDA on filtered)
        auto fano_res = rarecell::compute_fano_all_genes_cuda(MPI_COMM_WORLD, outs);

        // Gini (CUDA on RAW, reduce to rank 0)
        auto gini_res = rarecell::compute_gini_raw_int_cuda_rank0(MPI_COMM_WORLD, R, outs, /*hist_L_cap=*/128);

        // Palma (CUDA on RAW, distributed owners -> rank 0)
        auto palma_res = rarecell::compute_palma_raw_int_cuda_distributed_reduce_rank0(
            MPI_COMM_WORLD, R, outs, cfg, /*hist_L_cap=*/128);

        // ---------- Two-pass LOWESS detrending (RAW-based log2max) ----------
        std::vector<float> gini_resid_rank0 =
            rarecell::lowess_twopass_detrend_metric_rank0(
                MPI_COMM_WORLD, R, outs, gini_res.gini, /*outlier_q=*/0.75, /*span=*/0.90);

        std::vector<float> palma_resid_rank0 =
            rarecell::lowess_twopass_detrend_metric_rank0(
                MPI_COMM_WORLD, R, outs, palma_res.palma, /*outlier_q=*/0.75, /*span=*/0.90);

        if (rank == 0) {
            // ---- Top 10 by residuals (Gini) ----
            std::vector<size_t> idx_g(gini_resid_rank0.size());
            for (size_t i = 0; i < idx_g.size(); ++i) idx_g[i] = i;
            std::partial_sort(idx_g.begin(), idx_g.begin() + std::min<size_t>(10, idx_g.size()), idx_g.end(),
                [&](size_t a, size_t b) { return gini_resid_rank0[a] > gini_resid_rank0[b]; });

            std::cout << "\nTop 10 genes by detrended Gini residual:\n";
            for (size_t k = 0; k < std::min<size_t>(10, idx_g.size()); ++k) {
                size_t i = idx_g[k];
                std::cout << (k + 1) << ". " << outs.gene_names_filtered[i]
                    << "  resid=" << gini_resid_rank0[i] << "\n";
            }

            // ---- Top 10 by residuals (Palma) ----
            std::vector<size_t> idx_p(palma_resid_rank0.size());
            for (size_t i = 0; i < idx_p.size(); ++i) idx_p[i] = i;
            std::partial_sort(idx_p.begin(), idx_p.begin() + std::min<size_t>(10, idx_p.size()), idx_p.end(),
                [&](size_t a, size_t b) { return palma_resid_rank0[a] > palma_resid_rank0[b]; });

            std::cout << "\nTop 10 genes by detrended Palma residual:\n";
            for (size_t k = 0; k < std::min<size_t>(10, idx_p.size()); ++k) {
                size_t i = idx_p[k];
                std::cout << (k + 1) << ". " << outs.gene_names_filtered[i]
                    << "  resid=" << palma_resid_rank0[i] << "\n";
            }

            // ---- Write CSVs ----
            try {
                namespace fs = std::filesystem;
                fs::path outdir = cfg.output_folder.empty() ? fs::path(".") : fs::path(cfg.output_folder);
                std::error_code ec; fs::create_directories(outdir, ec);


                try {
                    std::ofstream csv(std::filesystem::path(cfg.output_folder) / "fano_all_genes.csv");
                    csv << "gene,fano\n";
                    for (std::size_t g = 0; g < outs.gene_names_filtered.size(); ++g) {
                        csv << outs.gene_names_filtered[g] << "," << fano_res.fano[g] << "\n";
                    }
                }
                catch (...) {}

                // Gini residuals
                {
                    std::ofstream csv(outdir / "gini_lowess_residuals.csv", std::ios::out | std::ios::trunc);
                    if (csv) {
                        csv << "gene,gini_residual\n";
                        for (size_t i = 0; i < gini_resid_rank0.size(); ++i) {
                            csv << outs.gene_names_filtered[i] << "," << gini_resid_rank0[i] << "\n";
                        }
                    }
                    else {
                        std::cerr << "[rank 0] Warning: cannot open gini_lowess_residuals.csv\n";
                    }
                }

                // Palma residuals
                {
                    std::ofstream csv(outdir / "palma_lowess_residuals.csv", std::ios::out | std::ios::trunc);
                    if (csv) {
                        csv << "gene,palma_residual\n";
                        for (size_t i = 0; i < palma_resid_rank0.size(); ++i) {
                            csv << outs.gene_names_filtered[i] << "," << palma_resid_rank0[i] << "\n";
                        }
                    }
                    else {
                        std::cerr << "[rank 0] Warning: cannot open palma_lowess_residuals.csv\n";
                    }
                }
            }
            catch (...) {
                std::cerr << "[rank 0] Warning: exception while writing detrended residual CSVs\n";
            }
        }


        
        // ---------- Feature selection (separate panels) ----------
        auto panels = rarecell::select_feature_panels_rank0_bcast(
            MPI_COMM_WORLD,
            outs,
            gini_resid_rank0,
            fano_res.fano,
            palma_resid_rank0,
            cfg);

        // Small summary (rank 0)
        if (rank == 0) {
            std::cout << "\nSelected features:\n"
                << "  Gini (detrended):  " << panels.gini_genes.size() << " genes\n"
                << "  Fano:              " << panels.fano_genes.size() << " genes\n"
                << "  Palma (detrended): " << panels.palma_genes.size() << " genes\n";

            auto print_head = [&](const char* title, const std::vector<std::string>& v) {
                std::cout << "  " << title << " (first 10): ";
                for (size_t i = 0; i < std::min<size_t>(10, v.size()); ++i) {
                    if (i) std::cout << ", ";
                    std::cout << v[i];
                }
                std::cout << "\n";
                };
            print_head("Gini det.", panels.gini_genes);
            print_head("Fano", panels.fano_genes);
            print_head("Palma det.", panels.palma_genes);
        }


        const double gamma = 0.90;   // as requested

        // --- Compute panel cutoffs (CUDA hist + rank-0 reduction) ---
        int cutoff_gini = rarecell::compute_gamma_cutoff_selected_cuda(
            MPI_COMM_WORLD, R, outs, panels.gini_idx, gamma, 128);
        int cutoff_fano = rarecell::compute_gamma_cutoff_selected_cuda(
            MPI_COMM_WORLD, R, outs, panels.fano_idx, gamma, 128);
        int cutoff_palma = rarecell::compute_gamma_cutoff_selected_cuda(
            MPI_COMM_WORLD, R, outs, panels.palma_idx, gamma, 128);

        // --- Binarize locally (fast CPU, row-major CSR) ---
        auto B_gini = rarecell::build_binary_panel_cpu(outs, panels.gini_idx, cutoff_gini);
        auto B_fano = rarecell::build_binary_panel_cpu(outs, panels.fano_idx, cutoff_fano);
        auto B_palma = rarecell::build_binary_panel_cpu(outs, panels.palma_idx, cutoff_palma);

        // Optional: summary on rank 0
        if (rank == 0) {
            std::cout << "\nBinarization summary (gamma=" << gamma << "):\n"
                << "  Gini panel:  features=" << B_gini.B_local.n_cols
                << " cutoff=" << B_gini.cutoff << "\n"
                << "  Fano panel:  features=" << B_fano.B_local.n_cols
                << " cutoff=" << B_fano.cutoff << "\n"
                << "  Palma panel: features=" << B_palma.B_local.n_cols
                << " cutoff=" << B_palma.cutoff << "\n";
        }

        const int k_neighbors = 30;
        const int block_size = 4096;  // matches your Python default

        auto G_gini = rarecell::build_knn_graph_jaccard_mpi(MPI_COMM_WORLD, B_gini, k_neighbors, block_size);
        auto G_fano = rarecell::build_knn_graph_jaccard_mpi(MPI_COMM_WORLD, B_fano, k_neighbors, block_size);
        auto G_palma = rarecell::build_knn_graph_jaccard_mpi(MPI_COMM_WORLD, B_palma, k_neighbors, block_size);

        // Quick summary per rank
        if (rank == 0) {
            auto nnz_local = (long long)G_gini.indices.size();
            std::cout << "\nKNN (Jaccard, k=" << k_neighbors << ", block=" << block_size << ") built.\n";
        }
        
        


    }
    catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] Error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Finalize();
    return 0;
}
