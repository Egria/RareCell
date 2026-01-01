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
#include "graph/mix.hpp"
#include "cluster/leiden.hpp"
#include "refine/refine_pca_cosine.hpp"

using rarecell::H5ADReader;
using rarecell::H5ADReadResult;
using rarecell::FilterConfig;
using rarecell::FilterEngine;

static void gather_and_write_labels_csv(
    MPI_Comm comm,
    const std::vector<int32_t>& labels_local,
    const std::vector<std::string>& cell_ids_local,   // pass {} if you don't want IDs
    const std::string& out_csv_path
) {
    int rank = 0, world = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world);

    const int local_n = (int)labels_local.size();

    // ---- gather counts (cells per rank) ----
    std::vector<int> counts, displs;
    if (rank == 0) {
        counts.resize(world);
        displs.resize(world);
    }

    MPI_Gather(&local_n, 1, MPI_INT,
        rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
        0, comm);

    int total_n = 0;
    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < world; ++r) displs[r] = displs[r - 1] + counts[r - 1];
        total_n = displs[world - 1] + counts[world - 1];
    }

    // ---- gather labels ----
    std::vector<int32_t> labels_global;
    if (rank == 0) labels_global.resize((size_t)total_n);

#ifdef MPI_INT32_T
    MPI_Datatype mpi_i32 = MPI_INT32_T;
#else
    MPI_Datatype mpi_i32 = MPI_INT; // MSVC int is 32-bit on Windows
    static_assert(sizeof(int) == 4, "MPI_INT expected to be 32-bit");
#endif

    MPI_Gatherv((void*)labels_local.data(), local_n, mpi_i32,
        rank == 0 ? (void*)labels_global.data() : nullptr,
        rank == 0 ? counts.data() : nullptr,
        rank == 0 ? displs.data() : nullptr,
        mpi_i32,
        0, comm);

    // ---- optionally gather cell IDs ----
    bool have_ids = (!cell_ids_local.empty() && (int)cell_ids_local.size() == local_n);

    std::vector<int32_t> id_len_global;
    std::vector<char>    id_chars_global;

    if (have_ids) {
        // local lengths
        std::vector<int32_t> id_len_local((size_t)local_n);
        long long local_bytes_ll = 0;
        for (int i = 0; i < local_n; ++i) {
            id_len_local[(size_t)i] = (int32_t)cell_ids_local[(size_t)i].size();
            local_bytes_ll += (long long)id_len_local[(size_t)i];
        }
        if (local_bytes_ll > (long long)std::numeric_limits<int>::max()) {
            throw std::runtime_error("Local cell-id bytes exceed INT_MAX; chunking needed.");
        }
        const int local_bytes = (int)local_bytes_ll;

        // pack local chars
        std::vector<char> id_chars_local((size_t)local_bytes);
        int pos = 0;
        for (int i = 0; i < local_n; ++i) {
            const std::string& s = cell_ids_local[(size_t)i];
            if (!s.empty()) {
                std::memcpy(id_chars_local.data() + pos, s.data(), s.size());
            }
            pos += (int)s.size();
        }

        // gather lengths (same counts/displs as labels)
        if (rank == 0) id_len_global.resize((size_t)total_n);
#ifdef MPI_INT32_T
        MPI_Datatype mpi_i32_len = MPI_INT32_T;
#else
        MPI_Datatype mpi_i32_len = MPI_INT;
#endif
        MPI_Gatherv((void*)id_len_local.data(), local_n, mpi_i32_len,
            rank == 0 ? (void*)id_len_global.data() : nullptr,
            rank == 0 ? counts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            mpi_i32_len,
            0, comm);

        // gather byte counts
        std::vector<int> byte_counts, byte_displs;
        int total_bytes = 0;
        if (rank == 0) {
            byte_counts.resize(world);
            byte_displs.resize(world);
        }

        MPI_Gather((void*)&local_bytes, 1, MPI_INT,
            rank == 0 ? byte_counts.data() : nullptr, 1, MPI_INT,
            0, comm);

        if (rank == 0) {
            byte_displs[0] = 0;
            for (int r = 1; r < world; ++r) byte_displs[r] = byte_displs[r - 1] + byte_counts[r - 1];
            total_bytes = byte_displs[world - 1] + byte_counts[world - 1];
            id_chars_global.resize((size_t)total_bytes);
        }

        MPI_Gatherv((void*)id_chars_local.data(), local_bytes, MPI_CHAR,
            rank == 0 ? (void*)id_chars_global.data() : nullptr,
            rank == 0 ? byte_counts.data() : nullptr,
            rank == 0 ? byte_displs.data() : nullptr,
            MPI_CHAR,
            0, comm);
    }

    // ---- rank0 write CSV ----
    if (rank == 0) {
        std::ofstream out(out_csv_path, std::ios::out);
        if (!out) throw std::runtime_error("Failed to open output file: " + out_csv_path);

        if (have_ids) out << "cell_id,label\n";
        else          out << "cell_index,label\n";

        if (have_ids) {
            // reconstruct strings from length+char stream
            long long pos = 0;
            for (int i = 0; i < total_n; ++i) {
                const int32_t L = id_len_global[(size_t)i];
                std::string id;
                if (L > 0) {
                    id.assign(id_chars_global.data() + pos, id_chars_global.data() + pos + L);
                }
                pos += (long long)L;
                out << id << "," << labels_global[(size_t)i] << "\n";
            }
        }
        else {
            for (int i = 0; i < total_n; ++i) {
                out << i << "," << labels_global[(size_t)i] << "\n";
            }
        }

        out.close();
        std::cout << "[rank 0] Wrote labels to: " << out_csv_path
            << " (N=" << total_n << ")\n";
    }
}

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
        
        
        // Our knn builder stores Jaccard similarity already, so graphs_are_distance=false.
        const bool graphs_are_distance = false;

        // Recommended: keep graph sparsity similar to original kNN graphs
        const int prune_topk = 30;

        auto G_mix = rarecell::mix_three_knn_graphs_local(
            G_gini, G_fano, G_palma,
            cfg.gini_balance, cfg.fano_balance, cfg.palma_balance,
            graphs_are_distance,
            prune_topk,
            /*drop_self_loops=*/true
        );

        if (rank == 0) {
            std::cout << "Mixed graph built (local nnz rank0)=" << (long long)G_mix.indices.size()
                << " | weights: gini=" << cfg.gini_balance
                << " fano=" << cfg.fano_balance
                << " palma=" << cfg.palma_balance
                << " | prune_topk=" << prune_topk
                << "\n";
        }

        rarecell::LeidenParams lp;
        lp.is_distance = false;         // your graph stores similarity weights (Jaccard sim), not distances
        lp.assume_symmetric = false;     // if your kNN builder already symmetrized (union/max)
        lp.force_symmetrize = true;    // not needed when assume_symmetric=true
        lp.resolution = 1.0;
        lp.beta = 0.01;
        lp.n_iterations = -1;
        lp.seed = 0;
        lp.verbose = (rank == 0);

        auto labels_local = rarecell::leiden_cluster_mpi(G_mix, MPI_COMM_WORLD, lp);

        if (rank == 0) {
            std::cout << "[rank 0] received labels_local size=" << labels_local.size()
                << " (this is only rank0 slice)\n";
        }

        std::string out_csv = cfg.output_folder + "/leiden_labels.csv";

        gather_and_write_labels_csv(MPI_COMM_WORLD, labels_local, R.cell_ids_local, out_csv);
        
        rarecell::RefineParams rp;
        rp.use_arctan = false;       // start false
        rp.n_pcs = 20;
        rp.k_knn = 20;
        rp.resolution = 1.5;
        rp.mix_alpha = 0.7;
        rp.min_child_size_abs = 10;
        rp.min_child_size_frac_parent = 0.005;
        rp.seed = 12277;
        rp.verbose = (rank == 0);

        auto labels_refined_local = rarecell::refine_pca_cosine_mpi(
            MPI_COMM_WORLD,
            outs.X_local_filtered,          // CSRMatrixF32 (cells x genes) after filtering
            outs.gene_names_filtered,              // vector<string>
            labels_local,              // int32 major labels local
            panels.palma_genes,          // vector<string>
            &G_mix,                    // global candidate graph
            rp
        );
        std::string out_csv_refined = cfg.output_folder + "/leiden_labels_refined.csv";

        gather_and_write_labels_csv(MPI_COMM_WORLD, labels_refined_local, R.cell_ids_local, out_csv_refined);
    }
    catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] Error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Finalize();
    return 0;
}
