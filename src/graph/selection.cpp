#include "graph/selection.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;

namespace rarecell {

    static std::vector<int> top_k_indices_desc(const std::vector<float>& score, int K) {
        const int n = (int)score.size();
        if (K <= 0 || n <= 0) return {};
        K = std::min(K, n);

        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);

        auto getv = [&](int i)->float {
            float v = score[(size_t)i];
            if (std::isnan(v) || std::isinf(v)) return -std::numeric_limits<float>::infinity();
            return v;
            };
        auto cmp = [&](int a, int b) {
            float va = getv(a), vb = getv(b);
            if (va != vb) return va > vb;
            return a < b;
            };

        if (K < n) {
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), cmp);
            idx.resize(K);
        }
        else {
            std::sort(idx.begin(), idx.end(), cmp);
        }
        return idx;
    }

    static void write_list(const fs::path& path, const std::vector<std::string>& genes) {
        std::error_code ec; fs::create_directories(path.parent_path(), ec);
        std::ofstream f(path, std::ios::out | std::ios::trunc);
        if (!f) return;
        for (const auto& g : genes) f << g << "\n";
    }

    FeaturePanels select_feature_panels_rank0_bcast(
        MPI_Comm comm,
        const FilterOutputs& outs,
        const std::vector<float>& gini_detrended_rank0,
        const std::vector<float>& fano_allranks,
        const std::vector<float>& palma_detrended_rank0,
        const FilterConfig& cfg)
    {
        FeaturePanels P;
        int rank = 0, world = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &world);

        const int G = (int)outs.gene_names_filtered.size();

        // Clamp K's to [0, G]
        int Kg = std::min(std::max(0, cfg.gini_nfeatures), G);
        int Kf = std::min(std::max(0, cfg.fano_nfeatures), G);
        int Kp = std::min(std::max(0, cfg.palma_nfeatures), G);

        // Rank 0 selects
        std::vector<int> idx_g, idx_f, idx_p;
        if (rank == 0) {
            if ((int)gini_detrended_rank0.size() != G)
                throw std::runtime_error("gini_detrended size != #kept genes");
            if ((int)fano_allranks.size() != G)
                throw std::runtime_error("fano size != #kept genes");
            if ((int)palma_detrended_rank0.size() != G)
                throw std::runtime_error("palma_detrended size != #kept genes");

            idx_g = top_k_indices_desc(gini_detrended_rank0, Kg);
            idx_f = top_k_indices_desc(fano_allranks, Kf);
            idx_p = top_k_indices_desc(palma_detrended_rank0, Kp);
        }

        // Broadcast sizes
        int sz_g = (rank == 0 ? (int)idx_g.size() : 0);
        int sz_f = (rank == 0 ? (int)idx_f.size() : 0);
        int sz_p = (rank == 0 ? (int)idx_p.size() : 0);
        MPI_Bcast(&sz_g, 1, MPI_INT, 0, comm);
        MPI_Bcast(&sz_f, 1, MPI_INT, 0, comm);
        MPI_Bcast(&sz_p, 1, MPI_INT, 0, comm);

        // Broadcast indices
        if (rank != 0) { idx_g.resize(sz_g); idx_f.resize(sz_f); idx_p.resize(sz_p); }
        if (sz_g > 0) MPI_Bcast(idx_g.data(), sz_g, MPI_INT, 0, comm);
        if (sz_f > 0) MPI_Bcast(idx_f.data(), sz_f, MPI_INT, 0, comm);
        if (sz_p > 0) MPI_Bcast(idx_p.data(), sz_p, MPI_INT, 0, comm);

        // Fill outputs on all ranks
        P.gini_idx = std::move(idx_g);
        P.fano_idx = std::move(idx_f);
        P.palma_idx = std::move(idx_p);

        P.gini_genes.reserve(P.gini_idx.size());
        for (int i : P.gini_idx)  P.gini_genes.push_back(outs.gene_names_filtered[(size_t)i]);
        P.fano_genes.reserve(P.fano_idx.size());
        for (int i : P.fano_idx)  P.fano_genes.push_back(outs.gene_names_filtered[(size_t)i]);
        P.palma_genes.reserve(P.palma_idx.size());
        for (int i : P.palma_idx) P.palma_genes.push_back(outs.gene_names_filtered[(size_t)i]);

        // Rank 0 writes lists
        if (rank == 0) {
            fs::path outdir = cfg.output_folder.empty() ? fs::path(".") : fs::path(cfg.output_folder);
            if (!P.gini_genes.empty())
                write_list(outdir / ("features_gini_detrended_top" + std::to_string((int)P.gini_genes.size()) + ".txt"),
                    P.gini_genes);
            if (!P.fano_genes.empty())
                write_list(outdir / ("features_fano_top" + std::to_string((int)P.fano_genes.size()) + ".txt"),
                    P.fano_genes);
            if (!P.palma_genes.empty())
                write_list(outdir / ("features_palma_detrended_top" + std::to_string((int)P.palma_genes.size()) + ".txt"),
                    P.palma_genes);
        }

        return P;
    }

} // namespace rarecell