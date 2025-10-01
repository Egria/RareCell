#include "metrics/fano.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rarecell {

    FanoResult compute_fano_all_genes(MPI_Comm comm, const FilterOutputs& outs) {
        FanoResult R;

        // Dimensions
        const auto& X = outs.X_local_filtered;
        const int64_t n_rows_local = X.row1 - X.row0;     // kept local cells
        const int64_t n_genes = X.n_cols;            // kept genes (global-identical ordering)

        // Edge case: nothing kept
        long long N_local_ll = static_cast<long long>(n_rows_local);
        long long N_total_ll = 0;
        MPI_Allreduce(&N_local_ll, &N_total_ll, 1, MPI_LONG_LONG, MPI_SUM, comm);
        R.N_total = static_cast<int64_t>(N_total_ll);
        if (n_genes == 0 || R.N_total == 0) {
            R.fano.assign(0, 0.0f);
            return R;
        }

        // Local accumulators per gene: S1 = sum x, S2 = sum x^2
        std::vector<double> S1_local((std::size_t)n_genes, 0.0);
        std::vector<double> S2_local((std::size_t)n_genes, 0.0);

        // Single pass over local nnz
        for (int64_t r = 0; r < n_rows_local; ++r) {
            const int64_t s = X.indptr[(std::size_t)r];
            const int64_t e = X.indptr[(std::size_t)r + 1];
            for (int64_t p = s; p < e; ++p) {
                const int32_t g = X.indices[(std::size_t)p];
                const float   v = X.data[(std::size_t)p];
                S1_local[(std::size_t)g] += (double)v;
                S2_local[(std::size_t)g] += (double)v * (double)v;
            }
        }

        // Global reduce
        std::vector<double> S1_global((std::size_t)n_genes, 0.0);
        std::vector<double> S2_global((std::size_t)n_genes, 0.0);
        MPI_Allreduce(S1_local.data(), S1_global.data(), (int)n_genes, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(S2_local.data(), S2_global.data(), (int)n_genes, MPI_DOUBLE, MPI_SUM, comm);

        // Compute Fano per gene (exact, population variance)
        R.fano.resize((std::size_t)n_genes);
        const double N = (double)R.N_total;
        for (int64_t g = 0; g < n_genes; ++g) {
            const double mu = S1_global[(std::size_t)g] / N;
            if (mu <= 0.0) {
                R.fano[(std::size_t)g] = 0.0f; // all zeros -> define Fano = 0
            }
            else {
                double ex2 = S2_global[(std::size_t)g] / N;
                double var = ex2 - mu * mu;
                if (var < 0.0) var = 0.0;          // numeric guard
                R.fano[(std::size_t)g] = static_cast<float>(var / mu);
            }
        }

        return R;
    }

    std::vector<std::pair<std::string, float>>
        top_k_fano(const std::vector<std::string>& gene_names,
            const std::vector<float>& fano, std::size_t k) {
        const std::size_t n = fano.size();
        k = std::min(k, n);

        // Indices 0..n-1
        std::vector<std::size_t> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = i;

        // Partial sort by descending fano
        auto cmp = [&](std::size_t a, std::size_t b) {
            if (fano[a] != fano[b]) return fano[a] > fano[b];
            return a < b;
            };
        if (k < n) {
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), cmp);
            idx.resize(k);
        }
        else {
            std::sort(idx.begin(), idx.end(), cmp);
        }

        std::vector<std::pair<std::string, float>> out;
        out.reserve(idx.size());
        for (auto i : idx) {
            out.emplace_back(gene_names[i], fano[i]);
        }
        return out;
    }

} // namespace rarecell
