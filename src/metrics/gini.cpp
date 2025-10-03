#include "metrics/gini.hpp"

#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rarecell {

    // Exact Gini using integer histograms.
    // For each gene g, let N be total kept cells (global), S = sum of raw counts,
    // and y_(i) be the values sorted asc with zeros included.
    // Using the identity:
    //   G = ( sum_{i=1..N} (2*i - N - 1) * y_(i) ) / ( N * S )
    //
    // With integer histograms, we avoid sorting:
    // walk values v = 1..vmax, keep a global cumulative position 'pos_before' that
    // starts at Z (zeros), where Z = N - nnz_total(g). For each value v with count c:
    //   sum_i_in_bin = c*pos_before + c*(c+1)/2
    //   contribution = v * ( 2*sum_i_in_bin - (N+1)*c )
    // Accumulate over bins ascending by v.

    struct LocalGeneHist {
        // only positive values recorded: value -> count
        std::unordered_map<int, long long> by_value;
        long long nnz = 0;     // number of positive entries seen locally
        double    sum = 0.0;   // sum of raw counts locally
        int       vmax = 0;    // max positive count value seen locally
    };

    GiniResult compute_gini_raw_int_rank0(MPI_Comm comm,
        const H5ADReadResult& R,
        const FilterOutputs& outs) {
        GiniResult res;
        int rank = 0, world = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &world);

        const auto& Xin = R.X_local;                 // RAW counts (float, integral values)
        const auto& keepRows = outs.kept_local_rows;      // local row ids to keep
        const auto& remap = outs.col_remap_final;      // old global col -> new kept col (>=0) or -1
        const int64_t n_keep_genes = static_cast<int64_t>(outs.gene_names_filtered.size());

        // Global kept cell count
        long long N_local = static_cast<long long>(keepRows.size());
        long long N_total = 0;
        MPI_Allreduce(&N_local, &N_total, 1, MPI_LONG_LONG, MPI_SUM, comm);
        res.N_total = static_cast<int64_t>(N_total);

        if (n_keep_genes == 0 || N_total == 0) {
            // Nothing to compute
            if (rank == 0) res.gini.assign(0, 0.0f);
            return res;
        }

        // 1) Build local per-gene integer histograms (only for kept cells & kept genes)
        std::vector<LocalGeneHist> H(static_cast<std::size_t>(n_keep_genes));

        for (std::size_t i = 0; i < keepRows.size(); ++i) {
            int64_t r = keepRows[i];               // local row id in Xin
            int64_t s = Xin.indptr[(std::size_t)r];
            int64_t e = Xin.indptr[(std::size_t)r + 1];
            for (int64_t p = s; p < e; ++p) {
                int32_t oldc = Xin.indices[(std::size_t)p];
                int32_t newc = remap[(std::size_t)oldc];
                if (newc < 0) continue;              // gene not kept

                // RAW integer counts — Xin.data is float but holds integral counts
                float  vf = Xin.data[(std::size_t)p];
                if (vf <= 0.0f) continue;            // safety (CSR shouldn't have zeros)
                int    v = static_cast<int>(std::llround(static_cast<double>(vf)));
                if (v <= 0) continue;

                auto& G = H[(std::size_t)newc];
                ++G.nnz;
                G.sum += static_cast<double>(v);
                if (v > G.vmax) G.vmax = v;
                // update histogram
                auto it = G.by_value.find(v);
                if (it == G.by_value.end()) G.by_value.emplace(v, 1);
                else                        ++(it->second);
            }
        }

        // 2) Reduce per-gene aggregates needed on rank 0: global nnz, sum, vmax.
        std::vector<long long> nnz_local(n_keep_genes, 0), nnz_global(n_keep_genes, 0);
        std::vector<double>    sum_local(n_keep_genes, 0.0), sum_global(n_keep_genes, 0.0);
        std::vector<int>       vmax_local(n_keep_genes, 0), vmax_global(n_keep_genes, 0);

        for (int64_t g = 0; g < n_keep_genes; ++g) {
            nnz_local[(std::size_t)g] = H[(std::size_t)g].nnz;
            sum_local[(std::size_t)g] = H[(std::size_t)g].sum;
            vmax_local[(std::size_t)g] = H[(std::size_t)g].vmax;
        }

        MPI_Allreduce(nnz_local.data(), nnz_global.data(), (int)n_keep_genes, MPI_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(sum_local.data(), sum_global.data(), (int)n_keep_genes, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(vmax_local.data(), vmax_global.data(), (int)n_keep_genes, MPI_INT, MPI_MAX, comm);

        // 3) For each gene, MPI_Reduce the (0..vmax) counts to rank 0 and compute exact Gini there.
        if (rank == 0) res.gini.assign((std::size_t)n_keep_genes, 0.0f);

        for (int64_t g = 0; g < n_keep_genes; ++g) {
            const int vmax = vmax_global[(std::size_t)g];
            const int L = vmax + 1; // indices 0..vmax (we won't use 0 here)
            // Local dense counts vector (only positive values)
            std::vector<long long> cnt_local(L, 0);
            for (const auto& kv : H[(std::size_t)g].by_value) {
                int v = kv.first;
                long long c = kv.second;
                if (v > 0 && v < L) cnt_local[(std::size_t)v] += c;
            }

            // Reduce to rank 0
            std::vector<long long> cnt_global;
            if (rank == 0) cnt_global.resize(L, 0);
            MPI_Reduce(cnt_local.data(),
                rank == 0 ? cnt_global.data() : nullptr,
                L, MPI_LONG_LONG, MPI_SUM, 0, comm);

            if (rank == 0) {
                const long long N = N_total;
                const double    S = sum_global[(std::size_t)g];
                if (N <= 0 || S <= 0.0) {
                    res.gini[(std::size_t)g] = 0.0f;
                    continue;
                }
                const long long nnz = nnz_global[(std::size_t)g];
                long long zeros = N - nnz;

                // Accumulate weighted sum over bins in ascending value order
                long long pos_before = zeros; // number of zeros precede
                long double weighted = 0.0L;

                for (int v = 1; v <= vmax; ++v) {
                    long long c = cnt_global[(std::size_t)v];
                    if (c == 0) continue;
                    // sum of indices for this bin: i = pos_before+1 .. pos_before+c
                    // sum_i = c*pos_before + c*(c+1)/2
                    long long sum_i = pos_before * c + (c * (c + 1)) / 2;
                    // contribution for (2i - N - 1): v * ( 2*sum_i - (N+1)*c )
                    long double term = (long double)v * ((long double)2 * (long double)sum_i
                        - ((long double)N + 1.0L) * (long double)c);
                    weighted += term;
                    pos_before += c;
                }

                long double denom = (long double)N * (long double)S;
                long double G = (denom > 0.0L) ? (weighted / denom) : 0.0L;

                // Numerical safety: clamp to [0, 1]
                if (G < 0.0L) G = 0.0L;
                if (G > 1.0L) G = 1.0L;

                res.gini[(std::size_t)g] = static_cast<float>(G);
            }
        }

        return res;
    }

    std::vector<std::pair<std::string, float>>
        top_k_gini(const std::vector<std::string>& gene_names,
            const std::vector<float>& gini, std::size_t k) {
        std::vector<std::pair<std::string, float>> out;
        if (gini.empty() || gene_names.size() != gini.size()) return out;

        std::vector<std::size_t> idx(gini.size());
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;

        auto cmp = [&](std::size_t a, std::size_t b) {
            if (gini[a] != gini[b]) return gini[a] > gini[b];
            return a < b;
            };
        if (k < idx.size()) {
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), cmp);
            idx.resize(k);
        }
        else {
            std::sort(idx.begin(), idx.end(), cmp);
        }

        out.reserve(idx.size());
        for (auto i : idx) out.emplace_back(gene_names[i], gini[i]);
        return out;
    }

} // namespace rarecell