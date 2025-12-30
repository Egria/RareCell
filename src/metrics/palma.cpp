#include "metrics/palma.hpp"
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>  // optional, for rank-0 stats

namespace rarecell {

    struct LocalGeneHistP {
        std::unordered_map<int, long long> by_value; // v -> count (v>=1)
        long long nnz = 0;
        double    sum = 0.0;
        int       vmax = 0;
    };

    static inline long long ceil_ll(double x) { return (long long)std::ceil(x); }

    // -----------------------------------------------------------------------------
    // Optimized distributed Palma:
    //   * Allreduce scalars (N, nnz_g, sum_g, vmax_g)
    //   * Early-out (EXACT) when winsor=0 and zeros >= lower*N and nnz <= upper*N
    //   * Batch small histograms (vmax+1 <= L_CAP) per owner to cut #collectives
    //   * Fallback to per-gene reduce for large histograms
    // -----------------------------------------------------------------------------
    PalmaResult compute_palma_raw_int_distributed_batched_rank0(MPI_Comm comm,
        const H5ADReadResult& R,
        const FilterOutputs& outs,
        const FilterConfig& cfg)
    {
        PalmaResult res;
        int rank = 0, world = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &world);

        const auto& Xin = R.X_local;            // RAW integer counts (float-stored)
        const auto& keepRows = outs.kept_local_rows; // local kept rows
        const auto& remap = outs.col_remap_final; // old col -> kept col
        const int64_t G = (int64_t)outs.gene_names_filtered.size();

        // Global kept-cell count (N)
        long long N_local = (long long)keepRows.size();
        long long N_total = 0;
        MPI_Allreduce(&N_local, &N_total, 1, MPI_LONG_LONG, MPI_SUM, comm);
        res.N_total = (int64_t)N_total;

        if (G == 0 || N_total == 0) {
            if (rank == 0) res.palma.assign(0, 0.0f);
            return res;
        }

        // ---------------- 1) Build local integer histograms ----------------
        std::vector<LocalGeneHistP> H((size_t)G);

        for (size_t i = 0; i < keepRows.size(); ++i) {
            int64_t r = keepRows[i];
            int64_t s = Xin.indptr[(size_t)r];
            int64_t e = Xin.indptr[(size_t)r + 1];
            for (int64_t p = s; p < e; ++p) {
                int32_t oldc = Xin.indices[(size_t)p];
                int32_t newc = remap[(size_t)oldc];
                if (newc < 0) continue;
                float vf = Xin.data[(size_t)p];
                if (vf <= 0.0f) continue;
                int v = (int)std::llround((double)vf);
                if (v <= 0) continue;
                auto& gh = H[(size_t)newc];
                ++gh.nnz;
                gh.sum += (double)v;
                if (v > gh.vmax) gh.vmax = v;
                auto it = gh.by_value.find(v);
                if (it == gh.by_value.end()) gh.by_value.emplace(v, 1);
                else ++(it->second);
            }
        }

        // ---------------- 2) Allreduce scalars per gene --------------------
        std::vector<long long> nnz_local(G, 0), nnz_global(G, 0);
        std::vector<double>    sum_local(G, 0.0), sum_global(G, 0.0);
        std::vector<int>       vmax_local(G, 0), vmax_global(G, 0);
        for (int64_t g = 0; g < G; ++g) {
            nnz_local[(size_t)g] = H[(size_t)g].nnz;
            sum_local[(size_t)g] = H[(size_t)g].sum;
            vmax_local[(size_t)g] = H[(size_t)g].vmax;
        }
        MPI_Allreduce(nnz_local.data(), nnz_global.data(), (int)G, MPI_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(sum_local.data(), sum_global.data(), (int)G, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(vmax_local.data(), vmax_global.data(), (int)G, MPI_INT, MPI_MAX, comm);

        // Tail targets and knobs
        const long long N = N_total;
        const long long need_bottom = ceil_ll((double)cfg.palma_lower * (double)N);
        const long long need_top = ceil_ll((double)cfg.palma_upper * (double)N);
        const double    alpha = (double)cfg.palma_alpha;
        const double    winsor_p = (double)cfg.palma_winsor;

        // ---------------- 3) Early-out (exact, when safe) ------------------
        // Safe only when winsor_p == 0 (winsorization could change tails).
        // Condition: zeros >= need_bottom AND nnz <= need_top
        //   => S_lower = 0 and S_upper = S  (no histogram needed)
        std::vector<uint8_t> need_hist((size_t)G, 1);
        std::vector<float>   palma_local((size_t)G, 0.0f); // each rank will set only owned genes later

        long long early_out_cnt = 0;
        if (winsor_p == 0.0) {
            for (int64_t g = 0; g < G; ++g) {
                const double S = sum_global[(size_t)g];
                if (S <= 0.0) { // all zero -> neutral
                    // Owner will set Palma=1; mark as no-hist
                    need_hist[(size_t)g] = 0;
                    ++early_out_cnt;
                    continue;
                }
                const long long nnz = nnz_global[(size_t)g];
                const long long zeros = N - nnz;
                const bool bottom_zero = (zeros >= need_bottom);
                const bool top_all = (nnz <= need_top);
                if (bottom_zero && top_all) {
                    // Palma = (S/S + a) / (0/S + a) = (1+a) / (a)
                    double P = ((1.0 + alpha) / (alpha > 0.0 ? alpha : 1.0)); // guard alpha=0 => infinite; choose finite fallback
                    if (!std::isfinite(P)) P = 1.0;
                    // Only the owner will contribute this entry
                    int owner = (int)(g % world);
                    if (rank == owner) palma_local[(size_t)g] = (float)P;
                    need_hist[(size_t)g] = 0;
                    ++early_out_cnt;
                }
            }
        }

        // ---------------- 4) Prepare batched reductions --------------------
        // Heuristic caps (tuned for low latency): small histograms go in batches.
        const int   L_CAP = 128;          // max (vmax+1) included in batches
        const size_t TARGET_BYTES = 1 << 20;      // ~1 MiB target payload per batch
        const size_t ELEM_SIZE = sizeof(long long);
        const size_t ELEMS_PER_GENE = (size_t)L_CAP;
        const size_t GENES_PER_BATCH = std::max<size_t>(1, TARGET_BYTES / (ELEMS_PER_GENE * ELEM_SIZE));

        // For each owner, collect indices of genes that: (need_hist==1) and (vmax+1 <= L_CAP)
        std::vector<std::vector<int64_t>> owner_small(world), owner_large(world);
        for (int64_t g = 0; g < G; ++g) {
            if (!need_hist[(size_t)g]) continue;
            int owner = (int)(g % world);
            const int L = vmax_global[(size_t)g] + 1;
            if (L <= L_CAP) owner_small[(size_t)owner].push_back(g);
            else            owner_large[(size_t)owner].push_back(g);
        }

        // ---------------- 5) Process SMALL genes in batches ----------------
        // For each owner, submit batches of size <= GENES_PER_BATCH. Each reduction reduces a
        // contiguous buffer [batch_size * L_CAP] of long long counts (padded).
        auto sum_bottom = [&](const long long zeros, const long long nnz,
            const long long* cnt, int L) -> double {
                if (need_bottom <= 0) return 0.0;
                if (zeros >= need_bottom) return 0.0;
                long long rem = need_bottom - zeros;
                double acc = 0.0;
                for (int v = 1; v < L && rem > 0; ++v) {
                    long long c = cnt[v];
                    if (!c) continue;
                    if (c <= rem) { acc += (double)c * (double)v; rem -= c; }
                    else { acc += (double)rem * (double)v; rem = 0; }
                }
                return acc;
            };
        auto sum_top = [&](const long long nnz, const long long* cnt, int L, double S_used)->double {
            if (need_top <= 0) return 0.0;
            if (nnz <= need_top) return S_used;
            long long rem = need_top;
            double acc = 0.0;
            for (int v = L - 1; v >= 1 && rem > 0; --v) {
                long long c = cnt[v];
                if (!c) continue;
                if (c <= rem) { acc += (double)c * (double)v; rem -= c; }
                else { acc += (double)rem * (double)v; rem = 0; }
            }
            return acc;
            };

        // Helper: winsorize upper tail (on owner)
        auto winsorize_upper = [&](long long zeros, const long long nnz, long long* cnt, int L, double& S_used) {
            if (winsor_p <= 0.0) return;
            long long q_high = (long long)std::ceil((1.0 - winsor_p) * (double)N);
            long long cum = zeros;
            int v_high = 0;
            for (int v = 1; v < L; ++v) {
                cum += cnt[v];
                if (cum >= q_high) { v_high = v; break; }
            }
            if (v_high == 0 && q_high <= zeros) {
                // collapse positives to zero
                for (int v = 1; v < L; ++v) cnt[v] = 0;
                S_used = 0.0;
            }
            else if (v_high > 0) {
                long long move = 0;
                double delta_sum = 0.0;
                for (int v = v_high + 1; v < L; ++v) {
                    long long c = cnt[v];
                    if (!c) continue;
                    move += c;
                    delta_sum += (double)c * (double)(v_high - v);
                    cnt[v] = 0;
                }
                if (move > 0) {
                    cnt[v_high] += move;
                    S_used += delta_sum;
                }
            }
            };

        // Batch loop
        for (int owner = 0; owner < world; ++owner) {
            auto& genes = owner_small[(size_t)owner];
            size_t pos = 0;
            while (pos < genes.size()) {
                const size_t batch_sz = std::min(GENES_PER_BATCH, genes.size() - pos);

                // Prepare send buffer of size batch_sz * L_CAP
                std::vector<long long> sendbuf(batch_sz * (size_t)L_CAP, 0);
                for (size_t b = 0; b < batch_sz; ++b) {
                    const int64_t g = genes[pos + b];
                    const int L = std::min(L_CAP, vmax_global[(size_t)g] + 1);
                    // Fill this gene's row (offset = b*L_CAP)
                    auto row = &sendbuf[b * (size_t)L_CAP];
                    for (const auto& kv : H[(size_t)g].by_value) {
                        int v = kv.first; long long c = kv.second;
                        if (v > 0 && v < L) row[v] += c;
                    }
                }

                // Reduce to owner
                std::vector<long long> recvbuf;
                if (rank == owner) recvbuf.resize(sendbuf.size(), 0);
                MPI_Reduce(sendbuf.data(),
                    rank == owner ? recvbuf.data() : nullptr,
                    (int)sendbuf.size(), MPI_LONG_LONG, MPI_SUM, owner, comm);

                // Owner computes Palmas for these genes
                if (rank == owner) {
                    for (size_t b = 0; b < batch_sz; ++b) {
                        const int64_t g = genes[pos + b];
                        const int L = std::min(L_CAP, vmax_global[(size_t)g] + 1);
                        long long* cnt = &recvbuf[b * (size_t)L_CAP];

                        const double S_raw = sum_global[(size_t)g];
                        const long long nnz = nnz_global[(size_t)g];
                        const long long zeros = N - nnz;

                        double Palma = 1.0;
                        if (S_raw > 0.0 && N > 0) {
                            double S_used = S_raw;
                            winsorize_upper(zeros, nnz, cnt, L, S_used);
                            const double S_lower = sum_bottom(zeros, nnz, cnt, L);
                            const double S_upper = sum_top(nnz, cnt, L, S_used);
                            if (S_used > 0.0) {
                                const double num = (S_upper / S_used) + alpha;
                                const double den = (S_lower / S_used) + alpha;
                                Palma = (den > 0.0) ? (num / den) : 1.0;
                            }
                            else {
                                Palma = 1.0;
                            }
                            if (!std::isfinite(Palma)) Palma = 1.0;
                        }
                        palma_local[(size_t)g] = (float)Palma;
                    }
                }

                pos += batch_sz;
            }
        }

        // ---------------- 6) Process LARGE histograms one-by-one ----------
        for (int owner = 0; owner < world; ++owner) {
            for (int64_t g : owner_large[(size_t)owner]) {
                const int L = vmax_global[(size_t)g] + 1;
                // Local dense row for this gene
                std::vector<long long> cnt_local((size_t)L, 0);
                for (const auto& kv : H[(size_t)g].by_value) {
                    int v = kv.first; long long c = kv.second;
                    if (v > 0 && v < L) cnt_local[(size_t)v] += c;
                }
                std::vector<long long> cnt_owner;
                if (rank == owner) cnt_owner.resize((size_t)L, 0);

                MPI_Reduce(cnt_local.data(),
                    rank == owner ? cnt_owner.data() : nullptr,
                    L, MPI_LONG_LONG, MPI_SUM, owner, comm);

                if (rank == owner) {
                    const double S_raw = sum_global[(size_t)g];
                    const long long nnz = nnz_global[(size_t)g];
                    const long long zeros = N - nnz;
                    double Palma = 1.0;
                    if (S_raw > 0.0 && N > 0) {
                        double S_used = S_raw;
                        winsorize_upper(zeros, nnz, cnt_owner.data(), L, S_used);
                        const double S_lower = sum_bottom(zeros, nnz, cnt_owner.data(), L);
                        const double S_upper = sum_top(nnz, cnt_owner.data(), L, S_used);
                        if (S_used > 0.0) {
                            const double num = (S_upper / S_used) + alpha;
                            const double den = (S_lower / S_used) + alpha;
                            Palma = (den > 0.0) ? (num / den) : 1.0;
                        }
                        else {
                            Palma = 1.0;
                        }
                        if (!std::isfinite(Palma)) Palma = 1.0;
                    }
                    palma_local[(size_t)g] = (float)Palma;
                }
            }
        }

        // ---------------- 7) Final gather to rank 0 -----------------------
        std::vector<float> palma_rank0;
        if (rank == 0) palma_rank0.resize((size_t)G, 0.0f);
        MPI_Reduce(palma_local.data(),
            rank == 0 ? palma_rank0.data() : nullptr,
            (int)G, MPI_FLOAT, MPI_SUM, 0, comm);

        if (rank == 0) {
            res.palma = std::move(palma_rank0);

            // Optional: a quick summary so you can see the optimization effect
            long long need_hist_cnt = 0;
            for (auto v : need_hist) if (v) ++need_hist_cnt;
            std::cout << "[palma] early_out=" << early_out_cnt
                << " need_hist=" << need_hist_cnt
                << " (small batched=" << owner_small[0].size()
                << " + ... per owner; large total=";
            size_t tot_large = 0;
            for (int o = 0; o < world; ++o) tot_large += owner_large[(size_t)o].size();
            std::cout << tot_large << ")\n";
        }

        return res;
    }


    std::vector<std::pair<std::string, float>>
        top_k_palma(const std::vector<std::string>& gene_names,
            const std::vector<float>& palma, std::size_t k) {
        std::vector<std::pair<std::string, float>> out;
        if (palma.empty() || gene_names.size() != palma.size()) return out;

        std::vector<size_t> idx(palma.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;

        auto cmp = [&](size_t a, size_t b) {
            if (palma[a] != palma[b]) return palma[a] > palma[b];
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
        for (auto i : idx) out.emplace_back(gene_names[i], palma[i]);
        return out;
    }

} // namespace rarecell